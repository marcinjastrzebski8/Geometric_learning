"""
Perform a ray study of many models on the small classification toy dataset.

Run from the terminal with one argument: path to a run config file.
The config file needs to be a .json file

Config needs to contain:
(parallelisation info)
- n_cpus: int
- n_cpus_per_model: int
- n_models: int
(results path info)
- output_models_dir: str
- local_model_save_dir: str [TODO: make optional, only when not using ray]
"""
import wandb
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler
import pandas as pd
from pathlib import Path
import ray
from ray import train as ray_train
from ray import tune
from utils import SymmetricDatasetJax
from jax.example_libraries import optimizers
import numpy as np
from src.geometric_classifier import GeometricClassifierAutotwirlJax, BasicClassifier
from src.twirling import c4_rep_on_qubits, C4On9QEquivGate1Local, C4On9QEquivGate2Local
from src.utils import loss_dict
from src.ansatze import GeometricAnsatzConstructor
from pqc_training.trainer import JaxTrainer
import argparse
import json
import pennylane as qml
import itertools

api = wandb.Api()
path_to_package = Path('.').absolute()


def main(json_config):
    #NOTE: SET THIS TO SWITCH BETWEEN THE TWO CURRENT IMPLEMENTATIONS OF THE GEO CLASSIFIERS
    geo_classifier_implementation = json_config['geo_classifier_implementation']

    # keeping all these constant as I want this to be a study of standard vs geo models not search for best ever model
    lr = 0.001
    n_epochs = 50
    batch_size = 50
    train_size = 500

    # I don't use the validation option as im not interested in the validation loss
    # but i think things will break if i dont define these
    validation_size = 1
    eval_interval = 50

    image_size = json_config['image_size']
    #NOTE: precomputed method only supported for 3x3 for now
    num_wires = image_size*image_size

    # produce data
    data = SymmetricDatasetJax(train_size + validation_size, image_size)

    # define search space for hyperparameters
    single_qubit_pauli = [qml.RX, qml.RY]
    two_qubit_pauli = [combo[0] + combo[1]
                       for combo in itertools.combinations_with_replacement(['X', 'Y', 'Z'], r=2)]

    #for pre-twirled bank method
    gate_placement_1local = ['corner', 'side', 'centre']
    gate_placement_2local = [
        'side_centre',
        'ring_neighbours_corner',
        'ring_neighbours_side',
        'ring_second_neighbours_corner',
        'ring_second_neighbours_side',
        'ring_third_neighbours_corner',
        'ring_third_neighbours_side',
        'ring_fourth_neighbours_corner',
        'ring_fourth_neighbours_side'
    ]

    #consider layers with up to 4 1-local and 2-local gate-blocks [single gates, twirled]
    all_1local_placement_combinations_up_to_4_gates = [combo for r in range(1,4) for combo in itertools.combinations_with_replacement(gate_placement_1local, r=r)]
    all_2local_placement_combinations_up_to_4_gates = [combo for r in range(1,4) for combo in itertools.product(gate_placement_2local, repeat=r)]

    search_space_for_nontwirled_base_model = {'single_qubit_pauli': tune.choice(single_qubit_pauli),
                        'two_qubit_pauli': tune.choice(two_qubit_pauli),
                        'embedding_pauli': tune.choice(single_qubit_pauli),
                        'n_layers': tune.randint(1, 4),
                        'twirled_bool': tune.grid_search([True, False])
                        }

    if geo_classifier_implementation == 'auto_twirl':
        search_space = search_space_for_nontwirled_base_model

    elif geo_classifier_implementation == 'auto_twirl_untwirled_only':
        search_space_for_nontwirled_base_model['twirled_bool'] = tune.grid_search([False])
        search_space = search_space_for_nontwirled_base_model

    elif geo_classifier_implementation == 'precomputed_bank':
        search_space = {'n_layers': tune.randint(1, 4),
        'embedding_pauli': tune.choice(single_qubit_pauli),
        'gate_1local_placements':tune.choice(all_1local_placement_combinations_up_to_4_gates),
        'gate_2local_placements':tune.choice(all_2local_placement_combinations_up_to_4_gates),
        'group_equiv_1local_gate': tune.grid_search([C4On9QEquivGate1Local]),
        'group_equiv_2local_gate': tune.grid_search([C4On9QEquivGate2Local])
        }

    else:
        raise NotImplementedError()

    # define trainable which searches through hyperparameters
    def train_ray(config):
        """
        Takes config which defines search space. Needs to be of this format for ray.
        """

        # NOTE: HARCODING THE INVARIANT MEASUREMENT FOR NOW
        if image_size == 2:
            invariant_measurement = qml.Z(
                0)@qml.Z(1)@qml.Z(2)@qml.Z(3)
        elif image_size == 3:
            invariant_measurement = qml.Z(4)
        else:
            raise NotImplementedError('Only supporting image sizes 2 and 3 for now.')

        
        if geo_classifier_implementation in ('auto_twirl', 'auto_twirl_untwirled_only'):
            # NOTE: params_per_layer is tied to the ansatz used and hardcoded here
            params_per_layer = num_wires*2

            model = GeometricClassifierAutotwirlJax('RotEmbedding', 'GeneralCascadingAnsatz', num_wires, config['twirled_bool'], c4_rep_on_qubits, group_commuting_meas=invariant_measurement, image_size=image_size)

        elif geo_classifier_implementation == 'precomputed_bank':

            num_1local_gates = len(config['gate_1local_placements'])
            num_2local_gates = len(config['gate_2local_placements'])
            params_per_layer = num_1local_gates + num_2local_gates

            gate_1local_instructions = [
                {'gate_placement': placement, 
                'gate': np.random.choice(single_qubit_pauli,1)[0]} 
                for placement in config['gate_1local_placements']
                ]

            gate_2local_instructions = [
                {'gate_placement': placement, 
                'pauli_word':np.random.choice(two_qubit_pauli,1)[0]} 
                for placement in config['gate_2local_placements']
                ]
            
            config['gate_1local_instructions'] = gate_1local_instructions
            config['gate_2local_instructions'] = gate_2local_instructions
            
            model = BasicClassifier('RotEmbedding', GeometricAnsatzConstructor,num_wires, invariant_measurement)
        else:
            raise NotImplementedError('wrong implementation passed')

        init_params = np.random.uniform(0,1, (config['n_layers'], params_per_layer))
        
        model_fn = model.prediction_circuit

        optimiser_fn = optimizers.adam(lr)
        loss_fn = loss_dict['bce_loss']

        trainer = JaxTrainer(init_params,
                             train_size,
                             validation_size,
                             1,
                             n_epochs,
                             batch_size,
                             [],
                             eval_interval,
                             str(path_to_package)+'/'+json_config['local_model_save_dir'])

        params, history, info = trainer.train(data,
                                              model_fn,
                                              loss_fn,
                                              optimiser_fn,
                                              config,
                                              True)



    # set up ray with weights and biases logging
    ray.init(num_cpus=json_config['n_cpus'])
    scheduler = ASHAScheduler(
        time_attr="training_iteration", max_t=200, grace_period=5)
    trainable_with_resources = tune.with_resources(
        train_ray, {'cpu': json_config['n_cpus_per_model']})
    run_config = ray_train.RunConfig(storage_path=path_to_package, name=json_config['output_models_dir'], callbacks=[WandbLoggerCallback(project=json_config['output_models_dir'])], checkpoint_config=ray_train.CheckpointConfig(
        checkpoint_score_attribute='loss', num_to_keep=5))
    tuner = tune.Tuner(trainable_with_resources, param_space=search_space,
                       tune_config=tune.TuneConfig(metric='loss', mode='min', num_samples=json_config['n_models'], scheduler=scheduler), run_config=run_config)

    # run ray hyperparameter search
    tuner.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_config_path')
    parse_args = parser.parse_args()
    # load config
    with open(parse_args.run_config_path, 'r', encoding='utf-8') as f:
        load_config = json.load(f)
    main(load_config)
