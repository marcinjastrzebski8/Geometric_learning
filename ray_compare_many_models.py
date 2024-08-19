"""
Perform a ray study of many models on the small 2x2 classification toy dataset.

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
from src.geometric_classifier import GeometricClassifierJax
from src.twirling import c4_on_4_qubits
from src.utils import loss_dict
from pqc_training.trainer import JaxTrainer
import argparse
import json
import pennylane as qml
import itertools

api = wandb.Api()
path_to_package = Path('.').absolute()

def main(json_config):

    #keeping all these constant as I want this to be a study of standard vs geo models not search for best ever model
    lr = 0.001
    n_epochs = 50
    batch_size = 50
    train_size = 500

    #I don't use the validation option as im not interested in the validation loss
    #but i think things will break if i dont define these
    validation_size = 1 
    eval_interval = 50

    image_size = 2
    num_wires = image_size*image_size

    # produce data
    data = SymmetricDatasetJax(train_size + validation_size, image_size)

    # define trainable which searches through hyperparameters
    def train_ray(config):
        """
        Takes config which defines search space. Needs to be of this format for ray.
        """

        # NOTE: params_per_layer is tied to the ansatz used and hardcoded here
        init_params = np.random.uniform(
            0, 1, (config['n_layers'], 8))

        model = GeometricClassifierJax('RotEmbeddingWEnt', 'GeneralCascadingAnsatz', num_wires, config['twirled_bool'], c4_on_4_qubits)
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


    # define search space for hyperparameters
    single_qubit_pauli = [qml.RX, qml.RY, qml.RZ]
    two_qubit_pauli = [combo[0] + combo[1] for combo in itertools.combinations_with_replacement(['X','Y','Z'], r=2)]

    search_space = {'single_qubit_pauli': tune.choice(single_qubit_pauli), 
                    'two_qubit_pauli': tune.choice(two_qubit_pauli),
                    'embedding_pauli': tune.choice(single_qubit_pauli),
                    'n_layers': tune.randint(1, 6),
                    'twirled_bool': tune.grid_search([True,False])
                    }

    # set up ray with weights and biases logging
    ray.init(num_cpus=json_config['n_cpus'])
    scheduler = ASHAScheduler(time_attr = "training_oteration",max_t = train_size*n_epochs/batch_size)
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
