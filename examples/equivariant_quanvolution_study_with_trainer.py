"""
Ray study of equivariant quanvolution+classical nn classifier on 21x21 microboone slices.
Top be compared with Callum's non-equivariant top performers on the same data.


Run from the terminal with one argument: path to a run config file.
The config file needs to be a .json file

Config needs to contain:
(parallelisation info)
- n_cpus: int
- n_cpus_per_model: int
- n_models: int
(results path info)
- output_models_dir: str
- local_model_save_dir: str [TODO: is this needed when using ray?]
"""

import argparse
import json
import numpy as np
from pathlib import Path
import itertools

import ray
from ray import train as ray_train
from ray import tune
import wandb
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
import torch
from torch import nn, optim
from pennylane import RX, RY, RZ, PauliZ

from src.quanvolution import EquivariantQuanvolution2DTorchLayer
from src.geometric_classifier import BasicClassifierTorch, BasicModelTorchLayer
from src.ansatze import MatchCallumAnsatz, GeometricAnsatzConstructor
from src.torch_architectures import ConvolutionalEQNEC, ConvolutionalEQEQ
# NOTE atm im restricted to working with 9-qubit representations of C4 which restrics architecture design
from src.twirling import c4_rep_on_qubits, C4On9QEquivGate1Local, C4On9QEquivGate2Local
from pqc_training.trainer import NewTrainer
from data.datasets import MicrobooneTrainData, MicrobooneValData
from .utils import calculate_image_output_shape

api = wandb.Api()
path_to_package = Path('.').absolute()


def train_model(model, dataset, criterion, optimizer, epochs, batch_size, val_dataset=None):
    """
    This changed from first study script to utilise the new trainer class.
    """
    model.train()

    trainer = NewTrainer(k_folds=1,
                         epochs=epochs,
                         eval_interval=1,
                         )
    # NOTE: TRAIN DATA SIZE HARDCODED
    trainer.train(model,
                  dataset,
                  optimizer,
                  criterion,
                  500,
                  0,
                  batch_size,
                  val_dataset,
                  True,
                  'val')


def prep_equiv_quanv_model(config, json_config, is_first_layer):
    """
    config is the hypopt config
    json_config is the run json_config
    """
    n_layers = config['n_layers']
    n_reuploads = config['n_reuploads']
    patch_circuit = BasicClassifierTorch(feature_map='RotEmbedding',
                                         ansatz=MatchCallumAnsatz,
                                         size=4,
                                         n_reuploads=n_reuploads)
    patch_circuit_properties = {
        'n_layers': n_layers, 'ansatz_block': [RY, RZ], 'embedding_pauli': RX}

    if is_first_layer:
        input_channel_side_len = json_config['image_size']
        quantum_circs = [
            patch_circuit for i in range(config['n_filters0'])]
        quantum_circs_properties = [
            patch_circuit_properties for i in range(config['n_filters0'])]
        stride = json_config['stride_quanv0']

    else:
        #NOTE: ATM SIZE OF KERNEL HARDCODED (THROUGHOUT, NOT JUST THIS LINE)
        input_channel_side_len = calculate_image_output_shape(json_config['image_size'], 2, json_config['stride_quanv0'])
        quantum_circs = [
            patch_circuit for i in range(config['n_filters1'])]
        quantum_circs_properties = [
            patch_circuit_properties for i in range(config['n_filters1'])]
        stride = json_config['stride_quanv1']

    group_info = {'size': 4}
    # NOTE: parameter shape is kind of hardcoded, could do some lookup table based on ansatz
    quanv_layer = EquivariantQuanvolution2DTorchLayer(group_info,
                                                      is_first_layer,
                                                      quantum_circs,
                                                      quantum_circs_properties,
                                                      (input_channel_side_len,
                                                       input_channel_side_len),
                                                      stride,
                                                      2,
                                                      [{'params': (
                                                          n_reuploads, n_layers, 4, 2)} for filter in quantum_circs],
                                                      config['param_init_max_vals'])
    return quanv_layer


def prep_equiv_quant_classifier(config):
    # NOTE image size is restricted by my C4 bank (I only have it for 9 qubits)
    # this restricts convolutional layer architectures
    image_size = 9
    n_layers = config['classifier_n_layers']
    n_reuploads = config['classifier_n_reuploads']
    gates_1local = config['1local_gates']
    gates_2local = config['2local_gates']

    gate_1local_instructions = [
        {'gate_placement': placement,
            'gate': gates_1local[id]}
        for id, placement in enumerate(config['1local_placements'])
    ]

    gate_2local_instructions = [
        {'gate_placement': placement,
         'pauli_word': gates_2local[id]}
        for id, placement in enumerate(config['2local_placements'])
    ]
    circuit = BasicClassifierTorch(feature_map='RotEmbedding',
                                   ansatz=GeometricAnsatzConstructor,
                                   size=image_size,
                                   n_reuploads=n_reuploads, 
                                   measurement = PauliZ(4))
    circuit_properties = {
        'n_layers': n_layers,
        'embedding_pauli': RX,
        'group_equiv_1local_gate': C4On9QEquivGate1Local,
        'group_equiv_2local_gate': C4On9QEquivGate2Local,
        'gate_1local_instructions': gate_1local_instructions,
        'gate_2local_instructions': gate_2local_instructions
    }

    weight_shapes = {'params': (n_reuploads, n_layers, len(
        gate_1local_instructions)+len(gate_2local_instructions))}
    # NOTE: using default init param values
    quant_classifier_layer = BasicModelTorchLayer(
        circuit, circuit_properties, weight_shapes)

    return quant_classifier_layer


def main(json_config):
    print('json config', json_config)
    architecture_codeword = json_config['architecture_codeword']

    def train_ray(config):
        """
        Takes config which defines search space. Needs to be of this format for ray.
        NOTE: this is defined inside main because that way I can access arguments from json_config easily
        """
        criterion = nn.BCEWithLogitsLoss()
        if architecture_codeword == 'EQNEC':
            architecture_config = {'quanv0': prep_equiv_quanv_model(config, json_config, True),
                                   'quanv1': prep_equiv_quanv_model(config, json_config, False),
                                   'dense_units': config['dense_units'],
                                   'image_size': json_config['image_size'],
                                   'use_dropout0': config['use_dropout0'],
                                   'use_dropout1': config['use_dropout1'],
                                   'dropout0': config['dropout0'],
                                   'dropout1': config['dropout1'],
                                   'n_filters1': config['n_filters1']}
            model = ConvolutionalEQNEC(architecture_config)
        elif architecture_codeword == 'EQEQ':
            architecture_config = {'quanv0': prep_equiv_quanv_model(config, json_config, True),
                                   'quanv1': prep_equiv_quanv_model(config, json_config, False),
                                   'quantum_classifier': prep_equiv_quant_classifier(config)}
            model = ConvolutionalEQEQ(architecture_config)
        else:
            raise ValueError('architecture not supported: ',
                             architecture_codeword)

        optimiser = optim.Adam(
            model.parameters(), lr=config['lr'])

        train_model(model, MicrobooneTrainData(), criterion,
                    optimiser, json_config['n_epochs'], json_config['batch_size'], val_dataset=MicrobooneValData()[:])

    # search space params
    # NOTE: NOT YET HOW TO HANDLE MULTIPLE POSSIBLE ARCHITECTURES ATM, PROBABLY SOME LOOKUP DICT
    #   FOR NOW JUST IFS

    # common hyperparams
    lr = tune.loguniform(0.001, 0.1)
    search_space = {'lr': lr}

    if architecture_codeword[:2] == 'EQ':
        # hyperparams for the quanvolution layer
        search_space['n_layers'] = tune.choice([1, 2, 3, 4])
        search_space['n_filters0'] = tune.choice([
            1, 2])
        search_space['n_filters1'] = tune.choice([
            1, 2, 3])
        search_space['n_reuploads'] = tune.choice([1,2,3])
        search_space['param_init_max_vals'] = tune.choice(
            [0.001, 0.1, np.pi/4, np.pi/2, 2*np.pi])

    elif architecture_codeword[:2] == 'EC':
        # TODO
        pass
    else:
        raise ValueError(
            'architecture codeword in the json_config is not supported: ', architecture_codeword)
    if architecture_codeword[2:] == 'NEC':
        # hyperparams for the classical classification network
        search_space['dense_units'] = tune.choice([[128, 32], [8, 8]])
        search_space['use_dropout0'] = search_space['use_dropout1'] = tune.choice([
            True, False])
        search_space['dropout0'] = search_space['dropout1'] = tune.uniform(
            1e-4, 0.5)
    elif architecture_codeword[2:] == 'EQ':
        # hyperparams for quantum classification network
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
        single_qubit_pauli = [RX, RY, RZ]
        two_qubit_pauli = [combo[0] + combo[1]
                           for combo in itertools.combinations_with_replacement(['X', 'Y', 'Z'], r=2)]

        max_gates_per_block = 4
        # this finds all combos [X,X,X,X],...,[X,Y,X,Z],...
        combos_for_1local_blocks = itertools.product(
            single_qubit_pauli, repeat=max_gates_per_block)
        combos_for_2local_blocks = itertools.product(
            two_qubit_pauli, repeat=max_gates_per_block)
        # this finds combos like [corner,],..., [corner,corner,corner,corner],...
        all_1local_placement_combos = [combo for r in range(
            1, max_gates_per_block+1) for combo in itertools.combinations_with_replacement(gate_placement_1local, r=r)]
        all_2local_placement_combos = [combo for r in range(
            1, max_gates_per_block+1) for combo in itertools.combinations_with_replacement(gate_placement_2local, r=r)]

        # maybe not the best way to ensure correct gate is used? - this is pretty much fixed by the group, right?
        # TODO: SEE IF NEEDED AT ALL
        group_equiv_1local_gate = tune.grid_search([C4On9QEquivGate1Local])
        group_equiv_2local_gate = tune.grid_search([C4On9QEquivGate2Local])

        search_space['1local_gates'] = tune.choice(combos_for_1local_blocks)
        search_space['2local_gates'] = tune.choice(combos_for_2local_blocks)
        search_space['1local_placements'] = tune.choice(
            all_1local_placement_combos)
        search_space['2local_placements'] = tune.choice(
            all_2local_placement_combos)
        search_space['classifier_n_layers'] = tune.choice([1, 2, 3, 4])
        search_space['classifier_n_reuploads'] = tune.choice([1])

    scheduler = ASHAScheduler(
        time_attr='training_iteration', grace_period=5, metric='loss', mode='min')

    # set up ray with weights and biases logging
    ray.init(num_cpus=json_config['n_cpus'])
    trainable_with_resources = tune.with_resources(
        train_ray, {'cpu': json_config['n_cpus_per_model']})
    run_config = ray_train.RunConfig(storage_path=path_to_package, name=json_config['output_models_dir'], callbacks=[
        WandbLoggerCallback(project=json_config['output_models_dir'])], checkpoint_config=ray_train.CheckpointConfig(
        checkpoint_score_attribute='loss'))
    tuner = tune.Tuner(trainable_with_resources, param_space=search_space, tune_config=tune.TuneConfig(
        num_samples=json_config['n_models'], scheduler=scheduler), run_config=run_config)
    tuner.fit()

    train_ray(json_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_config_path')
    parse_args = parser.parse_args()
    # load config
    with open(parse_args.run_config_path, 'r', encoding='utf-8') as f:
        load_config = json.load(f)
    main(load_config)
