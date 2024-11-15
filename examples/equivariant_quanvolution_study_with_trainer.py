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
import tempfile

import ray
from ray import train as ray_train
from ray import tune
import wandb
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
import torch
from torch import nn, optim
from pennylane import RX, RY, RZ

from src.quanvolution import EquivariantQuanvolution2DTorchLayer
from src.geometric_classifier import BasicClassifierTorch
from src.ansatze import MatchCallumAnsatz
from src.torch_architectures import ConvolutionalEQNEC
from pqc_training.trainer import NewTrainer
from data.datasets import MicrobooneTrainData, MicrobooneValData

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


def main(json_config):
    print('json config', json_config)

    def train_ray(config):
        """
        Takes config which defines search space. Needs to be of this format for ray.
        NOTE: this is defined inside main because that way I can access arguments from json_config easily
        """
        print('THIS IS CONFIG', config)

        def prep_equiv_quanv_model(is_first_layer):
            n_layers = config['n_layers']
            n_reuploads = config['n_reuploads']
            # CHANGE TO CALLUM ANSATZ ONCE HE SENDS DETAILS
            patch_circuit = BasicClassifierTorch(feature_map='RotEmbedding',
                                                 ansatz=MatchCallumAnsatz,
                                                 size=4,
                                                 n_reuploads=n_reuploads)
            patch_circuit_properties = {
                'n_layers': n_layers, 'ansatz_block': [RY, RZ], 'embedding_pauli': RX}
            # both layers take same size in because we'll use stride=1

            if is_first_layer:
                input_channel_side_len = json_config['image_size']
                quantum_circs = [patch_circuit]
                quantum_circs_properties = [patch_circuit_properties]

            else:
                # NOTE: this is hardcoded based on filter size and stride - im sure there's a general formula
                input_channel_side_len = json_config['image_size'] - 1
                # using the same ansatz for each pose
                quantum_circs = [patch_circuit for i in range(4)]
                quantum_circs_properties = [patch_circuit_properties
                                            for i in range(4)]

            group_info = {'size': 4}
            # NOTE: parameter shape is kind of hardcoded, could do some lookup table based on ansatz
            quanv_layer = EquivariantQuanvolution2DTorchLayer(group_info,
                                                              is_first_layer,
                                                              quantum_circs,
                                                              quantum_circs_properties,
                                                              (input_channel_side_len,
                                                               input_channel_side_len),
                                                              1,
                                                              2,
                                                              [{'params': (
                                                                  n_reuploads, n_layers, 4, 2)}],
                                                              config['param_init_max_vals'])
            return quanv_layer

        criterion = nn.BCEWithLogitsLoss()
        architecture_config = {'quanv0': prep_equiv_quanv_model(True),
                               'quanv1': prep_equiv_quanv_model(False),
                               'dense_units': config['dense_units'],
                               'image_size': json_config['image_size'],
                               'use_dropout0': config['use_dropout0'],
                               'use_dropout1': config['use_dropout1'],
                               'dropout0': config['dropout0'],
                               'dropout1': config['dropout1']}
        model = ConvolutionalEQNEC(architecture_config)
        optimiser = optim.Adam(model.parameters(), lr=config['lr'])
        # TODO: CHECK HOW MANY EPOCHS CALLUM DID
        train_model(model, MicrobooneTrainData(), criterion,
                    optimiser, json_config['n_epochs'], json_config['batch_size'], val_dataset=MicrobooneValData()[:])

    # search space params
    lr = tune.loguniform(0.001, 0.1)
    dense_units = tune.choice([[128, 32], [8, 8]])
    param_init_max_vals = tune.choice([0.001, 0.1, np.pi/4, np.pi/2, 2*np.pi])
    n_reuploads = tune.choice([1])
    n_layers = tune.choice([1, 2, 3, 4])
    dropout_bool = tune.choice([True, False])
    dropout_amount = tune.uniform(1e-4, 0.5)

    search_space = {'lr': lr,
                    'dense_units': dense_units,
                    'param_init_max_vals': param_init_max_vals,
                    'n_reuploads': n_reuploads,
                    'n_layers': n_layers,
                    'use_dropout0': dropout_bool,
                    'use_dropout1': dropout_bool,
                    'dropout0': dropout_amount,
                    'dropout1': dropout_amount}

    scheduler = ASHAScheduler(
        time_attr='training_iteration', grace_period=5)

    # set up ray with weights and biases logging
    ray.init(num_cpus=json_config['n_cpus'])
    trainable_with_resources = tune.with_resources(
        train_ray, {'cpu': json_config['n_cpus_per_model']})
    run_config = ray_train.RunConfig(storage_path=path_to_package, name=json_config['output_models_dir'], callbacks=[
        WandbLoggerCallback(project=json_config['output_models_dir'])], checkpoint_config=ray_train.CheckpointConfig(
        checkpoint_score_attribute='loss'))
    tuner = tune.Tuner(trainable_with_resources, param_space=search_space, tune_config=tune.TuneConfig(
        metric='loss', mode='min', num_samples=json_config['n_models'], scheduler=scheduler), run_config=run_config)
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
