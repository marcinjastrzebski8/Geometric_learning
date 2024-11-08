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
- local_model_save_dir: str [TODO: make optional, only when not using ray]
"""

import pennylane as qml
import argparse
import itertools
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
from sklearn.metrics import accuracy_score
from pennylane import RX, RY, RZ

from src.quanvolution import EquivariantQuanvolution2DTorchLayer
from src.geometric_classifier import BasicClassifierTorch
from src.ansatze import MatchCallumAnsatz, SimpleAnsatz1

api = wandb.Api()
path_to_package = Path('.').absolute()


def train_model(model, train_dict, criterion, optimizer, epochs, batch_size=500):
    """
    Stolen from Callum and modified. Ideally I'd revisit my trainer.py module - it should encompass anything like this example.
    """
    model.train()
    # NOTE: TRAIN DATA SIZE HARDCODED
    batches = [{'data': train_dict['data']
                [i:i+batch_size], 'labels': train_dict['labels'][i:i+batch_size]} for i in range(0, 500, batch_size)]
    for epoch in range(epochs):
        print('epoch: ', epoch)
        running_loss = 0.0
        # batch loop
        for batch in batches:
            images = batch['data']
            labels = batch['labels']
            optimizer.zero_grad()
            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            with tempfile.TemporaryDirectory() as tempdir:
                ray_train.report(
                    metrics={'loss': running_loss}, checkpoint=Checkpoint.from_directory(tempdir))

        epoch_loss = running_loss / len(train_dict['data'])

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')


def evaluate_model(model, test_dict):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in zip(test_dict['data'], test_dict['labels']):
            preds = model(images)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test accuracy: {accuracy:.4f}')
    print(all_labels)
    print(all_preds)


def prep_microboone_data(which_set: str):
    """
    Microboone data, taken directly form Callum to ensure like-to-like comparison.
    """

    path_to_microboone_data = path_to_package/'data/microboone_from_callum/'

    data_file = path_to_microboone_data/f'{which_set}_data.pt'
    labels_file = path_to_microboone_data/f'{which_set}_labels.pt'

    data = torch.load(data_file)
    # need to add the channel dimension (grayscale image in this case)
    data = data.view(data.shape[0], 1, data.shape[1], data.shape[2])
    print('DATA IS SHAPE ', data.shape)
    labels = torch.load(labels_file)
    data_dict = {'data': data, 'labels': labels}

    return data_dict


def main(json_config):
    print('json config', json_config)

    def train_ray(config):
        """
        Takes config which defines search space. Needs to be of this format for ray.
        NOTE: this is defined inside main because that way I can access arguments from json_config easily
        """

        def prep_equiv_quanv_model(is_first_layer):
            n_layers = config['n_layers']
            n_reuploads = config['n_reuploads']
            # CHANGE TO CALLUM ANSATZ ONCE HE SENDS DETAILS
            patch_circuit = BasicClassifierTorch(feature_map='RotEmbedding',
                                                 ansatz=SimpleAnsatz1,
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
            quanv_layer = EquivariantQuanvolution2DTorchLayer(group_info,
                                                              is_first_layer,
                                                              quantum_circs,
                                                              quantum_circs_properties,
                                                              (input_channel_side_len,
                                                               input_channel_side_len),
                                                              1,
                                                              2,
                                                              [{'params': (n_reuploads, n_layers, 2*4)}],
                                                              config['param_init_max_vals'])
            return quanv_layer

        class EquivQuanvClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = prep_equiv_quanv_model(is_first_layer=True)
                self.conv2 = prep_equiv_quanv_model(is_first_layer=False)
                self.dropout0 = False
                self.dropout1 = False
                dense_shape_0 = config['dense_units'][0]
                dense_shape_1 = config['dense_units'][1]
                dense_input_shape = 4 * \
                    (json_config['image_size']-2)*(json_config['image_size']-2)
                if config['use_dropout0']:
                    self.dropout0 = nn.Dropout(config['dropout0'])

                if config['use_dropout1']:
                    self.dropout1 = nn.Dropout(config['dropout1'])

                # NOTE: input size is hardcoded for kernel size + stride. TODO: Could generalise.
                self.fc0 = nn.Linear(
                    dense_input_shape, dense_shape_0)
                self.fc1 = nn.Linear(
                    dense_shape_0, dense_shape_1)
                self.fc2 = nn.Linear(dense_shape_1, 1)

            def forward(self, x):
                # NOTE: Callum is using batch normalisation here which is not equivariant by default,
                # could use the escnn package to do that if needed
                x = nn.functional.relu(self.conv1(x))
                x = nn.functional.relu(self.conv2(x))
                x = x.permute(1, 0, 2, 3, 4)
                x = torch.flatten(x, 1)
                x = nn.functional.relu(self.fc0(x))
                if self.dropout0:
                    x = self.dropout0(x)
                x = nn.functional.relu(self.fc1(x))
                if self.dropout1:
                    x = self.dropout1(x)
                x = self.fc2(x)
                return x

        # trainer here
        tr_dict = prep_microboone_data('train')
        criterion = nn.BCEWithLogitsLoss()
        model = EquivQuanvClassifier()
        optimiser = optim.Adam(model.parameters(), lr=config['lr'])
        # TODO: CHECK HOW MANY EPOCHS CALLUM DID
        train_model(model, tr_dict, criterion,
                    optimiser, json_config['n_epochs'])

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
        checkpoint_score_attribute='loss', num_to_keep=5))
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
