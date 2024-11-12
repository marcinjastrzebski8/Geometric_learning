"""
Tests only the newest addition to pqc_training/trainer.py which was written by me only (but heavily inspired by Callum/Mohammad code).
NOTE: testing using some microboone data.
"""

import pytest
import numpy as np
from pathlib import Path
from copy import deepcopy

import torch
from torch import optim, nn
from pennylane import RX, RY, RZ

from src.torch_architectures import ConvolutionalEQNEC
from src.ansatze import MatchCallumAnsatz
from src.geometric_classifier import BasicClassifierTorch
from src.quanvolution import EquivariantQuanvolution2DTorchLayer
from pqc_training.trainer import NewTrainer
from data.datasets import MicrobooneTrainData
microboone_image_size = 21

dummy_n_reuploads = 1
dummy_n_layers = 1
dummy_lr = 0.01


def prep_dummy_model(is_first_layer):
    """
    This repurposed from equivariant_quanvolution_first_study.py.
    """
    patch_circuit = BasicClassifierTorch(feature_map='RotEmbedding',
                                         ansatz=MatchCallumAnsatz,
                                         size=4,
                                         n_reuploads=dummy_n_reuploads
                                         )
    patch_circuit_properties = {
        'n_layers': dummy_n_layers, 'ansatz_block': [RY, RZ], 'embedding_pauli': RX}
    if is_first_layer:
        input_channel_side_len = microboone_image_size
        quantum_circs = [patch_circuit]
        quantum_circs_properties = [patch_circuit_properties]

    else:
        # NOTE: this is hardcoded based on filter size and stride - im sure there's a general formula
        input_channel_side_len = microboone_image_size - 1
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
                                                          dummy_n_reuploads, dummy_n_layers, 4, 2)}],
                                                      )
    return quanv_layer


architecture_config = {
    'quanv0': prep_dummy_model(True),
    'quanv1': prep_dummy_model(False),
    'dense_units': (8, 8),
    'image_size': 21,
    'use_dropout0': True,
    'use_dropout1': True,
    'dropout0': 0.1,
    'dropout1': 0.2

}
my_full_model = ConvolutionalEQNEC(architecture_config)

optimizer = optim.Adam(my_full_model.parameters(), lr=dummy_lr)
# TODO: TEST FOR K-FOLD VALIDATION AND VARIOUS EVAL INTERVALS
savedir = Path('.').absolute() / 'test'
trainer_to_test = NewTrainer(1, 2, 1, str(savedir))


def test_save_loss_works():
    trainer_to_test.save_losses(savedir)

    train_losses_path = savedir/'train_losses.npy'
    val_losses_path = savedir/'val_losses.npy'

    assert train_losses_path.exists()
    assert val_losses_path.exists()


def test_save_model_works():
    trainer_to_test.save_model(my_full_model, str(savedir))

    model_path = savedir/'model_state.pth'

    assert model_path.exists()


def test_saved_model_makes_sense():
    model_state_dict_before_saving = deepcopy(my_full_model.state_dict())

    trainer_to_test.save_model(my_full_model, str(savedir))

    model_path = savedir/'model_state.pth'

    my_full_model.load_state_dict(
        torch.load(str(model_path)))
    model_state_dict_after_saving = my_full_model.state_dict()
    for key in model_state_dict_after_saving:
        assert torch.allclose(
            model_state_dict_before_saving[key], model_state_dict_after_saving[key])


# TODO: TRY DIFFERENT BATCH SIZES
def test_train_throws_no_error():
    """
    This will pass if no exceptions raised.
    """
    trainer_to_test.train(my_full_model,
                          MicrobooneTrainData(),
                          optim.Adam(my_full_model.parameters(), dummy_lr),
                          nn.BCEWithLogitsLoss(),
                          50,
                          0,
                          10,
                          )
