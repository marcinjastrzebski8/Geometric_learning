import pytest
import torch
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane import RY

from quanvolution import EquivariantQuanvolution2DTorchLayer, create_structured_patches
from geometric_classifier import BasicClassifierTorch
from ansatze import SimpleAnsatz1

n_layers = 4
equiv_quanv_circuits = [BasicClassifierTorch(
    feature_map='RotEmbedding', ansatz=SimpleAnsatz1, size=9, measurement=qml.PauliZ(4)) for i in range(4)]
equiv_circuits_properties = [
    {'n_layers': n_layers, 'embedding_pauli': RY} for i in range(4)]
some_first_equiv_layer = EquivariantQuanvolution2DTorchLayer({'size': 4},
                                                             True,
                                                             equiv_quanv_circuits,
                                                             equiv_circuits_properties,
                                                             (9, 9),
                                                             3,
                                                             3,
                                                             [{'params': (n_layers, n_layers*18)}])


input_9x9 = torch.rand(1, 1, 9, 9)
# TODO: check all rotations
rotated_input = input_9x9.rot90(1, (2, 3))
print(input_9x9)
print(rotated_input)


def test_create_structured_patches_works_correct():
    structured_patches = create_structured_patches(patches)


def test_quanv_first_layer_equivariant():
    output = some_first_equiv_layer(input_9x9)
    rotated_output = some_first_equiv_layer(rotated_input)
    plot_output_channels(input_9x9, output, 'output')
    plot_output_channels(rotated_input, rotated_output, 'rotated_output')
    print(output, output.shape)
    print(rotated_output, output.shape)
    assert torch.all(rotated_output.eq(output.rot90(1, (3, 4))))


def plot_output_channels(input, output, name):
    fig, ax = plt.subplots(2, 4)
    ax[0][0].imshow(input[0][0].detach().numpy())
    for ax_id in range(4):
        ax[1][ax_id].imshow(
            output[0][0][ax_id].detach().numpy(), vmin=0, vmax=1)

    plt.savefig(name, dpi=300)
