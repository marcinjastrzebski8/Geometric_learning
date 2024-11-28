"""
NOTE: test written with the C4 group in mind. If studying different group, need to change code.
"""
import functools
import math

import pytest
import torch
from torch import nn
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import RY, RX

from quanvolution import EquivariantQuanvolution2DTorchLayer, create_structured_patches
from geometric_classifier import BasicClassifierTorch
from ansatze import SimpleAnsatz0, SimpleAnsatz1
from examples.utils import calculate_image_output_shape

group_info = {'size': 4}


def prep_first_layer_conv(n_output_filters, input_size, stride, kernel_size):
    """
    prepares an equivariant (first) quanvolutional network layer with one or two (structured) filters
    """
    # these chosen at random - should not matter for equivarince
    n_layers = torch.randint(1, 5, (1,))
    n_reuploads = torch.randint(1, 3, (1,))

    weight_shapes_list = [
        {'params': (n_reuploads, n_layers, 2*kernel_size*kernel_size)}, {'params': (n_reuploads, n_layers, kernel_size*kernel_size)}]

    # circuits acting as filters
    quantum_circs = [
        BasicClassifierTorch(
            feature_map='RotEmbeddingWEnt', ansatz=SimpleAnsatz1, size=kernel_size*kernel_size),
        BasicClassifierTorch(
            feature_map='RotEmbeddingWEnt', ansatz=SimpleAnsatz0, size=kernel_size*kernel_size),
    ]
    quantum_circs_properties = [{'n_layers': n_layers, 'embedding_pauli': RY}, {
        'n_layers': n_layers-4, 'embedding_pauli': RX}]
    print('USING INPUT SIZE ', input_size)
    quanv_layer = EquivariantQuanvolution2DTorchLayer(group_info,
                                                      True,
                                                      quantum_circs[:n_output_filters],
                                                      quantum_circs_properties[:n_output_filters],
                                                      (input_size, input_size),
                                                      stride,
                                                      kernel_size,
                                                      weight_shapes_list,
                                                      )

    return quanv_layer


def prep_second_layer_conv(n_output_filters, input_size, stride, kernel_size):
    """
    prepares an equivariant (second) quanvolutional network layer with one or two (structured) filters
    """
    # these chosen at random - should not matter for equivarince
    n_layers = torch.randint(1, 5, (1,))
    n_reuploads = torch.randint(1, 3, (1,))

    init_method = functools.partial(
        torch.nn.init.uniform_, b=2 * math.pi)
    weight_shapes_list = [
        {'params': (n_reuploads, n_layers, 2*kernel_size*kernel_size)}, {'params': (n_reuploads, n_layers, kernel_size*kernel_size)}]

    # circuits acting as filters
    quantum_circs = [
        BasicClassifierTorch(
            feature_map='RotEmbeddingWEnt', ansatz=SimpleAnsatz1, size=kernel_size*kernel_size),
        BasicClassifierTorch(
            feature_map='RotEmbeddingWEnt', ansatz=SimpleAnsatz0, size=kernel_size*kernel_size),
    ]
    quantum_circs_properties = [{'n_layers': n_layers, 'embedding_pauli': RY}, {
        'n_layers': n_layers-4, 'embedding_pauli': RX}]
    print('USING INPUT SIZE ', input_size)
    quanv_layer = EquivariantQuanvolution2DTorchLayer(group_info,
                                                      False,
                                                      quantum_circs[:n_output_filters],
                                                      quantum_circs_properties[:n_output_filters],
                                                      (input_size, input_size),
                                                      stride,
                                                      kernel_size,
                                                      weight_shapes_list,
                                                      )

    return quanv_layer


def prep_model(one_layer_model: bool,
               n_filters_0,
               kernel_0_size,
               stride_0,
               image_size,
               n_filters_1=None,
               kernel_1_size=None,
               stride_1=None,
               ):

    first_layer_output_size = calculate_image_output_shape(
        image_size, kernel_0_size, stride_0)

    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv0 = prep_first_layer_conv(n_filters_0,
                                               image_size,
                                               stride_0,
                                               kernel_0_size
                                               )
            self.conv1 = False
            if not one_layer_model:
                self.conv1 = prep_second_layer_conv(n_filters_1,
                                                    first_layer_output_size,
                                                    stride_1,
                                                    kernel_1_size)

        def forward(self, x):
            # TODO: CHECK WITH NONLINEARITY
            x = self.conv0(x)
            if self.conv1:
                x = self.conv1(x)
            return x

    return FullModel()


def prep_symmetric_dataset(patch_size):
    """
    Makes a dataset of size 4 where each datapoint is the same image in a different pose.
    """
    datapoint_e = torch.rand(patch_size, patch_size)
    datapoints = [datapoint_e.rot90(i) for i in range(4)]

    symmetric_dataset = torch.zeros(4, 1, patch_size, patch_size)
    for i in range(4):
        symmetric_dataset[i] = datapoints[i]

    return symmetric_dataset


@pytest.mark.parametrize('n_patches', [
    10,
    100,
    1000,
    10000,
])
@pytest.mark.parametrize('patch_size', [
    2,
    3,
    4,
    5
])
def test_create_structured_patches_shape_correct(n_patches, patch_size, group_size=4):
    first_layer_patches = torch.rand(n_patches, patch_size*patch_size)
    subsequent_layer_patches = torch.rand(4, n_patches, patch_size*patch_size)

    structured_first_layer_patches = create_structured_patches(
        first_layer_patches, patch_size)

    structured_subsequent_layer_patches = create_structured_patches(
        subsequent_layer_patches, patch_size)

    assert structured_first_layer_patches.shape == (
        group_size, first_layer_patches.shape[0], patch_size, patch_size)
    # this could be 4,4 but sort of generalises to different group reps between layers
    # mainly to remember Gin and Gout can be different in general
    assert structured_subsequent_layer_patches.shape == (subsequent_layer_patches.shape[0], 4,
                                                         subsequent_layer_patches.shape[1], patch_size, patch_size)


@pytest.mark.parametrize('image_size, kernel_size, stride', [
    (10, 2, 1),
    (10, 2, 2),
    (11, 2, 1),
    (12, 3, 3)
])
@pytest.mark.parametrize('n_filters', [1, 2])
def test_quanv_first_layer_equivariant(image_size, kernel_size, stride, n_filters):
    print('HERE USING IMAGE SIZE ', image_size)
    model = prep_model(True,
                       n_filters,
                       kernel_size,
                       stride,
                       image_size)
    data = prep_symmetric_dataset(image_size)
    output = model(data)

    """
    #NOTE: THIS TO VISUALISE
    first_layer_output_size = calculate_image_output_shape(
        image_size, kernel_size, stride)
    for channel_id in range(1):
        fig, ax = plt.subplots(3, 4)
        for i in range(4):
            ax[0][i].imshow(output[i][0]
                            [channel_id].view(first_layer_output_size, first_layer_output_size).detach())
            ax[1][i].imshow(output[i][1]
                            [channel_id].view(first_layer_output_size, first_layer_output_size).detach())
            # ax[2][i].imshow(first_layer_output[i][2]
            #                [channel_id].view(9, 9).detach())
        fig.suptitle(f'first layer, channel {channel_id}')
    plt.show()
    """
    # check each filter separately
    for filter_id in range(n_filters):
        filter_output = output[:, :, filter_id, :, :]
        # check components of filter match across datapoints (remember datapoints related via group on purpose)
        for filter_component_id in range(4):
            # compare components of first datapoint to the following datapoints
            for datapoint_id in range(1, 4):
                corresponding_matching_id = (
                    filter_component_id + datapoint_id) % 4
                assert torch.allclose(filter_output[filter_component_id][filter_component_id].rot90(datapoint_id),
                                      filter_output[corresponding_matching_id][corresponding_matching_id])


@pytest.mark.parametrize('image_size, kernel_sizes, strides', [
    (10, [2, 2], [1, 1]),
    (10, [2, 2], [2, 1]),
    (11, [2, 2], [1, 1]),
    (12, [3, 2], [3, 2])
])
@pytest.mark.parametrize('ns_filters', [[1, 1], [1, 2], [2, 1], [2, 2]])
def test_quanv_subsequent_layer_equivariant(image_size, kernel_sizes, strides, ns_filters):
    model = prep_model(False, ns_filters[0], kernel_sizes[0], strides[0],
                       image_size, ns_filters[1], kernel_sizes[1], strides[1])
    data = prep_symmetric_dataset(image_size)
    output = model(data)
    # check each filter separately
    for filter_id in range(ns_filters[1]):
        filter_output = output[:, :, filter_id, :, :]
        # check components of filter match across datapoints (remember datapoints related via group on purpose)
        for filter_component_id in range(4):
            # compare components of first datapoint to the following datapoints
            for datapoint_id in range(1, 4):
                corresponding_matching_id = (
                    filter_component_id + datapoint_id) % 4
                assert torch.allclose(filter_output[filter_component_id][filter_component_id].rot90(datapoint_id),
                                      filter_output[corresponding_matching_id][corresponding_matching_id])
