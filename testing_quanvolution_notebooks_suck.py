"""
Jupyter does not cooperate when running torch (often breaks unexpectedly).
This is thus a script-version of what I was doing in a notebook to try and figure out why
my equivariant quanvolution works only for 180deg rotations (and not for 90,270).

For now this is just for first-layer equivariance.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import functools
import pennylane as qml
from pennylane import RY, RX
import math

from src.quanvolution import create_structured_patches
from src.geometric_classifier import BasicClassifierTorch
from src.ansatze import SimpleAnsatz1, SimpleAnsatz0


def classical_filter_convolution(filter, patches):
    """
    Dummy convolution to check that the equivariance holds.
    """
    output = torch.zeros(len(patches))
    print('GETTING PATCHES LEN : ', len(patches))
    for patch_id, patch in enumerate(patches):
        patch_output = torch.dot(filter, patch)
        output[patch_id] = patch_output
    return output


dummy_filters = torch.rand(4, 9)


# ansatz to use as filter
init_method = functools.partial(
    torch.nn.init.uniform_, b=2 * math.pi)
n_layers = 6
weight_shapes_list = [
    {'params': (n_layers, n_layers*18)}, {'params': (n_layers, n_layers*9)}]

# trying out using multiple filters
quantum_circs = [
    BasicClassifierTorch(
        feature_map='RotEmbeddingWEnt', ansatz=SimpleAnsatz1, size=9, measurement=qml.PauliZ(4)),
    BasicClassifierTorch(
        feature_map='RotEmbeddingWEnt', ansatz=SimpleAnsatz0, size=9, measurement=qml.PauliZ(4)),
]
quantum_circs_properties = [{'n_layers': n_layers, 'embedding_pauli': RY}, {
    'n_layers': n_layers-4, 'embedding_pauli': RX}]

quantum_filters = [quantum_circ.prediction_circuit(
    quantum_circ_properties) for quantum_circ, quantum_circ_properties in zip(quantum_circs, quantum_circs_properties)]
torch_layers = [qml.qnn.TorchLayer(quantum_filter, weight_shapes, init_method=init_method)
                for quantum_filter, weight_shapes in zip(quantum_filters, weight_shapes_list)]

subsequent_filters = [[quantum_circ.prediction_circuit(quantum_circ_properties) for i in range(
    4)] for quantum_circ, quantum_circ_properties in zip(quantum_circs, quantum_circs_properties)]
subsequent_torch_layers = [[qml.qnn.TorchLayer(
    quantum_filter[i], weight_shapes, init_method=init_method) for i in range(4)] for quantum_filter, weight_shapes in zip(subsequent_filters, weight_shapes_list)]

temp_subsequent_layer = [qml.qnn.TorchLayer(
    quantum_circs[0].prediction_circuit(quantum_circs_properties[0]), weight_shapes_list[0], init_method=init_method) for i in range(4)]
####################################################################################################

# imagine dataset with two datapoints that happen to be related by a rotation
datapoint_e = torch.range(1, 729).view(27, 27)/729
# datapoint_e = torch.rand(27, 27)
datapoint_r = datapoint_e.rot90(1, (0, 1))
datapoint_r2 = datapoint_e.rot90(2, (0, 1))
datapoint_r3 = datapoint_e.rot90(3, (0, 1))
n_data = 2
data = torch.zeros(2, 1, 27, 27)
data[0] = datapoint_e
data[1] = datapoint_r
# data[2] = datapoint_r2

fig, ax = plt.subplots(1, 3)

ax[0].imshow(data[0][0])
ax[1].imshow(data[1][0])
# ax[2].imshow(data[2][0])


# unfolded as in main loop
unfolded = torch.nn.functional.unfold(data, kernel_size=(3, 3), stride=(3, 3))
print('UNFOLDED SHAPE: ', unfolded.shape)
unfolded = unfolded.view(2, 1, 9, 81)

# unfolded patches are just what needs to be passed to a filter
# fig, ax = plt.subplots(1, 9)
# for i in range(9):
#    ax[i].imshow(unfolded[0][0][i].view(9, 9), vmin=0, vmax=1)
# plt.show(block=True)


first_layer_output = torch.zeros(4, n_data, 2, 9, 9)
# this happens inside input_channel loop (so different filters from previous layer, for subsequent layers all 4 poses are passed)
# it also happens inside the pose loop, meaning that the output of the following makes up one pose in the final output
for filter_id, torch_layer in enumerate(torch_layers):
    filter_output = torch.zeros(4, n_data, 81)
    for input_channel_id in range(1):
        inner_loop_content = unfolded[:, input_channel_id, :, :,]

        patches_to_convolve = inner_loop_content.permute(
            0, 2, 1).reshape(-1, 3*3)

        structured_patches_to_convolve = create_structured_patches(
            patches_to_convolve).view(-1, 9)
        """
        # Adjust figsize to make windows larger
        fig, ax = plt.subplots(2, 36, figsize=(36, 2))
        for i in range(36):
            ax[0][i].imshow(
                structured_patches_to_convolve[i].view(3, 3), vmin=0, vmax=1)
            ax[1][i].imshow(
                structured_rotated_patches_to_convolve[i].view(3, 3), vmin=0, vmax=1)

            # Remove axes
            ax[0][i].axis('off')
            ax[1][i].axis('off')
        plt.show(block=True)
        """

        channel_output = torch_layer(
            structured_patches_to_convolve).view(4, -1, 81)
        filter_output += channel_output
        print('OUTPUT SHAPE: ', channel_output.shape)
    first_layer_output[:, :, filter_id, :,
                       :] = filter_output.view(4, n_data, 9, 9)


# this works fine
for channel_id in range(2):
    fig, ax = plt.subplots(3, 4)
    for i in range(4):
        ax[0][i].imshow(first_layer_output[i][0]
                        [channel_id].view(9, 9).detach())
        ax[1][i].imshow(first_layer_output[i][1]
                        [channel_id].view(9, 9).detach())
        # ax[2][i].imshow(first_layer_output[i][2]
        #                [channel_id].view(9, 9).detach())
    fig.suptitle(f'first layer, channel {channel_id}')


##################################################################################################
# subsequent layer
"""
#NOTE: THIS BIT OF CODE WORKS - SAVING IN ORDER TO DEV FURTHER
# this is to pretend there's only one channel
first_layer_output = first_layer_output[:, :, 0, :, :].view(
    4, 2, 1, 9, 9)

# this to treat input components as different channels
first_layer_output = first_layer_output.permute(1, 0, 2, 3, 4).reshape(
    n_data, 4, 9, 9)

# this extracts (batch_size, n_input_channels*prod(kernel_sizes), h_iter*w_iter)
first_layer_output_unfolded = torch.nn.functional.unfold(
    first_layer_output, 3, 3)

# separate input channels,
# permute to put input component into first dimension again
first_layer_output_unfolded = first_layer_output_unfolded.view(
    n_data, 4, 1, 9, 9).permute(1, 0, 2, 3, 4)

# inside the loop
inner_loop_content = first_layer_output_unfolded[:, :, 0, :, :]
# this puts input into circuit into last dim
patches_to_convolve = inner_loop_content.reshape(4, -1, 9)

fig, ax = plt.subplots(1, n_data*9, figsize=(20, 8))
for component_id in range(1, 2):
    for patch_id in range(n_data*9):
        ax[patch_id].imshow(patches_to_convolve[component_id]
                            [patch_id].view(3, 3).detach())
        ax[patch_id].axis(
            'off')

rotated_patches_to_convolve = create_structured_patches(
    patches_to_convolve).view(4, -1, 9)

# look at one input component only
fig, ax = plt.subplots(1, n_data*4*9, figsize=(20, 8))
for component_id in range(1, 2):
    for patch_id in range(n_data*4*9):
        ax[patch_id].imshow(rotated_patches_to_convolve[component_id]
                            [patch_id].view(3, 3).detach())
        ax[patch_id].axis(
            'off')

conv_output = torch.zeros(4, n_data, 9)
for input_component_id in range(4):
    output_from_component = temp_subsequent_layer[input_component_id](
        rotated_patches_to_convolve[input_component_id]).view(4, -1, 9)
    # output_from_component = classical_filter_convolution(
    #    dummy_filters[input_component_id], rotated_patches_to_convolve[input_component_id]).view(4, -1, 9)
    conv_output += output_from_component


fig, ax = plt.subplots(n_data, 4)
for datapoint_id in range(n_data):
    for component_id in range(4):
        ax[datapoint_id][component_id].imshow(
            conv_output[component_id][datapoint_id].view(3, 3).detach())
fig.suptitle('temp second layer')
plt.show(block=True)
"""
"""
with_filter_ids_output = first_layer_output.view(4, n_data, 2, 9, 9)
# second_layer_input = torch.nn.functional.relu(with_filter_ids_output)
second_layer_input = with_filter_ids_output
input_channels = second_layer_input.permute(1, 0, 2, 3, 4).reshape(
    n_data, 8, second_layer_input.shape[3], second_layer_input.shape[4])
input_channels_unfolded = torch.nn.functional.unfold(input_channels, 3, 3)
print('UNFOLDED: ', input_channels_unfolded.shape)

# NOTE adding the permute
input_channels_unfolded = input_channels_unfolded.view(
    n_data, 4, 2, 9, 9).permute(1, 0, 2, 3, 4)
for datapoint_id in range(n_data):
    for channel_id in range(2):
        fig, ax = plt.subplots(4, 9)
        for filter_component_id in range(4):
            for patch_id in range(9):
                ax[filter_component_id][patch_id].imshow(input_channels_unfolded[filter_component_id][datapoint_id][channel_id]
                                                         [patch_id].view(3, 3).detach(), vmin=-0.1, vmax=0.1)
        plt.suptitle(f'unfolded, datapoint {
                     datapoint_id}, channel {channel_id}')

second_layer_output = torch.zeros(4, n_data, 2, 3, 3)
"""


for filter_id, torch_layer in enumerate(subsequent_torch_layers[:1]):
    filter_output = torch.zeros(4, n_data, 9)
    for input_channel_id in range(1):
        inner_loop_content = input_channels_unfolded[:,
                                                     :, input_channel_id, :, :,]
        patches_to_convolve = inner_loop_content.permute(
            0, 1, 3, 2).reshape(4, -1, 9)
        # this comes out as (|components|, |pose|, batch_size*number of patches, prod(kernel_sizes))
        # so for me (4,4,3*9,9)
        rotated_patches_to_convolve = create_structured_patches(
            patches_to_convolve).view(4, -1, 9)

        fig, ax = plt.subplots(1, n_data*4*9, figsize=(20, 8))
        # only looking at a single component
        for component_id in range(1, 2):
            for patch_id in range(n_data*4*9):
                ax[patch_id].imshow(rotated_patches_to_convolve[component_id]
                                    [patch_id].view(3, 3).detach())
                ax[patch_id].axis(
                    'off')

        fig.suptitle('patches rotated')

        channel_output = torch.zeros(4, n_data, 9)
        for input_component_id in range(4):
            # NOTE NAMING HERE IS GETTING CONFUSING
            # outputs_from_filter = torch_layer[input_component_id](
            #    rotated_patches_to_convolve[input_component_id]).view(4, 3, 9)
            outputs_from_filter = temp_subsequent_layer[input_component_id](
                rotated_patches_to_convolve[input_component_id]).view(4, n_data, 9)
            channel_output += outputs_from_filter
        filter_output += channel_output
    # second_layer_output[:, :, filter_id, :, :] = filter_output.view(4, 3, 3, 3)
    second_layer_output[:, :, filter_id, :,
                        :] = channel_output.view(4, n_data, 3, 3)

"""
        fig, ax = plt.subplots(2, 4)
        for i in range(4):
            ax[0][i].imshow(output[i][0].view(3, 3).detach())
            ax[1][i].imshow(output[i][1].view(3, 3).detach())
        fig.suptitle('second layer')
        plt.show(block=True)

for channel_id in range(2):
    fig, ax = plt.subplots(n_data, 4)
    for datapoint_id in range(n_data):
        for component_id in range(4):
            ax[datapoint_id][component_id].imshow(second_layer_output[component_id][datapoint_id]
                                                  [channel_id].view(3, 3).detach())
    fig.suptitle(f'second layer, channel {channel_id}')
plt.show(block=True)
"""
