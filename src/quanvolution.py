"""
Code based heavily on https://github.com/ericardomuten/qcnn-hep/blob/main/qcnn_drc/quantum_convolution.py.
"""

import math
import numpy as np
import torch.nn as nn
import jax
import pennylane as qml
from typing import Sequence
import torch
import functools
from copy import deepcopy


def cyclic_permute(tensor, shift):
    """
    From chatgpt, used in create_structured_patches.
    """
    indices = [(i + shift) % tensor.shape[0] for i in range(tensor.shape[0])]
    return tensor[indices]


def create_structured_patches(patches_to_convolve, patch_size=3):
    """
    Creates *poses* of the patches to convolve.

    Takes a tensor of shape (batch_size*number_of_patches_in_image, prod(kernel_sizes))
    with output of shape (|G|_out, batch_size*number_of_patches_in_image, prod(kernel_sizes))
    OR
    Takes a tensor of shape (|G|_in, batch_size*number_of_patches_in_image, , prod(kernel_sizes))
    with output of shape (|G|_in,|G|_out, batch_size*number_of_patches_in_image, prod(kernel_sizes))
    where |G|_out is the size of the group and hence len(patch_transformations).
    NOTE: FOR NOW ONLY ROTATIONS SUPPORTED (EASY WITH TORCH)
    TODO: IMPLEMENT SUCH THAT
        The transformations need to be passed as a list of functions, in the order in which they appear in the group. 
        The transformations will be applied to the patches BACKWARDS to achieve the same effect as when acting on the 'quantum filters'.

    """
    input_patches_dims = deepcopy(patches_to_convolve.shape)
    print(input_patches_dims)
    # need to un-flatten last dimension to rotate it
    patches_to_convolve = patches_to_convolve.view(
        *input_patches_dims[:-1], patch_size, patch_size)
    rotation_patches_dims = patches_to_convolve.shape

    # what is this
    structured_patches_to_convolve = torch.zeros(4, *rotation_patches_dims)

    print('GETTING SHAPE ', patches_to_convolve.shape)
    for n_rotations in range(4):
        rotated_patches = patches_to_convolve.rot90(
            range(4)[-n_rotations], tuple(range(len(rotation_patches_dims))[-2:]))
        if len(rotation_patches_dims) == 3:
            structured_patches_to_convolve[n_rotations,
                                           :, :, :] = rotated_patches
        else:
            rotated_patches = cyclic_permute(rotated_patches, n_rotations)
            structured_patches_to_convolve[:,
                                           n_rotations, :, :, :] = rotated_patches
            # rearrange such that it can be passed to the torch layer effectively
            # a single filter sees all patches belonging to (i,i) pairs where the first coordinate is
            # in the input component space and the second is in group basis
            # TODO: HERE, NEED TO REARRANGE. GOT ALL NECESSARY PATCHES BUT THEY NEED TO BE IN CORRECT ORDER
            # LEADING DIMENSION NEEDS TO BE THE FILTER - WHICH IS NOT A DIMENSION THAT EXISTS AT THE MOMENT
    return structured_patches_to_convolve


class Quanvolution2DTorchLayer(nn.Module):
    """
    Convolutional layer with filters being quantum circuits.

    quantum_circs: List of circuit class objects (I don't have a common base class yet)
    quantum_circs_properties: List of dicts which are the properties of their corresponding circuits, needed to define the qnodes which act as filters
    input_channel_shape: TODO - rethink with new batching
    stride:
    kernel_size:
    weight_shapes_list:
    """

    def __init__(self,
                 quantum_circs: Sequence,
                 quantum_circs_properties: Sequence[dict],
                 input_channel_shape: tuple,
                 stride,
                 kernel_size,
                 weight_shapes_list: Sequence[dict]):
        super().__init__()
        self.input_channel_shape = input_channel_shape
        self.strides = (stride, stride) if isinstance(
            stride, int) else stride
        self.kernel_sizes = (kernel_size, kernel_size) if isinstance(
            kernel_size, int) else kernel_size
        # TODO: ALLOW FOR SOME USER INPUT
        init_method = functools.partial(
            torch.nn.init.uniform_, b=2 * math.pi)

        # Calculate the number of convolution iterations
        # TODO: THIS DOESNT SUPPORT PADDING
        self.h_iter = int(1 + (self.input_channel_shape[0] -
                               self.kernel_sizes[0]) / self.strides[0])
        self.w_iter = int(1 + (self.input_channel_shape[1] -
                               self.kernel_sizes[1]) / self.strides[1])

        print('OUTPUT SIZE:', self.h_iter, self.w_iter)
        # the circuits qnodes
        quantum_filters = [quantum_circ.prediction_circuit(
            quantum_circ_properties) for quantum_circ, quantum_circ_properties in zip(quantum_circs, quantum_circs_properties)]
        # qnodes turned into torch layers MAYBE NOT THE BEST NAME?
        self.torch_layers = [qml.qnn.TorchLayer(
            quantum_filter, weight_shapes, init_method=init_method) for quantum_filter, weight_shapes in zip(quantum_filters, weight_shapes_list)]
        self.bias = nn.Parameter(torch.ones(len(self.torch_layers))*0.1)

    def quantum_convolution(self, input_channel, torch_layer):
        """
        Quantum circuit which can act as the convolutional filter.
        Input_channel is a set of patches for a single input channel for all data points,
        shape: (batch_size, prod(kernel_sizes), h_iter*w_iter)
        Output is a single channel of a batch (so a (batch_size, h_iter, w_iter)-array)
        """

        # each data point split into the different patches we'll be convolving over
        # permuting to have the input to a circuit be a single patch values
        patches_to_convolve = input_channel.permute(
            0, 2, 1).reshape(-1, self.kernel_sizes[0]*self.kernel_sizes[1])

        output = torch_layer(patches_to_convolve).view(-1,
                                                       self.h_iter*self.w_iter)
        return output

    def quanvolution_layer(self, input_channels):
        """
        A pass through one layer of convolution using quantum filters.
        Expects input of shape (batch_size, n_input_channels, base_space_height, base_space_width)
        Output is of shape (batch_size, n_input_channels, h_iter, w_iter)
        """
        batch_size, n_input_channels = input_channels.shape[:2]
        print('INPUT CHANNELS SHAPE ', input_channels.shape)
        # this extracts (batch_size, n_input_channels*prod(kernel_sizes), h_iter*w_iter)
        input_channels_unfolded = torch.nn.functional.unfold(
            input_channels, kernel_size=self.kernel_sizes, stride=self.strides)
        print('UNFOLDED: ', input_channels_unfolded.shape)
        # this separates the input channels
        input_channels_unfolded = input_channels_unfolded.view(
            batch_size, n_input_channels, np.prod(self.kernel_sizes), self.h_iter*self.w_iter)

        output = torch.zeros(batch_size, len(
            self.torch_layers), self.h_iter, self.w_iter)

        # Filter iteration
        for filter_id, torch_layer in enumerate(self.torch_layers):

            filter_output = torch.zeros(batch_size, self.h_iter*self.w_iter)
            # Input channel iteration
            for input_channel_id in range(n_input_channels):

                channel_output = self.quantum_convolution(input_channels_unfolded[:, input_channel_id, :, :,],
                                                          torch_layer)
                filter_output = filter_output + channel_output
            output[:, filter_id, :, :] = filter_output.view(
                batch_size, self.h_iter, self.w_iter)
        output = output + self.bias.view(1, len(self.torch_layers), 1, 1)

        return output

    # for nn.Module compatibility
    def forward(self, x):
        return self.quanvolution_layer(x)


class EquivariantQuanvolution2DTorchLayer(nn.Module):
    """
    Equivariant convolutional layer with filters being quantum circuits.

    quantum_circs: List of circuit class objects (I don't have a common base class yet)
    quantum_circs_properties: List of dicts which are the properties of their corresponding circuits, needed to define the qnodes which act as filters
    input_channel_shape:
    stride:
    kernel_size:
    weight_shapes_list:

    NOTE: THE class does not check whether the circuits provided are in fact equivariant. Needs to be tested separately.
    """

    def __init__(self,
                 group,
                 is_first_layer_quanv: bool,
                 quantum_circs: Sequence,
                 quantum_circs_properties: Sequence[dict],
                 input_channel_shape: tuple,
                 stride,
                 kernel_size,
                 weight_shapes_list: Sequence[dict]):
        super().__init__()
        self.group = group
        self.is_first_layer_quanv = is_first_layer_quanv
        self.input_channel_shape = input_channel_shape
        self.strides = (stride, stride) if isinstance(
            stride, int) else stride
        self.kernel_sizes = (kernel_size, kernel_size) if isinstance(
            kernel_size, int) else kernel_size
        # TODO: ALLOW FOR SOME USER INPUT
        init_method = functools.partial(
            torch.nn.init.uniform_, b=2 * math.pi)

        # Calculate the number of convolution iterations
        # TODO: THIS DOESNT SUPPORT PADDING
        self.h_iter = int(1 + (self.input_channel_shape[0] -
                               self.kernel_sizes[0]) / self.strides[0])
        self.w_iter = int(1 + (self.input_channel_shape[1] -
                               self.kernel_sizes[1]) / self.strides[1])

        print('OUTPUT SIZE:', self.h_iter, self.w_iter)
        # the circuits qnodes
        if self.is_first_layer_quanv:
            quantum_filters = [quantum_circ.prediction_circuit(
                quantum_circ_properties) for quantum_circ, quantum_circ_properties in zip(quantum_circs, quantum_circs_properties)]
            # qnodes turned into torch layers MAYBE NOT THE BEST NAME?
            self.torch_layers = [qml.qnn.TorchLayer(
                quantum_filter, weight_shapes, init_method=init_method) for quantum_filter, weight_shapes in zip(quantum_filters, weight_shapes_list)]
        else:
            # this is a bit messy...
            quantum_filters = [
                [quantum_circ.prediction_circuit(quantum_circ_properties) for i in range(self.group['size'])] for quantum_circ, quantum_circ_properties in zip(quantum_circs, quantum_circs_properties)]
            self.torch_layers = [[qml.qnn.TorchLayer(
                quantum_filter[i], weight_shapes, init_method=init_method) for i in range(self.group['size'])] for quantum_filter, weight_shapes in zip(quantum_filters, weight_shapes_list)]

        self.bias = nn.Parameter(torch.ones(len(self.torch_layers))*0.1)

    def quantum_convolution(self, input_channel, torch_layer):
        """
        Quantum circuit which can act as the convolutional filter.
        Input_channel is a set of patches for all of the poses of a single input channel for all data points,
        shape: (batch_size, |G|, prod(kernel_sizes), h_iter*w_iter) [subsequent layers]
        OR a set of patches of a single input channel for all data points [first layer]
        Output is a single channel (so a (batch_size, |G|, h_iter*w_iter)-shaped tensor).
        Torch layer is a TorchLayer object or Sequence[TorchLayer], if self.is_first_layer = True/False
        """

        # NOTE THAT WE CAN'T 'ROTATE' THE CIRCUIT BECAUSE IT'S NOT A FUNCTION OF GROUP ELEMENTS
        # WE CAN INSTEAD ROTATE THE SELECTED PATCH OF FEATURE MAP BUT I THINK YOU NEED TO DO IT BACKWARDS

        # in case the layer acts on a simple image (function on z2)
        if self.is_first_layer_quanv:

            # each data point split into the different patches we'll be convolving over
            # permuting to have the input to a circuit be a single patch values
            patches_to_convolve = input_channel.permute(
                0, 2, 1).reshape(-1,  np.prod(self.kernel_sizes))

            # this creates patches which are equivalent to 'rotating the filter'
            # but this way we can reuse the same circuit multiple times (also can't really rotate circuit anyways)
            rotated_patches_to_convolve = create_structured_patches(
                patches_to_convolve).view(-1, np.prod(self.kernel_sizes))
            print('SHAPE OF ROTATED PATCHES IS: ',
                  rotated_patches_to_convolve.shape)

            output = torch_layer(
                rotated_patches_to_convolve).view(self.group['size'], -1, self.h_iter*self.w_iter)
        # in case this layer is already receiving a function on a compound group (output of first and subsequent layers)
        else:
            # data has shape (|G|, batch_size, prod(kernel_sizes), h_iter*w_iter)
            batch_size = input_channel.shape[1]
            print('batch_size: ', batch_size)
            patches_to_convolve = input_channel.permute(
                0, 1, 3, 2).reshape(self.group['size'], -1, np.prod(self.kernel_sizes))

            rotated_patches_to_convolve = create_structured_patches(
                patches_to_convolve).view(self.group['size'], -1, np.prod(self.kernel_sizes))

            # applies the correct filter pose to the correct input channel pose
            # then sums to obtain full output for a single pose
            output = torch.zeros(
                self.group['size'], batch_size, self.h_iter*self.w_iter)

            for input_component_id in range(self.group['size']):
                outputs_from_filter = torch_layer[input_component_id](
                    rotated_patches_to_convolve[input_component_id]).view(self.group['size'], -1, self.h_iter*self.w_iter)
                output += outputs_from_filter

        return output

    def quanvolution_layer(self, input_channels):
        """
        A pass through one layer of convolution using quantum filters.
        Expects input of shape (batch_size, n_input_channels, base_space_height, base_space_width)
        OR (|G|, batch_size, n_input_channels,  base_space_height, base_space_width)
        Output is of shape (|G|, batch_size, n_input_channels, h_iter, w_iter)
        """
        # TODO: the shapes could be handled better
        if self.is_first_layer_quanv:
            batch_size, n_input_channels = input_channels.shape[:2]
        else:
            batch_size, n_input_channels = input_channels.shape[1:3]
        print('INPUT CHANNELS SHAPE ', input_channels.shape)

        if not self.is_first_layer_quanv:
            # temporarily permute and flatten the group dimension along the input_channel dimension to allow unfolding
            input_channels = input_channels.permute(1, 0, 2, 3, 4).reshape(
                batch_size, n_input_channels*self.group['size'], input_channels.shape[3], input_channels.shape[4])

        # this extracts (batch_size, n_input_channels*prod(kernel_sizes), h_iter*w_iter)
        input_channels_unfolded = torch.nn.functional.unfold(
            input_channels, kernel_size=self.kernel_sizes, stride=self.strides)
        print('UNFOLDED: ', input_channels_unfolded.shape)

        # this separates the input channels
        if self.is_first_layer_quanv:
            unfolded_shape = (
                batch_size, n_input_channels, np.prod(self.kernel_sizes), self.h_iter*self.w_iter)
        else:
            unfolded_shape = (batch_size, self.group['size'], n_input_channels,  np.prod(
                self.kernel_sizes), self.h_iter*self.w_iter)

        input_channels_unfolded = input_channels_unfolded.view(
            unfolded_shape)

        if not self.is_first_layer_quanv:
            input_channels_unfolded = input_channels_unfolded.permute(
                1, 0, 2, 3, 4)

        # this common regardless of input shape
        output = torch.zeros(self.group['size'], batch_size, len(
            self.torch_layers),  self.h_iter, self.w_iter)

        # Filter iteration
        for filter_id, torch_layer in enumerate(self.torch_layers):

            filter_output = torch.zeros(
                self.group['size'], batch_size, self.h_iter*self.w_iter)
            print('FILTER OUTPUT SHAPE:', filter_output.shape)
            # Input channel iteration
            for input_channel_id in range(n_input_channels):

                if self.is_first_layer_quanv:
                    channel_output = self.quantum_convolution(input_channels_unfolded[:, input_channel_id, :, :,],
                                                              torch_layer)
                    print('CHANNEL OUTPUT SHAPE:', channel_output.shape)
                else:
                    channel_output = self.quantum_convolution(input_channels_unfolded[:, :, input_channel_id, :, :,],
                                                              torch_layer)
                filter_output = filter_output + channel_output
            output[:, :, filter_id, :, :] = filter_output.view(
                self.group['size'], batch_size, self.h_iter, self.w_iter)
        # TODO: THINK THIS WILL NEED CHANGING
        output = output + self.bias.view(1, len(self.torch_layers), 1, 1)

        return output

    # for nn.Module compatibility

    def forward(self, x):
        return self.quanvolution_layer(x)
