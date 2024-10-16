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


def create_structured_patches(patches_to_convolve):
    # TODO: this function rotates the patches which will be processed by quantum filter
    # in the usual formulation of the quanvolution, it is the filter that rotates
    # but we can't really do so with a quantum filter (we're not really doing a convolution)
    # I think that rotating the filters in the opposite direction achieves the same result
    return patches_to_convolve


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

    TODO: KEEP DEVVING
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

        # TODO: DEVVING HERE - TWO CASES: FIRST LAYER AND SUBSEQUENT LAYER
        # IN EACH CASE NEED TO APPLY THE FILTER |G| TIMES, EACH TIME ACTED ON WITH THE CORRECT GROUP ACTION
        # NOTE THAT WE CAN'T 'ROTATE' THE CIRCUIT BECAUSE IT'S NOT A FUNCTION OF GROUP ELEMENTS
        # WE CAN INSTEAD ROTATE THE SELECTED PATCH OF FEATURE MAP BUT I THINK YOU NEED TO DO IT BACKWARDS

        # in case the layer acts on a simple image (function on z2)
        if self.is_first_layer_quanv:

            # each data point split into the different patches we'll be convolving over
            # permuting to have the input to a circuit be a single patch values

            patches_to_convolve = input_channel.permute(
                0, 2, 1).reshape(-1, self.kernel_sizes[0]*self.kernel_sizes[1])

            # TODO: this function should return |G|*n_patches patches where n_patches is the -1 in above reshape
            rotated_patches_to_convolve = create_structured_patches(
                patches_to_convolve)

            # NOTE: extra dimension coming from pixels being vector-valued [separate to possible multiple input channels]
            output = torch_layer(rotated_patches_to_convolve).view(-1, self.group.size,
                                                                   self.h_iter*self.w_iter)
        # in case this layer is already receiving a function on a compound group (output of first and subsequent layers)
        else:
            # data has shape (batch_size, prod(kernel_sizes), |G|, h_iter*w_iter)
            # thus filters should be in a square matrix (|G|, |G|)
            for pose_id in range(self.group.size):
                torch_layer[pose_id]
            pass

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
