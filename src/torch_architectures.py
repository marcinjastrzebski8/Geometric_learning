"""
Holds various types of torch architectures combining classical and quantum layers.
NOTE: all circuits assume C4 symmetries which might manifest itself in some hardcoding
"""
import torch
from torch import nn


class ConvolutionalEQNEC(nn.Module):
    """
    EQ -> NEC architecture with two quanvolutional layers and three FF layers.
    """

    def __init__(self, architecture_config):
        super().__init__()
        self.quanv0 = architecture_config['quanv0']
        self.quanv1 = architecture_config['quanv1']
        self.dropout0 = False
        self.dropout1 = False
        dense_shape_0 = architecture_config['dense_units'][0]
        dense_shape_1 = architecture_config['dense_units'][1]
        # NOTE: this shape is hardcoded for kernel size 2 and stride 1 and 2 layers
        dense_input_shape = 4 * \
            (architecture_config['image_size']-2) * \
            (architecture_config['image_size']-2)
        if architecture_config['use_dropout0']:
            self.dropout0 = nn.Dropout(architecture_config['dropout0'])

        if architecture_config['use_dropout1']:
            self.dropout1 = nn.Dropout(architecture_config['dropout1'])

        # NOTE: input size is hardcoded for kernel size + stride. TODO: Could generalise.
        self.fc0 = nn.Linear(
            dense_input_shape, dense_shape_0)
        self.fc1 = nn.Linear(
            dense_shape_0, dense_shape_1)
        self.fc2 = nn.Linear(dense_shape_1, 1)

    def forward(self, x):
        # NOTE: Callum is using batch normalisation here which is not equivariant by default,
        # could use the escnn package to do that if needed
        x = nn.functional.relu(self.quanv0(x))
        x = nn.functional.relu(self.quanv1(x))
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


architectures_lookup = {'ConvolutionalEQNEC': ConvolutionalEQNEC}


class ConvolutionalEQEQ(nn.Module):
    """
    EQ -> EQ architecture with two quanvolutional layers and a single quantum classifier of variable length.
    """

    def __init__(self, architecture_config):
        super().__init__()
        self.quanv0 = architecture_config['quanv0']
        self.quanv1 = architecture_config['quanv1']
        self.quantum_classifier = architecture_config['quantum_classifier']

    def forward(self, x):
        # NOTE: Callum is using batch normalisation here which is not equivariant by default,
        # could use the escnn package to do that if needed
        x = nn.functional.relu(self.quanv0(x))
        x = nn.functional.relu(self.quanv1(x))
        x = x.permute(1, 0, 2, 3, 4)
        x = torch.flatten(x, 1)
        x = self.quantum_classifier(x)
        return x


class ConvolutionalECEQ(nn.Module):
    pass
