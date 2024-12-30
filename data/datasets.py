"""
Contains implementations of Dataset classes (torch,jax) which are used during training.
Instead of loading inside scripts, these should be used.
"""
import numpy as np
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from pennylane import numpy as qnp
from jax import numpy as jnp
import jax_dataloader as jdl
import random
from PIL import Image

path_to_datasets = Path('.').absolute()/'data'


class RotateByC4:
    # from chatgpt
    def __call__(self, img):
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)
        return F.rotate(img, angle)


def rotate_270(matrix):
    return np.array([list(col) for col in reversed(list(zip(*matrix)))])


def rotate_90(matrix):
    return np.array([list(reversed(col)) for col in rotate_270(matrix)])


def rotate_180(matrix):
    return np.array([list(reversed(row)) for row in reversed(matrix)])


def apply_random_group_action(matrix):
    rotations_to_choose_from = [rotate_90, rotate_180, rotate_270, lambda x: x]
    choice_of_rotation = np.random.choice(rotations_to_choose_from)
    return choice_of_rotation(matrix)


def mock_image_dataset(num_data, random_seed, image_length):
    """
    Produces mock 'track' and 'shower' data.
    Each image can be rotated by 90/180/270 degrees. This makes the dataset symmetric wrt discreet rotations.
    The symmetry group is: D_4.
    """
    # same random seed shouldn't be a problem as the numbers are being
    # distributed very differently for each dataset
    tracks = make_mock_tracks(num_data, random_seed, image_length, True)
    showers = make_mock_showers(num_data, random_seed, image_length, True)
    tracks_dict = {'data': list(tracks), 'label': 0}
    showers_dict = {'data': list(showers), 'label': 1}
    tracks_df = pd.DataFrame(tracks_dict)
    showers_df = pd.DataFrame(showers_dict)
    full_df = pd.concat([tracks_df, showers_df])
    scrambled_df = full_df.sample(frac=1).reset_index(drop=True)
    return scrambled_df


def make_mock_tracks(num_data, random_seed, image_length, w_symmetry=False):
    """
    Copied directly from bars data for generative modelling project.
    A 'track' is a straight vertical line.
    Not meant to be an accurate model. Just vaguely related to the problem we will deal with.
    The individual images are flattened and need to be reshaped with .reshape(image_length, image_length)
    """
    # NOTE: THIS IS UGLY
    # which columns will be populated with numbers
    np.random.seed(random_seed)
    columns = np.random.choice([0, 1], num_data)
    # the numbers representing the bars
    bars = np.random.uniform(
        0.1, 1, image_length*num_data).reshape(num_data, image_length)
    data = np.zeros((num_data, image_length, image_length))
    for column, bar, data_point in zip(columns, bars, data):
        for row_id, row in enumerate(data_point):
            row[column] = bar[row_id]
    data = data.reshape(num_data, image_length*image_length)
    if w_symmetry:
        for image_id, image in enumerate(data):
            data[image_id] = apply_random_group_action(
                data[image_id].reshape(image_length, image_length)).reshape(image_length*image_length)
    return data


def make_mock_showers(num_data, random_seed, image_length, w_symmetry=False):
    """
    Make images which very vaguely represent a shower region.
    The individual images are flattened and need to be reshaped with .reshape(image_length, image_length)
    """
    # NOTE: this is a bit ugly
    np.random.seed(random_seed)
    data = np.zeros((num_data, image_length * image_length))
    for image_id, image in enumerate(data):
        # pick how many random pixels will have nonzero value
        how_many_pixels = np.random.randint(2, 4)
        # pick which pixels
        which_pixels = np.random.choice(
            range(len(image)), how_many_pixels, replace=False)
        # pick values for these pixels
        shower_values = np.random.uniform(0.1, 1, how_many_pixels)
        for value_id, pixel_id in enumerate(which_pixels):
            image[pixel_id] = shower_values[value_id]
        if w_symmetry:
            data[image_id] = apply_random_group_action(
                image.reshape(image_length, image_length)).reshape(image_length*image_length)
    return data

# NOTE: can these three classes be obtained with less code? I need each to be a class but so much of the code repeats!
# templates??


class MicrobooneTrainData(Dataset):

    def __init__(self, size: int):
        data = torch.load(
            path_to_datasets/f'microboone_from_callum/train_data_{size}x{size}.pt')
        # add the channel dimension
        self.data = data.view(data.shape[0], 1, data.shape[1], data.shape[2])
        labels = torch.load(
            path_to_datasets/f'microboone_from_callum/train_labels_{size}x{size}.pt')
        self.labels = torch.flatten(labels)
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label

    def split(self, train_size, validation_size):
        dataset_size = self.data.shape[0]
        train_idx = np.random.choice(
            range(dataset_size), train_size, replace=False)
        remaining_idxs = np.array(
            list(set(range(dataset_size)) - set(train_idx)), dtype=int)
        val_idx = np.random.choice(
            remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx


class MicrobooneValData(Dataset):

    def __init__(self, size: int):
        data = torch.load(
            path_to_datasets/f'microboone_from_callum/val_data_{size}x{size}.pt')
        # add the channel dimension
        self.data = data.view(data.shape[0], 1, data.shape[1], data.shape[2])
        labels = torch.load(
            path_to_datasets/f'microboone_from_callum/val_labels_{size}x{size}.pt')
        self.labels = torch.flatten(labels)
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label

    def split(self, train_size, validation_size):
        dataset_size = self.data.shape[0]
        train_idx = np.random.choice(
            range(dataset_size), train_size, replace=False)
        remaining_idxs = np.array(
            list(set(range(dataset_size)) - set(train_idx)), dtype=int)
        val_idx = np.random.choice(
            remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx


class MicrobooneTestData(Dataset):

    def __init__(self, size: int):
        data = torch.load(
            path_to_datasets/f'microboone_from_callum/test_data_{size}x{size}.pt')
        # add the channel dimension
        self.data = data.view(data.shape[0], 1, data.shape[1], data.shape[2])
        labels = torch.load(
            path_to_datasets/f'microboone_from_callum/test_labels_{size}x{size}.pt')
        self.labels = torch.flatten(labels)
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label

    def split(self, train_size, validation_size):
        dataset_size = self.data.shape[0]
        train_idx = np.random.choice(
            range(dataset_size), train_size, replace=False)
        remaining_idxs = np.array(
            list(set(range(dataset_size)) - set(train_idx)), dtype=int)
        val_idx = np.random.choice(
            remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx


class SymmetricDataset(Dataset):
    def __init__(self, size, image_length):
        data_df = mock_image_dataset(size, 8, image_length)
        self.data = np.array(data_df['data'].tolist())
        self.labels = np.array(data_df['label'].tolist())
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return qnp.array(sample), label

    def split(self, train_size, validation_size):
        dataset_size = self.data.shape[0]
        train_idx = np.random.choice(
            range(dataset_size), train_size, replace=False)
        remaining_idxs = np.array(
            list(set(range(dataset_size)) - set(train_idx)))
        val_idx = np.random.choice(
            remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx


class SymmetricDatasetJax(jdl.Dataset):
    """
    TODO: Change functionality somehow to allow either torch or jax and not have to repeat code.
    """

    def __init__(self, size, image_length):
        data_df = mock_image_dataset(size, 8, image_length)
        self.data = np.array(data_df['data'].tolist())
        self.labels = np.array(data_df['label'].tolist())
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return jnp.array(sample), label

    def split(self, train_size, validation_size):
        dataset_size = self.data.shape[0]
        train_idx = np.random.choice(
            range(dataset_size), train_size, replace=False)
        remaining_idxs = np.array(
            list(set(range(dataset_size)) - set(train_idx)))
        val_idx = np.random.choice(
            remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx


def simple_symmetric_dataset(size):
    df_dict = {'data': [], 'label': []}
    positive_class = np.random.uniform(-1, 0, (size, 2))
    negative_class = np.random.uniform(0, 1, (size, 2))
    df_dict['data'] += (list(positive_class))
    df_dict['data'] += (list(negative_class))
    df_dict['label'] += [1 for i in range(size)]
    df_dict['label'] += [-1 for i in range(size)]
    df = pd.DataFrame(df_dict)
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    return df


class SimpleSymmetricDataset(Dataset):
    def __init__(self, size):
        data_df = simple_symmetric_dataset(size)
        self.data = np.array(data_df['data'].tolist())
        self.labels = np.array(data_df['label'].tolist())
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return qnp.array(sample), label

    def split(self, train_size, validation_size):
        dataset_size = self.data.shape[0]
        train_idx = np.random.choice(
            range(dataset_size), train_size, replace=False)
        remaining_idxs = np.array(
            list(set(range(dataset_size)) - set(train_idx)))
        val_idx = np.random.choice(
            remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx

# NOTE: again, very sloppy way to have train-val-test split from rotated mnist


class legacy_RotatedMNISTTrain(Dataset):
    """
    Keeping this just in case. This is how the rotmnist dataset has been created. 
    Slices for val were 500:600 and test 600:
    These have been saved and are now read by RotatedMNIST...() classes.
    """

    def __init__(self):
        self.transform = transforms.Compose([
            RotateByC4(),
            transforms.ToTensor()
        ])
        full_dataset = datasets.MNIST(
            root=str(path_to_datasets), train=False, download=False)
        mask = (full_dataset.targets == 0) | (full_dataset.targets == 1)
        self.data = full_dataset.data[mask][:500]
        self.labels = full_dataset.targets[mask][:500].float()
        # From gpt, not sure wht the transform has to go here but it returns bytes when applied at load time
        self.data = torch.stack(
            [self.transform(Image.fromarray(img.numpy(), mode='L')) for img in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        return img, label

    def split(self, train_size, validation_size):
        dataset_size = self.data.shape[0]
        train_idx = np.random.choice(
            range(dataset_size), train_size, replace=False)
        remaining_idxs = np.array(
            list(set(range(dataset_size)) - set(train_idx)), dtype=int)
        val_idx = np.random.choice(
            remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx


class RotatedMNISTTrain(Dataset):

    def __init__(self):
        self.data = torch.load(
            path_to_datasets/'rot_MNIST/train_data.pt')
        self.labels = torch.load(
            path_to_datasets/'rot_MNIST/train_labels.pt')
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label

    def split(self, train_size, validation_size):
        dataset_size = self.data.shape[0]
        train_idx = np.random.choice(
            range(dataset_size), train_size, replace=False)
        remaining_idxs = np.array(
            list(set(range(dataset_size)) - set(train_idx)), dtype=int)
        val_idx = np.random.choice(
            remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx


class RotatedMNISTVal(Dataset):

    def __init__(self):
        self.data = torch.load(
            path_to_datasets/'rot_MNIST/val_data.pt')
        self.labels = torch.load(
            path_to_datasets/'rot_MNIST/val_labels.pt')
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label

    def split(self, train_size, validation_size):
        dataset_size = self.data.shape[0]
        train_idx = np.random.choice(
            range(dataset_size), train_size, replace=False)
        remaining_idxs = np.array(
            list(set(range(dataset_size)) - set(train_idx)), dtype=int)
        val_idx = np.random.choice(
            remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx


class RotatedMNISTTest(Dataset):

    def __init__(self):
        self.data = torch.load(
            path_to_datasets/'rot_MNIST/test_data.pt')
        self.labels = torch.load(
            path_to_datasets/'rot_MNIST/test_labels.pt')
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label

    def split(self, train_size, validation_size):
        dataset_size = self.data.shape[0]
        train_idx = np.random.choice(
            range(dataset_size), train_size, replace=False)
        remaining_idxs = np.array(
            list(set(range(dataset_size)) - set(train_idx)), dtype=int)
        val_idx = np.random.choice(
            remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx



dataset_lookup = {'MicrobooneTrainData': MicrobooneTrainData,
                  'MicrobooneValData': MicrobooneValData,
                  'MicrobooneTestData': MicrobooneTestData,
                  'RotatedMNISTTrainData': RotatedMNISTTrain,
                  'RotatedMNISTValData': RotatedMNISTVal,
                  'RotatedMNISTTestData': RotatedMNISTTest}
