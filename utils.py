import numpy as np
from jax import numpy as jnp
from pennylane import numpy as qnp
import jax_dataloader as jdl
from torch.utils.data import Dataset
import pandas as pd
import os
import re
from pathlib import Path


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


def match_run_dir(model_name, path_to_saved_results):
    # outputs the directory for a model; like model_name_0001_lr=3....
    # path_to_package.parent/saved_results_name
    run_dir_name = 'NoneFound'
    for dir_name in os.listdir(path_to_saved_results):
        if re.search(model_name, dir_name) is not None:
            run_dir_name = dir_name
    return run_dir_name


def find_latest_checkpoint_dir(run_path):
    # outputs directory of the latest checkpoint; like checkpoint000010
    latest_checkpoint = 0
    latest_checkpoint_dir_name = ''
    for dir_name in os.listdir(run_path):
        search = re.search('checkpoint_0*([1-9][0-9]*)', dir_name)
        if search is not None:
            checkpoint_id = search.group(1)
            if int(checkpoint_id) > int(latest_checkpoint):
                latest_checkpoint = checkpoint_id
                latest_checkpoint_dir_name = dir_name
    return latest_checkpoint_dir_name


def get_best_config_and_params_from_run(model_name, path_to_saved_results):
    # e.g. some_model_2a34
    print(model_name)
    # NOTE: I didn't need the best models dir last time - not sure what changed
    run_dir = match_run_dir(model_name, path_to_saved_results)
    checkpoint_dir = find_latest_checkpoint_dir(
        path_to_saved_results / run_dir)  # e.g. checkpoint_000010
    best_params_dir = path_to_saved_results / \
        run_dir / checkpoint_dir / 'params.pkl'
    best_params = pd.read_pickle(best_params_dir)
    best_config_pkl = Path(best_params_dir).parent.parent/'params.pkl'
    best_config = pd.read_pickle(best_config_pkl)

    return best_config, best_params


def get_model_names_from_wandb(api, project_name):
    models_w_losses = {}
    for run in api.runs(f"ucl_hep_q_mj/{project_name}"):
        if run.state == 'finished':
            # get last loss obtained during hyperopt
            try:
                loss = run.history().loss.tolist()[-1]
                model = run.name
                models_w_losses[f'{model}'] = loss
            except (AttributeError):
                pass

    print(models_w_losses)

    return models_w_losses
