import os
import re
from pathlib import Path

import pandas as pd
import torch


def match_run_dir(model_name, path_to_saved_results):
    # outputs the directory for a model; like model_name_0001_lr=3....
    # path_to_package.parent/saved_results_name
    run_dir_name = 'NoneFound'
    for dir_name in os.listdir(path_to_saved_results):
        if re.search(model_name, dir_name) is not None:
            run_dir_name = dir_name
    return path_to_saved_results / run_dir_name


def find_checkpoint_dir(run_path, checkpoint_id):
    # outputs directory of the latest checkpoint; like checkpoint000010
    for dir_name in os.listdir(run_path):
        search = re.search('checkpoint_0*([1-9][0-9]*)', dir_name)
        if search is not None:
            found_checkpoint_id = search.group(1)
            if int(checkpoint_id) > int(latest_checkpoint):
                latest_checkpoint = checkpoint_id
                latest_checkpoint_dir_name = dir_name
    return latest_checkpoint_dir_name


def get_best_config_and_params_from_run(model_name, path_to_saved_results, checkpoint_id, state_dict_saved=False):
    """
    best_params is a pandas object [Series?] if state_dict_saved == False
    Otherwise it is a torch.state_dict() object [just a dict?]
    best_config is always a pandas object
    """
    # e.g. some_model_2a34
    print(model_name)
    # NOTE: I didn't need the best models dir last time - not sure what changed
    run_dir = match_run_dir(model_name, path_to_saved_results)
    #checkpoint_dir = find_latest_checkpoint_dir(run_dir)  # e.g. checkpoint_000010
    checkpoint_dir = 'checkpoint_000000'
    checkpoint_dir = checkpoint_dir[:-len(str(checkpoint_id))] + str(checkpoint_id)
    if state_dict_saved:
        best_params_dir = run_dir/checkpoint_dir/'model_state.pth'
        best_params = torch.load(best_params_dir)
    else:
        best_params_dir =  run_dir / checkpoint_dir / 'params.pkl'
        best_params = pd.read_pickle(best_params_dir)
    best_config_pkl = Path(best_params_dir).parent.parent/'params.pkl'
    best_config = pd.read_pickle(best_config_pkl)

    return best_config, best_params


def get_model_names_from_wandb(api, project_name, metric_name, which_metric_to_choose: int | str):
    """
    Not converged on best way to access training info yet. Exploring options.
    """
    models_w_metrics = {}
    for run in api.runs(f"ucl_hep_q_mj/{project_name}"):
        if run.state == 'finished':
            try:
                metrics = run.history()[metric_name].tolist()
                #user specifies exactly the point of train history to look at (same for all models)
                if isinstance(which_metric_to_choose, int):
                    metric = metrics[which_metric_to_choose]
                    checkpoint_id = which_metric_to_choose
                #user looks for smallest/largest metric achieved during training
                elif which_metric_to_choose == 'min':
                    metric = min(metrics)
                    checkpoint_id = metrics.index(min(metrics))
                elif which_metric_to_choose == 'max':
                    metric = max(metrics)
                    checkpoint_id = metrics.index(max(metrics))
                #user looks at last metric during training
                elif which_metric_to_choose == 'last':
                    metric = metrics[-1]
                    checkpoint_id = len(metrics) - 1
                else:
                    raise ValueError('which_loss_to_choose should be an int or one of (\'min\', \'last\')')
                model = run.name
                models_w_metrics[f'{model}'] = [metric, checkpoint_id]
            except (AttributeError, ValueError, KeyError) as e:
                print(e)

    print(models_w_metrics)

    return models_w_metrics

def calculate_image_output_shape(image_size, kernel_size, stride):
    """
    ASSUMES SQUARE IMAGES AND NO PADDING ALLOWED.
    """
    output_size = (image_size - kernel_size)/stride + 1
    if int(output_size) != output_size:
        raise ValueError(
            'Need (image_size-kernel_size)/stride to be an integer, got: ', output_size)
    return int(output_size)