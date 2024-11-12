
import pandas as pd
import os
import re
from pathlib import Path


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
