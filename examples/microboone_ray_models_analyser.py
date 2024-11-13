"""
Load best models from a ray run and test on test microboone data.
NOTE: Would be nice some general code to avoid rewriting every time I work on a new project.
"""
from sklearn.metrics import roc_auc_score, roc_curve
import wandb
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from examples.utils import get_model_names_from_wandb, get_best_config_and_params_from_run

api = wandb.Api()


def validate_ray_models(json_config, n_models_to_keep, test_dict):
    """
    Given a ray run and some metadata, check performance of the best found models on some test data.
    """
    path_to_ray_models = Path('ray_runs') / json_config['output_models_dir']
    models_w_losses = get_model_names_from_wandb(
        api, json_config['output_models_dir'])
    best_models = sorted(models_w_losses.items(), key=lambda item: item[1])[
        :int(n_models_to_keep)]

    fig, ax = plt.subplots(1, 1)

    for model_name_and_loss in best_models:
        best_config, best_params = get_best_config_and_params_from_run(
            model_name_and_loss[0], path_to_ray_models, True)
        model_name = model_name_and_loss[0]
        model = json_config['architecture']
        model.load_state_dict(best_params)
        # set to eval mode
        model.eval()
        outputs = model(test_dict['data']).view(-1)
        x_axis, y_axis, _ = roc_curve(test_dict['labels'], outputs)
        auc = roc_auc_score(test_dict['labels'], outputs)
        ax.plot(x_axis, y_axis, label=model_name)
    ax.legend()
    figname = json_config['output_models_dir'] + '_best_models_rocs'
    plt.savefig(figname, dpi=300)
