from sklearn.metrics import roc_auc_score, roc_curve
from src.geometric_classifier import GeometricClassifierAutotwirlJax
from src.embeddings import RotEmbedding
from src.twirling import c4_rep_on_qubits
from src.losses import sigmoid_activation
from utils import get_model_names_from_wandb, get_best_config_and_params_from_run, SymmetricDatasetJax
import wandb
import pennylane as qml
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import json


def validate_ray_models(json_config, n_models_to_keep):
    # connect to wandb
    api = wandb.Api()
    # get best models from ray
    path_to_ray_models = Path('ray_runs') / json_config['output_models_dir']
    models_w_losses = get_model_names_from_wandb(
        api, json_config['output_models_dir'])
    best_models = sorted(models_w_losses.items(), key=lambda item: item[1])[
        :int(n_models_to_keep)]
    image_size = json_config['image_size']

    # valid data
    # first 500 used for training - use rest for test
    data = SymmetricDatasetJax(1000, image_size)
    test_data = data[500:]

    n_wires = image_size**2
    if image_size == 2:
        group_commuting_meas = qml.Z(0)@qml.Z(1)@qml.Z(2)@qml.Z(3)
    elif image_size == 3:
        group_commuting_meas = qml.Z(4)
    else:
        raise NotImplementedError('image size not supported')

    fig, ax = plt.subplots(1, 1)

    for model_name_and_loss in best_models:
        best_config, best_params = get_best_config_and_params_from_run(
            model_name_and_loss[0], path_to_ray_models)
        model = GeometricClassifierAutotwirlJax('RotEmbedding', 'GeneralCascadingAnsatz', n_wires,
                                                best_config['twirled_bool'], c4_rep_on_qubits, group_commuting_meas, image_size=image_size)
        model_fn = model.prediction_circuit
        # TODO: DO THIS
        """
        if train_models_fully:
            if model_not_trained_for_200_steps:
                model.train()
        """
        # test on unseen data and get the roc auc
        preds = []
        for data_point in test_data[0]:
            pred = sigmoid_activation(
                model_fn(best_params, data_point, best_config))
            preds.append(pred)
        x_axis, y_axis, _ = roc_curve(test_data[1], preds)
        auc = roc_auc_score(test_data[1], preds)
        auc_text = 'AUC: '+format(auc, '.3f')
        if best_config['twirled_bool']:
            linestyle = '--'
        else:
            linestyle = '-'
        ax.plot(x_axis, y_axis, label=auc_text, linestyle=linestyle)

    ax.legend()
    figname = json_config['output_models_dir'] + '_best_models_rocs'
    plt.savefig(figname, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_config_path')
    parser.add_argument('n_models_to_keep')
    parse_args = parser.parse_args()
    # load config
    with open(parse_args.run_config_path, 'r', encoding='utf-8') as f:
        load_config = json.load(f)
    validate_ray_models(load_config, parse_args.n_models_to_keep)
