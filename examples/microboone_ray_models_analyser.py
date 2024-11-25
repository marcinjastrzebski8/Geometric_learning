"""
Load best models from a ray run and test on test microboone data.
NOTE: Would be nice some general code to avoid rewriting every time I work on a new project.
"""
import argparse
from pathlib import Path
import json

from sklearn.metrics import roc_auc_score, roc_curve
import wandb
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pennylane import RX, RY, RZ
import pandas as pd
import numpy as np

from examples.utils import get_model_names_from_wandb, get_best_config_and_params_from_run
from data.datasets import dataset_lookup
from src.torch_architectures import ConvolutionalEQNEC
from src.geometric_classifier import BasicClassifierTorch
from src.quanvolution import EquivariantQuanvolution2DTorchLayer
from src.ansatze import MatchCallumAnsatz

api = wandb.Api()

def prep_equiv_quanv_model(is_first_layer, n_layers, n_reuploads, param_init_max_vals):
    """
    Used to reproduce the pennylane TorchLayers used to define the model.

    NOTE: This atm is just copied from the study script... not sure how else I could be saving these TorchLayers
    Will have to figure out how to generalise.
    """

    # CHANGE TO CALLUM ANSATZ ONCE HE SENDS DETAILS
    patch_circuit = BasicClassifierTorch(feature_map='RotEmbedding',
                                            ansatz=MatchCallumAnsatz,
                                            size=4,
                                            n_reuploads=n_reuploads)
    patch_circuit_properties = {
        'n_layers': n_layers, 'ansatz_block': [RY, RZ], 'embedding_pauli': RX}
    # both layers take same size in because we'll use stride=1

    if is_first_layer:
        input_channel_side_len = 21
        quantum_circs = [patch_circuit]
        quantum_circs_properties = [patch_circuit_properties]

    else:
        # NOTE: this is hardcoded based on filter size and stride - im sure there's a general formula
        input_channel_side_len = 21 - 1
        # using the same ansatz for each pose
        quantum_circs = [patch_circuit for i in range(4)]
        quantum_circs_properties = [patch_circuit_properties
                                    for i in range(4)]

    group_info = {'size': 4}
    # NOTE: parameter shape is kind of hardcoded, could do some lookup table based on ansatz
    quanv_layer = EquivariantQuanvolution2DTorchLayer(group_info,
                                                        is_first_layer,
                                                        quantum_circs,
                                                        quantum_circs_properties,
                                                        (input_channel_side_len,
                                                        input_channel_side_len),
                                                        1,
                                                        2,
                                                        [{'params': (
                                                            n_reuploads, n_layers, 4, 2)}],
                                                        param_init_max_vals)
    return quanv_layer

def validate_ray_models(json_config, n_models_to_keep, test_dataset_name:str):
    """
    Given a ray run and some metadata, check performance of the best found models on some test data.
    """
    path_to_ray_models = Path('ray_runs') / json_config['output_models_dir']
    models_w_losses = get_model_names_from_wandb(
        api, json_config['output_models_dir'], 'val_acc', 'max')
    best_models = sorted(models_w_losses.items(), key=lambda item: item[1][0], reverse = True)[
        :int(n_models_to_keep)]

    fig, ax = plt.subplots(1, 1)
    aucs = []
    x_axes = []
    y_axes = []

    test_dataset = dataset_lookup[test_dataset_name]()
    test_dataset = torch.utils.data.Subset(test_dataset, range(10000))
    test_loader = DataLoader(test_dataset, batch_size = 10, shuffle = False)
    

    for model_name_and_loss in best_models:
        model_name, model_loss, loss_checkpoint_id = model_name_and_loss[0], model_name_and_loss[1][0], model_name_and_loss[1][1]
        print(f'on model {model_name}')
        best_config, best_params = get_best_config_and_params_from_run(
            model_name, path_to_ray_models, model_name_and_loss[1][1], True)

        architecture_config = {**best_config, 
        'quanv0': prep_equiv_quanv_model(True, best_config['n_layers'], best_config['n_reuploads'], best_config['param_init_max_vals']),
        'quanv1': prep_equiv_quanv_model(False, best_config['n_layers'], best_config['n_reuploads'], best_config['param_init_max_vals']),
        'image_size': 21}

        model = ConvolutionalEQNEC(architecture_config)
        model.load_state_dict(best_params)
        # set to eval mode
        model.eval()
        outputs_gathered = []
        labels_gathered = []
        for test_batch, test_labels in test_loader:
            outputs = model(test_batch).view(-1)
            outputs_gathered.append(outputs.detach().numpy())
            labels_gathered.append(test_labels.detach().numpy())
        outputs_gathered = np.array(outputs_gathered).flatten()
        labels_gathered = np.array(labels_gathered).flatten()

        x_axis, y_axis, _ = roc_curve(labels_gathered, outputs_gathered)
        x_axes.append(x_axis)
        y_axes.append(y_axis)
        auc = roc_auc_score(labels_gathered, outputs_gathered)
        aucs.append(auc)

    df_dict = {'auc':aucs,
        'x_axis': x_axes,
        'y_axis': y_axes,
        'model': best_models}
    
    #TODO: MAKE MORE ROBUST NAME
    pd.DataFrame(df_dict).to_pickle('aucs_from_microboone_analysis')
    best_aucs = sorted(aucs, key=lambda item: item, reverse=True)[
        :int(10)]
    for auc, model_name_and_loss, x_axis, y_axis in zip(aucs, best_models, x_axes, y_axes):
        model_name, model_loss, loss_checkpoint_id = model_name_and_loss[0], model_name_and_loss[1][0], model_name_and_loss[1][1]

        if auc in best_aucs:
            auc_text = 'AUC: '+format(auc, '.3f')
            ax.plot(x_axis, y_axis, label=auc_text + ', '+ model_name)
    ax.legend()
    figname = json_config['output_models_dir'] + '_best_models_rocs'
    plt.savefig(figname, dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_config_path')
    parser.add_argument('n_models_to_keep')
    parser.add_argument('test_dataset_name')
    parse_args = parser.parse_args()
    # load config
    with open(parse_args.run_config_path, 'r', encoding='utf-8') as f:
        load_config = json.load(f)
    validate_ray_models(load_config, parse_args.n_models_to_keep,
                        parse_args.test_dataset_name)