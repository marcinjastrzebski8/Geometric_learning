"""
Load best models from a ray run and test on test microboone data.
NOTE: Would be nice some general code to avoid rewriting every time I work on a new project.
"""
import argparse
from pathlib import Path
import json

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import wandb
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pennylane import RX, RY, RZ
import pandas as pd
import numpy as np

from examples.utils import get_model_names_from_wandb, get_best_config_and_params_from_run
from examples.equivariant_quanvolution_study_with_trainer import prep_equiv_quant_classifier, prep_equiv_quanv_model
from data.datasets import dataset_lookup
from src.torch_architectures import ConvolutionalEQNEC, ConvolutionalEQEQ
from src.geometric_classifier import BasicClassifierTorch
from src.quanvolution import EquivariantQuanvolution2DTorchLayer
from src.ansatze import MatchCallumAnsatz

api = wandb.Api()


def validate_ray_models(json_config, 
    n_models_to_keep, 
    test_dataset_name: str,
    test_data_size:int, 
    dataset_kwargs: None | dict = None):
    """
    Given a ray run and some metadata, check performance of the best found models on some test data.
    """
    output_models_dir = json_config['output_models_dir']
    path_to_ray_models = Path('ray_runs') / output_models_dir
    models_w_losses = get_model_names_from_wandb(
        api, output_models_dir, 'val_acc', 'max')
    best_models = sorted(models_w_losses.items(), key=lambda item: item[1][0], reverse=True)[
        :int(n_models_to_keep)]

    fig, ax = plt.subplots(1, 1)
    aucs = []
    accs = []
    x_axes = []
    y_axes = []

    if dataset_kwargs is None:
        dataset_kwargs = {}

    test_dataset = dataset_lookup[test_dataset_name+'TestData'](**dataset_kwargs)
    test_dataset = torch.utils.data.Subset(test_dataset, range(test_data_size))
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    for model_name_and_loss in best_models:
        model_name, model_loss, loss_checkpoint_id = model_name_and_loss[
            0], model_name_and_loss[1][0], model_name_and_loss[1][1]
        print(f'on model {model_name}')
        best_config, best_params = get_best_config_and_params_from_run(
            model_name, path_to_ray_models, loss_checkpoint_id, True)
        architecture_codeword = json_config['architecture_codeword']
        if architecture_codeword == 'EQNEC':
            architecture_config = {**best_config,
                                   'quanv0': prep_equiv_quanv_model(best_config, json_config, True),
                                   'quanv1': prep_equiv_quanv_model(best_config, json_config, False),
                                   'image_size': json_config['image_size']}

            model: ConvolutionalEQEQ | ConvolutionalEQNEC = ConvolutionalEQNEC(
                architecture_config)
        elif architecture_codeword == 'EQEQ':
            # best_config probably not needed here
            architecture_config = {**best_config,
                                   'quanv0': prep_equiv_quanv_model(best_config, json_config, True),
                                   'quanv1': prep_equiv_quanv_model(best_config, json_config, False),
                                   'quantum_classifier': prep_equiv_quant_classifier(best_config),
                                   'pooling_kernels_size': json_config['pooling_kernels_size'],
                                   'pooling_strides': json_config['pooling_strides']
                                   }
            model = ConvolutionalEQEQ(architecture_config)
        else:
            raise ValueError('Architecture not supported: ',
                             architecture_codeword)

        model.load_state_dict(best_params)
        # set to eval mode
        model.eval()
        outputs_gathered = []
        labels_gathered = []
        for test_batch, test_labels in test_loader:
            outputs = model(test_batch).view(-1)
            outputs_gathered.append(outputs.detach().numpy())
            labels_gathered.append(test_labels.detach().numpy())
        outputs_gathered_arr = np.array(outputs_gathered).flatten()
        labels_gathered_arr = np.array(labels_gathered).flatten()

        fpr, tpr, thresholds = roc_curve(
            labels_gathered_arr, outputs_gathered_arr)
        x_axes.append(fpr)
        y_axes.append(tpr)
        # find acc with best threshold assuming same weight for fpr and tpr
        threshold_id = np.argmax(tpr-fpr)
        threshold = thresholds[threshold_id]
        labels_predicted = [0 if output <
                            threshold else 1 for output in outputs_gathered_arr]
        acc = accuracy_score(labels_gathered_arr, labels_predicted)
        auc = roc_auc_score(labels_gathered_arr, outputs_gathered_arr)
        aucs.append(auc)
        accs.append(acc)

    df_dict = {'auc': aucs,
               'acc': accs,
               'fpr': x_axes,
               'tpr': y_axes,
               'model': best_models}

    pd.DataFrame(df_dict).to_pickle(f'microboone_analysis_{output_models_dir}')
    best_aucs = sorted(aucs, key=lambda item: item, reverse=True)[
        :int(10)]
    for auc, model_name_and_loss, x_axis, y_axis in zip(aucs, best_models, x_axes, y_axes):
        model_name, model_loss, loss_checkpoint_id = model_name_and_loss[
            0], model_name_and_loss[1][0], model_name_and_loss[1][1]

        if auc in best_aucs:
            auc_text = 'AUC: '+format(auc, '.3f')
            ax.plot(x_axis, y_axis, label=auc_text + ', ' + model_name)
    ax.legend()
    figname = output_models_dir + '_best_models_rocs'
    plt.savefig(figname, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_config_path')
    parser.add_argument('n_models_to_keep', type=int)
    parser.add_argument('test_dataset_name')
    parser.add_argument('test_data_size', type=int)
    parser.add_argument("--microboone_patch_size", type=int)
    parse_args = parser.parse_args()
    # load config
    with open(parse_args.run_config_path, 'r', encoding='utf-8') as f:
        load_config = json.load(f)
    if parse_args.test_dataset_name == 'Microboone':
        dataset_kwargs: dict | None = {
            'patch_size': parse_args.microboone_patch_size}
    else:
        dataset_kwargs = None
    validate_ray_models(load_config, parse_args.n_models_to_keep,
                        parse_args.test_dataset_name, parse_args.test_data_size, dataset_kwargs)
