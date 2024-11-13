from sklearn.metrics import roc_auc_score, roc_curve
from src.geometric_classifier import GeometricClassifierAutotwirlJax, BasicClassifier
from src.embeddings import RotEmbedding
from src.twirling import c4_rep_on_qubits, C4On9QEquivGate1Local, C4On9QEquivGate2Local
from src.losses import sigmoid_activation
from src.ansatze import GeometricAnsatzConstructor
from examples.utils import get_model_names_from_wandb, get_best_config_and_params_from_run, SymmetricDatasetJax
import wandb
import pennylane as qml
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import json


def construct_gate_instructions(best_config, run_with_errors=False, model_name=None):

    if run_with_errors:
        assert model_name is not None
        # SOME INFO WAS NOT SAVED DURING FIRST RUN
        # Need to reconstruct with manually looked up info.
        if model_name == 'train_ray_ee957_00067':
            pauli_words_used = ['XY', 'XZ', 'XZ']
            gates_used = [qml.RX, qml.RY, qml.RX]
        elif model_name == 'train_ray_ee957_00026':
            pauli_words_used = ['ZZ', 'YZ', 'ZZ']
            gates_used = [qml.RY, qml.RX, qml.RX]
        elif model_name == 'train_ray_ee957_00011':
            pauli_words_used = ['ZZ', 'XX', 'ZZ']
            gates_used = [qml.RX, qml.RY, qml.RY]
        elif model_name == 'train_ray_ee957_00017':
            pauli_words_used = ['ZZ', 'YY', 'ZZ']
            gates_used = [qml.RX, qml.RY]
        elif model_name == 'train_ray_ee957_00003':
            pauli_words_used = ['YY', 'XY', 'XY']
            gates_used = [qml.RX, qml.RY, qml.RX]
    else:
        # NOTE: gate options are always produced at full length (maximum length considered for hyperopt run)
        # hence they need to be matched to the actual length of a given config
        # not the cleanest looking solution but idk how to make hyperparams depend on each other in ray
        n_1local_gates_used = len(best_config['gate_1local_placements'])
        n_2local_gates_used = len(best_config['gate_2local_placements'])
        gates_used = best_config['gates'][:n_1local_gates_used]
        pauli_words_used = best_config['pauli_words'][:n_2local_gates_used]

    gate_1local_instructions = [{'gate': gate, 'gate_placement': placement} for gate, placement in zip(
        gates_used, best_config['gate_1local_placements'])]

    gate_2local_instructions = [{'pauli_word': pauli_word, 'gate_placement': placement}
                                for pauli_word, placement in zip(pauli_words_used, best_config['gate_2local_placements'])]

    return gate_1local_instructions, gate_2local_instructions


def validate_ray_models(json_config, n_models_to_keep, geo_classifier_implementation):
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
    # first 500 was used for training - use rest for test
    data = SymmetricDatasetJax(1500, image_size)
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

        # two main implementations of the geometric model
        if 'auto_twirl' in geo_classifier_implementation:
            model = GeometricClassifierAutotwirlJax('RotEmbedding', 'GeneralCascadingAnsatz', n_wires,
                                                    best_config['twirled_bool'], c4_rep_on_qubits, group_commuting_meas, image_size=image_size)
        elif geo_classifier_implementation == 'precomputed_bank':
            # first run had an error - some info was not saved
            run_with_errors = json_config['output_models_dir'] == '3x3_geo_precomputed_bank_method'

            model_name = model_name_and_loss[0]
            gate_1local_instructions, gate_2local_instructions = construct_gate_instructions(
                best_config, run_with_errors, model_name)

            best_config['gate_1local_instructions'] = gate_1local_instructions
            best_config['gate_2local_instructions'] = gate_2local_instructions
            model = BasicClassifier(
                'RotEmbedding', GeometricAnsatzConstructor, n_wires, group_commuting_meas)
        else:
            raise NotImplementedError()
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
        try:
            if best_config['twirled_bool']:
                linestyle = '--'
            else:
                linestyle = '-'
        except KeyError:
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
    classifier_implementation = load_config['geo_classifier_implementation']
    validate_ray_models(load_config, parse_args.n_models_to_keep,
                        classifier_implementation)
