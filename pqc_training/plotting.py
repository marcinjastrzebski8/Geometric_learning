from experiments import ExperimentPlotter, get_experiment_results_dir, get_roc_data
from examples.utils import (
    DatasetFactory,
    Permute,
    convert_h5_to_lists,
    DynamicEntanglement,
    entanglement_capability,
    expressivity,
    entanglement_measure,
)
from new_test_qae import QuantumTester, ClassicalTester
import numpy as np
from sklearn.metrics import roc_curve, auc
import dill
from pathlib import Path
import matplotlib.pyplot as plt
import pennylane as qml

# from testing import test_on_new_anomalies
from pennylane import numpy as qnp
from examples.utils import weight_init
import itertools
from functools import partial
import os
from scipy.interpolate import CubicSpline

plt.rcParams.update({"text.usetex": True, "font.family": "Lucida Grande"})
plt.rcParams["figure.dpi"] = 150


def quantum_test_losses(model, k_folds, test_data, test_labels):
    tester = QuantumTester(
        model["model_fn"],
        model["loss_fn"],
        model["params"],
        model,
    )
    bg_fold_losses = []
    sg_fold_losses = []
    aucs = []
    for k in range(k_folds):
        bg_losses, sg_losses = tester.compute_losses(test_data[k], test_labels[k])

        labels = np.concatenate(
            (np.zeros(bg_losses.shape[0]), np.ones(sg_losses.shape[0]))
        )

        preds = np.concatenate((bg_losses, sg_losses))
        fpr, tpr, _ = roc_curve(labels, preds, drop_intermediate=False)
        auc_score = auc(fpr, tpr)
        aucs.append(auc_score)
        bg_fold_losses.append(bg_losses)
        sg_fold_losses.append(sg_losses)
    return bg_fold_losses, sg_fold_losses, aucs, fpr, tpr


def classical_test_losses(model, k_folds, test_data, test_labels):
    tester = ClassicalTester(model["model_fn"], model["loss_fn"], model["params"])

    bg_fold_losses = []
    sg_fold_losses = []
    aucs = []
    for k in range(k_folds):
        bg_losses, sg_losses = tester.compute_losses(test_data[k], test_labels[k])
        labels = np.concatenate(
            (np.zeros(bg_losses.shape[0]), np.ones(sg_losses.shape[0]))
        )

        preds = np.concatenate((bg_losses, sg_losses))
        fpr, tpr, _ = roc_curve(test_labels[k], preds, drop_intermediate=False)
        auc_score = auc(fpr, tpr)
        print("auc score: ", auc_score)
        bg_fold_losses.append(bg_losses)
        sg_fold_losses.append(sg_losses)
        aucs.append(auc_score)

    return bg_fold_losses, sg_fold_losses, aucs


def load_model(path):
    with open(path, "rb") as f:
        model = dill.load(f)
    return model


def load_pickle_file(path):
    with open(path, "rb") as f:
        file = dill.load(f)
    return file


def compute_auc_scores(ids, k_folds, bg_loss, sg_loss):
    auc_list = []
    for i, id_name in enumerate(ids):
        for j in range(k_folds):
            fpr, tpr = get_roc_data(bg_loss[i][j], sg_loss[i][j])

            auc_list.append(auc(fpr, tpr))

    return auc_list


def fig_higgs_inp4_roc(
    quantum_model_paths,
    classical_model_paths,
    variables,
    k_folds,
    losses_path,
    signal,
    signal_dataset,
    background_dataset,
    quantum_labels,
    classical_labels,
    filename,
):
    quantum_models = []
    for model_path in quantum_model_paths:
        print(model_path)
        quantum_models.append(load_model(model_path))

    classical_models = []
    for model_path in classical_model_paths:
        classical_models.append(load_model(model_path))

    ids = [
        "original",
        "new",
        "no_entanglement",
        "hea",
        "shallow",
        "deep",
    ]
    # ["#4f8c9d", "#91ca32", "#841ea4", "#34daea", "#3f436d", "#dc8bfe"]
    q_palette = ["#58b5e1", "#d65f41", "#387561", "#21f0b6"]
    c_palette = ["#4f8c9d", "#dc8bfe"]  # ["#58b5e1", "#d65f41"]

    with open(signal_dataset, "rb") as f:
        signal_data = dill.load(f)

    with open(background_dataset, "rb") as f:
        background_data = dill.load(f)

    test_data_bank = []
    test_label_bank = []
    for idx, background_fold in enumerate(background_data):
        signal_fold = signal_data[idx]
        new_test_data = np.vstack((background_fold, signal_fold))
        new_test_labels = np.vstack(
            (np.zeros(background_fold.shape[0]), np.ones(signal_fold.shape[0]))
        ).ravel()
        test_data_bank.append(new_test_data)
        test_label_bank.append(new_test_labels)

    test_bg_q_losses = []
    test_sg_q_losses = []
    test_bg_c_losses = []
    test_sg_c_losses = []
    print("computing losses")

    for i, model in enumerate(quantum_models):
        bg_path = Path(f"{losses_path}/{ids[i]}_bg_losses.pkl")
        sg_path = Path(f"{losses_path}/{ids[i]}_sg_losses.pkl")  # alter name
        if bg_path.is_file() & sg_path.is_file():
            bg_loss = load_pickle_file(bg_path)
            sg_loss = load_pickle_file(sg_path)
        else:
            print("file not found!")
            bg_loss, sg_loss = quantum_test_losses(
                model, k_folds, test_data_bank, test_label_bank
            )
            with open(f"{losses_path}/{ids[i]}_bg_losses.pkl", "wb") as f:
                dill.dump(bg_loss, f)

            with open(f"{losses_path}/{ids[i]}_sg_losses.pkl", "wb") as f:
                dill.dump(sg_loss, f)

        test_bg_q_losses.append(bg_loss)
        test_sg_q_losses.append(sg_loss)

    for i, model in enumerate(classical_models):
        i = len(quantum_models) + i
        bg_path = Path(f"{losses_path}/{ids[i]}_bg_losses.pkl")
        sg_path = Path(f"{losses_path}/{ids[i]}_sg_losses.pkl")

        if bg_path.is_file() & sg_path.is_file():
            bg_loss = load_pickle_file(bg_path)
            sg_loss = load_pickle_file(sg_path)

        else:
            print("file not found!")
            bg_loss, sg_loss = classical_test_losses(
                model, k_folds, test_data_bank, test_label_bank
            )

        test_bg_c_losses.append(bg_loss)
        test_sg_c_losses.append(sg_loss)

    experiment_plotter = ExperimentPlotter(
        test_bg_q_losses,
        test_sg_q_losses,
        test_bg_c_losses,
        test_sg_c_losses,
        quantum_labels,
        classical_labels,
        k_folds,
        q_palette,
        c_palette,
    )
    experiment_plotter.plot_performance(
        f"/unix/qcomp/users/cduffy/anomaly_detection/report_figures/{filename}.pdf",
    )


def plot_train_size_experiment(og_experiment_dir, new_experiment_dir, k_folds):
    train_sizes = np.array(
        [
            1,
            5,
            10,
            25,
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            700,
            800,
            900,
            1000,
        ]
    )

    plotting_config = {
        "experiment_var": {"train_size": train_sizes},
        "k_folds": k_folds,
    }

    (
        og_test_bg_loss,
        og_test_sg_loss,
        og_test_bgc_loss,
        og_test_sgc_loss,
    ) = convert_h5_to_lists(
        f"{og_experiment_dir}/experiment_losses.h5", plotting_config
    )

    (
        new_test_bg_loss,
        new_test_sg_loss,
        new_test_bgc_loss,
        new_test_sgc_loss,
    ) = convert_h5_to_lists(
        f"{new_experiment_dir}/experiment_losses.h5", plotting_config
    )

    original_auc_scores = compute_auc_scores(
        train_sizes, k_folds, og_test_bg_loss, og_test_sg_loss
    )

    new_auc_scores = compute_auc_scores(
        train_sizes, k_folds, new_test_bg_loss, new_test_sg_loss
    )

    train_sizes = np.repeat(train_sizes, k_folds)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.semilogx(train_sizes, original_auc_scores, "o", markersize=1)
    ax.semilogx(train_sizes, new_auc_scores, "o", markersize=1)
    ax.set_ylim(0.8, 1)
    fig.savefig("train_sizes.pdf")


def get_rejection_rate(
    original_q_model_path,
    new_q_model_path,
    shallow_c_model_path,
    deep_c_model_path,
    variables,
    k_folds,
    losses_path,
    filename,
    tpr_working_point=0.8,
):
    original_q_model = load_model(original_q_model_path)
    new_q_model = load_model(new_q_model_path)
    shallow_c_model = load_model(shallow_c_model_path)
    deep_c_model = load_model(deep_c_model_path)

    quantum_models = [original_q_model, new_q_model]
    classical_models = [shallow_c_model, deep_c_model]

    ids = ["original", "new", "shallow", "deep"]

    # now to compute losses
    dataset_factory = DatasetFactory()
    dataset_config = {
        "root_dir": "data/higgs_dataset",
        "scale": True,
        "transform": Permute(variables),
    }
    test_data = dataset_factory.create_dataset("higgs", **dataset_config)

    test_data_bank = []
    test_label_bank = []
    for _ in range(k_folds):
        k_test_data, k_test_labels = test_data.get_test_chunk(5000, 5000)
        test_data_bank.append(k_test_data)
        test_label_bank.append(k_test_labels)

    test_bg_q_losses = []
    test_sg_q_losses = []
    test_bg_c_losses = []
    test_sg_c_losses = []
    print("computing losses")
    # save these losses for later use
    for i, model in enumerate(quantum_models):
        bg_path = Path(f"{losses_path}/{ids[i]}_bg_losses.pkl")
        sg_path = Path(f"{losses_path}/{ids[i]}_sg_losses.pkl")
        if bg_path.is_file() & sg_path.is_file():
            bg_loss = load_pickle_file(bg_path)
            sg_loss = load_pickle_file(sg_path)
        else:
            print("file not found!")
            bg_loss, sg_loss = quantum_test_losses(
                model, k_folds, test_data_bank, test_label_bank
            )

        test_bg_q_losses.append(bg_loss)
        test_sg_q_losses.append(sg_loss)

        tester = QuantumTester(
            model["model_fn"], model["loss_fn"], model["params"], model
        )
        quantum_rejection = []
        f = open(filename, "a")
        for k in range(k_folds):
            preds = np.concatenate((bg_loss[k], sg_loss[k]))
            fpr, tpr, _ = roc_curve(test_label_bank[k], preds, drop_intermediate=False)
            one_over_fpr_mean, one_over_fpr = tester.get_fpr_around_tpr_point(
                fpr, tpr, tpr_working_point
            )
            quantum_rejection.append(one_over_fpr)

        quantum_rejection = np.concatenate(quantum_rejection).ravel()
        print(f"{ids[i]}: {np.mean(quantum_rejection)}+\-{np.std(quantum_rejection)}")
        f.write(
            f"{ids[i]}: {np.mean(quantum_rejection)}+/-{np.std(quantum_rejection)}\n"
        )
    for i, model in enumerate(classical_models):
        i = len(quantum_models) + i
        bg_path = Path(f"{losses_path}/{ids[i]}_bg_losses.pkl")
        sg_path = Path(f"{losses_path}/{ids[i]}_sg_losses.pkl")

        if bg_path.is_file() & sg_path.is_file():
            bg_loss = load_pickle_file(bg_path)
            sg_loss = load_pickle_file(sg_path)

        else:
            print("file not found!")
            bg_loss, sg_loss = classical_test_losses(
                model, k_folds, test_data_bank, test_label_bank
            )
        test_bg_c_losses.append(bg_loss)
        test_sg_c_losses.append(sg_loss)

        tester = ClassicalTester(model["model_fn"], model["loss_fn"], model["params"])
        classical_rejection = []
        for k in range(k_folds):
            preds = np.concatenate((bg_loss[k], sg_loss[k]))
            fpr, tpr, _ = roc_curve(test_label_bank[k], preds, drop_intermediate=False)
            one_over_fpr_mean, one_over_fpr = tester.get_fpr_around_tpr_point(fpr, tpr)
            classical_rejection.append(one_over_fpr)
        classical_rejection = np.concatenate(classical_rejection).ravel()

        print(
            f"{ids[i]}: {np.mean(classical_rejection)}+/-{np.std(classical_rejection)}"
        )
        f.write(
            f"{ids[i]}: {np.mean(classical_rejection)}+/-{np.std(classical_rejection)}\n"
        )

    f.close()


def get_entanglement_capabilities(
    original_q_model_path,
    new_q_model_path,
    filename,
    save_dir,
):
    original_q_model = load_model(original_q_model_path)
    new_q_model = load_model(new_q_model_path)
    ids = ["original", "new"]
    plt.close()
    fig, ax = plt.subplots()
    models = [original_q_model, new_q_model]

    hist_colors = ["#68affc", "#1c5f1e"]
    vline_colors = ["#f764de", "#74ce4b"]
    for i, model in enumerate(models):
        snapshot_file = Path(f"{save_dir}_snapshot_{ids[i]}.pkl")
        distribution_file = Path(f"{save_dir}_dist_{ids[i]}.pkl")

        if snapshot_file.is_file():
            print(f"Snapshot file found for {ids[i]}!")
            current_entanglement = load_pickle_file(snapshot_file)
        else:
            print(f"Snapshot no file found for {ids[i]}!")
            entanglement_snapshot = DynamicEntanglement(model)
            current_entanglement = entanglement_snapshot(model["params"])

            with open(f"{save_dir}_snapshot_{ids[i]}.pkl", "wb") as f:
                dill.dump(current_entanglement, f)

        print(f"{ids[i]}: {current_entanglement}")

        if distribution_file.is_file():
            print(f"Distribution file found for {ids[i]}!")
            entanglements = load_pickle_file(distribution_file)
            mean_entanglement = np.mean(entanglements)
            std_entanglement = np.std(entanglements)

        else:
            print(f"No distribution file found for {ids[i]}!")

            def circuit(params, wires, config):
                model["ansatz_fn"](params, range(wires), config)
                return qml.state()

            dev = qml.device("default.qubit", wires=model["input_size"])
            circuit_fn = qml.QNode(circuit, dev)
            (
                mean_entanglement,
                std_entanglement,
                min_entanglement,
                max_entanglement,
                entanglements,
            ) = entanglement_capability(
                circuit_fn, 10, model, model["input_size"], n_shots=1000
            )

            with open(f"{save_dir}_dist_{ids[i]}.pkl", "wb") as f:
                dill.dump(entanglements, f)

        print(f"{ids[i]}: {mean_entanglement}")
        ax.hist(
            entanglements,
            bins=70,
            density=True,
            label=f"{ids[i]} ansatz, mean: {mean_entanglement}+/-{std_entanglement}",
            alpha=0.5,
            color=hist_colors[i],
            histtype="step",
            # stacked=True,
        )
        ax.axvline(
            current_entanglement,
            color=vline_colors[i],
            label=f"trained {ids[i]} ansatz: {current_entanglement}",
        )

    plt.legend()
    fig.savefig(filename)
    plt.close()


def get_expressivities(original_q_model_path, new_q_model_path):
    original_q_model = load_model(original_q_model_path)
    new_q_model = load_model(new_q_model_path)
    ids = ["original", "new"]
    models = [original_q_model, new_q_model]
    for i, model in enumerate(models):

        def circuit(params, wires, config):
            model["ansatz_fn"](params, range(wires), config)
            return qml.density_matrix(range(wires))

        dev = qml.device("default.qubit", wires=model["input_size"])
        circuit_fn = qml.QNode(circuit, dev)
        exp = expressivity(
            circuit_fn,
            model,
            model["input_size"],
            model["params"].shape,
            model["input_size"],
        )
        print(f"{ids[i]}: {exp}")


def fig_qcd_roc(
    quantum_model_paths,
    classical_model_paths,
    k_folds,
    losses_path,
    filename,
    signal,
    signal_dataset,
    background_dataset,
    quantum_labels,
    classical_labels,
):
    quantum_models = []
    for model_path in quantum_model_paths:
        model = load_model(model_path)
        quantum_models.append(model)

    classical_models = []
    for model_path in classical_model_paths:
        model = load_model(model_path)
        classical_models.append(model)

    ids = ["new", "shallow", "deep"]

    q_palette = ["#58b5e1"]  # "#58b5e1", "#d65f41", "#387561", "#e91451", "#89b786"
    c_palette = ["#4f8c9d", "#dc8bfe"]

    # now to compute losses
    with open(signal_dataset, "rb") as f:
        signal_data = dill.load(f)

    with open(background_dataset, "rb") as f:
        background_data = dill.load(f)

    test_data_bank = []
    test_label_bank = []
    for idx, background_fold in enumerate(background_data):
        signal_fold = signal_data[idx]
        new_test_data = np.vstack((background_fold, signal_fold))
        new_test_labels = np.vstack(
            (np.zeros(background_fold.shape[0]), np.ones(signal_fold.shape[0]))
        ).ravel()
        test_data_bank.append(new_test_data)
        test_label_bank.append(new_test_labels)

    test_bg_q_losses = []
    test_sg_q_losses = []
    test_bg_c_losses = []
    test_sg_c_losses = []
    print("computing losses")

    for i, model in enumerate(quantum_models):
        bg_path = Path(f"{losses_path}/{ids[i]}_tuned_{signal}_bg_losses.pkl")
        sg_path = Path(f"{losses_path}/{ids[i]}_tuned_{signal}_sg_losses.pkl")
        print(bg_path.is_file())
        if bg_path.is_file() & sg_path.is_file():
            bg_loss = load_pickle_file(bg_path)
            sg_loss = load_pickle_file(sg_path)
        else:
            print(ids[i], " file not found!")
            bg_loss, sg_loss = quantum_test_losses(
                model, k_folds, test_data_bank, test_label_bank
            )
            with open(f"{losses_path}/{ids[i]}_bg_losses.pkl", "wb") as f:
                dill.dump(bg_loss, f)

            with open(f"{losses_path}/{ids[i]}_{signal}_losses.pkl", "wb") as f:
                dill.dump(sg_loss, f)

        test_bg_q_losses.append(bg_loss)
        test_sg_q_losses.append(sg_loss)

    for i, model in enumerate(classical_models):
        i = len(quantum_models) + i
        bg_path = Path(f"{losses_path}/{ids[i]}_bg_losses.pkl")
        sg_path = Path(f"{losses_path}/{ids[i]}_{signal}_losses.pkl")

        if bg_path.is_file() & sg_path.is_file():
            bg_loss = load_pickle_file(bg_path)
            sg_loss = load_pickle_file(sg_path)

        else:
            print("file not found!")
            bg_loss, sg_loss = classical_test_losses(
                model, k_folds, test_data_bank, test_label_bank
            )
            with open(f"{losses_path}/{ids[i]}_bg_losses.pkl", "wb") as f:
                dill.dump(bg_loss, f)

            with open(f"{losses_path}/{ids[i]}_{signal}_losses.pkl", "wb") as f:
                dill.dump(sg_loss, f)

        test_bg_c_losses.append(bg_loss)
        test_sg_c_losses.append(sg_loss)

    experiment_plotter = ExperimentPlotter(
        test_bg_q_losses,
        test_sg_q_losses,
        test_bg_c_losses,
        test_sg_c_losses,
        quantum_labels,
        classical_labels,
        k_folds,
        q_palette,
        c_palette,
    )
    experiment_plotter.plot_performance(
        f"/unix/qcomp/users/cduffy/anomaly_detection/report_figures/{filename}.pdf",
    )


def plot_roc_quantum_only(
    quantum_model_paths,
    k_folds,
    losses_path,
    filename,
    signal,
    signal_dataset,
    background_dataset,
    quantum_labels,
):
    quantum_models = []
    for model_path in quantum_model_paths:
        model = load_model(model_path)
        quantum_models.append(model)

    ids = ["original", "new", "no_entanglement", "hea"]

    q_palette = ["#58b5e1", "#d65f41", "#387561", "#21f0b6"]

    # now to compute losses
    with open(signal_dataset, "rb") as f:
        signal_data = dill.load(f)

    with open(background_dataset, "rb") as f:
        background_data = dill.load(f)

    test_data_bank = []
    test_label_bank = []
    for idx, background_fold in enumerate(background_data):
        signal_fold = signal_data[idx]
        new_test_data = np.vstack((background_fold, signal_fold))
        new_test_labels = np.vstack(
            (np.zeros(background_fold.shape[0]), np.ones(signal_fold.shape[0]))
        ).ravel()
        test_data_bank.append(new_test_data)
        test_label_bank.append(new_test_labels)

    test_bg_q_losses = []
    test_sg_q_losses = []
    test_bg_c_losses = []
    test_sg_c_losses = []
    print("computing losses")

    for i, model in enumerate(quantum_models):
        bg_path = Path(f"{losses_path}/{ids[i]}_bg_losses.pkl")
        sg_path = Path(f"{losses_path}/{ids[i]}_{signal}_losses.pkl")

        if bg_path.is_file() & sg_path.is_file():
            bg_loss = load_pickle_file(bg_path)
            sg_loss = load_pickle_file(sg_path)
            print("FOLD NUMBER", len(sg_loss))

        else:
            print("file not found!")
            bg_loss, sg_loss = quantum_test_losses(
                model, k_folds, test_data_bank, test_label_bank
            )
            with open(f"{losses_path}/{ids[i]}_bg_losses.pkl", "wb") as f:
                dill.dump(bg_loss, f)

            with open(f"{losses_path}/{ids[i]}_{signal}_losses.pkl", "wb") as f:
                dill.dump(sg_loss, f)

        test_bg_q_losses.append(bg_loss)
        test_sg_q_losses.append(sg_loss)

    experiment_plotter = ExperimentPlotter(
        test_bg_q_losses,
        test_sg_q_losses,
        test_bg_c_losses,
        test_sg_c_losses,
        quantum_labels,  # "hea"
        ["deep", "shallow"],
        k_folds,
        q_palette,
        None,
    )
    experiment_plotter.plot_quantum_roc(
        f"/unix/qcomp/users/cduffy/anomaly_detection/report_figures/{filename}.pdf",
    )


def get_entanglement_data_capabilities(
    original_q_model_path,
    new_q_model_path,
    background_dataset,
    save_dir,
    filename,
):
    original_q_model = load_model(original_q_model_path)
    new_q_model = load_model(new_q_model_path)
    ids = ["original", "new"]

    with open(background_dataset, "rb") as f:
        background_data = dill.load(f)

    plt.close()
    fig, ax = plt.subplots()
    models = [original_q_model, new_q_model]

    hist_colors = ["#68affc", "#1c5f1e"]

    for i, model in enumerate(models):
        entanglement_file = Path(f"{save_dir}_{ids[i]}.pkl")

        if entanglement_file.is_file():
            print(f"Entanglement file found for {ids[i]}!")
            entanglements = load_pickle_file(entanglement_file)
        else:
            print(f"No entanglement file found for {ids[i]}!")
            entanglements = []
            params = model["params"]

            dev = qml.device("default.qubit", wires=model["input_size"])

            @qml.qnode(device=dev)
            def density_matrix(params, features, wires, model):
                model["embedding_fn"](features, range(wires))
                model["ansatz_fn"](params, range(wires), model)
                return qml.state()

            for _, fold in enumerate(background_data):
                for _, datapoint in enumerate(fold):
                    rho = density_matrix(params, datapoint, model["input_size"], model)
                    entanglements.append(entanglement_measure(rho))

            with open(f"{save_dir}_{ids[i]}.pkl", "wb") as f:
                dill.dump(entanglements, f)

        ax.hist(
            entanglements,
            bins=200,
            label=f"{ids[i]} ansatz, mean: {np.round(np.mean(entanglements), 3)}+/-{np.round(np.std(entanglements), 4)}",
            alpha=0.5,
            color=hist_colors[i],
            density=True,
            histtype="step",
        )

    ax.set_xlabel(r"$E(x)|_{\theta_{trained}}$")
    plt.legend()
    fig.savefig(filename)
    plt.close()


def get_entanglement_loss_corr(
    original_q_model_path,
    new_q_model_path,
    background_dataset,
    save_dir,
    filename,
):
    original_q_model = load_model(original_q_model_path)
    new_q_model = load_model(new_q_model_path)
    ids = ["original", "new"]

    with open(background_dataset, "rb") as f:
        background_data = dill.load(f)

    plt.close()
    fig, ax = plt.subplots()
    models = [original_q_model, new_q_model]

    hist_colors = ["#68affc", "#1c5f1e"]

    for i, model in enumerate(models):
        entanglement_file = Path(f"{save_dir}_entanglement_{ids[i]}.pkl")
        losses_file = Path(f"{save_dir}_losses_{ids[i]}.pkl")

        losses = []
        entanglements = []
        params = model["params"]

        if entanglement_file.is_file():
            print(f"Entanglement file found for {ids[i]}!")
            entanglements = load_pickle_file(entanglement_file)
            losses = load_pickle_file(losses_file)
        else:
            print(f"No entanglement file found for {ids[i]}!")
            dev = qml.device("default.qubit", wires=model["input_size"])

            @qml.qnode(device=dev)
            def density_matrix(params, features, wires, model):
                model["embedding_fn"](features, range(wires))
                model["ansatz_fn"](params, range(wires), model)
                return qml.state()

            for _, fold in enumerate(background_data):
                for _, datapoint in enumerate(fold):
                    rho = density_matrix(params, datapoint, model["input_size"], model)
                    entanglements.append(entanglement_measure(rho))

                    loss = model["loss_fn"](
                        model["params"],
                        qnp.array([datapoint]),
                        model["model_fn"],
                        model,
                    )
                    losses.append(loss)

            with open(entanglement_file, "wb") as f:
                dill.dump(entanglements, f)

            with open(losses_file, "wb") as f:
                dill.dump(losses, f)

        ax.plot(
            entanglements,
            losses,
            "o",
            markersize=2,
            alpha=0.6,
            label=ids[i],
            color=hist_colors[i],
        )

    plt.legend()
    fig.savefig(filename)
    plt.close()


def get_entanglement_capabilities_given_data(
    original_q_model_path,
    new_q_model_path,
    background_dataset,
    num_samples,
    save_dir,
    filename,
):
    original_q_model = load_model(original_q_model_path)
    new_q_model = load_model(new_q_model_path)
    ids = ["original", "new"]
    vline_colors = ["#68affc", "#74ce4b"]

    with open(background_dataset, "rb") as f:
        background_data = dill.load(f)

    plt.close()
    fig, ax = plt.subplots()
    models = [original_q_model, new_q_model]

    hist_colors = ["#68affc", "#1c5f1e"]

    for i, model in enumerate(models):
        entanglement_file = Path(f"{save_dir}_{ids[i]}.pkl")

        if entanglement_file.is_file():
            print(f"Entanglement file found for {ids[i]}!")
            mean_entanglements = load_pickle_file(entanglement_file)
            snapshot = load_pickle_file(f"{save_dir}_snapshot_{ids[i]}.pkl")
        else:
            print(f"No entanglement file found for {ids[i]}!")
            dev = qml.device("default.qubit", wires=model["input_size"])

            @qml.qnode(device=dev)
            def density_matrix(params, features, wires, model):
                model["embedding_fn"](features, range(wires))
                model["ansatz_fn"](params, range(wires), model)
                return qml.state()

            mean_entanglements = []
            for _ in range(num_samples):
                params = weight_init(
                    0,
                    2 * np.pi,
                    "uniform",
                    model["ansatz_fn"].shape(model["input_size"], model["layers"]),
                )

                entanglements = []

                for _, fold in enumerate(background_data):
                    for _, datapoint in enumerate(fold):
                        rho = density_matrix(
                            params, datapoint, model["input_size"], model
                        )
                        entanglements.append(entanglement_measure(rho))

                mean_entanglements.append(np.mean(entanglements))

            with open(entanglement_file, "wb") as f:
                dill.dump(mean_entanglements, f)

            params = model["params"]
            snapshot_entanglements = []
            for _, fold in enumerate(background_data):
                for _, datapoint in enumerate(fold):
                    rho = density_matrix(params, datapoint, model["input_size"], model)
                    snapshot_entanglements.append(entanglement_measure(rho))

            snapshot = np.mean(snapshot_entanglements)
            with open(f"{save_dir}_snapshot_{ids[i]}.pkl", "wb") as f:
                dill.dump(snapshot, f)

        ax.hist(
            mean_entanglements,
            bins=150,
            label=rf"sampled {ids[i]} ansatz, $\mi=${np.round(np.mean(mean_entanglements), 3)}+/-{np.round(np.std(mean_entanglements), 4)}",
            alpha=0.5,
            color=hist_colors[i],
            histtype="step",
        )
        ax.axvline(
            snapshot,
            label=f"trained {ids[i]} ansatz: {np.round(snapshot, 3)}",
            color=vline_colors[i],
        )
    ax.set_xlabel(r"$E(\theta, x)$")
    plt.legend()
    fig.savefig(filename)
    plt.close()


def magic_measure(model, params, data=None):
    # params = model["params"]
    wires = model["input_size"]
    d = 2**wires
    dev = qml.device("default.qubit", wires=model["input_size"])

    @qml.qnode(device=dev)
    def density_matrix(params, features, wires, model):
        # function for computing the purity of the state
        model["embedding_fn"](features, range(wires))
        model["ansatz_fn"](params, range(wires), model)
        return qml.density_matrix(wires=range(wires))

    @qml.qnode(device=dev)
    def pauli_density_matrix(params, features, wires, model, pauli_operators):
        # function for computing the stabilizer purity
        model["embedding_fn"](features, range(wires))
        model["ansatz_fn"](params, range(wires), model)

        return qml.expval(pauli_operators)

    pauli_operators = [qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ]
    pauli_strings = itertools.product(pauli_operators, repeat=model["input_size"])

    # computing S_2
    renyi_2_entropy = -qnp.log2(qnp.trace(density_matrix(params, data, wires, model)))
    # compute w
    total = 0
    for string in pauli_strings:
        for j in range(model["input_size"]):  # for j in range(4 * model["input_size"]):
            # string_idx = j % model["input_size"]
            # pauli = string[string_idx]
            pauli = string[j]
            if j == 0:
                current_operator = pauli(j)
            else:
                current_operator = current_operator @ pauli(j)
        state = pauli_density_matrix(params, data, wires, model, current_operator) ** 4
        total += state

    w = total * (d ** (-2))
    m_2 = -qnp.log2(w) - renyi_2_entropy - qnp.log2(d)
    return m_2


def get_magic_capabilities_given_data(
    original_q_model_path,
    new_q_model_path,
    background_dataset,
    num_samples,
    save_dir,
    filename,
):
    original_q_model = load_model(original_q_model_path)
    new_q_model = load_model(new_q_model_path)
    ids = ["original", "new"]
    vline_colors = ["#f764de", "#74ce4b"]

    with open(background_dataset, "rb") as f:
        background_data = dill.load(f)

    plt.close()
    fig, ax = plt.subplots()
    models = [original_q_model, new_q_model]

    hist_colors = ["#68affc", "#1c5f1e"]

    for i, model in enumerate(models):
        magic_file = Path(f"{save_dir}_{ids[i]}.pkl")

        if magic_file.is_file():
            print(f"Magic file found for {ids[i]}!")
            entanglements = load_pickle_file(magic_file)
        else:
            print(f"No magic file found for {ids[i]}!")

            mean_magics = []
            for j in range(num_samples):
                params = weight_init(
                    0,
                    2 * np.pi,
                    "uniform",
                    model["ansatz_fn"].shape(model["input_size"], model["layers"]),
                )

                magics = []
                for k, fold in enumerate(background_data):
                    print(f"sample: {j}, fold: {k}, model: {ids[i]}")
                    for _, datapoint in enumerate(fold):
                        m2 = magic_measure(model, params, datapoint)
                        magics.append(m2)

                mean_magics.append(np.mean(magics))

            with open(f"{save_dir}_dist_{ids[i]}.pkl", "wb") as f:
                dill.dump(mean_magics, f)

            params = model["params"]
            snapshot_magics = []
            for _, fold in enumerate(background_data):
                for _, datapoint in enumerate(fold):
                    m2 = magic_measure(model, params, datapoint)
                    snapshot_magics.append(m2)

            snapshot = np.mean(snapshot_magics)
            with open(f"{save_dir}_snapshot_{ids[i]}.pkl", "wb") as f:
                dill.dump(snapshot, f)

        ax.hist(
            mean_magics,
            bins=70,
            density=True,
            label=f"{ids[i]} ansatz, mean: {np.mean(mean_magics)}+/-{np.std(mean_magics)}",
            alpha=0.5,
            color=hist_colors[i],
            histtype="step",
            stacked=True,
        )
        ax.axvline(
            snapshot,
            label=f"trained {ids[i]} ansatz: {snapshot}",
            color=vline_colors[i],
        )

    plt.legend()
    fig.savefig(filename)
    plt.close()


def get_magic_dist_for_data(
    original_q_model_path,
    new_q_model_path,
    background_dataset,
    save_dir,
    filename,
):
    original_q_model = load_model(original_q_model_path)
    new_q_model = load_model(new_q_model_path)
    ids = ["original", "new"]

    with open(background_dataset, "rb") as f:
        background_data = dill.load(f)

    # plt.close()
    fig, ax = plt.subplots()
    models = [original_q_model, new_q_model]

    hist_colors = ["#68affc", "#1c5f1e"]

    for i, model in enumerate(models):
        magic_file = Path(f"{save_dir}_{ids[i]}.pkl")

        if magic_file.is_file():
            print(f"Magic file found for {ids[i]}!")
            magics = load_pickle_file(magic_file)
        else:
            print(f"No magic file found for {ids[i]}!")
            params = model["params"]

            magics = []
            for j, fold in enumerate(background_data):
                for k, datapoint in enumerate(fold):
                    m2 = magic_measure(model, params, datapoint)
                    magics.append(m2)
                    print(f"model: {ids[i]}, fold: {j}, point: {k}, m2: {m2}")

            with open(f"{save_dir}_{ids[i]}.pkl", "wb") as f:
                dill.dump(magics, f)

        ax.hist(
            np.real(magics),
            bins=100,
            density=True,
            label=f"{ids[i]} ansatz, mean: {np.mean(magics)}+/-{np.std(magics)}",
            alpha=0.5,
            color=hist_colors[i],
            histtype="step",
            stacked=True,
        )

    d = 2 ** new_q_model["input_size"]
    max_magic = np.log2(d + 1) - np.log2(2)
    ax.vlines(x=max_magic, ymin=0, ymax=0.5)
    ax.set_xlim(0, max_magic + 1)
    plt.legend()
    fig.savefig(filename)
    plt.close()


def plot_magic_dist(
    rootdir, snapshot_paths, save_dir, f1_bins, f2_bins, frame1_ymax, frame2_ymax
):
    new_means = []
    original_means = []
    hist_colors = ["#68affc", "#1c5f1e"]
    max_x = np.log2((2 ** (4)) + 1) - np.log2(2)
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.__contains__("_new_"):
                with open(filepath, "rb") as f:
                    m = np.real(dill.load(f))
                    mean_magic = np.mean(m)
                    new_means.append(mean_magic)

            elif filepath.__contains__("_original_"):
                with open(filepath, "rb") as f:
                    m = np.real(dill.load(f))
                    mean_magic = np.mean(m)
                    original_means.append(mean_magic)

    snapshots = []
    for snapshot_path in snapshot_paths:
        with open(snapshot_path, "rb") as f:
            snapshot = np.real(dill.load(f))
            snapshots.append(snapshot)

    labels = ["orignal", "new"]

    fig1 = plt.figure(1)
    frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
    dist_mean = np.round(np.mean(new_means), 3)
    dist_std = np.round(np.std(new_means), 4)

    og_dist_mean = np.round(np.mean(original_means), 3)
    og_dist_std = np.round(np.std(original_means), 4)
    plt.hist(
        new_means,
        bins=f1_bins,
        color=hist_colors[1],
        histtype="step",
        label=rf"sampled new ansatz $\mu=${dist_mean}$\pm${dist_std}",
        alpha=0.8,
    )
    plt.hist(
        original_means,
        bins=f1_bins,
        color=hist_colors[0],
        histtype="step",
        label=rf"sampled original ansatz $\mu=${og_dist_mean}$\pm${og_dist_std}",
        alpha=0.8,
    )
    frame1.set_xticklabels([])
    frame1.set_ylim(0, frame1_ymax)
    frame1.set_xlim(0, max_x)
    plt.legend()

    frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
    for i, snap in enumerate(snapshots):
        dist_mean = np.round(np.mean(snap), 3)
        plt.hist(
            snap,
            bins=f2_bins,
            color=hist_colors[i],
            histtype="step",
            alpha=0.8,
            label=rf"trained {labels[i]} ansatz $\mu=${dist_mean}",
        )
    frame2.set_ylim(0, frame2_ymax)
    frame2.set_xlim(0, max_x)
    frame2.set_xlabel(rf"$M_2$")
    plt.legend()
    plt.savefig(save_dir)
    plt.close()


def plot_magic_dist_2(
    rootdir, snapshot_paths, save_dir, f1_bins, f2_bins, frame1_ymax, frame2_ymax
):
    new_means = []
    original_means = []
    hist_colors = ["#68affc", "#1c5f1e"]
    max_x = np.log2((2 ** (4)) + 1) - np.log2(2)
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.__contains__("_new_"):
                with open(filepath, "rb") as f:
                    m = np.real(dill.load(f))
                    mean_magic = np.mean(m)
                    new_means.append(mean_magic)

            elif filepath.__contains__("_original_"):
                with open(filepath, "rb") as f:
                    m = np.real(dill.load(f))
                    mean_magic = np.mean(m)
                    original_means.append(mean_magic)

    snapshots = []

    with open(snapshot_paths[0], "rb") as f:
        snapshot = np.real(dill.load(f))
        snapshots.append(snapshot)

    with open(
        snapshot_paths[1],
        "rb",
    ) as f:
        info = dill.load(f)

    print(info.keys())
    dynamic_magic = info["dynamic_magic"]
    losses = info["loss_reconstructions"]

    print(dynamic_magic.shape)

    intervals = 40
    for i in range(intervals + 1):
        epoch_m = dynamic_magic[:, i, :].flatten()
    snapshots.append(epoch_m)

    labels = ["orignal", "new"]

    fig1 = plt.figure(1)
    frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
    dist_mean = np.round(np.mean(new_means), 3)
    dist_std = np.round(np.std(new_means), 4)

    og_dist_mean = np.round(np.mean(original_means), 3)
    og_dist_std = np.round(np.std(original_means), 4)
    plt.hist(
        new_means,
        bins=f1_bins,
        color=hist_colors[1],
        histtype="step",
        label=rf"sampled new ansatz $\mu=${dist_mean}$\pm${dist_std}",
        alpha=0.8,
    )
    plt.hist(
        original_means,
        bins=f1_bins,
        color=hist_colors[0],
        histtype="step",
        label=rf"sampled original ansatz $\mu=${og_dist_mean}$\pm${og_dist_std}",
        alpha=0.8,
    )
    frame1.set_xticklabels([])
    frame1.set_ylim(0, frame1_ymax)
    frame1.set_xlim(0, max_x)
    plt.legend()

    frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
    for i, snap in enumerate(snapshots):
        dist_mean = np.round(np.mean(snap), 3)
        plt.hist(
            snap,
            bins=f2_bins,
            color=hist_colors[i],
            histtype="step",
            alpha=0.8,
            label=rf"trained {labels[i]} ansatz $\mu=${dist_mean}",
        )
    frame2.set_ylim(0, frame2_ymax)
    frame2.set_xlim(0, max_x)
    frame2.set_xlabel(rf"$M_2$")
    plt.legend()
    plt.savefig(save_dir)
    plt.close()


def plot_entanglement_dist(
    rootdir, snapshot_path, save_dir, bins=100, frame1_ymax=110, frame2_ymax=2000
):
    new_means = []
    original_means = []
    hist_colors = ["#68affc", "#1c5f1e"]
    labels = ["original", "new"]
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.__contains__(f"_new_"):
                with open(filepath, "rb") as f:
                    m = dill.load(f)
                    mean_entanglement = np.mean(m)
                    new_means.append(mean_entanglement)

            if filepath.__contains__(f"_original_"):
                with open(filepath, "rb") as f:
                    m = dill.load(f)
                    mean_entanglement = np.mean(m)
                    original_means.append(mean_entanglement)

    snapshots = []
    for p in snapshot_path:
        with open(p, "rb") as f:
            new_snapshot = dill.load(f)
            snapshots.append(new_snapshot)

    fig1 = plt.figure(1)
    frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
    dist_mean = np.round(np.mean(new_means), 3)
    dist_std = np.round(np.std(new_means), 4)

    og_dist_mean = np.round(np.mean(original_means), 3)
    og_dist_std = np.round(np.std(original_means), 4)
    plt.hist(
        original_means,
        bins=bins,
        color=hist_colors[0],
        histtype="step",
        label=rf"sampled original ansatz $\mu=${og_dist_mean}$\pm${og_dist_std}",
        alpha=0.8,
    )
    plt.hist(
        new_means,
        bins=bins,
        color=hist_colors[1],
        histtype="step",
        label=rf"sampled new ansatz $\mu=${dist_mean}$\pm${dist_std}",
        alpha=0.8,
    )

    frame1.set_xticklabels([])
    frame1.set_ylim(0, frame1_ymax)

    plt.legend(loc="upper right")

    frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
    for i, snap in enumerate(snapshots):
        dist_mean = np.round(np.mean(snap), 3)
        dist_std = np.round(np.std(snap), 4)
        print(dist_mean)
        plt.hist(
            snap,
            bins=100,
            color=hist_colors[i],
            histtype="step",
            alpha=0.8,
            label=rf"trained {labels[i]} ansatz $\mu=${dist_mean}",
        )
    frame2.set_ylim(0, frame2_ymax)
    frame2.set_xlabel(r"Q")

    plt.legend(loc="upper right")
    plt.savefig(save_dir)
    plt.close()


def plot_entanglement_dist_w_data(
    dist_paths, snapshot_paths, save_dir, bins=100, frame1_ymax=80, frame2_ymax=2000
):
    hist_colors = ["#68affc", "#1c5f1e"]
    labels = ["original", "new"]

    entanglement_dist_list = []
    entanglement_dist_means = []
    entanglement_dist_stds = []
    for p in dist_paths:
        with open(p, "rb") as f:
            entanglement_dist = dill.load(f)
            entanglement_dist_list.append(entanglement_dist)
            entanglement_dist_means.append(np.round(np.mean(entanglement_dist), 3))
            entanglement_dist_stds.append(np.round(np.std(entanglement_dist), 4))

    snapshots = []
    for p in snapshot_paths:
        with open(p, "rb") as f:
            new_snapshot = dill.load(f)
            snapshots.append(new_snapshot)

    fig1 = plt.figure(1)
    frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))

    for i, dist in enumerate(entanglement_dist_list):
        plt.hist(
            dist,
            bins=bins,
            color=hist_colors[i],
            histtype="step",
            label=rf"sampled {labels[i]} ansatz $\mu=${entanglement_dist_means[i]}$\pm${entanglement_dist_stds[i]}",
        )

    frame1.set_xticklabels([])
    frame1.set_ylim(0, frame1_ymax)
    plt.legend(loc="upper right")

    frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
    for i, snap in enumerate(snapshots):
        dist_mean = np.round(np.mean(snap), 3)
        dist_std = np.round(np.std(snap), 4)
        y, x, _ = plt.hist(
            snap,
            bins=100,
            color=hist_colors[i],
            histtype="step",
            alpha=0.8,
            label=rf"trained {labels[i]} ansatz $\mu=${dist_mean}",
        )
    frame2.set_ylim(0, frame2_ymax)
    frame2.set_xlabel(r"Q")

    plt.legend(loc="upper right")
    plt.savefig(save_dir)
    plt.close()


def plot_qcd_inp8_entanglement_dist(
    original_dist_path,
    new_dist_path,
    snapshot_paths,
    save_dir,
    bins=100,
    frame1_ymax=80,
    frame2_ymax=2000,
):
    hist_colors = ["#68affc", "#1c5f1e"]
    labels = ["original", "new"]

    original_means = []
    with open(original_dist_path, "rb") as f:
        entanglement_dist = dill.load(f)
        original_means.append(entanglement_dist)

    new_means = []
    print("rootdir: ", new_dist_path)
    for subdir, dirs, files in os.walk(new_dist_path):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.__contains__(f"_new_"):
                with open(filepath, "rb") as f:
                    m = dill.load(f)
                    mean_entanglement = np.mean(m)
                    new_means.append(mean_entanglement)

    snapshots = []
    for p in snapshot_paths:
        with open(p, "rb") as f:
            snapshot = dill.load(f)
            print(f"snap samples: {len(snapshot)}")
            snapshots.append(snapshot[:5000])

    fig1 = plt.figure(1)
    frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
    dist_mean = np.round(np.mean(new_means), 3)
    dist_std = np.round(np.std(new_means), 4)

    og_dist_mean = np.round(np.mean(original_means), 3)
    og_dist_std = np.round(np.std(original_means), 4)
    plt.hist(
        original_means,
        bins=bins,
        color=hist_colors[0],
        histtype="step",
        label=rf"sampled original ansatz $\mu=${og_dist_mean}$\pm${og_dist_std}",
        alpha=0.8,
    )
    plt.hist(
        new_means,
        bins=bins,
        color=hist_colors[1],
        histtype="step",
        label=rf"sampled new ansatz $\mu=${dist_mean}$\pm${dist_std}",
        alpha=0.8,
    )

    frame1.set_xticklabels([])
    frame1.set_ylim(0, frame1_ymax)
    plt.legend(loc="upper left")

    frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
    for i, snap in enumerate(snapshots):
        dist_mean = np.round(np.mean(snap), 3)
        y, x, _ = plt.hist(
            snap,
            bins=100,
            color=hist_colors[i],
            histtype="step",
            alpha=0.8,
            label=rf"trained {labels[i]} ansatz $\mu=${dist_mean}",
        )
    frame2.set_ylim(0, frame2_ymax)
    frame2.set_xlabel(r"Q")

    plt.legend(loc="upper left")
    plt.savefig(save_dir)
    plt.close()


def plot_qcd_inp16_entanglement_dist(
    original_dist_path,
    new_dist_path,
    snapshot_paths,
    save_dir,
    bins=100,
    frame1_ymax=80,
    frame2_ymax=2000,
):
    hist_colors = ["#68affc", "#1c5f1e"]
    labels = ["original", "new"]

    original_means = []
    new_means = []
    print("rootdir: ", original_dist_path)
    for subdir, dirs, files in os.walk(original_dist_path):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.__contains__(f"_original_"):
                with open(filepath, "rb") as f:
                    m = dill.load(f)
                    mean_entanglement = np.mean(m)
                    original_means.append(mean_entanglement)

    new_means = []
    print("rootdir: ", new_dist_path)
    for subdir, dirs, files in os.walk(new_dist_path):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.__contains__(f"_new_"):
                with open(filepath, "rb") as f:
                    m = dill.load(f)
                    mean_entanglement = np.mean(m)
                    new_means.append(mean_entanglement)

    snapshots = []
    for p in snapshot_paths:
        with open(p, "rb") as f:
            snapshot = dill.load(f)
            print(f"snap samples: {len(snapshot)}")
            snapshots.append(snapshot[:5000])

    fig1 = plt.figure(1)
    frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
    dist_mean = np.round(np.mean(new_means), 3)
    dist_std = np.round(np.std(new_means), 4)

    og_dist_mean = np.round(np.mean(original_means), 3)
    og_dist_std = np.round(np.std(original_means), 4)
    plt.hist(
        original_means,
        bins=bins,
        color=hist_colors[0],
        histtype="step",
        label=rf"sampled original ansatz $\mu=${og_dist_mean}$\pm${og_dist_std}",
        alpha=0.8,
    )
    plt.hist(
        new_means,
        bins=bins,
        color=hist_colors[1],
        histtype="step",
        label=rf"sampled new ansatz $\mu=${dist_mean}$\pm${dist_std}",
        alpha=0.8,
    )

    frame1.set_xticklabels([])
    frame1.set_ylim(0, frame1_ymax)
    plt.legend(loc="upper left")

    frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
    for i, snap in enumerate(snapshots):
        dist_mean = np.round(np.mean(snap), 3)
        y, x, _ = plt.hist(
            snap,
            bins=100,
            color=hist_colors[i],
            histtype="step",
            alpha=0.8,
            label=rf"trained {labels[i]} ansatz $\mu=${dist_mean}",
        )
    frame2.set_ylim(0, frame2_ymax)
    frame2.set_xlabel(r"Q")

    plt.legend(loc="upper left")
    plt.savefig(save_dir)
    plt.close()


def plot_qcd_inp16_entanglement_dist_v2(
    original_dist_path,
    new_dist_path,
    snapshot_paths,
    save_dir,
    bins=100,
    frame1_ymax=80,
    frame2_ymax=2000,
):
    hist_colors = ["#68affc", "#1c5f1e"]
    labels = ["original", "new"]

    original_means = []
    new_means = []
    print("rootdir: ", original_dist_path)
    for subdir, dirs, files in os.walk(new_dist_path):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.__contains__(f"_original_"):
                with open(filepath, "rb") as f:
                    m = dill.load(f)
                    mean_entanglement = np.mean(m)
                    original_means.append(mean_entanglement)

    new_means = []
    print("rootdir: ", new_dist_path)
    for subdir, dirs, files in os.walk(new_dist_path):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.__contains__(f"_new_"):
                with open(filepath, "rb") as f:
                    m = dill.load(f)
                    mean_entanglement = np.mean(m)
                    new_means.append(mean_entanglement)

    snapshots = []
    for p in snapshot_paths:
        with open(p, "rb") as f:
            snapshot = dill.load(f)
            print(f"snap samples: {len(snapshot)}")
            snapshots.append(snapshot[:5000])

    fig1 = plt.figure(1)
    frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
    dist_mean = np.round(np.mean(new_means), 3)
    dist_std = np.round(np.std(new_means), 4)

    og_dist_mean = np.round(np.mean(original_means), 3)
    og_dist_std = np.round(np.std(original_means), 4)
    plt.hist(
        original_means,
        bins=bins,
        color=hist_colors[0],
        histtype="step",
        label=rf"sampled original ansatz $\mu=${og_dist_mean}$\pm${og_dist_std}",
        alpha=0.8,
    )
    plt.hist(
        new_means,
        bins=bins,
        color=hist_colors[1],
        histtype="step",
        label=rf"sampled new ansatz $\mu=${dist_mean}$\pm${dist_std}",
        alpha=0.8,
    )

    frame1.set_xticklabels([])
    frame1.set_ylim(0, frame1_ymax)
    plt.legend(loc="upper left")

    frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
    for i, snap in enumerate(snapshots):
        dist_mean = np.round(np.mean(snap), 3)
        y, x, _ = plt.hist(
            snap,
            bins=100,
            color=hist_colors[i],
            histtype="step",
            alpha=0.8,
            label=rf"trained {labels[i]} ansatz $\mu=${dist_mean}",
        )
    frame2.set_ylim(0, frame2_ymax)
    frame2.set_xlabel(r"Q")

    plt.legend(loc="upper left")
    plt.savefig(save_dir)
    plt.close()


def plot_snapshot_scaling(
    input_sizes, original_snapshot_paths, new_snapshot_paths, save_dir
):
    hist_colors = ["#68affc", "#1c5f1e"]
    labels = ["original", "new"]

    paths = [original_snapshot_paths, new_snapshot_paths]

    fig, ax = plt.subplots()
    for j, snapshot_paths in enumerate(paths):
        snapshots = []
        for p in snapshot_paths:
            with open(p, "rb") as f:
                new_snapshot = dill.load(f)
                snapshots.append(new_snapshot)

        means = []
        stds = []
        for i, snap in enumerate(snapshots):
            means.append(np.round(np.mean(snap), 3))
            stds.append(np.round(np.std(snap), 4))

        ax.errorbar(
            input_sizes,
            means,
            yerr=stds,
            color=hist_colors[j],
            label=f"{labels[j]}",
        )
    plt.legend()
    fig.savefig(save_dir)
    plt.close()


def dynamic_entanglement_plot(entanglement_paths, labels, intervals=40):
    colors = ["#1c5f1e", "#68affc"]
    fig, ax = plt.subplots()
    for idx, path in enumerate(entanglement_paths):
        with open(
            path,
            "rb",
        ) as f:
            info = dill.load(f)

        print(info.keys())
        dynamic_entanglement = info["dynamic_entanglement"]
        losses = info["loss_reconstructions"]

        print(dynamic_entanglement.shape)

        mean_e = []
        std_e = []
        mean_loss = []
        for i in range(intervals + 1):
            epoch_e = dynamic_entanglement[:, i, :].flatten()
            epoch_loss = losses[:, i, :].flatten()
            mean_e.append(np.mean(epoch_e))
            std_e.append(np.std(epoch_e))
            mean_loss.append(np.mean(epoch_loss))

        mean_e = np.array(mean_e)
        std_e = np.array(std_e)

        x = range(0, (2 * intervals) + 1, 2)
        mean_e_s = CubicSpline(x, mean_e)
        mean_loss_s = CubicSpline(x, mean_loss)
        xs = np.linspace(0, (2 * intervals) + 1, 1000)

        sc = ax.scatter(
            xs, mean_e_s(xs), c=mean_loss_s(xs), s=40, cmap="plasma", vmin=0, vmax=1
        )
        ax.scatter(x, mean_e, marker="o", c=colors[idx], s=15, label=labels[idx])

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("loss", rotation=270, labelpad=15, fontsize=12)
    plt.legend()
    ax.set_xlabel("epochs", fontsize=12)
    ax.set_ylabel(r"Q", fontsize=12)
    plt.savefig("new_e_plot.pdf")
    plt.close()


def dynamic_magic_plot(magic_paths, labels, intervals=40):
    colors = ["#1c5f1e", "#68affc"]
    fig, ax = plt.subplots()
    for idx, path in enumerate(magic_paths):
        with open(
            path,
            "rb",
        ) as f:
            info = dill.load(f)

        print(info.keys())
        dynamic_magic = info["dynamic_magic"]
        losses = info["loss_reconstructions"]

        print(dynamic_magic.shape)

        mean_e = []
        std_e = []
        mean_loss = []
        for i in range(intervals + 1):
            epoch_e = dynamic_magic[:, i, :].flatten()
            epoch_loss = losses[:, i, :].flatten()
            mean_e.append(np.mean(epoch_e))
            std_e.append(np.std(epoch_e))
            mean_loss.append(np.mean(epoch_loss))

        mean_e = np.array(mean_e)
        std_e = np.array(std_e)

        x = range(0, (2 * intervals) + 1, 2)
        mean_e_s = CubicSpline(x, mean_e)
        mean_loss_s = CubicSpline(x, mean_loss)
        xs = np.linspace(0, (2 * intervals) + 1, 1000)

        sc = ax.scatter(
            xs, mean_e_s(xs), c=mean_loss_s(xs), s=40, cmap="plasma", vmin=0, vmax=1
        )
        ax.scatter(x, mean_e, marker="o", c=colors[idx], s=15, label=labels[idx])

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("loss", rotation=270, labelpad=15, fontsize=12)
    plt.legend()
    ax.set_xlabel("epochs", fontsize=12)
    ax.set_ylabel(r"$M_2$", fontsize=12)
    plt.savefig("new_m_plot.pdf")
    plt.close()


if __name__ == "__main__":

    plot_magic_dist(
        "/unix/qcomp/users/cduffy/anomaly_detection/higgs_inp4_magic_data",
        [
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/higgs_inp4_data_dist_given_param_new.pkl",
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/higgs_inp4_data_dist_given_param_original.pkl",
        ],
        "higgs_inp4_magic_dist.pdf",
        f1_bins=20,
        f2_bins=100,
        frame1_ymax=70,
        frame2_ymax=2500,
    )

    plot_magic_dist(
        "/unix/qcomp/users/cduffy/anomaly_detection/qcd_inp8_v2_magic_data",  # /unix/qcomp/users/cduffy/anomaly_detection/qcd_inp8_magic_data",
        [
            "/unix/qcomp/users/cduffy/anomaly_detection/qcd_inp8_magic_original_snapshot.pkl",  # /unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/qcd_inp8_data_dist_given_param_original.pkl",
            "/unix/qcomp/users/cduffy/anomaly_detection/qcd_inp8_magic_new_snapshot.pkl",  # "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/qcd_inp8_data_dist_given_param_new.pkl",
        ],
        "qcd_inp8_magic_dist.pdf",
        f1_bins=50,
        f2_bins=50,
        frame1_ymax=125,
        frame2_ymax=150,
    )

    plot_magic_dist_2(
        "/unix/qcomp/users/cduffy/anomaly_detection/qcd_inp8_v2_magic_data",  # /unix/qcomp/users/cduffy/anomaly_detection/qcd_inp8_magic_data",
        [
            "/unix/qcomp/users/cduffy/anomaly_detection/qcd_inp8_magic_original_snapshot.pkl",  # /unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/qcd_inp8_data_dist_given_param_original.pkl",
            "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/test/run_47/info.pkl",  # "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/qcd_inp8_data_dist_given_param_new.pkl",
        ],
        "qcd_inp8_magic_dist_2.pdf",
        f1_bins=50,
        f2_bins=50,
        frame1_ymax=125,
        frame2_ymax=150,
    )

    dynamic_entanglement_plot(
        [
            "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/test/run_34/info.pkl",
            "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/test/run_36/info.pkl",
        ],
        ["new", "original"],
    )

    dynamic_magic_plot(
        [
            "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/test/run_48/info.pkl",
            "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/test/run_47/info.pkl",
        ],
        ["new", "original"],
    )

    plot_entanglement_dist_w_data(
        [
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp4_param_dist_w_data_original.pkl",
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp4_param_dist_w_data_new.pkl",
        ],
        [
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp4_entanglement_w_data_original.pkl",
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp4_entanglement_w_data_new.pkl",
        ],
        "higgs_inp4_entanglement_dist.pdf",
        bins=30,
        frame1_ymax=100,
        frame2_ymax=2500,
    )
    plot_entanglement_dist(
        "/unix/qcomp/users/cduffy/anomaly_detection/higgs_inp6_entanglement_data",
        [
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp6_entanglement_w_data_original.pkl",
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp6_entanglement_w_data_new.pkl",
        ],
        "higgs_inp6_entanglement_dist.pdf",
        bins=25,
        frame1_ymax=120,
        frame2_ymax=2500,
    )
    print("INP 8")
    plot_entanglement_dist(
        "/unix/qcomp/users/cduffy/anomaly_detection/higgs_inp8_entanglement_data",
        [
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp8_entanglement_w_data_original.pkl",
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp8_entanglement_w_data_new.pkl",
        ],
        "higgs_inp8_entanglement_dist.pdf",
        bins=25,
        frame1_ymax=140,
        frame2_ymax=2300,
    )
    input_sizes = [4, 6, 8]
    original_snapshot_paths = [
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp4_entanglement_w_data_original.pkl",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp6_entanglement_w_data_original.pkl",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp8_entanglement_w_data_original.pkl",
    ]
    new_snapshot_paths = [
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp4_entanglement_w_data_new.pkl",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp6_entanglement_w_data_new.pkl",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp8_entanglement_w_data_new.pkl",
    ]
    save_dir = "higgs_snapshot_scaling.pdf"
    plot_snapshot_scaling(
        input_sizes, original_snapshot_paths, new_snapshot_paths, save_dir
    )

    plot_entanglement_dist_w_data(
        [
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp8_param_dist_w_data_original.pkl",
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp8_param_dist_w_data_new.pkl",
        ],
        [
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp8_entanglement_w_data_original.pkl",
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp8_entanglement_w_data_new.pkl",
        ],
        "qcd_inp8_entanglement_dist.pdf",
        frame1_ymax=50,
        frame2_ymax=1800,
    )

    plot_qcd_inp8_entanglement_dist(
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp8_param_dist_w_data_original.pkl",
        "/unix/qcomp/users/cduffy/anomaly_detection/qcd_inp8_entanglement_data_v2",
        [
            "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp8_entanglement_w_data_original.pkl",
            "/unix/qcomp/users/cduffy/anomaly_detection/inp8_new_entanglement_snapshot_v2.pkl",
        ],
        "qcd_inp8_entanglement_dist_v2.pdf",
        bins=50,
        frame1_ymax=180,
        frame2_ymax=250,
    )

    plot_qcd_inp16_entanglement_dist(
        "/unix/qcomp/users/cduffy/anomaly_detection/qcd_inp16_entanglement_data",
        "/unix/qcomp/users/cduffy/anomaly_detection/qcd_inp16_entanglement_data_v2",
        [
            "/unix/qcomp/users/cduffy/anomaly_detection/inp16_original_entanglement_snapshot.pkl",
            "/unix/qcomp/users/cduffy/anomaly_detection/inp16_new_entanglement_snapshot_v2.pkl",
        ],
        "qcd_inp16_entanglement_dist_v2.pdf",
        bins=20,
        frame1_ymax=100,
        frame2_ymax=550,
    )

    with open(
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp8_entanglement_w_data_new.pkl",
        "rb",
    ) as f:
        new_snapshot = dill.load(f)
        print("SHAPE: ", len(new_snapshot))
    plot_entanglement_dist(
        "/unix/qcomp/users/cduffy/anomaly_detection/qcd_inp16_entanglement_data",
        [
            "/unix/qcomp/users/cduffy/anomaly_detection/inp16_original_entanglement_snapshot.pkl",
            "/unix/qcomp/users/cduffy/anomaly_detection/inp16_new_entanglement_snapshot.pkl",  # "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp16_entanglement_w_data_original.pkl",
            # "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp16_entanglement_w_data_new.pkl",
        ],
        "qcd_inp16_entanglement_dist.pdf",
        bins=25,
        frame1_ymax=80,
        frame2_ymax=1000,
    )
    with open(
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp16_entanglement_w_data_new.pkl",
        "rb",
    ) as f:
        new_snapshot = dill.load(f)
        print("SHAPE: ", len(new_snapshot))

    input_sizes = [4, 8]
    original_snapshot_paths = [
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp8_entanglement_w_data_original.pkl",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp16_entanglement_w_data_original.pkl",
    ]
    new_snapshot_paths = [
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp8_entanglement_w_data_new.pkl",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp16_entanglement_w_data_new.pkl",
    ]
    save_dir = "qcd_snapshot_scaling.pdf"
    plot_snapshot_scaling(
        input_sizes, original_snapshot_paths, new_snapshot_paths, save_dir
    )

    print("ROC PLOTS!")
    original_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_lat1/quantum/run_1/model.pkl"
    new_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_lat1/quantum/run_10/model.pkl"
    hea_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_lat1/quantum/run_7/model.pkl"
    no_e_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp8/run_4/model.pkl"
    reuploader_q_model_path = ""
    shallow_c_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_lat1/classical/run_2/model.pkl"
    deep_c_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_lat1/classical/run_3/model.pkl"
    variables = [6, 3, 0, 1]
    k_folds = 5
    losses_path = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/higgs_inp4_roc_losses"
    filename = "higgs_inp4_roc"
    background_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/higgs_inp4_bg_data.pkl"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/higgs_inp4_signal_data.pkl"

    quantum_model_paths = [
        original_q_model_path,
        new_q_model_path,
        no_e_model_path,
        hea_q_model_path,
    ]
    classical_model_paths = [shallow_c_model_path, deep_c_model_path]

    quantum_labels = ["original (4)", "new (4)", "no entanglement (4)", "hea (4)"]
    classical_labels = ["shallow (59)", "deep (135)"]

    fig_higgs_inp4_roc(
        quantum_model_paths,
        classical_model_paths,
        variables,
        k_folds,
        losses_path,
        "higgs",
        signal_dataset,
        background_dataset,
        quantum_labels,
        classical_labels,
        filename,
    )
    """
    get_entanglement_data_capabilities(
        original_q_model_path,
        new_q_model_path,
        background_dataset,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp4_entanglement_w_data",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/higgs_inp4_entanglements_w_data.pdf",
    )

    get_entanglement_capabilities_given_data(
        original_q_model_path,
        new_q_model_path,
        background_dataset,
        1000,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp4_param_dist_w_data",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/higgs_inp4_entanglements_dist_w_data.pdf",
    )
    """
    # get_magic_capabilities_given_data(
    #    original_q_model_path,
    #    new_q_model_path,
    #    background_dataset,
    #    500,
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/higgs_inp4_param_dist_w_data",
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_plots/higgs_inp4_magic_dist_w_data.pdf",
    # )

    """
    get_rejection_rate(
        original_q_model_path,
        new_q_model_path,
        shallow_c_model_path,
        deep_c_model_path,
        variables,
        k_folds,
        losses_path,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/higgs_inp4_fpr.txt",
    )

    get_expressivities(
        original_q_model_path,
        new_q_model_path,
    )

    get_entanglement_capabilities(
        original_q_model_path,
        new_q_model_path,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/higgs_inp4_entanglements.pdf",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp4_entanglement",
    )

    get_entanglement_loss_corr(
        original_q_model_path,
        new_q_model_path,
        background_dataset,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp4_corr",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/higgs_inp4_entanglements_loss_corr.pdf",
    )
    
    
    get_magic_capabilities_given_data(
        original_q_model_path,
        new_q_model_path,
        background_dataset,
        100,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/higgs_inp4_param_dist_w_data",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_plots/higgs_inp4_magic_dist_w_data.pdf",
    )
    """

    # higgs inp6 roc curve
    original_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp6/quantum/run_12/model.pkl"
    new_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp6/quantum/run_22/model.pkl"
    no_e_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp6/run_3/model.pkl"
    hea_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp6/quantum/run_13/model.pkl"
    shallow_c_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp6/classical/run_13/model.pkl"
    deep_c_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp6/classical/run_14/model.pkl"
    variables = [6, 3, 0, 1, 2, 4]
    k_folds = 4
    losses_path = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/higgs_inp6_roc_losses"
    filename = "higgs_inp6_roc"
    background_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/higgs_inp6_bg_data.pkl"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/higgs_inp6_sg_data.pkl"
    quantum_model_paths = [
        original_q_model_path,
        new_q_model_path,
        no_e_model_path,
        hea_q_model_path,
    ]
    classical_model_paths = [shallow_c_model_path, deep_c_model_path]

    quantum_labels = ["original (6)", "new (6)", "no entanglement (6)", "hea (6)"]
    classical_labels = ["shallow (70)", "deep (540)"]

    fig_higgs_inp4_roc(
        quantum_model_paths,
        classical_model_paths,
        variables,
        k_folds,
        losses_path,
        "higgs",
        signal_dataset,
        background_dataset,
        quantum_labels,
        classical_labels,
        filename,
    )
    """
    get_entanglement_data_capabilities(
        original_q_model_path,
        new_q_model_path,
        background_dataset,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp6_entanglement_w_data",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/higgs_inp6_entanglements_w_data.pdf",
    )
    """
    # print("inp 6")
    # get_entanglement_capabilities_given_data(
    #    original_q_model_path,
    #    new_q_model_path,
    #    background_dataset,
    #    1000,
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp6_param_dist_w_data",
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/higgs_inp6_entanglements_dist_w_data.pdf",
    # )

    # get_magic_capabilities_given_data(
    #    original_q_model_path,
    #    new_q_model_path,
    #    background_dataset,
    #    500,
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/higgs_inp6_param_dist_w_data",
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_plots/higgs_inp6_magic_dist_w_data.pdf",
    # )

    """
    working_point = 0.3
    print(f"working point: {working_point}")
    get_rejection_rate(
        original_q_model_path,
        new_q_model_path,
        shallow_c_model_path,
        deep_c_model_path,
        variables,
        k_folds,
        losses_path,
        filename,
        working_point,
    )
    working_point = 0.8
    print(f"working point: {working_point}")

    get_rejection_rate(
        original_q_model_path,
        new_q_model_path,
        shallow_c_model_path,
        deep_c_model_path,
        variables,
        k_folds,
        losses_path,
        filename,
        working_point,
    )

    working_point = 0.9
    print(f"working point: {working_point}")

    get_rejection_rate(
        original_q_model_path,
        new_q_model_path,
        shallow_c_model_path,
        deep_c_model_path,
        variables,
        k_folds,
        losses_path,
        filename,
        working_point,
    )

    working_point = 0.99
    print(f"working point: {working_point}")

    get_rejection_rate(
        original_q_model_path,
        new_q_model_path,
        shallow_c_model_path,
        deep_c_model_path,
        variables,
        k_folds,
        losses_path,
        filename,
        working_point,
    )
    print("-----")
    
    get_entanglement_capabilities(
        original_q_model_path,
        new_q_model_path,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/higgs_inp6_entanglements.pdf",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp6_entanglement",
    )

    get_entanglement_loss_corr(
        original_q_model_path,
        new_q_model_path,
        background_dataset,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp6_corr",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/higgs_inp6_entanglements_loss_corr.pdf",
    )

    get_magic_capabilities_given_data(
        original_q_model_path,
        new_q_model_path,
        background_dataset,
        100,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/higgs_inp6_param_dist_w_data",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_plots/higgs_inp6_magic_dist_w_data.pdf",
    )

    print("----")
    """

    # higgs inp8 roc curve
    original_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp8/quantum/run_1/model.pkl"
    new_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp8/quantum/run_9/model.pkl"
    # need to recompute losses for this model
    no_e_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp8/run_3/model.pkl"
    hea_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp8/quantum/run_2/model.pkl"
    reuploader_q_model_path = ""
    shallow_c_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp8/classical/run_1/model.pkl"
    deep_c_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/higgs/ansatz_experiment_inp8/classical/run_2/model.pkl"
    variables = [6, 3, 0, 1, 2, 4, 5, 7]
    k_folds = 3
    losses_path = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/higgs_inp8_roc_losses"
    filename = "higgs_inp8_roc"
    background_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/higgs_inp8_bg_data.pkl"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/higgs_inp8_sg_data.pkl"
    quantum_model_paths = [
        original_q_model_path,
        new_q_model_path,
        no_e_model_path,
        hea_q_model_path,
    ]
    classical_model_paths = [shallow_c_model_path, deep_c_model_path]

    quantum_labels = ["original (8)", "new (8)", "no entanglement (8)", "hea (8)"]
    classical_labels = ["shallow (76)", "deep (942)"]

    fig_higgs_inp4_roc(
        quantum_model_paths,
        classical_model_paths,
        variables,
        k_folds,
        losses_path,
        "higgs",
        signal_dataset,
        background_dataset,
        quantum_labels,
        classical_labels,
        filename,
    )
    """
    get_entanglement_data_capabilities(
        original_q_model_path,
        new_q_model_path,
        background_dataset,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp8_entanglement_w_data",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/higgs_inp8_entanglements_w_data.pdf",
    )
    """
    # print("inp 8")
    # get_entanglement_capabilities_given_data(
    #    original_q_model_path,
    #    new_q_model_path,
    #    background_dataset,
    #    1000,
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp8_param_dist_w_data",
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/higgs_inp8_entanglements_dist_w_data.pdf",
    # )

    # get_magic_capabilities_given_data(
    # original_q_model_path,
    # new_q_model_path,
    # background_dataset,
    # 500,
    # "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/higgs_inp8_param_dist_w_data",
    # "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_plots/higgs_inp8_magic_dist_w_data.pdf",
    # )

    """
    # correlation between loss of a datapoint and the total entanglement generated
    # by the state
    get_entanglement_loss_corr(
        original_q_model_path,
        new_q_model_path,
        background_dataset,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp8_corr",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/higgs_inp8_entanglements_loss_corr.pdf",
    )

    # given the trained parameters what is the entanglement distribution for each
    # datapoint
    get_entanglement_data_capabilities(
        original_q_model_path,
        new_q_model_path,
        background_dataset,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp8_entanglement_w_data",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/higgs_inp8_entanglements_w_data.pdf",
    )

    # Each point in the distribution is for a given set of parameters find the mean
    # generated entanglement once all datapoints considered.
    get_entanglement_capabilities_given_data(
        original_q_model_path,
        new_q_model_path,
        background_dataset,
        1000,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/higgs_inp8_param_dist_w_data",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/higgs_inp8_entanglements_dist_w_data.pdf",
    )

    # get_magic_capabilities_given_data(
    #    original_q_model_path,
    #    new_q_model_path,
    #    background_dataset,
    #    100,
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/qcd_inp8_param_dist_w_data",
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_plots/qcd_inp8_magic_dist_w_data.pdf",
    # )
    """
    """
    # plotting for wg rx_mebed 4feat roc 3 layers
    # no e model has not been made
    original_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/ansatz_experiment/quantum/run_6/model.pkl"
    new_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/ansatz_experiment/quantum/run_9/model.pkl"
    no_e_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/ansatz_experiment/quantum/run_16/model.pkl"
    reuploader_q_model_path = ""
    shallow_c_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/ansatz_experiment/classical/run_16/model.pkl"
    deep_c_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/ansatz_experiment/classical/run_16/model.pkl"
    k_folds = 5
    losses_path = (
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/qcd_inp4_roc_losses"
    )
    background_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/qcd_inp4_signal_data.pkl"

    filename = "wg_inp4_roc"
    signal = "wg"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/wg_inp4_signal_data.pkl"

    entanglement_data = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp4_entanglement"
    
    get_entanglement_capabilities(
        original_q_model_path,
        new_q_model_path,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/qcd_inp4_entanglements.pdf",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp4_entanglement",
    )
    # (new plot idea) plot mean entanglements over data for sampled parameters

    get_entanglement_loss_corr(
        original_q_model_path,
        new_q_model_path,
        background_dataset,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp4_corr",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/qcd_inp4_entanglements_loss_corr.pdf",
    )

    get_entanglement_data_capabilities(
        original_q_model_path,
        new_q_model_path,
        background_dataset,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp4_entanglement_w_data",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/qcd_inp4_entanglements_w_data.pdf",
    )

    get_entanglement_capabilities_given_data(
        original_q_model_path,
        new_q_model_path,
        background_dataset,
        1000,
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp4_param_dist_w_data",
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/qcd_inp4_entanglements_dist_w_data.pdf",
    )
    

    fig_qcd_roc(
        original_q_model_path,
        new_q_model_path,
        no_e_model_path,
        shallow_c_model_path,
        deep_c_model_path,
        k_folds,
        losses_path,
        filename,
        signal,
        signal_dataset,
        background_dataset,
    )

    # inp4 3 layers rx embed ng dataset
    filename = "ng_inp4_roc"
    signal = "ng"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/ng_inp4_signal_data.pkl"

    fig_qcd_roc(
        original_q_model_path,
        new_q_model_path,
        no_e_model_path,
        shallow_c_model_path,
        deep_c_model_path,
        k_folds,
        losses_path,
        filename,
        signal,
        signal_dataset,
        background_dataset,
    )

    filename = "zzz_inp4_roc"
    signal = "zzz"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/zzz_inp4_signal_data.pkl"

    fig_qcd_roc(
        original_q_model_path,
        new_q_model_path,
        no_e_model_path,
        shallow_c_model_path,
        deep_c_model_path,
        k_folds,
        losses_path,
        filename,
        signal,
        signal_dataset,
        background_dataset,
    )

    """
    print("QCD!")
    print("inp 8")
    ###########################################################
    ### input 8 features 4 qubits, particle embedding
    # plotting for zzz 1 layer
    # currently no no entanglement model
    # no e model not been made
    original_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/zzz/ansatz_experiment_lat2/quantum/run_1/model.pkl"
    new_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/zzz/ansatz_experiment_lat2/quantum/run_9/model.pkl"
    # does not exist so far
    no_e_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/zzz/ansatz_experiment_lat2/quantum/run_1/model.pkl"
    # needs losses computed for other signals
    hea_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/zzz/ansatz_experiment_lat2/quantum/run_2/model.pkl"
    shallow_c_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/zzz/ansatz_experiment_lat2/classical/run_6/model.pkl"
    deep_c_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/zzz/ansatz_experiment_lat2/classical/run_6/model.pkl"
    k_folds = 5
    losses_path = (
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/qcd_inp8_roc_losses"
    )
    background_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/qcd_inp8_signal_data.pkl"

    quantum_model_paths = [
        original_q_model_path,
        new_q_model_path,
        no_e_model_path,
        hea_q_model_path,
    ]
    classical_model_paths = [shallow_c_model_path, deep_c_model_path]

    quantum_labels = ["new (12)"]
    classical_labels = ["shallow (15)", "deep (183)"]

    filename = "zzz_inp8_roc"
    signal = "zzz"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/zzz_inp8_signal_data.pkl"

    # get_magic_capabilities_given_data(
    # original_q_model_path,
    # new_q_model_path,
    # background_dataset,
    # 500,
    # "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/qcd_inp8_param_dist_w_data",
    # "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_plots/qcd_inp8_magic_dist_w_data.pdf",
    # )

    # get_entanglement_loss_corr(
    #    original_q_model_path,
    #    new_q_model_path,
    #    background_dataset,
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp8_corr",
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/qcd_inp8_entanglements_loss_corr.pdf",
    # )

    # given the trained parameters what is the entanglement distribution for each
    # datapoint
    # get_entanglement_data_capabilities(
    #    original_q_model_path,
    #    new_q_model_path,
    #    background_dataset,
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp8_entanglement_w_data",
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/qcd_inp8_entanglements_w_data.pdf",
    # )

    # Each point in the distribution is for a given set of parameters find the mean
    # generated entanglement once all datapoints considered.
    # get_entanglement_capabilities_given_data(
    #    original_q_model_path,
    #    new_q_model_path,
    #    background_dataset,
    #    1000,
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp8_param_dist_w_data",
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/qcd_inp8_entanglements_dist_w_data.pdf",
    # )

    # get_magic_capabilities_given_data(
    #    original_q_model_path,
    #    new_q_model_path,
    #    background_dataset,
    #    100,
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/qcd_inp8_param_dist_w_data",
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_plots/qcd_inp8_magic_dist_w_data.pdf",
    # )

    filename = "zzz_inp8_roc"
    signal = "zzz"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/zzz_inp8_signal_data.pkl"

    plot_roc_quantum_only(
        quantum_model_paths,
        k_folds,
        losses_path,
        "zzz_inp8_q_roc",
        signal,
        signal_dataset,
        background_dataset,
        ["original (4)", "new (4)", "no entanglement (4)", "hea (4)"],
    )

    fig_qcd_roc(
        [new_q_model_path],
        classical_model_paths,
        k_folds,
        losses_path,
        filename,
        signal,
        signal_dataset,
        background_dataset,
        quantum_labels,
        classical_labels,
    )

    ### input 8 qubits 4, 1 layer
    # ng
    filename = "ng_inp8_roc"
    signal = "ng"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/ng_inp8_signal_data.pkl"

    plot_roc_quantum_only(
        quantum_model_paths,
        k_folds,
        losses_path,
        "ng_inp8_q_roc",
        signal,
        signal_dataset,
        background_dataset,
        ["original (4)", "new (4)", "no entanglement (4)", "hea (4)"],
    )

    fig_qcd_roc(
        [new_q_model_path],
        classical_model_paths,
        k_folds,
        losses_path,
        filename,
        signal,
        signal_dataset,
        background_dataset,
        quantum_labels,
        classical_labels,
    )

    ### input 8 qubits 4, 1 layer
    # wg
    filename = "wg_inp8_roc"
    signal = "wg"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/wg_inp8_signal_data.pkl"

    plot_roc_quantum_only(
        quantum_model_paths,
        k_folds,
        losses_path,
        "wg_inp8_q_roc",
        signal,
        signal_dataset,
        background_dataset,
        ["original (4)", "new (4)", "no entanglement (4)", "hea (4)"],
    )

    fig_qcd_roc(
        [new_q_model_path],
        classical_model_paths,
        k_folds,
        losses_path,
        filename,
        signal,
        signal_dataset,
        background_dataset,
        quantum_labels,
        classical_labels,
    )

    ###########################################################

    ### input 16 features 8 qubits, particle embedding
    # plotting for zzz 1 layer
    # currently no entanglement model
    # no e model been made yet
    original_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/zzz/ansatz_experiment_inp8/quantum/run_1/model.pkl"
    new_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/zzz/ansatz_experiment_inp8/quantum/run_11/model.pkl"
    no_e_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/zzz/ansatz_experiment_lat2/quantum/run_1/model.pkl"
    hae_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/zzz/ansatz_experiment_inp8/quantum/run_2/model.pkl"
    reuploader_q_model_path = ""
    shallow_c_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/zzz/ansatz_experiment_inp8/classical/run_7/model.pkl"
    deep_c_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/zzz/ansatz_experiment_inp8/classical/run_7/model.pkl"
    k_folds = 1
    losses_path = (
        "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/qcd_inp16_roc_losses"
    )
    background_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/qcd_inp16_signal_data.pkl"

    filename = "zzz_inp16_roc"
    signal = "zzz"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/zzz_inp16_signal_data.pkl"

    quantum_model_paths = [
        original_q_model_path,
        new_q_model_path,
        no_e_model_path,
        hea_q_model_path,
    ]
    classical_model_paths = [shallow_c_model_path, deep_c_model_path]

    quantum_labels = ["new (32)"]
    classical_labels = ["shallow (42)", "deep (537)"]

    print("inp 16")
    # get_entanglement_data_capabilities(
    #    original_q_model_path,
    #    new_q_model_path,
    #    background_dataset,
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp16_entanglement_w_data",
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/qcd_inp16_entanglements_w_data.pdf",
    # )
    # get_entanglement_capabilities_given_data(
    #    original_q_model_path,
    #    new_q_model_path,
    #    background_dataset,
    #    1000,
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp16_param_dist_w_data",
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_plots/qcd_inp16_entanglements_dist_w_data.pdf",
    # )

    # get_magic_capabilities_given_data(
    #    original_q_model_path,
    #    new_q_model_path,
    #    background_dataset,
    #    500,
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_data/qcd_inp16_param_dist_w_data",
    #    "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/magic_plots/qcd_inp16_magic_dist_w_data.pdf",
    # )

    plot_roc_quantum_only(
        quantum_model_paths,
        k_folds,
        losses_path,
        "zzz_inp16_q_roc",
        signal,
        signal_dataset,
        background_dataset,
        ["original (8)", "new (8)", "no entanglement (8)", "hea (8)"],
    )

    fig_qcd_roc(
        [new_q_model_path],
        classical_model_paths,
        k_folds,
        losses_path,
        filename,
        signal,
        signal_dataset,
        background_dataset,
        quantum_labels,
        classical_labels,
    )

    ## ng features 16 qubits 8
    filename = "ng_inp16_roc"
    signal = "ng"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/ng_inp16_signal_data.pkl"

    plot_roc_quantum_only(
        quantum_model_paths,
        k_folds,
        losses_path,
        "ng_inp16_q_roc",
        signal,
        signal_dataset,
        background_dataset,
        ["original (8)", "new (8)", "no entanglement (8)", "hea (8)"],
    )

    fig_qcd_roc(
        [new_q_model_path],
        classical_model_paths,
        k_folds,
        losses_path,
        filename,
        signal,
        signal_dataset,
        background_dataset,
        quantum_labels,
        classical_labels,
    )

    ## wg features 16 qubits 8
    filename = "wg_inp16_roc"
    signal = "wg"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/wg_inp16_signal_data.pkl"

    plot_roc_quantum_only(
        quantum_model_paths,
        k_folds,
        losses_path,
        "wg_inp16_q_roc",
        signal,
        signal_dataset,
        background_dataset,
        ["original (8)", "new (8)", "no entanglement (8)", "hea (8)"],
    )

    fig_qcd_roc(
        [new_q_model_path],
        classical_model_paths,
        k_folds,
        losses_path,
        filename,
        signal,
        signal_dataset,
        background_dataset,
        quantum_labels,
        classical_labels,
    )

    ###########################################################
    """
    ### input 16 features 8 qubits, particle embedding
    # plotting for zzz 2 layer
    # currently no entanglement model
    original_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/ansatz_experiment_inp8_2layers/quantum/run_1/model.pkl"
    new_q_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/ansatz_experiment_inp8_2layers/quantum/run_16/model.pkl"
    no_e_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/ansatz_experiment_inp8_2layers/quantum/run_3/model.pkl"
    reuploader_q_model_path = ""
    shallow_c_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/ansatz_experiment_inp8_2layers/classical/run_13/model.pkl"
    deep_c_model_path = "/unix/qcomp/users/cduffy/anomaly_detection/saved_models/wg/ansatz_experiment_inp8_2layers/classical/run_13/model.pkl"
    k_folds = 1
    losses_path = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/qcd_inp16_roc_losses_2layers"
    background_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/qcd_inp16_signal_data.pkl"

    filename = "zzz_inp16_2_layers_roc"
    signal = "zzz"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/zzz_inp16_signal_data.pkl"

    entanglement_data = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/entanglement_data/qcd_inp6_2_layers_entanglement"
    get_entanglement_capabilities(
        original_q_model_path,
        new_q_model_path,
        "qcd_inp16_2_layers_entanglements.pdf",
        entanglement_data,
    )

    fig_qcd_roc(
        original_q_model_path,
        new_q_model_path,
        no_e_model_path,
        shallow_c_model_path,
        deep_c_model_path,
        k_folds,
        losses_path,
        filename,
        signal,
        signal_dataset,
        background_dataset,
    )

    filename = "ng_inp16_2_layers_roc"
    signal = "ng"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/ng_inp16_signal_data.pkl"

    fig_qcd_roc(
        original_q_model_path,
        new_q_model_path,
        no_e_model_path,
        shallow_c_model_path,
        deep_c_model_path,
        k_folds,
        losses_path,
        filename,
        signal,
        signal_dataset,
        background_dataset,
    )

    filename = "wg_inp16_2_layers_roc"
    signal = "wg"
    signal_dataset = "/unix/qcomp/users/cduffy/anomaly_detection/report_figures/data/wg_inp16_signal_data.pkl"

    fig_qcd_roc(
        original_q_model_path,
        new_q_model_path,
        no_e_model_path,
        shallow_c_model_path,
        deep_c_model_path,
        k_folds,
        losses_path,
        filename,
        signal,
        signal_dataset,
        background_dataset,
    )
    """
