import pennylane as qml
import time
import subprocess
from circuits import fidelity_circuit, ansatz_dict, embedding_dict, model_dict
from losses import loss_dict
import h5py
from examples.utils import setup_save_dir
from classical_models import original_autoencoder as autoencoder
from classical_models import mse
import haiku as hk
import jax
import optax
import dill
from examples.utils import (
    weight_init,
    DynamicEntanglement,
    EntanglementEntropy,
    setup_save_dir,
    ConfigReader,
    convert_h5_to_lists,
    DatasetFactory,
    Permute,
)
from new_test_qae import QuantumTester, ClassicalTester
from pathlib import Path
import os
from classical_models import classical_loss_dict
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from new_train_qae import QuantumTrainer, ClassicalTrainer
import yaml
from jax import config as jax_config
from pennylane import numpy as qnp
from jax import numpy as jnp
from subprocess import Popen

jax_config.update("jax_enable_x64", True)

plt.rcParams.update({"text.usetex": True, "font.family": "Lucida Grande"})
plt.rcParams["figure.dpi"] = 150


def setup_experiment_results_dir(dataset, experiment):
    root_dir = Path(
        os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        + "/anomaly_detection"
        + "/experiment_results/"
        + dataset
        + f"/{experiment}"
    )

    if not root_dir.exists():
        os.makedirs(str(root_dir))

    subdir, dirs, files = next(os.walk(root_dir))

    run_index = f"run_{len(dirs)+1}"
    save_dir = (
        Path(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
            + "/anomaly_detection"
            + "/experiment_results/"
            + dataset
        )
        / experiment
        / run_index
    )
    if not save_dir.exists():
        os.makedirs(str(save_dir))

    return save_dir


def get_experiment_results_dir(dataset, experiment):
    root_dir = Path(
        os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        + "/anomaly_detection"
        + "/experiment_results/"
        + dataset
        + f"/{experiment}"
    )

    if not root_dir.exists():
        os.makedirs(str(root_dir))

    subdir, dirs, files = next(os.walk(root_dir))

    run_index = f"run_{len(dirs)}"
    save_dir = (
        Path(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
            + "/anomaly_detection"
            + "/experiment_results/"
            + dataset
        )
        / experiment
        / run_index
    )
    if not save_dir.exists():
        os.makedirs(str(save_dir))

    return save_dir


class Experiment:
    def __init__(
        self,
        c_trainer,
        q_trainer,
        config,
        q_config,
        c_config,
        experiment_dir,
        distributed=False,
    ):
        self.q_trainer = q_trainer
        self.c_trainer = c_trainer
        self.config = config
        self.q_config = q_config
        self.c_config = c_config
        self.experiment_dir = experiment_dir
        self.independent_var = self.config["experiment_var"]
        self.distributed = distributed

        self.test_bg_loss = []
        self.test_sg_loss = []

        self.test_bgc_loss = []
        self.test_sgc_loss = []

    def get_model(self):
        dev = qml.device("default.qubit", wires=self.q_config["total_wires"])
        model_fn = qml.QNode(model_dict[self.q_config["model"]], dev)
        return model_fn

    def get_loss(self):
        loss_fn = loss_dict[self.q_config["loss"]]
        return loss_fn

    def get_optimiser(self):
        opt = getattr(qml, self.q_config["optimiser"])
        return opt(self.q_config["lr"])

    def update_quantum_config(self, config, key, var):
        config[key] = var
        config["loss_fn"] = self.get_loss()
        config["model_fn"] = self.get_model()
        config["optimiser_fn"] = self.get_optimiser()
        config["ansatz_fn"] = ansatz_dict[self.q_config["ansatz"]]
        config["embedding_fn"] = embedding_dict[self.q_config["embedding"]]
        return config

    def update_classical_config(self, config, key, var):
        if key in config:
            config[key] = var

        model = autoencoder(config["input_size"], config["latent_size"])
        config["model_fn"] = hk.without_apply_rng(hk.transform(model))

        config["loss_fn"] = classical_loss_dict[config["loss"]]
        config["optimiser_fn"] = getattr(
            optax, config["optimiser"])(config["lr"])
        return config

    def save_experiment_losses(self, losses_dir):
        # might need to change if only one classical model needed
        experiment_losses = h5py.File(
            f"{losses_dir}/experiment_losses.h5", "w")
        key = list(self.independent_var.keys())[0]
        var_list_len = len(self.independent_var[key])
        var_set_len = len(set(self.independent_var[key]))
        if var_list_len == var_set_len:
            for idx, var in enumerate(self.independent_var[key]):
                for k in range(self.config["k_folds"]):
                    experiment_losses.create_dataset(
                        f"classical/bg_losses/{key}_{var}/fold_{k}",
                        data=self.test_bgc_loss[idx][k],
                    )
                    experiment_losses.create_dataset(
                        f"classical/sg_losses/{key}_{var}/fold_{k}",
                        data=self.test_sgc_loss[idx][k],
                    )
                    experiment_losses.create_dataset(
                        f"quantum/bg_losses/{key}_{var}/fold_{k}",
                        data=self.test_bg_loss[idx][k],
                    )
                    experiment_losses.create_dataset(
                        f"quantum/sg_losses/{key}_{var}/fold_{k}",
                        data=self.test_sg_loss[idx][k],
                    )
        else:
            for idx, var in enumerate(self.independent_var[key]):
                new_var = var + str(idx)
                for k in range(self.config["k_folds"]):
                    experiment_losses.create_dataset(
                        f"classical/bg_losses/{key}_{new_var}/fold_{k}",
                        data=self.test_bgc_loss[idx][k],
                    )
                    experiment_losses.create_dataset(
                        f"classical/sg_losses/{key}_{new_var}/fold_{k}",
                        data=self.test_sgc_loss[idx][k],
                    )
                    experiment_losses.create_dataset(
                        f"quantum/bg_losses/{key}_{new_var}/fold_{k}",
                        data=self.test_bg_loss[idx][k],
                    )
                    experiment_losses.create_dataset(
                        f"quantum/sg_losses/{key}_{new_var}/fold_{k}",
                        data=self.test_sg_loss[idx][k],
                    )

    def construct_classical_training_dict(self, train_data, var, key):
        save_dir = setup_save_dir(
            self.config, self.config["save_dir"] + "/classical")
        self.c_config = self.update_classical_config(self.c_config, key, var)

        init_params = self.c_config["model_fn"].init(
            jax.random.PRNGKey(seed=0), train_data[0][0]
        )
        training_params = {
            "train_data": train_data,
            "train_size": self.c_config["train_size"],
            "validation_size": self.config["validation_size"],
            "model_fn": self.c_config["model_fn"],
            "loss_fn": self.c_config["loss_fn"],
            "optimiser_fn": self.c_config["optimiser_fn"],
            "epochs": self.c_config["epochs"],
            "batch_size": self.c_config["batch_size"],
            "init_params": init_params,
            "eval_interval": self.c_config["eval_interval"],
            "save_dir": save_dir,
            "disable_bar": self.c_config["disable_bar"],
        }

        return training_params, self.c_config["model_fn"], self.c_config["loss_fn"]

    def construct_quantum_training_dict(self, train_data, var, key):
        # neaten up make classical and quantum the same
        self.q_config = self.update_quantum_config(self.q_config, key, var)

        param_shape = self.q_config["ansatz_fn"].shape(
            self.q_config["input_size"], self.q_config["layers"]
        )

        init_params = weight_init(
            self.q_config["weight_init_range"][0],
            self.q_config["weight_init_range"][1],
            self.q_config["weight_init_dist"],
            param_shape,
        )

        save_dir = setup_save_dir(
            self.config, self.config["save_dir"] + "/quantum")

        training_params = {
            "train_data": train_data,
            "train_size": self.q_config["train_size"],
            "validation_size": self.config["validation_size"],
            "model_fn": self.q_config["model_fn"],
            "loss_fn": self.q_config["loss_fn"],
            "optimiser_fn": self.q_config["optimiser_fn"],
            "epochs": self.q_config["epochs"],
            "batch_size": self.q_config["batch_size"],
            "init_params": init_params,
            "circuit_properties": self.q_config,
            "eval_interval": self.q_config["eval_interval"],
            "save_dir": save_dir,
            "callbacks": [EntanglementEntropy],
            "disable_bar": self.q_config["disable_bar"],
        }

        with open(f"{save_dir}/circuit_diagram.txt", "w") as f:
            diagram = qml.draw(
                self.q_config["model_fn"], expansion_strategy="device", max_length=1000
            )
            f.write(diagram(init_params, train_data[0][0], self.q_config))

        return training_params, training_params["model_fn"], training_params["loss_fn"]

    def k_fold_testing(self, tester, test_data_bank, test_label_bank, save_dir):
        bg_fold_losses = []
        sg_fold_losses = []
        f, a = plt.subplots()
        for k in range(self.config["k_folds"]):
            bg_losses, sg_losses = tester.compute_losses(
                test_data_bank[k], test_label_bank[k]
            )

            labels = np.concatenate(
                (np.zeros(bg_losses.shape[0]), np.ones(sg_losses.shape[0]))
            )
            preds = np.concatenate((bg_losses, sg_losses))
            fpr, tpr, _ = roc_curve(labels, preds, drop_intermediate=False)
            auc_score = auc(fpr, tpr)
            a.plot(fpr, tpr, label=f"fold {k} auc: {auc_score}")
            auc_score = auc(bg_losses, sg_losses)

            bg_fold_losses.append(bg_losses)
            sg_fold_losses.append(sg_losses)

        a.legend()
        f.savefig(f"{save_dir}/auc_plot.pdf")
        plt.close()

        return bg_fold_losses, sg_fold_losses

    def run_experiment(self, train_data, test_data):
        # this function should be shortende
        key = list(self.independent_var.keys())[0]
        if key == "dataset":
            return self.run_dataset_experiment(train_data, test_data)
        elif self.distributed == True:
            return self.distributed_experiment(train_data, test_data)
        else:
            test_data_bank = []
            test_label_bank = []
            for _ in range(self.config["k_folds"]):
                k_test_data, k_test_labels = test_data.get_test_chunk(
                    self.config["test_size"], self.config["test_size"]
                )
                test_data_bank.append(k_test_data)
                test_label_bank.append(k_test_labels)

        for _, var in enumerate(self.independent_var[key]):
            # will want to note whether you should be retraining classical
            # based on whether experimental variable is quantum or not
            # dont care about vice versa for now maybe should though
            # would be easy once first one is done
            (
                classical_training_params,
                classical_model,
                classical_loss,
            ) = self.construct_classical_training_dict(train_data, var, key)
            (
                quantum_training_params,
                model_fn,
                loss_fn,
            ) = self.construct_quantum_training_dict(train_data, var, key)

            # maybe reorder and group by classical followed by quantum
            q_params, _, _ = self.q_trainer.train(**quantum_training_params)

            c_params, _, _ = self.c_trainer.train(**classical_training_params)

            q_tester = QuantumTester(
                model_fn, loss_fn, q_params, self.q_config)

            c_tester = ClassicalTester(
                classical_model, classical_loss, c_params)

            bg_fold_losses, sg_fold_losses = self.k_fold_testing(
                q_tester,
                test_data_bank,
                test_label_bank,
                quantum_training_params["save_dir"],
            )

            bgc_fold_losses, sgc_fold_losses = self.k_fold_testing(
                c_tester,
                test_data_bank,
                test_label_bank,
                classical_training_params["save_dir"],
            )

            self.test_bg_loss.append(bg_fold_losses)
            self.test_sg_loss.append(sg_fold_losses)

            self.test_bgc_loss.append(bgc_fold_losses)
            self.test_sgc_loss.append(sgc_fold_losses)

        parameters_file = open(
            f"{str(self.experiment_dir)}/experiment_setup.yaml", "w")
        yaml.dump(self.config, parameters_file)
        yaml.dump(self.c_config, parameters_file)
        yaml.dump(self.q_config, parameters_file)
        parameters_file.close()

        self.save_experiment_losses(self.experiment_dir)

        return (
            self.test_bg_loss,
            self.test_sg_loss,
            self.test_bgc_loss,
            self.test_sgc_loss,
        )

    def run_dataset_experiment(self, train_data, test_data):
        num_of_experiments = self.independent_var["number"]
        self.independent_var.pop("number")
        changes = self.independent_var["changes"]
        original_data_config = self.config["data"]
        train_datasets = []
        test_datasets = []
        classical_filepaths = []
        quantum_filepaths = []
        permutation_list = []
        dataset_factory = DatasetFactory()

        os.makedirs(f"{str(self.experiment_dir)}/temps")
        f = open(f"{str(self.experiment_dir)}/temps/completed_runs.txt", "w")
        f.close()

        for i in range(num_of_experiments):
            changes = self.independent_var["changes"]
            if "transform" in changes:
                permutation = np.arange(
                    int(2 * self.config["data"]["latent_rep"]))
                np.random.shuffle(permutation)
                new_data_config = original_data_config.copy()
                new_data_config["transform"] = Permute(permutation)
                permutation_list.append(permutation_list)

            new_data_config["partition"] = "train"
            train_dataset = dataset_factory.create_dataset(
                self.config["dataset"], **new_data_config
            )
            new_data_config["partition"] = "test"
            test_dataset = dataset_factory.create_dataset(
                self.config["dataset"], **new_data_config
            )

            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)

        for job in range(num_of_experiments):
            test_data_bank = []
            test_label_bank = []
            for _ in range(self.config["k_folds"]):
                k_test_data, k_test_labels = test_data.get_test_chunk(
                    self.config["test_size"], self.config["test_size"]
                )
                test_data_bank.append(k_test_data)
                test_label_bank.append(k_test_labels)

            var = train_datasets[job]

            (
                classical_training_params,
                classical_model,
                classical_loss,
            ) = self.construct_classical_training_dict(var, var, str(var))
            (
                quantum_training_params,
                model_fn,
                loss_fn,
            ) = self.construct_quantum_training_dict(var, var, str(var))

            classical_filepath = f"{self.experiment_dir}/temps/{job}"
            classical_filepaths.append(classical_filepath)
            if not Path(classical_filepath).exists():
                os.makedirs(classical_filepath)

            quantum_filepath = f"{self.experiment_dir}/temps/{job}"
            quantum_filepaths.append(quantum_filepath)
            if not Path(quantum_filepath).exists():
                os.makedirs(quantum_filepath)

            with open(f"{classical_filepath}/classical_training_params.pkl", "wb") as f:
                dill.dump(classical_training_params, f)

            with open(f"{quantum_filepath}/quantum_training_params.pkl", "wb") as f:
                dill.dump(quantum_training_params, f)

            with open(f"{classical_filepath}/classical_model.pkl", "wb") as f:
                dill.dump(classical_model, f)

            with open(f"{quantum_filepath}/quantum_model.pkl", "wb") as f:
                dill.dump(model_fn, f)

            with open(f"{classical_filepath}/classical_loss.pkl", "wb") as f:
                dill.dump(classical_loss, f)

            with open(f"{quantum_filepath}/quantum_loss.pkl", "wb") as f:
                dill.dump(loss_fn, f)

            with open(f"{classical_filepath}/k_folds.pkl", "wb") as f:
                dill.dump(self.config["k_folds"], f)

            with open(f"{quantum_filepath}/k_folds.pkl", "wb") as f:
                dill.dump(self.config["k_folds"], f)

        with open(f"{self.experiment_dir}/temps/test_dataset.pkl", "wb") as f:
            dill.dump(
                {
                    "test_data_bank": test_data_bank,
                    "test_label_bank": test_label_bank,
                },
                f,
            )

        f = open(f"{self.experiment_dir}/file_list.txt", "w")
        for item in classical_filepaths:
            f.write(item + "\n")
        f.close()

        parameters_file = open(
            f"{str(self.experiment_dir)}/experiment_setup.yaml", "w")
        self.config["experiment_var"]["permutations"] = permutation_list
        yaml.dump(self.config, parameters_file)
        yaml.dump(self.c_config, parameters_file)
        yaml.dump(self.q_config, parameters_file)
        parameters_file.close()

        for job in range(num_of_experiments):
            Popen(
                [
                    "qsub",
                    "-v",
                    f"ARGS1={self.experiment_dir},ARGS2={job+1}",
                    "run_distributed_experiment.sh",
                ],
            )

            print(f"job {job} launched")
        time.sleep(10)
        completed_runs = 0
        print("now we wait!")
        while completed_runs < num_of_experiments:
            f = open(f"{self.experiment_dir}/temps/completed_runs.txt", "r")
            contents = f.readlines()
            completed_runs = len(contents)
            print(f"runs completed: {completed_runs}")
            time.sleep(3600)

        f = open(f"{self.experiment_dir}/temps/completed_test_runs.txt", "w")
        f.close()

        for job, _ in range(num_of_experiments):
            Popen(
                [
                    "qsub",
                    "-v",
                    f"ARGS1={self.experiment_dir},ARGS2={job+1}",
                    "run_distributed_testing.sh",
                ],
            )
            print(f"job {job} launched")

        completed_runs = 0
        time.sleep(10)
        print("now we wait!")
        while completed_runs < num_of_experiments:
            f = open(
                f"{self.experiment_dir}/temps/completed_test_runs.txt", "r")
            contents = f.readlines()
            completed_runs = len(contents)
            print(f"runs completed: {completed_runs}")
            time.sleep(600)

        # open file_list.txt
        self.collate_test_results()

        return (
            self.test_bg_loss,
            self.test_sg_loss,
            self.test_bgc_loss,
            self.test_sgc_loss,
        )

    def distributed_experiment(self, train_data, test_data):
        key = list(self.independent_var.keys())[0]

        # this function should be shortende
        test_data_bank = []
        test_label_bank = []

        classical_filepaths = []
        quantum_filepaths = []

        os.makedirs(f"{str(self.experiment_dir)}/temps")
        f = open(f"{str(self.experiment_dir)}/temps/completed_runs.txt", "w")
        f.close()

        for _ in range(self.config["k_folds"]):
            k_test_data, k_test_labels = test_data.get_test_chunk(
                self.config["test_size"], self.config["test_size"]
            )
            test_data_bank.append(k_test_data)
            test_label_bank.append(k_test_labels)

        for job, var in enumerate(self.independent_var[key]):
            (
                classical_training_params,
                classical_model,
                classical_loss,
            ) = self.construct_classical_training_dict(train_data, var, key)
            (
                quantum_training_params,
                model_fn,
                loss_fn,
            ) = self.construct_quantum_training_dict(train_data, var, key)

            classical_filepath = f"{self.experiment_dir}/temps/{job}"
            classical_filepaths.append(classical_filepath)
            if not Path(classical_filepath).exists():
                os.makedirs(classical_filepath)

            quantum_filepath = f"{self.experiment_dir}/temps/{job}"
            quantum_filepaths.append(quantum_filepath)
            if not Path(quantum_filepath).exists():
                os.makedirs(quantum_filepath)

            with open(f"{classical_filepath}/classical_training_params.pkl", "wb") as f:
                dill.dump(classical_training_params, f)

            with open(f"{quantum_filepath}/quantum_training_params.pkl", "wb") as f:
                dill.dump(quantum_training_params, f)

            with open(f"{classical_filepath}/classical_model.pkl", "wb") as f:
                dill.dump(classical_model, f)

            with open(f"{quantum_filepath}/quantum_model.pkl", "wb") as f:
                dill.dump(model_fn, f)

            with open(f"{classical_filepath}/classical_loss.pkl", "wb") as f:
                dill.dump(classical_loss, f)

            with open(f"{quantum_filepath}/quantum_loss.pkl", "wb") as f:
                dill.dump(loss_fn, f)

            with open(f"{classical_filepath}/k_folds.pkl", "wb") as f:
                dill.dump(self.config["k_folds"], f)

            with open(f"{quantum_filepath}/k_folds.pkl", "wb") as f:
                dill.dump(self.config["k_folds"], f)

        with open(f"{self.experiment_dir}/temps/test_dataset.pkl", "wb") as f:
            dill.dump(
                {
                    "test_data_bank": test_data_bank,
                    "test_label_bank": test_label_bank,
                },
                f,
            )

        f = open(f"{self.experiment_dir}/file_list.txt", "w")
        for item in classical_filepaths:
            f.write(item + "\n")
        f.close()

        parameters_file = open(
            f"{str(self.experiment_dir)}/experiment_setup.yaml", "w")
        yaml.dump(self.config, parameters_file)
        yaml.dump(self.c_config, parameters_file)
        yaml.dump(self.q_config, parameters_file)
        parameters_file.close()

        for job, _ in enumerate(self.independent_var[key]):
            Popen(
                [
                    "qsub",
                    "-v",
                    f"ARGS1={self.experiment_dir},ARGS2={job+1}",
                    "run_distributed_training.sh",
                ],
            )

            print(f"job {job} launched")
        time.sleep(10)
        completed_runs = 0
        print("now we wait!")
        while completed_runs < len(self.independent_var[key]):
            f = open(f"{self.experiment_dir}/temps/completed_runs.txt", "r")
            contents = f.readlines()
            completed_runs = len(contents)
            print(f"runs completed: {completed_runs}")
            time.sleep(3600)

        f = open(f"{self.experiment_dir}/temps/completed_test_runs.txt", "w")
        f.close()

    def distributed_testing(self):
        # now we are testing

        key = list(self.independent_var.keys())[0]
        for job, _ in enumerate(self.independent_var[key]):
            Popen(
                [
                    "qsub",
                    "-v",
                    f"ARGS1={self.experiment_dir},ARGS2={job+1}",
                    "run_distributed_testing.sh",
                ],
            )
            print(f"job {job} launched")

        completed_runs = 0
        time.sleep(10)
        print("now we wait!")
        while completed_runs < len(self.independent_var[key]):
            f = open(
                f"{self.experiment_dir}/temps/completed_test_runs.txt", "r")
            contents = f.readlines()
            completed_runs = len(contents)
            print(f"runs completed: {completed_runs}")
            time.sleep(600)

        # open file_list.txt
        self.collate_test_results()

        return (
            self.test_bg_loss,
            self.test_sg_loss,
            self.test_bgc_loss,
            self.test_sgc_loss,
        )

    def collate_test_results(self):
        f = open(f"{self.experiment_dir}/file_list.txt", "r")
        classical_filepaths = f.read().splitlines()
        f.close()

        for config_filepath in classical_filepaths:
            with open(f"{config_filepath}/bgc_fold_losses.pkl", "rb") as f:
                bgc_fold_losses = dill.load(f)

            with open(f"{config_filepath}/sgc_fold_losses.pkl", "rb") as f:
                sgc_fold_losses = dill.load(f)

            with open(f"{config_filepath}/bg_fold_losses.pkl", "rb") as f:
                bg_fold_losses = dill.load(f)

            with open(f"{config_filepath}/sg_fold_losses.pkl", "rb") as f:
                sg_fold_losses = dill.load(f)

            self.test_bg_loss.append(bg_fold_losses)
            self.test_sg_loss.append(sg_fold_losses)

            self.test_bgc_loss.append(bgc_fold_losses)
            self.test_sgc_loss.append(sgc_fold_losses)

        self.save_experiment_losses(self.experiment_dir)

        return (
            self.test_bg_loss,
            self.test_sg_loss,
            self.test_bgc_loss,
            self.test_sgc_loss,
        )

    @staticmethod
    def single_train_run(config_filepath):
        with open(f"{config_filepath}/classical_model.pkl", "rb") as f:
            classical_model = dill.load(f)

        with open(f"{config_filepath}/quantum_model.pkl", "rb") as f:
            quantum_model = dill.load(f)

        with open(f"{config_filepath}/classical_loss.pkl", "rb") as f:
            classical_loss = dill.load(f)

        with open(f"{config_filepath}/quantum_loss.pkl", "rb") as f:
            quantum_loss = dill.load(f)

        with open(f"{config_filepath}/k_folds.pkl", "rb") as f:
            folds = dill.load(f)

        with open(f"{config_filepath}/quantum_training_params.pkl", "rb") as f:
            quantum_training_config = dill.load(f)

        with open(f"{config_filepath}/classical_training_params.pkl", "rb") as f:
            classical_training_config = dill.load(f)

        with open(f"{config_filepath[:-2]}/test_dataset.pkl", "rb") as f:
            test_data_dict = dill.load(f)

        # we need to reintitialize the model and init_params, they
        # do not work for some reason dill loaded
        circuit_properties = quantum_training_config["circuit_properties"]
        total_wires = circuit_properties["total_wires"]
        dev = qml.device("default.qubit", wires=total_wires)
        model_fn = qml.QNode(fidelity_circuit, dev)

        init_params = weight_init(
            0,
            np.pi,
            "uniform",
            circuit_properties["ansatz_fn"].shape(
                circuit_properties["input_size"], circuit_properties["layers"]
            ),
        )

        quantum_training_config["model_fn"] = model_fn
        quantum_training_config["init_params"] = init_params

        c_trainer = ClassicalTrainer(int(folds))
        c_params, _, _ = c_trainer.train(**classical_training_config)

        q_trainer = QuantumTrainer(int(folds))
        q_params, _, _ = q_trainer.train(**quantum_training_config)

        with open(f"{config_filepath}/classical_params.pkl", "wb") as f:
            dill.dump(c_params, f)

        with open(f"{config_filepath}/quantum_params.pkl", "wb") as f:
            dill.dump(q_params, f)

        f = open(f"{config_filepath[:-2]}/completed_runs.txt", "a")
        f.write("1\n")
        f.close()

        return q_params, c_params

    @staticmethod
    def single_test_run(config_filepath):
        with open(f"{config_filepath}/classical_model.pkl", "rb") as f:
            classical_model = dill.load(f)

        with open(f"{config_filepath}/quantum_model.pkl", "rb") as f:
            quantum_model = dill.load(f)

        with open(f"{config_filepath}/classical_loss.pkl", "rb") as f:
            classical_loss = dill.load(f)

        with open(f"{config_filepath}/quantum_loss.pkl", "rb") as f:
            quantum_loss = dill.load(f)

        with open(f"{config_filepath}/k_folds.pkl", "rb") as f:
            folds = dill.load(f)

        with open(f"{config_filepath}/quantum_training_params.pkl", "rb") as f:
            quantum_training_config = dill.load(f)

        with open(f"{config_filepath}/classical_training_params.pkl", "rb") as f:
            classical_training_config = dill.load(f)

        with open(f"{config_filepath[:-2]}/test_dataset.pkl", "rb") as f:
            test_data_dict = dill.load(f)

        with open(f"{config_filepath[:-2]}/test_dataset.pkl", "rb") as f:
            test_data_dict = dill.load(f)

        with open(f"{config_filepath}/classical_params.pkl", "rb") as f:
            c_params = dill.load(f)

        with open(f"{config_filepath}/quantum_params.pkl", "rb") as f:
            q_params = dill.load(f)

        test_data_bank = test_data_dict["test_data_bank"]
        test_label_bank = test_data_dict["test_label_bank"]
        c_tester = ClassicalTester(classical_model, classical_loss, c_params)
        q_tester = QuantumTester(
            quantum_model,
            quantum_loss,
            q_params,
            quantum_training_config["circuit_properties"],
        )

        def k_fold_testing(folds, tester, test_data_bank, test_label_bank, save_dir):
            bg_fold_losses = []
            sg_fold_losses = []
            f, a = plt.subplots()
            for k in range(folds):
                bg_losses, sg_losses = tester.compute_losses(
                    test_data_bank[k], test_label_bank[k]
                )

                labels = np.concatenate(
                    (np.zeros(bg_losses.shape[0]), np.ones(sg_losses.shape[0]))
                )

                preds = np.concatenate((bg_losses, sg_losses))
                fpr, tpr, _ = roc_curve(labels, preds, drop_intermediate=False)
                auc_score = auc(fpr, tpr)
                a.plot(fpr, tpr, label=f"fold {k} auc: {auc_score}")

                bg_fold_losses.append(bg_losses)
                sg_fold_losses.append(sg_losses)

            a.legend()
            f.savefig(f"{save_dir}/auc_plot.pdf")
            plt.close()

            return bg_fold_losses, sg_fold_losses

        bg_fold_losses, sg_fold_losses = k_fold_testing(
            folds,
            q_tester,
            test_data_bank,
            test_label_bank,
            quantum_training_config["save_dir"],
        )

        bgc_fold_losses, sgc_fold_losses = k_fold_testing(
            folds,
            c_tester,
            test_data_bank,
            test_label_bank,
            classical_training_config["save_dir"],
        )

        with open(f"{config_filepath}/bgc_fold_losses.pkl", "wb") as f:
            dill.dump(bgc_fold_losses, f)

        with open(f"{config_filepath}/sgc_fold_losses.pkl", "wb") as f:
            dill.dump(sgc_fold_losses, f)

        with open(f"{config_filepath}/bg_fold_losses.pkl", "wb") as f:
            dill.dump(bg_fold_losses, f)

        with open(f"{config_filepath}/sg_fold_losses.pkl", "wb") as f:
            dill.dump(sg_fold_losses, f)

        f = open(f"{config_filepath[:-2]}/completed_test_runs.txt", "a")
        f.write("1\n")
        f.close()


def get_roc_data(bg_loss, sg_loss):
    """We have to interpolate to assert x coordinates between folds match"""
    labels = np.concatenate(
        (np.zeros(bg_loss.shape[0]), np.ones(sg_loss.shape[0])))
    preds = np.concatenate((bg_loss, sg_loss))
    fpr, tpr, _ = roc_curve(labels, preds, drop_intermediate=False)

    new_tpr = np.linspace(0, 1, int(2 * bg_loss.shape[0] + 1))
    new_fpr = np.interp(new_tpr, tpr, fpr)

    return new_fpr, new_tpr


def get_mean_and_error(data):
    return [np.mean(data, axis=0), np.std(data, axis=0)]


class ExperimentPlotter:
    def __init__(
        self, bg_loss, sg_loss, bgc_loss, sgc_loss, ids, k_folds, q_palette, c_palette
    ):
        self.ids = np.array(ids)
        self.k_folds = k_folds
        self.bg_loss = bg_loss
        self.sg_loss = sg_loss
        self.bgc_loss = bgc_loss
        self.sgc_loss = sgc_loss
        self.q_palette = q_palette
        self.c_palette = c_palette
        print("K_FOLDS", self.k_folds)

    def compute(self):
        # very long
        self.all_tpr_means_q = []
        self.all_fpr_means_q = []
        self.all_bg_rej_means_q = []

        self.all_fpr_err_q = []
        self.all_bg_rej_err_q = []

        self.all_auc_means_q = []
        self.all_auc_err_q = []

        self.all_tpr_means_c = []
        self.all_fpr_means_c = []
        self.all_bg_rej_means_c = []

        self.all_fpr_err_c = []
        self.all_bg_rej_err_c = []

        self.all_auc_means_c = []
        self.all_auc_err_c = []

        for i, id_name in enumerate(self.ids):
            fpr_q = []
            auc_q = []
            tpr_q = []

            fpr_c = []
            auc_c = []
            tpr_c = []

            for j in range(self.k_folds):
                fq_q, tq_q = get_roc_data(
                    self.bg_loss[i][j], self.sg_loss[i][j])

                auc_q.append(auc(fq_q, tq_q))
                fpr_q.append(fq_q)
                tpr_q.append(tq_q)

                fq_c, tq_c = get_roc_data(
                    self.bgc_loss[i][j], self.sgc_loss[i][j])

                auc_c.append(auc(fq_c, tq_c))
                fpr_c.append(fq_c)
                tpr_c.append(tq_c)

            auc_data_q = get_mean_and_error(np.array(auc_q))

            bg_rej_q = get_mean_and_error(1 - np.array(fpr_q))
            bg_rej_q[0][-1] = 1

            fpr_data_q = get_mean_and_error(1.0 / np.array(fpr_q))
            bg_rej_q[0][-1] = 1

            tpr_mean_q = np.mean(np.array(tpr_q), axis=0)

            self.all_tpr_means_q.append(tpr_mean_q)
            self.all_fpr_means_q.append(fpr_data_q[0])
            self.all_bg_rej_means_q.append(bg_rej_q[0])

            self.all_fpr_err_q.append(fpr_data_q[1])
            self.all_bg_rej_err_q.append(bg_rej_q[1])

            self.all_auc_means_q.append(auc_data_q[0])
            self.all_auc_err_q.append(auc_data_q[1])

            #######
            auc_data_c = get_mean_and_error(np.array(auc_c))

            bg_rej_c = get_mean_and_error(1 - np.array(fpr_c))

            fpr_data_c = get_mean_and_error(1.0 / np.array(fpr_c))

            tpr_mean_c = np.mean(np.array(tpr_c), axis=0)

            self.all_tpr_means_c.append(tpr_mean_c)
            self.all_fpr_means_c.append(fpr_data_c[0])
            self.all_bg_rej_means_c.append(bg_rej_c[0])

            self.all_fpr_err_c.append(fpr_data_c[1])
            self.all_bg_rej_err_c.append(bg_rej_c[1])

            self.all_auc_means_c.append(auc_data_c[0])
            self.all_auc_err_c.append(auc_data_c[1])

    def get_top_k_scores(
        self,
        top_k,
        auc_means,
        auc_errs,
        tpr_means,
        fpr_means,
        fpr_errs,
        palette,
    ):
        idxs = np.argsort(auc_means)[-top_k:]
        experiment_ids = self.ids[idxs]
        auc_means = np.array(auc_means)[idxs]
        auc_errs = np.array(auc_errs)[idxs]
        tpr_means = np.array(tpr_means)[idxs]
        fpr_means = np.array(fpr_means)[idxs]
        fpr_errs = np.array(fpr_errs)[idxs]
        palette = np.array(palette)[idxs]
        return (
            experiment_ids,
            auc_means,
            auc_errs,
            tpr_means,
            fpr_means,
            fpr_errs,
            palette,
        )

    def plot_quantum(self, q_experiment_ids, save_path):
        self.compute()

        fig, ax = plt.subplots(1, figsize=(8, 5))
        styles = ["solid", "dashed"]

        ax, q_legend = self.plot_models(
            ax,
            self.all_tpr_means_q,
            self.all_fpr_means_q,
            self.all_fpr_err_q,
            self.all_auc_means_q,
            self.all_auc_err_q,
            q_experiment_ids,
            "solid",
            self.q_palette,
        )

        dummy_res_lines = [
            Line2D([0, 1], [0, 1], linestyle=s, color="black") for s in styles[:2]
        ]
        lines = ax.get_lines()
        ax.semilogy(
            np.linspace(0, 1, num=int(self.bg_loss[0][0].shape[0])),
            1.0 / np.linspace(0, 1, int(self.bg_loss[0][0].shape[0])),
            linewidth=1.5,
            linestyle="--",
            color="0.75",
        )

        legend1 = ax.legend(
            dummy_res_lines,
            [r"Quantum", r"Classical"],
            frameon=False,
            loc="upper right",
            handlelength=1.5,
            fontsize=16,
            title_fontsize=14,
        )

        legend2 = ax.legend(
            [lines[i] for i in range(len(self.q_palette))],
            q_legend,
            loc="lower left",
            frameon=True,
            title=r"$Quantum \; Models$",
            fontsize=15,
            title_fontsize=14,
            markerscale=0.5,
        )

        legend3 = ax.legend(
            [lines[top_k_quantum + i] for i in range(len(self.c_palette))],
            c_legend,
            loc="upper left",
            frameon=True,
            title=r"$Classical Models$",
            fontsize=15,
            title_fontsize=14,
            markerscale=0.5,
        )
        legend2.get_frame().set_alpha(0.35)
        legend3.get_frame().set_alpha(0.35)

        ax.set_xlabel("TPR")
        ax.set_ylabel("1/FPR")

        legend1._legend_box.align = "left"
        legend2._legend_box.align = "left"
        legend3._legend_box.align = "left"

        for leg in legend1.legendHandles:
            leg.set_linewidth(2.2)
            leg.set_color("gray")
        # for leg in legend2.legendHandles:
        #    leg.set_linewidth(2.2)

        ax.add_artist(legend1)
        ax.add_artist(legend2)
        ax.add_artist(legend3)
        ax.set_yscale("log")

        fig.tight_layout()
        fig.savefig(f"{save_path}")
        plt.show()

    def plot_performance(self, save_path, top_k_quantum=None, top_k_classical=None):
        self.compute()

        if top_k_quantum:
            (
                q_experiment_ids,
                self.all_auc_means_q,
                self.all_auc_err_q,
                self.all_tpr_means_q,
                self.all_fpr_means_q,
                self.all_fpr_err_q,
                self.q_palette,
            ) = self.get_top_k_scores(
                top_k_quantum,
                self.all_auc_means_q,
                self.all_auc_err_q,
                self.all_tpr_means_q,
                self.all_fpr_means_q,
                self.all_fpr_err_q,
                self.q_palette,
            )
            q_experiment_ids = ["original ansatz", "new ansatz"]

        if top_k_classical:
            (
                c_experiment_ids,
                self.all_auc_means_c,
                self.all_auc_err_c,
                self.all_tpr_means_c,
                self.all_fpr_means_c,
                self.all_fpr_err_c,
                self.c_palette,
            ) = self.get_top_k_scores(
                top_k_classical,
                self.all_auc_means_c,
                self.all_auc_err_c,
                self.all_tpr_means_c,
                self.all_fpr_means_c,
                self.all_fpr_err_c,
                self.c_palette,
            )

            c_experiment_ids = ["shallow network", "deep network"]

        q_experiment_ids = ["original ansatz", "new ansatz"]
        # ["shallow network", "deep network"]
        c_experiment_ids = ["neural network"]

        fig, ax = plt.subplots(1, figsize=(8, 5))
        styles = ["solid", "dashed"]

        ax, q_legend = self.plot_models(
            ax,
            self.all_tpr_means_q,
            self.all_fpr_means_q,
            self.all_fpr_err_q,
            self.all_auc_means_q,
            self.all_auc_err_q,
            q_experiment_ids,
            "solid",
            self.q_palette,
        )

        ax, c_legend = self.plot_models(
            ax,
            self.all_tpr_means_c,
            self.all_fpr_means_c,
            self.all_fpr_err_c,
            self.all_auc_means_c,
            self.all_auc_err_c,
            c_experiment_ids,
            "dashed",
            self.c_palette,
        )

        dummy_res_lines = [
            Line2D([0, 1], [0, 1], linestyle=s, color="black") for s in styles[:2]
        ]
        lines = ax.get_lines()
        ax.semilogy(
            np.linspace(0, 1, num=int(self.bg_loss[0][0].shape[0])),
            1.0 / np.linspace(0, 1, int(self.bg_loss[0][0].shape[0])),
            linewidth=1.5,
            linestyle="--",
            color="0.75",
        )

        legend1 = ax.legend(
            dummy_res_lines,
            [r"Quantum", r"Classical"],
            frameon=False,
            loc="upper right",
            handlelength=1.5,
            fontsize=16,
            title_fontsize=14,
        )

        legend2 = ax.legend(
            [lines[i] for i in range(len(self.q_palette))],
            q_legend,
            loc="lower left",
            frameon=True,
            title=r"$Quantum \; Models$",
            fontsize=15,
            title_fontsize=14,
            markerscale=0.5,
        )

        legend3 = ax.legend(
            [lines[top_k_quantum + i] for i in range(len(self.c_palette))],
            c_legend,
            loc="upper left",
            frameon=True,
            title=r"$Classical Models$",
            fontsize=15,
            title_fontsize=14,
            markerscale=0.5,
        )
        legend2.get_frame().set_alpha(0.35)
        legend3.get_frame().set_alpha(0.35)

        ax.set_xlabel("TPR")
        ax.set_ylabel("1/FPR")

        legend1._legend_box.align = "left"
        legend2._legend_box.align = "left"
        legend3._legend_box.align = "left"

        for leg in legend1.legendHandles:
            leg.set_linewidth(2.2)
            leg.set_color("gray")
        # for leg in legend2.legendHandles:
        #    leg.set_linewidth(2.2)

        ax.add_artist(legend1)
        ax.add_artist(legend2)
        ax.add_artist(legend3)
        ax.set_yscale("log")

        fig.tight_layout()
        fig.savefig(f"{save_path}")
        plt.show()

    def plot_models(
        self,
        ax,
        tpr_means,
        fpr_means,
        fpr_errs,
        auc_means,
        auc_errs,
        ids,
        linestyle,
        palette,
    ):
        model_legend = []
        for i, id_name in enumerate(ids):
            ax.plot(
                tpr_means[i],
                fpr_means[i],
                label=f"{id_name}: Q {
                    np.round(auc_means[i], 3)}+/-{np.round(auc_errs[i], 4)}",
                linewidth=1.5,
                color=palette[i],
                linestyle=linestyle,
            )

            ax.fill_between(
                tpr_means[i],
                fpr_means[i] - fpr_errs[i],
                fpr_means[i] + fpr_errs[i],
                alpha=0.2,
                color=palette[i],
                linestyle=linestyle,
            )
            model_legend.append(
                f"{id_name} "
                f"auc score: {np.round(auc_means[i], 3)}"
                f"± {np.round(auc_errs[i], 4)}"
            )
        return ax, model_legend


class ExperimentPlotter:
    def __init__(
        self,
        bg_loss,
        sg_loss,
        bgc_loss,
        sgc_loss,
        quantum_ids,
        classical_ids,
        k_folds,
        q_palette,
        c_palette,
    ):
        self.quantum_ids = np.array(quantum_ids)
        self.classical_ids = np.array(classical_ids)
        self.k_folds = k_folds
        self.bg_loss = bg_loss
        self.sg_loss = sg_loss
        self.bgc_loss = bgc_loss
        self.sgc_loss = sgc_loss
        self.q_palette = q_palette
        self.c_palette = c_palette
        print("K_FOLDS", self.k_folds)

    def compute_quantum(self):
        # very long
        self.all_tpr_means_q = []
        self.all_fpr_means_q = []
        self.all_bg_rej_means_q = []

        self.all_fpr_err_q = []
        self.all_bg_rej_err_q = []

        self.all_auc_means_q = []
        self.all_auc_err_q = []

        for i, id_name in enumerate(self.quantum_ids):
            fpr_q = []
            auc_q = []
            tpr_q = []

            for j in range(self.k_folds):
                fq_q, tq_q = get_roc_data(
                    self.bg_loss[i][j], self.sg_loss[i][j])

                auc_q.append(auc(fq_q, tq_q))
                fpr_q.append(fq_q)
                tpr_q.append(tq_q)

            auc_data_q = get_mean_and_error(np.array(auc_q))

            bg_rej_q = get_mean_and_error(1 - np.array(fpr_q))
            bg_rej_q[0][-1] = 1

            fpr_data_q = get_mean_and_error(1.0 / np.array(fpr_q))
            bg_rej_q[0][-1] = 1

            tpr_mean_q = np.mean(np.array(tpr_q), axis=0)

            self.all_tpr_means_q.append(tpr_mean_q)
            self.all_fpr_means_q.append(fpr_data_q[0])
            self.all_bg_rej_means_q.append(bg_rej_q[0])

            self.all_fpr_err_q.append(fpr_data_q[1])
            self.all_bg_rej_err_q.append(bg_rej_q[1])

            self.all_auc_means_q.append(auc_data_q[0])
            self.all_auc_err_q.append(auc_data_q[1])

    def compute_classical(self):
        self.all_tpr_means_c = []
        self.all_fpr_means_c = []
        self.all_bg_rej_means_c = []

        self.all_fpr_err_c = []
        self.all_bg_rej_err_c = []

        self.all_auc_means_c = []
        self.all_auc_err_c = []

        for i, id_name in enumerate(self.classical_ids):
            fpr_c = []
            auc_c = []
            tpr_c = []

            for j in range(self.k_folds):
                fq_c, tq_c = get_roc_data(
                    self.bgc_loss[i][j], self.sgc_loss[i][j])

                auc_c.append(auc(fq_c, tq_c))
                fpr_c.append(fq_c)
                tpr_c.append(tq_c)

            auc_data_c = get_mean_and_error(np.array(auc_c))

            bg_rej_c = get_mean_and_error(1 - np.array(fpr_c))

            fpr_data_c = get_mean_and_error(1.0 / np.array(fpr_c))

            tpr_mean_c = np.mean(np.array(tpr_c), axis=0)

            self.all_tpr_means_c.append(tpr_mean_c)
            self.all_fpr_means_c.append(fpr_data_c[0])
            self.all_bg_rej_means_c.append(bg_rej_c[0])

            self.all_fpr_err_c.append(fpr_data_c[1])
            self.all_bg_rej_err_c.append(bg_rej_c[1])

            self.all_auc_means_c.append(auc_data_c[0])
            self.all_auc_err_c.append(auc_data_c[1])

    def get_top_k_scores(
        self,
        top_k,
        auc_means,
        auc_errs,
        tpr_means,
        fpr_means,
        fpr_errs,
        palette,
    ):
        idxs = np.argsort(auc_means)[-top_k:]
        experiment_ids = self.ids[idxs]
        auc_means = np.array(auc_means)[idxs]
        auc_errs = np.array(auc_errs)[idxs]
        tpr_means = np.array(tpr_means)[idxs]
        fpr_means = np.array(fpr_means)[idxs]
        fpr_errs = np.array(fpr_errs)[idxs]
        palette = np.array(palette)[idxs]
        return (
            experiment_ids,
            auc_means,
            auc_errs,
            tpr_means,
            fpr_means,
            fpr_errs,
            palette,
        )

    def plot_quantum_roc(self, save_path):
        self.compute_quantum()

        fig, ax = plt.subplots(1, figsize=(8, 8))
        styles = ["solid", "dashed"]

        ax, q_legend = self.plot_models(
            ax,
            self.all_tpr_means_q,
            self.all_fpr_means_q,
            self.all_fpr_err_q,
            self.all_auc_means_q,
            self.all_auc_err_q,
            self.quantum_ids,
            "solid",
            self.q_palette,
        )

        dummy_res_lines = [
            Line2D([0, 1], [0, 1], linestyle=s, color="black") for s in styles[:2]
        ]
        lines = ax.get_lines()
        ax.semilogy(
            np.linspace(0, 1, num=int(self.bg_loss[0][0].shape[0])),
            1.0 / np.linspace(0, 1, int(self.bg_loss[0][0].shape[0])),
            linewidth=1.5,
            linestyle="--",
            color="0.75",
        )

        legend2 = ax.legend(
            [lines[i] for i in range(len(self.q_palette))],
            q_legend,
            loc="lower left",
            frameon=True,
            title=r"$Quantum \; Models \; (params)$",
            fontsize=17,
            title_fontsize=17,
            markerscale=0.5,
        )

        legend2.get_frame().set_alpha(0.35)

        ax.set_xlabel("TPR", fontsize=17)
        ax.set_ylabel("1/FPR", fontsize=17)

        legend2._legend_box.align = "left"

        ax.add_artist(legend2)
        ax.set_yscale("log")

        fig.tight_layout()
        fig.savefig(f"{save_path}")
        plt.show()

    def plot_performance(self, save_path, top_k_quantum=None, top_k_classical=None):
        self.compute_quantum()
        self.compute_classical()

        if top_k_quantum:
            (
                q_experiment_ids,
                self.all_auc_means_q,
                self.all_auc_err_q,
                self.all_tpr_means_q,
                self.all_fpr_means_q,
                self.all_fpr_err_q,
                self.q_palette,
            ) = self.get_top_k_scores(
                top_k_quantum,
                self.all_auc_means_q,
                self.all_auc_err_q,
                self.all_tpr_means_q,
                self.all_fpr_means_q,
                self.all_fpr_err_q,
                self.q_palette,
            )

        if top_k_classical:
            (
                c_experiment_ids,
                self.all_auc_means_c,
                self.all_auc_err_c,
                self.all_tpr_means_c,
                self.all_fpr_means_c,
                self.all_fpr_err_c,
                self.c_palette,
            ) = self.get_top_k_scores(
                top_k_classical,
                self.all_auc_means_c,
                self.all_auc_err_c,
                self.all_tpr_means_c,
                self.all_fpr_means_c,
                self.all_fpr_err_c,
                self.c_palette,
            )

        fig, ax = plt.subplots(1, figsize=(8, 8))
        styles = ["solid", "dashed"]

        ax, q_legend = self.plot_models(
            ax,
            self.all_tpr_means_q,
            self.all_fpr_means_q,
            self.all_fpr_err_q,
            self.all_auc_means_q,
            self.all_auc_err_q,
            self.quantum_ids,
            "solid",
            self.q_palette,
        )

        ax, c_legend = self.plot_models(
            ax,
            self.all_tpr_means_c,
            self.all_fpr_means_c,
            self.all_fpr_err_c,
            self.all_auc_means_c,
            self.all_auc_err_c,
            self.classical_ids,
            "dashed",
            self.c_palette,
        )

        dummy_res_lines = [
            Line2D([0, 1], [0, 1], linestyle=s, color="black") for s in styles[:2]
        ]
        lines = ax.get_lines()
        ax.semilogy(
            np.linspace(0, 1, num=int(self.bg_loss[0][0].shape[0])),
            1.0 / np.linspace(0, 1, int(self.bg_loss[0][0].shape[0])),
            linewidth=1.5,
            linestyle="--",
            color="0.75",
        )

        legend1 = ax.legend(
            dummy_res_lines,
            [r"Quantum", r"Classical"],
            frameon=False,
            loc="upper right",
            handlelength=1.5,
            fontsize=17,
            title_fontsize=17,
        )

        legend2 = ax.legend(
            [lines[i] for i in range(len(self.q_palette))],
            q_legend,
            loc="lower left",
            frameon=True,
            title=r"$Quantum \; Models \; (params)$",
            fontsize=17,
            title_fontsize=17,
            markerscale=0.5,
        )

        legend3 = ax.legend(
            [lines[len(self.q_palette) + i]
             for i in range(len(self.c_palette))],
            c_legend,
            loc="upper left",
            frameon=True,
            title=r"$Classical \; Models \; (params)$",
            fontsize=17,
            title_fontsize=17,
            markerscale=0.5,
        )
        legend2.get_frame().set_alpha(0.35)
        legend3.get_frame().set_alpha(0.35)

        ax.set_xlabel("TPR", fontsize=16)
        ax.set_ylabel("1/FPR", fontsize=16)

        legend1._legend_box.align = "left"
        legend2._legend_box.align = "left"
        legend3._legend_box.align = "left"

        for leg in legend1.legendHandles:
            leg.set_linewidth(2.2)
            leg.set_color("gray")
        # for leg in legend2.legendHandles:
        #    leg.set_linewidth(2.2)

        ax.add_artist(legend1)
        ax.add_artist(legend2)
        ax.add_artist(legend3)
        ax.set_yscale("log")

        fig.tight_layout()
        fig.savefig(f"{save_path}")
        plt.show()

    def plot_models(
        self,
        ax,
        tpr_means,
        fpr_means,
        fpr_errs,
        auc_means,
        auc_errs,
        ids,
        linestyle,
        palette,
    ):
        model_legend = []
        for i, id_name in enumerate(ids):
            ax.plot(
                tpr_means[i],
                fpr_means[i],
                label=f"{id_name}: {
                    np.round(auc_means[i], 3)}+/-{np.round(auc_errs[i], 4)}",
                linewidth=1.5,
                color=palette[i],
                linestyle=linestyle,
            )

            ax.fill_between(
                tpr_means[i],
                fpr_means[i] - fpr_errs[i],
                fpr_means[i] + fpr_errs[i],
                alpha=0.2,
                color=palette[i],
                linestyle=linestyle,
            )
            model_legend.append(
                f"{id_name}: "
                f"{np.round(auc_means[i], 3)}"
                # f"±{np.round(auc_errs[i],4)}"
            )
        return ax, model_legend


def train(config, q_config, c_config):
    # maybe in config file layout train and test dataset configs
    # seperately?
    # maybe construct datasets within experiment? maybe just give option
    dataset_config = config["data"]

    train_dataset_config = dataset_config.copy()
    train_dataset_config["partition"] = "train"

    test_dataset_config = dataset_config.copy()
    test_dataset_config["partition"] = "test"

    dataset_factory = DatasetFactory()

    train_dataset = dataset_factory.create_dataset(
        config["dataset"], **train_dataset_config
    )

    test_dataset = dataset_factory.create_dataset(
        config["dataset"], **test_dataset_config
    )
    q_trainer = QuantumTrainer(k_folds=config["k_folds"])
    c_trainer = ClassicalTrainer(k_folds=config["k_folds"])

    experiment_dir = setup_experiment_results_dir(
        config["dataset"], config["save_dir"])
    print("Experiment!", experiment_dir)

    train_size_experiment = Experiment(
        c_trainer,
        q_trainer,
        config,
        q_config,
        c_config,
        experiment_dir,
        distributed=True,
    )
    train_size_experiment.run_experiment(train_dataset, test_dataset)

    # (
    #    test_bg_loss,
    #    test_sg_loss,
    #    test_bgc_loss,
    #    test_sgc_loss,
    # ) = train_size_experiment.run_experiment(train_dataset, test_dataset)


def test(config, q_config, c_config, distribued=True):
    if distribued:
        experiment_dir = get_experiment_results_dir(
            config["dataset"], config["save_dir"]
        )
        print(f"testing on: {experiment_dir}")
        q_trainer = QuantumTrainer(k_folds=config["k_folds"])
        c_trainer = ClassicalTrainer(k_folds=config["k_folds"])

        experiment = Experiment(
            c_trainer,
            q_trainer,
            config,
            q_config,
            c_config,
            experiment_dir,
            distributed=True,
        )
        experiment.distributed_testing()
        # experiment.collate_test_results()

        losses_dir = get_experiment_results_dir(
            config["dataset"], config["save_dir"])

        key = list(config["experiment_var"].keys())[0]
        ids = [str(i) for i in config["experiment_var"][key]]
        k_folds = config["k_folds"]

        test_bg_loss, test_sg_loss, test_bgc_loss, test_sgc_loss = convert_h5_to_lists(
            f"{losses_dir}/experiment_losses.h5", config
        )

        experiment_plotter = ExperimentPlotter(
            test_bg_loss,
            test_sg_loss,
            test_bgc_loss,
            test_sgc_loss,
            ids,
            k_folds,
            config["q_palette"],
            config["c_palette"],
        )
        experiment_plotter.plot_performance(f"{losses_dir}/roc.pdf", 2, 1)

    else:
        losses_dir = get_experiment_results_dir(
            config["dataset"], config["save_dir"])

        key = list(config["experiment_var"].keys())[0]
        ids = [str(i) for i in config["experiment_var"][key]]
        k_folds = config["k_folds"]

        test_bg_loss, test_sg_loss, test_bgc_loss, test_sgc_loss = convert_h5_to_lists(
            f"{losses_dir}/experiment_losses.h5", config
        )

        experiment_plotter = ExperimentPlotter(
            test_bg_loss,
            test_sg_loss,
            test_bgc_loss,
            test_sgc_loss,
            ids,
            k_folds,
            config["q_palette"],
            config["c_palette"],
        )
        experiment_plotter.plot_performance(f"{losses_dir}/roc.pdf", 2, 1)


def main(config, q_config, c_config):
    train(config, q_config, c_config)
    test(config, q_config, c_config)
    """
    print("Welcome to the experiment!\n")
    options = ["train & test", "train", "test", "test_runs"]
    for i in range(len(options)):
        print(f"{i}. {options[i]}")
    choice = input("Please select a functionality: ")

    valid_inputs = [str(i) for i in range(len(options))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")
    choice = int(choice)
    if choice == 0:
        train(config, q_config, c_config)
        test(config)
    elif choice == 1:
        train(config, q_config, c_config)
    elif choice == 2:
        test(config)
    """


if __name__ == "__main__":
    config_reader = ConfigReader()
    g_config, c_config, q_config = config_reader.compile_config()
    main(g_config, q_config, c_config)
