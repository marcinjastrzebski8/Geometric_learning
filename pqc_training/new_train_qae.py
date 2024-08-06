import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from utils import DataLoader
import yaml
from circuits import (
    ansatz_dict,
    embedding_dict,
    swap_test,
    fidelity_circuit,
)
from losses import fidelity_loss, mutual_info_loss, qfidelity_loss
import argparse
from argparse import Namespace
from tqdm import tqdm
import sys
import os
from pathlib import Path
from utils import (
    transform,
    HiggsDataset,
    Scale,
    ConfigReader,
    setup_save_dir,
    weight_init,
    Permute,
)
from dataclasses import dataclass
from typing import Tuple, Callable, Dict, NamedTuple, List
from utils import DynamicEntanglement
import torch
from abc import ABC, abstractmethod
import jax
import optax
from jax import numpy as jnp
import dill
from typing import List, Tuple
from torch.utils.data import Dataset
import cloudpickle
import jaxpruner
import functools

# config may have to list which features it wishes to train on
# as well as specify a dataset to work with
# maybe we have a folder per model containing graphs for each
# e.g. loss history


class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    CYAN = "\033[0;36m"
    RESET = "\033[0m"


class Trainer(ABC):
    def __init__(self, k_folds: int = 1):
        self.k_folds = k_folds

    def save_params(self, params: dict | qnp.ndarray, save_dir: str) -> None:
        with open(f"{str(save_dir)}/params.pkl", "wb") as f:
            dill.dump(params, f)

    def save_loss(self, loss_history: np.ndarray, save_dir: str) -> None:
        filename = f"{str(save_dir)}/validation_loss"
        np.save(filename, loss_history)

    def save_setup(self, save_dir):
        parameters_file = open(f"{str(save_dir)}/setup.yaml", "w")
        file_dict = {}
        for item in locals().items():
            file_dict[item[0]] = str(item[1])

        yaml.dump(file_dict, parameters_file)
        parameters_file.close()

    def setup_callbacks(self, circuit_properties, eval_interval):
        for i, call in enumerate(self.callbacks):
            c = call(circuit_properties)
            self.callbacks[i] = c
            self.info[str(c)] = np.full(
                (self.k_folds, int((self.epochs / eval_interval) + 1), 1000), np.NaN
            )

    def save_losses_and_info(self, save_dir):
        np.save(f"{save_dir}/train_losses.npy", self.train_loss_hist)
        np.save(f"{save_dir}/train_loss_intervlas.npy", self.train_loss_intervals)
        np.save(f"{save_dir}/val_losses.npy", self.val_loss_histories)
        np.save(f"{save_dir}/val_loss_intervals.npy", self.k_loss_intervals)
        with open(f"{save_dir}/info.pkl", "wb") as f:
            cloudpickle.dump(self.info, f)

    def save_model(self, model_fn, loss_fn, circuit_properties, save_dir):
        with open(f"{str(save_dir)}/model.pkl", "wb") as f:
            if circuit_properties:
                saved_model = circuit_properties.copy()
            else:
                saved_model = {}
            saved_model["model_fn"] = model_fn
            saved_model["loss_fn"] = loss_fn
            saved_model["params"] = self.best_params
            methods = [dill.dump, cloudpickle.dump]
            for method in methods:
                try:
                    return method(model_fn, f)
                except:
                    TypeError
            raise ("All methods failed")

    @abstractmethod
    def validate(self):
        pass

    def train(
        self,
        train_data: Dataset,
        train_size: int,
        validation_size: int,
        model_fn: object,
        loss_fn: object,
        optimiser_fn: object,
        epochs: int,
        batch_size: int,
        init_params: dict,
        eval_interval: int,
        save_dir: str,
        circuit_properties: dict = None,
        callbacks: list = None,
        disable_bar: bool = False,
        prune=False,
    ) -> Tuple[qnp.ndarray, np.ndarray, dict]:
        """function needs to be simplified

        Args:
            train_data (Dataset): _description_
            train_size (int): _description_
            validation_size (int): _description_
            model_fn (object): _description_
            loss_fn (object): _description_
            optimiser_fn (object): _description_
            epochs (int): _description_
            batch_size (int): _description_
            init_params (dict): _description_
            eval_interval (int): _description_
            save_dir (str): _description_
            circuit_properties (dict, optional): _description_. Defaults to None.
            callbacks (list, optional): _description_. Defaults to None.
            disable_bar (bool, optional): _description_. Defaults to False.

        Returns:
            Tuple[qnp.ndarray, np.ndarray, dict]: _description_
        """

        self.train_size = train_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_params = init_params
        self.save_dir = save_dir
        self.callbacks = callbacks
        self.disable_bar = disable_bar

        self.val_loss_histories = np.full(
            (self.k_folds, int((self.epochs / eval_interval) + 1)), np.inf
        )

        self.train_loss_hist = np.zeros((self.k_folds, self.epochs + 1))
        self.k_loss_intervals = np.tile(
            np.arange(0, self.epochs + 1, eval_interval), (self.k_folds, 1)
        )
        self.train_loss_intervals = np.tile(
            np.arange(0, self.epochs + 1, 1), (self.k_folds, 1)
        )

        self.best_params = self.init_params
        self.eval_interval = eval_interval
        self.info = {}

        self.save_setup(save_dir)

        if self.callbacks:
            self.setup_callbacks(circuit_properties, eval_interval)

        ### setup performance log
        print(Colors.GREEN + "BEGINNING TRAINING" + Colors.RESET)
        self.best_performance_log = tqdm(
            total=0,
            position=0,
            bar_format="{desc}",
            leave=False,
            disable=self.disable_bar,
        )
        fold_bar = tqdm(
            total=self.k_folds,
            desc="Fold",
            leave=False,
            position=1,
            ascii=" ->",
            disable=self.disable_bar,
        )

        for i in range(self.k_folds):
            self.current_fold = i
            train_ids, val_ids = train_data.split(self.train_size, validation_size)

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            train_loader = DataLoader(
                train_data,
                batch_size=int(self.batch_size),
                sampler=train_subsampler,
            )

            val_data = train_data[val_ids]

            params = self.train_loop(
                model_fn,
                train_loader,
                val_data,
                loss_fn,
                optimiser_fn,
                circuit_properties,
                prune,
            )
            fold_bar.update(1)
        fold_bar.close()

        print(Colors.RED + "TRAINING ENDING" + Colors.RESET)

        self.plot_loss(
            self.val_loss_histories,
            self.k_loss_intervals,
            f"{save_dir}/val_loss_plot.pdf",
        )
        self.plot_loss(
            self.train_loss_hist,
            self.train_loss_intervals,
            f"{save_dir}/train_loss_plot.pdf",
        )

        # if self.callbacks:
        #    self.plot_info(f"{save_dir}/info")

        self.save_losses_and_info(save_dir)
        self.save_model(model_fn, loss_fn, circuit_properties, save_dir)

        return (
            self.best_params,
            self.val_loss_histories,
            self.info,
        )

    @abstractmethod
    def train_loop(self):
        pass

    # maybe add method for validation loss metrics e.g. auc scores and curves?
    def plot_loss(self, loss_array, intervals, filename):
        for i in range(self.k_folds):
            plt.plot(intervals[i, :], loss_array[i, :], label=f"fold: {i}")

        plt.legend()
        plt.savefig(filename)
        plt.close()

    def plot_info(self, filename):
        for _, call in enumerate(self.callbacks):
            for i in range(self.k_folds):
                y = self.info[str(call)][i, :, :]
                x = self.k_loss_intervals[i, :]
                plt.plot(x, y, label=f"fold: {i}")
            plt.legend()
            plt.title(f"{str(call)}")
            plt.savefig(f"{filename}_{str(call)}.pdf")
            plt.close()


class QuantumTrainer(Trainer):
    def classical_update(
        self,
        loss_fn: object,
        params: qnp.ndarray,
        model_fn: object,
        sample_batch: np.array,
        optimiser_fn: object,
        circuit_properties: dict,
    ) -> Tuple[qnp.ndarray, float]:
        params, cost = optimiser_fn.step_and_cost(
            loss_fn,
            params,
            features=qnp.array(sample_batch),
            encoder=model_fn,
            properties=circuit_properties,
        )

        return params, cost

    def quantum_update(
        self,
        loss_fn: object,
        params: qnp.ndarray,
        model_fn: object,
        sample_batch: np.array,
        optimiser_fn: object,
        circuit_properties: dict,
    ) -> Tuple[qnp.ndarray, float]:
        loss_sample_batch = qnp.array(sample_batch, requires_grad=False)
        metric_sample_batch = qnp.array(sample_batch[0], requires_grad=False)
        cost_fn = lambda p: loss_fn(
            p,
            loss_sample_batch,
            model_fn,
            circuit_properties,
        )
        metric_fn = lambda p: qml.metric_tensor(model_fn, approx="block-diag")(
            p,
            metric_sample_batch,
            circuit_properties,
        )
        params, cost = optimiser_fn.step_and_cost(
            cost_fn, params, metric_tensor_fn=metric_fn
        )

        return params, cost

    def update_metrics(self, i: int, params: qnp.ndarray, val_data: np.ndarray) -> None:
        for _, call in enumerate(self.callbacks):
            self.info[str(call)][self.current_fold, int(i / self.eval_interval), :] = (
                call(params, val_data)
            )
        return

    def update_logs(
        self, i: int, params: qnp.ndarray, loss: object, performance_log: tqdm
    ):
        performance_log.set_description_str(f"step: {i}, loss: {loss}")

        if (i == 0) & (self.current_fold == 0):
            self.best_performance_log.set_description_str(
                f"Best Model saved at step: {i}, loss: {loss}"
            )
            self.save_params(params, self.save_dir)
            self.best_params = params

        if (i > 0) & (np.all(loss < self.val_loss_histories)):
            self.best_performance_log.set_description_str(
                f"Best Model saved at step: {i}, loss: {loss}"
            )
            self.save_params(params, self.save_dir)
            self.best_params = params

        return

    def validate(
        self,
        model: object,
        loss_fn: object,
        params: qnp.ndarray,
        val_data: np.ndarray,
        circuit_properties: dict,
    ) -> float:
        loss = loss_fn(params, qnp.array(val_data[:][0]), model, circuit_properties)

        return loss

    def train_loop(
        self,
        model_fn: object,
        train_loader: DataLoader,
        val_data: np.ndarray,
        loss_fn: object,
        optimiser_fn: object,
        circuit_properties: dict,
        prune: bool,
    ) -> qnp.ndarray:
        outer = tqdm(
            total=self.epochs,
            desc="Epoch",
            position=2,
            leave=False,
            disable=self.disable_bar,
            ascii=" -",
        )
        performance_log = tqdm(
            total=0,
            position=3,
            bar_format="{desc}",
            leave=False,
            disable=self.disable_bar,
        )

        if isinstance(optimiser_fn, qml.QNGOptimizer):
            update_fn = self.quantum_update
        else:
            update_fn = self.classical_update

        params = self.init_params
        param_shape = circuit_properties["ansatz_fn"].shape(
            circuit_properties["input_size"], circuit_properties["layers"]
        )
        params = weight_init(0, 2 * np.pi, "uniform", param_shape)
        for i in range(self.epochs + 1):
            inner = tqdm(
                total=self.train_size // self.batch_size,
                desc="Batch",
                position=3,
                leave=False,
                disable=self.disable_bar,
                ascii=" -",
            )

            for sample_batch, _ in train_loader:
                params, cost = update_fn(
                    loss_fn,
                    params,
                    model_fn,
                    sample_batch,
                    optimiser_fn,
                    circuit_properties,
                )

                inner.update(1)
            outer.update(1)

            self.train_loss_hist[self.current_fold, i] = cost

            if i % self.eval_interval == 0:
                loss = self.validate(
                    model_fn, loss_fn, params, val_data, circuit_properties
                )

                self.update_metrics(i, params, val_data)

                self.update_logs(i, params, loss, performance_log)

                self.val_loss_histories[
                    self.current_fold, int(i / self.eval_interval)
                ] = loss
        return params


class ClassicalTrainer(Trainer):
    def update_metrics(self, i, params):
        if self.callbacks:
            for _, call in enumerate(self.callbacks):
                self.info[str(call)][self.current_fold, int(i / self.eval_interval)] = (
                    call(params)
                )

    def update_logs(self, i, params, loss, performance_log, prune=None):
        n_params = int(
            jaxpruner.summarize_sparsity(params, only_total_sparsity=True)[
                "_nparams_nnz"
            ]
        )
        performance_log.set_description_str(
            f"step: {i}, loss: {loss}, params: {n_params}"
        )

        if (i == 0) & (self.current_fold == 0):
            self.best_performance_log.set_description_str(
                f"Best Model saved at step: {i}, loss: {loss}"
            )
            self.save_params(params, self.save_dir)
            self.best_params = params

        if prune:
            val_idx = np.int(np.floor(prune["update_end_step"] / self.eval_interval))
            if (i > 0) & (np.all(loss < self.val_loss_histories[val_idx:])):
                self.best_performance_log.set_description_str(
                    f"Best Model saved at step: {i}, loss: {loss}"
                )
                self.save_params(params, self.save_dir)
                self.best_params = params

        else:
            if (i > 0) & (np.all(loss < self.val_loss_histories)):
                self.best_performance_log.set_description_str(
                    f"Best Model saved at step: {i}, loss: {loss}"
                )
                self.save_params(params, self.save_dir)
                self.best_params = params

    def validate(
        self,
        model,
        loss_fn,
        params,
        val_data,
    ):
        loss = loss_fn(params, jnp.array(val_data[:][0]), model)
        return loss

    def train_loop(
        self,
        model_fn,
        train_loader,
        val_data,
        loss_fn,
        optimiser_fn,
        circuit_properties=None,
        prune=None,
    ):
        outer = tqdm(
            total=self.epochs,
            desc="Epoch",
            position=2,
            leave=False,
            disable=self.disable_bar,
            ascii=" -",
        )
        performance_log = tqdm(
            total=0,
            position=3,
            bar_format="{desc}",
            leave=False,
            disable=self.disable_bar,
        )
        if prune:
            sparsity_distribution = functools.partial(
                jaxpruner.sparsity_distributions.erk, sparsity=prune["sparsity"]
            )
            pruner = jaxpruner.MagnitudePruning(
                sparsity_distribution_fn=sparsity_distribution,
                scheduler=jaxpruner.sparsity_schedules.PolynomialSchedule(
                    update_freq=prune["update_freq"],
                    update_start_step=prune["update_start_step"],
                    update_end_step=prune["update_end_step"],
                ),
            )
            optimiser_fn = pruner.wrap_optax(optimiser_fn)

        params = self.init_params
        opt_state = optimiser_fn.init(params)

        @jax.jit
        def update(params, opt_state, data):
            grads = jax.grad(loss_fn)(params, data, model_fn)
            updates, opt_state = optimiser_fn.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        for i in range(self.epochs + 1):
            inner = tqdm(
                total=self.train_size // self.batch_size,
                desc="Batch",
                position=4,
                leave=False,
                disable=self.disable_bar,
                ascii=" -",
            )

            for sample_batch, _ in train_loader:
                batch = jnp.array(sample_batch)
                params, opt_state = update(params, opt_state, batch)
                if prune:
                    params = pruner.post_gradient_update(params, opt_state)

                cost = loss_fn(params, batch, model_fn)
                inner.update(1)
            outer.update(1)

            self.train_loss_hist[self.current_fold, i] = cost

            if i % self.eval_interval == 0:
                loss = self.validate(model_fn, loss_fn, params, val_data)

                self.update_metrics(i, params)
                if (type(prune) == dict) and (i > prune["update_end_step"]):
                    self.update_logs(i, params, loss, performance_log, prune)
                elif not prune:
                    self.update_logs(i, params, loss, performance_log)
                else:
                    pass

                self.val_loss_histories[
                    self.current_fold, int(i / self.eval_interval)
                ] = loss

        return params


def main():
    # trying out cross validation
    input_size = 4
    latent_size = 2
    trash_size = input_size - latent_size
    total_wires = input_size + trash_size + 1

    train_size = 100
    epochs = 20
    batch_size = 10

    k_folds = 1

    train_data = HiggsDataset(
        "data/higgs_dataset",
        partition="train",
        transform=Permute([6, 3, 0, 1]),
    )

    ansatz_fn = ansatz_dict["ansatz_2"]
    embedding_fn = embedding_dict["rx_embedding"]

    init_params = weight_init(
        0,
        np.pi,
        "uniform",
        ansatz_fn.shape(4, 1),
    )

    dev = qml.device("default.qubit", wires=total_wires)
    model_fn = qml.QNode(fidelity_circuit, dev)
    loss_fn = qfidelity_loss
    optimiser = qml.AdamOptimizer(0.01)
    save_dir = setup_save_dir({"dataset": "higgs"}, "testing")

    circuit_properties = {
        "embedding_fn": embedding_fn,
        "ansatz_fn": ansatz_fn,
        "input_size": input_size,
        "latent_size": latent_size,
        "trash_size": trash_size,
        "total_wires": total_wires,
        "layers": 1,
    }

    trainer = QuantumTrainer(k_folds=k_folds)

    trainer.train(
        train_data,
        train_size,
        100,
        model_fn,
        loss_fn,
        optimiser,
        epochs,
        batch_size,
        init_params,
        circuit_properties=circuit_properties,
        eval_interval=10,
        save_dir=save_dir,
        callbacks=[DynamicEntanglement],
    )


if __name__ == "__main__":
    main()
