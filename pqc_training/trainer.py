"""
Contains classes which can be used to train parametrised quantum circuits.
"""
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Sequence, List, Callable
import dill
from enum import Enum
import numpy as np
import torch
from tqdm import tqdm
import yaml
import pennylane as qml
from pennylane import numpy as qnp
import cloudpickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pqc_training.utils import weight_init
from ray import train as ray_train
from ray.train import Checkpoint
import tempfile
import jax
import jax_dataloader as jdl


class Colors(str, Enum):
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    CYAN = "\033[0;36m"
    RESET = "\033[0m"


class Trainer(ABC):
    def __init__(self, k_folds: int = 1):
        # TODO: Do all of those need to be attributes?
        self.k_folds = k_folds
        self.train_size = 0
        self.epochs = 0
        self.batch_size = 0
        self.init_params: dict = {}
        self.save_dir = ''
        self.callbacks: List = []
        self.disable_bar = False
        self.val_loss_histories = np.empty(0)
        self.train_loss_hist = np.empty(0)
        self.k_loss_intervals = np.empty(0)
        self.train_loss_intervals = np.empty(0)
        self.best_params = self.init_params
        self.eval_interval = 0
        self.info: dict = {}
        self.best_performance_log = tqdm()
        self.current_fold = 0

    def save_params(self, params: Union[dict, np.ndarray], save_dir: str) -> None:
        with open(f"{str(save_dir)}/params.pkl", "wb") as f:
            dill.dump(params, f)

    def save_loss(self, loss_history: np.ndarray, save_dir: str) -> None:
        filename = f"{str(save_dir)}/validation_loss"
        np.save(filename, loss_history)

    def save_setup(self, save_dir):
        with open(f"{str(save_dir)}/setup.yaml", "w", encoding='utf-8') as parameters_file:
            file_dict = {}
            for item in locals().items():
                file_dict[item[0]] = str(item[1])
            yaml.dump(file_dict, parameters_file)

    def setup_callbacks(self, circuit_properties, eval_interval):
        for i, call in enumerate(self.callbacks):
            c = call(circuit_properties)
            self.callbacks[i] = c
            self.info[str(c)] = np.full(
                (self.k_folds, int((self.epochs / eval_interval) + 1), 1000), np.NaN
            )

    def save_losses_and_info(self, save_dir):
        np.save(f"{save_dir}/train_losses.npy", self.train_loss_hist)
        np.save(f"{save_dir}/train_loss_intervlas.npy",
                self.train_loss_intervals)
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
                except TypeError as exc:
                    raise TypeError("All methods failed") from exc

    @abstractmethod
    def validate(self,
                 model: object,
                 loss_fn: Callable,
                 params: qnp.ndarray,  # pylint: disable=no-member
                 val_data: np.ndarray,
                 circuit_properties: dict,
                 ):
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
        circuit_properties: Optional[dict] = None,
        callbacks: List = [],
        disable_bar: bool = False,
        prune=False,
        use_ray=False,
        use_jax = False,
    ) -> Tuple[qnp.ndarray, np.ndarray, dict]:  # pylint: disable=no-member
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
            use_ray (bool, optional): when True will try to send reports to ray

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

        # setup performance log
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
            train_ids, val_ids = train_data.split(
                self.train_size, validation_size)
            if use_jax:
                train_loader = jdl.DataLoader(train_data,
                                              backend='jax',
                                              batch_size=int(self.batch_size),
                                              shuffle = True)
            else:
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
                use_ray,
            )
            fold_bar.update(1)
        fold_bar.close()

        print(Colors.RED + "TRAINING ENDING" + Colors.RESET)
        # NOTE: COMMENTED THESE OUT BC ERRORS WHEN USING RAY
        # self.plot_loss(
        #    self.val_loss_histories,
        #    self.k_loss_intervals,
        #    f"{save_dir}/val_loss_plot.pdf",
        # )
        # self.plot_loss(
        #    self.train_loss_hist,
        #    self.train_loss_intervals,
        #    f"{save_dir}/train_loss_plot.pdf",
        # )

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
    def train_loop(self, model_fn, train_loader, val_data, loss_fn, optimiser_fn, circuit_properties, prune, use_ray):
        pass

    # maybe add method for validation loss metrics e.g. auc scores and curves?
    # def plot_loss(self, loss_array, intervals, filename):
    #    for i in range(self.k_folds):
    #        plt.plot(intervals[i, :], loss_array[i, :], label=f"fold: {i}")
    #
    #    plt.legend()
    #    plt.savefig(filename)
    #    plt.close()

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
        params: qnp.ndarray,  # pylint: disable=no-member
        model_fn: object,
        sample_batch: np.array,
        optimiser_fn: object,
        circuit_properties: dict,
    ) -> Tuple[qnp.ndarray, float]:  # pylint: disable=no-member
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
        params: qnp.ndarray,  # pylint: disable=no-member
        model_fn: object,
        sample_batch: Sequence,
        optimiser_fn: object,
        circuit_properties: dict,
    ) -> Tuple[qnp.ndarray, float]:  # pylint: disable=no-member

        loss_sample_batch = qnp.array(sample_batch, requires_grad=False)
        metric_sample_batch = qnp.array(sample_batch[0], requires_grad=False)

        def cost_fn(p): return loss_fn(
            p,
            loss_sample_batch,
            model_fn,
            circuit_properties,
        )

        def metric_fn(p):
            tensor = qml.metric_tensor(model_fn, approx="block-diag")
            return tensor(p, metric_sample_batch, circuit_properties,)  # pylint: disable=not-callable

        params, cost = optimiser_fn.step_and_cost(
            cost_fn, params, metric_tensor_fn=metric_fn
        )

        return params, cost

    def update_metrics(self, i: int, params: qnp.ndarray, val_data: np.ndarray) -> None:  # pylint: disable=no-member
        for _, call in enumerate(self.callbacks):
            self.info[str(call)][self.current_fold, int(i / self.eval_interval), :] = (
                call(params, val_data)
            )
        return

    def update_logs(
        self, i: int, params: qnp.ndarray, loss: float, performance_log: tqdm  # pylint: disable=no-member
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
        loss_fn: Callable,
        params: qnp.ndarray,  # pylint: disable=no-member
        val_data: np.ndarray,
        circuit_properties: dict,
    ) -> float:
        loss = loss_fn(params, qnp.array(
            val_data[:][0]), model, circuit_properties)

        return loss

    def train_loop(
        self,
        model_fn: Callable,
        train_loader: DataLoader,
        val_data: np.ndarray,
        loss_fn: Callable,
        optimiser_fn: Callable,
        circuit_properties: dict,
        prune: bool,
        use_ray: bool
    ) -> qnp.ndarray:  # pylint: disable=no-member
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
        #param_shape = circuit_properties["ansatz_fn"].shape(
        #    circuit_properties["input_size"], circuit_properties["layers"]
        #)
        # NOTE: I'VE COMMENTED THIS OUT
        # params = weight_init(0, 2 * np.pi, "uniform", param_shape)
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
                if use_ray:
                    with tempfile.TemporaryDirectory() as tempdir:
                        self.save_params(self.best_params, tempdir)
                        ray_train.report(
                            metrics={'loss': cost}, checkpoint=Checkpoint.from_directory(tempdir))
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


class JaxTrainer(Trainer):
    #TODO: INTEGRATE ABSTRACT CLASS BETTER, MOSTLY TYPES TO NOT DEFAULT TO torch
    #TODO: if i want to jit, I need to understand what to do with arguments, otherwise error thrown
    #@jax.jit
    def quantum_update(self,
        loss_fn,#: object,
        opt_state,#: Callable,
        model_fn,#: object,
        sample_batch,#: Sequence,
        optimiser_fn,#: Sequence,
        circuit_properties,#: dict,
        step_id,#:int,
        ):
        print('got here0')

        #vec_model_fn = jax.vmap(model_fn)
        #@jax.jit
        def cost_fn(p): 
            return loss_fn(
            p,
            sample_batch,
            model_fn,
            circuit_properties,
        )
        
        opt_init, opt_update, get_params = optimiser_fn
        print('gothere1')
        net_params = get_params(opt_state)
        print('got here2')
        loss, grads = jax.value_and_grad(cost_fn)(net_params)
        return opt_update(step_id, grads, opt_state), loss
    
    def train_loop(
        
        self,
        model_fn: Callable,
        train_loader: DataLoader,
        val_data: np.ndarray,
        loss_fn: Callable,
        optimiser_fn: Callable,
        circuit_properties: dict,
        prune: bool,
        use_ray: bool
    ):
        """
        Main training loop.
        """
        update_fn = self.quantum_update

        opt_init, opt_update, get_params = optimiser_fn
        opt_state = opt_init(self.init_params)
        for epoch_id in range(self.epochs + 1):
            for batch_id, (sample_batch, _) in enumerate(train_loader):
                print(type(sample_batch))
                print(type(optimiser_fn))
                print(type(loss_fn))
                print(type(model_fn))
                print(type(circuit_properties))
                step_id = epoch_id*len(sample_batch) + batch_id
                print(type(step_id))
                print(step_id)
                opt_state, cost = update_fn(
                    loss_fn,
                    opt_state,
                    model_fn,
                    sample_batch,
                    optimiser_fn,
                    circuit_properties,
                    step_id,
                    )
                if use_ray:
                    with tempfile.TemporaryDirectory() as tempdir:
                        self.save_params(self.best_params, tempdir)
                        ray_train.report(
                            metrics={'loss': cost}, checkpoint=Checkpoint.from_directory(tempdir))
            print(cost)
            self.train_loss_hist[self.current_fold, epoch_id] = cost

            if epoch_id % self.eval_interval == 0:
                loss = self.validate(
                    model_fn, loss_fn, get_params(opt_state), val_data, circuit_properties
                )

                self.val_loss_histories[
                    self.current_fold, int(epoch_id / self.eval_interval)
                ] = loss

        return get_params(opt_state)
    
    def validate(
        self,
        model: object,
        loss_fn: Callable,
        params: qnp.ndarray,  # pylint: disable=no-member
        val_data: np.ndarray,
        circuit_properties: dict,
    ) -> float:
        
        loss = loss_fn(params, val_data[:][0], model, circuit_properties)

        return loss