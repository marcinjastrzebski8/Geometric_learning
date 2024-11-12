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
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from pqc_training.utils import weight_init
from ray import train as ray_train
from ray.train import Checkpoint
import tempfile
import jax
import jax_dataloader as jdl
from jax import numpy as jnp


def model_predictions_for_jit(params, features, encoder, properties):
    """
    Splitting up the funcitonality of the loss so I can jit it.
    This part sadly cannot be jitted as the encoder param is too abstract.
    """
    encoder_outputs = jnp.array([encoder(params, feat, properties)
                                 for feat in features])

    return encoder_outputs, features[1]


class Colors(str, Enum):
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    CYAN = "\033[0;36m"
    RESET = "\033[0m"


class Trainer(ABC):
    """
    NOTE: this Trainer class assumes working with qml.qnodes as models.
    NewTrainer set up for torch.Module models.
    """

    def __init__(self,
                 init_params: np.ndarray,
                 train_size: int = 1,
                 validation_size: int = 1,
                 k_folds: int = 1,
                 epochs: int = 1,
                 batch_size: int = 1,
                 callbacks: List = [],
                 eval_interval: int = 1,
                 save_dir: str = '',
                 disable_bar=False
                 ):

        # TODO: Do all of those need to be attributes?
        self.init_params = init_params
        self.train_size = train_size
        self.validation_size = validation_size
        self.k_folds = k_folds
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.save_dir = save_dir
        self.eval_interval = eval_interval
        self.disable_bar = disable_bar
        self.val_loss_histories = np.full(
            (self.k_folds, int((self.epochs / self.eval_interval) + 1)), np.inf
        )
        self.k_loss_intervals = np.tile(
            np.arange(0, self.epochs + 1,
                      self.eval_interval), (self.k_folds, 1)
        )
        self.train_loss_hist = np.zeros((self.k_folds, self.epochs + 1))
        self.train_loss_intervals = np.tile(
            np.arange(0, self.epochs + 1, 1), (self.k_folds, 1)
        )
        self.best_params = self.init_params
        self.info: dict = {}
        self.best_performance_log = tqdm()
        self.current_fold = 0

        self._use_jax = False

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
        data: Union[Dataset, jdl.Dataset],
        model_fn: object,
        loss_fn: object,
        optimiser_fn: object,
        circuit_properties: Optional[dict] = None,
        use_ray=False,
    ) -> Tuple[qnp.ndarray, np.ndarray, dict]:  # pylint: disable=no-member
        """function needs to be simplified

        Args:
            data (Dataset): _description_
            validation_size (int): _description_
            model_fn (object): _description_
            loss_fn (object): _description_
            optimiser_fn (object): _description_
            epochs (int): _description_
            batch_size (int): _description_
            eval_interval (int): _description_
            save_dir (str): _description_
            circuit_properties (dict, optional): _description_. Defaults to None.
            callbacks (list, optional): _description_. Defaults to None.
            disable_bar (bool, optional): _description_. Defaults to False.
            use_ray (bool, optional): when True will try to send reports to ray

        Returns:
            Tuple[qnp.ndarray, np.ndarray, dict]: _description_
        """

        self.save_setup(self.save_dir)

        if self.callbacks:
            self.setup_callbacks(circuit_properties, self.eval_interval)

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
            train_ids, val_ids = data.split(
                self.train_size, self.validation_size)

            val_data = jdl.ArrayDataset(*data[val_ids])
            train_data = jdl.ArrayDataset(*data[train_ids])

            if self._use_jax:
                train_loader = jdl.DataLoader(train_data,
                                              backend='jax',
                                              batch_size=int(self.batch_size),
                                              shuffle=True)
            else:
                # this seems overly complicated
                train_subsampler = torch.utils.data.SubsetRandomSampler(
                    train_ids)
                train_loader = DataLoader(
                    train_data,
                    batch_size=int(self.batch_size),
                    sampler=train_subsampler,
                )

            params = self.train_loop(
                model_fn,
                train_loader,
                val_data,
                loss_fn,
                optimiser_fn,
                circuit_properties,
                use_ray,
            )
            fold_bar.update(1)
        fold_bar.close()

        print(Colors.RED + "TRAINING ENDING" + Colors.RESET)

        self.save_losses_and_info(self.save_dir)
        self.save_model(model_fn, loss_fn, circuit_properties, self.save_dir)

        return (
            self.best_params,
            self.val_loss_histories,
            self.info,
        )

    @abstractmethod
    def train_loop(self, model_fn, train_loader, val_data, loss_fn, optimiser_fn, circuit_properties, use_ray):
        pass


class TorchQMLTrainer(Trainer):
    """
    Trainer which assumes pytorch pipleline and data as qnp.arrays
    NOTE: ongoing is organising this module such that I understand it and it makes sense to me
    inherited from Callum/Mohammad and there were stuff I didn't understand about it
    """

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
        # NOTE: not sure what this is used for

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
            # pylance suggests this is unreachable but that's probably not true and a case of bad typing
            update_fn = self.quantum_update
        else:
            update_fn = self.classical_update

        params = self.init_params
        # param_shape = circuit_properties["ansatz_fn"].shape(
        #    circuit_properties["input_size"], circuit_properties["layers"]
        # )
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
    """
    Trainer which uses Jax pipeline.
    """
    # @jax.jit

    def __init__(self,
                 init_params: np.ndarray,
                 train_size: int = 1,
                 validation_size: int = 1,
                 k_folds: int = 1,
                 epochs: int = 1,
                 batch_size: int = 1,
                 callbacks: List = [],
                 eval_interval: int = 1,
                 save_dir: str = '',
                 disable_bar=False):

        super().__init__(init_params,
                         train_size,
                         validation_size,
                         k_folds,
                         epochs,
                         batch_size,
                         callbacks,
                         eval_interval,
                         save_dir,
                         disable_bar)
        self._use_jax = True

    def classical_update(
        self,
        loss_fn,
        params,
        model_fn,
        sample_batch,
        optimiser_fn,
        circuit_properties
    ):
        """
        TODO: develop this when needing to compare to classical models - will this be much different to quantum update? prolly not
        I don't even think it'd be different. I just never made a hybrid model using jax.
        """
        pass

    def quantum_update(self,
                       loss_fn: object,
                       opt_state,  # : Callable,
                       model_fn,  # : object,
                       sample_batch,  # : Sequence,
                       optimiser_fn,  # : Sequence,
                       circuit_properties: dict,
                       step_id: int,
                       ):

        # vec_model_fn = jax.vmap(model_fn)
        def cost_fn(p):
            return loss_fn(
                p,
                sample_batch,
                model_fn,
                circuit_properties,
            )

        opt_init, opt_update, get_params = optimiser_fn
        net_params = get_params(opt_state)
        loss, grads = jax.value_and_grad(cost_fn)(net_params)
        return opt_update(step_id, grads, opt_state), loss

    def update_logs(
            self, i: int, params: qnp.ndarray, loss: float, performance_log: tqdm  # pylint: disable=no-member
    ):
        """
        NOTE: this just copied from TorchQMLTrainer
        """
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
        """
        Same as TorchQMLTrainer but data not passed as qnp.array
        """

        loss = loss_fn(params, val_data[:][0], model, circuit_properties)

        return loss

    def train_loop(
        self,
        model_fn: Callable,
        train_loader: DataLoader,
        val_data: np.ndarray,
        loss_fn: Callable,
        optimiser_fn: Callable,
        circuit_properties: dict,
        use_ray: bool
    ):
        """
        Main training loop.
        """
        update_fn = self.quantum_update

        opt_init, opt_update, get_params = optimiser_fn
        opt_state = opt_init(self.init_params)
        # NOTE these are the same as in TorchQMLTrainer
        # outer what exactly
        outer = tqdm(total=self.epochs,
                     desc="Epoch",
                     position=2,
                     leave=False,
                     disable=self.disable_bar,
                     ascii=" -"
                     )

        performance_log = tqdm(
            total=0,
            position=3,
            bar_format="{desc}",
            leave=False,
            disable=self.disable_bar)

        for epoch_id in range(self.epochs + 1):
            inner = tqdm(
                total=self.train_size // self.batch_size,
                desc="Batch",
                position=3,
                leave=False,
                disable=self.disable_bar,
                ascii=" -",
            )
            for batch_id, (sample_batch, _) in enumerate(train_loader):

                step_id = epoch_id*len(sample_batch) + batch_id
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
                            metrics={'loss': float(cost)}, checkpoint=Checkpoint.from_directory(tempdir))
                inner.update(1)
            outer.update(1)
            self.train_loss_hist[self.current_fold, epoch_id] = cost

            if epoch_id % self.eval_interval == 0:
                loss = self.validate(
                    model_fn, loss_fn, get_params(
                        opt_state), val_data, circuit_properties
                )

                self.update_logs(epoch_id, get_params(
                    opt_state), loss, performance_log)

                self.val_loss_histories[
                    self.current_fold, int(epoch_id / self.eval_interval)
                ] = loss

        return get_params(opt_state)


class JaxTrainerJit(Trainer):
    """
    Trainer which uses Jax pipeline which I want to make compatible with jitting.
    """

    def __init__(self,
                 init_params: np.ndarray,
                 train_size: int = 1,
                 validation_size: int = 1,
                 k_folds: int = 1,
                 epochs: int = 1,
                 batch_size: int = 1,
                 callbacks: List = [],
                 eval_interval: int = 1,
                 save_dir: str = '',
                 disable_bar=False):

        super().__init__(init_params,
                         train_size,
                         validation_size,
                         k_folds,
                         epochs,
                         batch_size,
                         callbacks,
                         eval_interval,
                         save_dir,
                         disable_bar)
        self._use_jax = True

    def classical_update(
        self,
        loss_fn,
        params,
        model_fn,
        sample_batch,
        optimiser_fn,
        circuit_properties
    ):
        """
        TODO: develop this when needing to compare to classical models - will this be much different to quantum update? prolly not
        """
        pass

    def quantum_update(self,
                       loss_fn: object,
                       opt_state,  # : Callable,
                       model_fn,  # : object,
                       sample_batch,  # : Sequence,
                       optimiser_fn,  # : Sequence,
                       circuit_properties: dict,
                       step_id: int,
                       ):

        # vec_model_fn = jax.vmap(model_fn)

        # NOTE: at the moment this is the only bit that I think I can jit
        # the jitting does not happen here, the loss function passed has to be decorated with jit
        # the only thing changed here is the functionality is split compared to the non-jitted implementation
        def cost_fn(params):
            outputs, targets = model_predictions_for_jit(
                params, sample_batch, model_fn, circuit_properties)

            return loss_fn(outputs, targets)

        opt_init, opt_update, get_params = optimiser_fn
        net_params = get_params(opt_state)
        loss, grads = jax.value_and_grad(cost_fn)(net_params)
        return opt_update(step_id, grads, opt_state), loss

    def update_logs(
            self, i: int, params: qnp.ndarray, loss: float, performance_log: tqdm  # pylint: disable=no-member
    ):
        """
        NOTE: this just copied from TorchQMLTrainer
        """
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
        """
        Same as TorchQMLTrainer but data not passed as qnp.array
        """
        outputs, targets = model_predictions_for_jit(
            params, val_data[:][0], model, circuit_properties)
        loss = loss_fn(outputs, targets)

        return loss

    def train_loop(
        self,
        model_fn: Callable,
        train_loader: DataLoader,
        val_data: np.ndarray,
        loss_fn: Callable,
        optimiser_fn: Callable,
        circuit_properties: dict,
        use_ray: bool
    ):
        """
        Main training loop.
        """
        update_fn = self.quantum_update

        opt_init, opt_update, get_params = optimiser_fn
        opt_state = opt_init(self.init_params)
        # NOTE these are the same as in TorchQMLTrainer
        # outer what exactly
        outer = tqdm(total=self.epochs,
                     desc="Epoch",
                     position=2,
                     leave=False,
                     disable=self.disable_bar,
                     ascii=" -"
                     )

        performance_log = tqdm(
            total=0,
            position=3,
            bar_format="{desc}",
            leave=False,
            disable=self.disable_bar)

        for epoch_id in range(self.epochs + 1):
            inner = tqdm(
                total=self.train_size // self.batch_size,
                desc="Batch",
                position=3,
                leave=False,
                disable=self.disable_bar,
                ascii=" -",
            )
            for batch_id, (sample_batch, _) in enumerate(train_loader):

                step_id = epoch_id*len(sample_batch) + batch_id
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
                inner.update(1)
            outer.update(1)
            self.train_loss_hist[self.current_fold, epoch_id] = cost

            if epoch_id % self.eval_interval == 0:
                loss = self.validate(
                    model_fn, loss_fn, get_params(
                        opt_state), val_data, circuit_properties
                )

                self.update_logs(epoch_id, get_params(
                    opt_state), loss, performance_log)

                self.val_loss_histories[
                    self.current_fold, int(epoch_id / self.eval_interval)
                ] = loss

        return get_params(opt_state)


class NewTrainer():
    """
    NOTE: DOES NOT INHERIT FROM TRAINER AS TRAINER DOES NOT SUPPORT NN.MODULE BASED CLASSIFIERS
    TODO: paused devving this, but maybe pick up if useful. I like the idea, just a tiny bit too much effort atm.
    Should just look for a good example online.
    Heavily inspired by Trainer class from Callum/Mohammad but for torch.Module-based models.
    Also written fully by me so I can understand everything that's happening inside.
    """

    def __init__(self,
                 k_folds: int = 1,
                 epochs: int = 1,
                 eval_interval: int = 1,
                 save_dir: str = ''):

        self.k_folds = k_folds
        self.current_fold = 0
        self.epochs = epochs
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        # use these if not saving with ray
        self.val_loss_histories = np.full((self.k_folds, int(
            (self.epochs/self.eval_interval)+1)), np.inf)
        self.train_loss_histories = np.full(
            (self.k_folds, self.epochs + 1), np.inf)

    def save_model(self, model, save_dir):
        with open(f"{str(save_dir)}/model_state.pth", "wb") as f:
            torch.save(model.state_dict(), f)

    def save_losses(self, save_dir):
        np.save(f"{save_dir}/train_losses.npy", self.train_loss_hist)
        np.save(f"{save_dir}/val_losses.npy", self.val_loss_histories)

    def validate(self, model, validation_dataset, criterion):
        outputs = model(validation_dataset[0]).view(-1)
        val_loss = criterion(outputs, validation_dataset[1])
        return val_loss.item()

    def train(self,
              model,
              data: Dataset,
              optimizer,
              criterion,
              train_size: int,
              validation_size: int,
              batch_size: int,
              standalone_val_dataset: Optional[Dataset] = None,
              use_ray=False):
        """
        Allows for k-fold validation training.
        If k is set to 1 but validation is desired, a separate standalone validation dataset can be passed.
        NOTE: train_size and validation size are not smart. Need to make sense with k_folds.

        """

        model.train()
        # k-fold validation training loop
        for i in range(self.k_folds):
            self.current_fold = i
            train_ids, val_ids = data.split(
                train_size=train_size, validation_size=validation_size)

            if (self.k_folds == 1) and validation_size == 0:
                val_data = standalone_val_dataset
            else:
                val_data: Dataset | Subset = Subset(data, val_ids)
            train_data = Subset(data, train_ids)

            train_loader = DataLoader(
                train_data, batch_size=int(batch_size))
            self.train_loop(model, train_loader, val_data,
                            optimizer, criterion, use_ray)

    def train_loop(self,
                   model,
                   train_loader,
                   val_data,
                   optimizer,
                   criterion,
                   use_ray):
        """
        Loop over epochs and batches.
        Assumes loss (criterion) which averages over batch.
        """

        for i in range(self.epochs+1):
            running_loss = 0.0
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_data).view(-1)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_data.size(0)
            # training loss
            epoch_loss = running_loss/len(train_loader.dataset)

            if i % self.eval_interval == 0:
                # validate
                if val_data is not None:
                    val_loss = self.validate(model, val_data, criterion)
                else:
                    val_loss = 0

                # model states are saved when training loss becomes smallest in history
                # NOTE: this might not be the best strategy
                # report losses to ray
                if use_ray:
                    with tempfile.TemporaryDirectory() as tempdir:
                        if np.all(epoch_loss < self.train_loss_histories):
                            self.save_model(model, tempdir)
                        ray_train.report(
                            metrics={'loss': epoch_loss, 'val_loss': val_loss}, checkpoint=Checkpoint.from_directory(tempdir))
                # report losses locally
                else:
                    if np.all(epoch_loss < self.train_loss_histories):
                        self.save_model(model, self.save_dir)

                    self.save_losses(self.save_dir)

                # update loss history
                self.train_loss_histories[self.current_fold, i] = epoch_loss
                self.val_loss_histories[
                    self.current_fold, int(i / self.eval_interval)
                ] = val_loss
