"""
Perform a ray study of many models on the small 2x2 classification toy dataset.

Run from the terminal with one argument: path to a run config file.
The config file needs to be a .json file

Config needs to contain:
(data/training info)
- data_size: int
- n_epochs: int
- eval_interval: int
(parallelisation info)
- n_cpus: int
- n_cpus_per_model: int
- n_models: int
(results path info)
- output_models_dir: str
- local_model_save_dir: str [TODO: make optional, only when not using ray]
"""
import wandb
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler
import pandas as pd
from pathlib import Path
import ray
from ray import train as ray_train
from ray import tune
from jax.example_libraries import optimizers
from src.geometric_classifier import GeometricClassifierJax
from src.twirling import c4_on_4_qubits
from src.utils import circuit_dict, loss_dict
from pqc_training.trainer import JaxTrainer
import argparse
import json

api = wandb.Api()
path_to_package = Path('.').absolute()


def main(json_config):
    # atm I need two versions of the ansatz, one passed as a string and another as my pennylane-compatible class object
    image_size = 2
    num_wires = image_size*image_size

    # produce data
    bars_data = SymmetricDatasetJax(
        json_config['data_size'], image_size)

    circuit_train_properties = {'total_wires': total_wires}

    # define trainable which searches through hyperparameters
    def train_bars_wide(config):
        """
        Takes config which defines search space. Needs to be of this format for ray.
        """
        ansatz_str = config['ansatz']
        ansatz_obj = circuit_dict[ansatz_str]

        # NOTE: params_per_layer needs to be tied to the ansatz used
        init_params = qnp.random.uniform(
            0, 1, (json_config['n_layers'], config['params_per_layer']))


        model = GeometricClassifierJax(feature_map, ansatz, num_wires, twirled_bool, c4_on_4_qubits)
        model_fn = model.prediction_circuit

        optimiser = optimizers.adam(lr)
        loss_fn = loss_dict['bce_loss']

        trainer = QuantumTrainer()

        train(trainer,
              bars_data,
              json_config['data_size'],
              json_config['validation_size'],
              model_fn,
              loss_dict[json_config['loss_fn']],
              optimiser,
              json_config['n_epochs'],
              config['batch_size'],
              init_params,
              circuit_train_properties,
              json_config['eval_interval'],
              path_to_package/json_config['local_model_save_dir'],
              True)
              
        #TODO: add this functionality
        preds, targets = test(model_fn, valid_data)
        auc = roc_auc_score(preds, targets)
        ray_train.report(metrics={'auc': auc})
        

    # define search space for hyperparameters
    noise_schedules = [schedule(
        json_config['initial_noise_param'], json_config['n_timesteps']) for schedule in [uniform_noise, increasing_noise, decreasing_noise]]
    time_embeddings = ['rx_w_ent_embedding', 'rx_embedding']

    search_space = {'noise_schedule': tune.choice(noise_schedules), 'time_embedding': tune.choice(
        time_embeddings), 'lr': tune.loguniform(1e-3, 1e-2), 'batch_size': tune.randint(10, 100)}

    # set up ray with weights and biases logging
    ray.init(num_cpus=json_config['n_cpus'])
    scheduler = ASHAScheduler()
    trainable_with_resources = tune.with_resources(
        train_bars_wide, {'cpu': json_config['n_cpus_per_model']})
    run_config = ray_train.RunConfig(storage_path=path_to_package.parent, name=json_config['output_models_dir'], callbacks=[WandbLoggerCallback(project=json_config['output_models_dir'])], checkpoint_config=ray_train.CheckpointConfig(
        checkpoint_score_attribute='loss', num_to_keep=5))
    tuner = tune.Tuner(trainable_with_resources, param_space=search_space,
                       tune_config=tune.TuneConfig(metric='loss', mode='min', num_samples=json_config['n_models'], scheduler=scheduler), run_config=run_config)

    # run ray hyperparameter search
    tuner.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_config_path')
    parse_args = parser.parse_args()
    # load config
    with open(parse_args.run_config_path, 'r', encoding='utf-8') as f:
        load_config = json.load(f)
    main(load_config)
