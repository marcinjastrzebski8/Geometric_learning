"""
basic_example was still too complicated
"""

from data.datasets import SimpleSymmetricDataset
from src.geometric_classifier import QuanumClassifier, GeometricClassifier
import pennylane as qml
from pennylane import numpy as jnp
from src.utils import loss_dict, circuit_dict
from pqc_training.trainer import QuantumTrainer

N_data = 10
train_size = 8
epochs = 10
batch_size = 1
eval_interval = 1
test_size = N_data-train_size

# NOTE: this needs to be 4 for now when using automatic twirl
num_wires = 4
feature_map = 'rx_embedding'
ansatz = 'SimpleAnsatz0'
num_layers = 1

lr = 0.001
loss_fn = loss_dict['bce_loss']


data = SimpleSymmetricDataset(N_data)
# this for hardcoded twirl
# model = QuanumClassifier(feature_map, ansatz, num_wires)
# this for manual twirl
model = GeometricClassifier(feature_map, ansatz, num_wires)

model_fn = model.prediction_circuit

optimiser = qml.AdamOptimizer(lr)
save_dir = '/Users/marcinjastrzebski/Desktop/ACADEMIA/THIRD_YEAR/Geometric_classifier/models_save_dir'

circuit_properties = {'ansatz_fn': circuit_dict[ansatz],
                      'total_wires': num_wires,
                      'num_layers': num_layers,
                      'loss_fn': loss_fn,
                      # note input size in trainer is what the ansatz acts on, not ideal naming overall here
                      'input_size': num_wires,
                      'layers': num_layers,
                      'batch_size': batch_size}

trainer = QuantumTrainer()
init_params = jnp.random.uniform(0, 1, (1, num_layers, num_wires))

params, history, info = trainer.train(train_data=data,
                                      train_size=train_size,
                                      validation_size=test_size,
                                      model_fn=model_fn,
                                      loss_fn=loss_fn,
                                      optimiser_fn=optimiser,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      init_params=init_params,
                                      circuit_properties=circuit_properties,
                                      eval_interval=eval_interval,
                                      save_dir=save_dir,
                                      callbacks=[],
                                      disable_bar=False
                                      )
