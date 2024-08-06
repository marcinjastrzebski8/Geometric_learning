"""
For now just make a simple classification pipeline with some basic model.
Later will expand to compare basic and geometric models.
We'll stick to 2x2 images at first.
"""

from utils import SymmetricDataset, SymmetricDatasetJax
from src.geometric_classifier import QuanumClassifier, GeometricClassifier, GeometricClassifierJax
import pennylane as qml
import numpy as np
from src.utils import loss_dict, circuit_dict
from pqc_training.trainer import QuantumTrainer, JaxTrainer
import jax
from jax.example_libraries import optimizers
jax.config.update("jax_traceback_filtering", "off")
jax.config.update('jax_platform_name', 'cpu')

N_data = 100
train_size = 80
epochs = 10
batch_size = 10
eval_interval = 2
test_size = N_data-train_size

image_size = 2
num_wires = int(image_size*image_size)
feature_map = 'rx_embedding'
ansatz = 'SimpleAnsatz0'
num_layers = 1

lr = 0.001
loss_fn = loss_dict['bce_loss']

use_jax = True

if use_jax:
    #loss_fn = jax.jit(loss_fn)
    loss_fn = loss_fn

data = SymmetricDatasetJax(N_data, image_size)
print(type(data))

# model = QuanumClassifier(feature_map, ansatz, num_wires)
if use_jax:
    model = GeometricClassifierJax(feature_map, ansatz, num_wires)
else:
    model = GeometricClassifier(feature_map, ansatz, num_wires)

model_fn = model.prediction_circuit
if use_jax:
    #this is made up of three functions; opt_init, opt_update, get_params
    optimiser  = optimizers.adam(lr)
else:
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
if use_jax:
    trainer = JaxTrainer()
else:
    trainer = QuantumTrainer()
init_params = np.random.uniform(0, 1, (1, num_layers, 2*num_wires))
# init_params = jnp.random.uniform(0, 1, (num_layers, num_wires))
# init_params = jnp.random.uniform(0, 1, (1, num_layers, 2*num_wires - 1))

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
                                      disable_bar=False,
                                      use_jax=True
                                      )
