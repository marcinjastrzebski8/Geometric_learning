"""
On simple 2x2 image data, compare two models:
- one which uses some simple ansatz
- one which uses a twirled version of that ansatz

Encoding and measurement for both is set up in a way which makes the twirled circuit an invariant classifier.
This means that encoding is equivariant and measurement is invariant.

For robustness, future studies should compare circuits which have gone through some hyperparam opt.
The non-twirled circuit should not have to need to have the equivariant encoding and invariant measurement.
"""
from src.utils import loss_dict
from basic_example import plot_metrics_from_runs, experiment_on_simple_data
import itertools
from sklearn.metrics import roc_curve
from utils import SymmetricDatasetJax

#TODO: TURN INTO A RAY STUDY WITH MORE EPOCHS, LAYERS, DATA (CLUSTER + JIT)
N_DATA = 40
TRAIN_SIZE = 32
N_EPOCHS = 5
BATCH_SIZE = 2
EVAL_INTERVAL = 1 #note i think epochs needs to be divisible by eval_interval
LR = 0.001
N_LAYERS = 2
valid_data = SymmetricDatasetJax(N_DATA, 2)[:N_DATA-TRAIN_SIZE]
embeddings = ['RXEmbedding'] #RXEmbeddingWEnt
ansatzes = ['SimpleAnsatz0'] #SimpleAnsatz1

for embedding, ansatz in itertools.product(embeddings, ansatzes):
    print('ON ', embedding, ansatz)
    for twirled_bool in [False, True]:
        experiment_on_simple_data(
            N_DATA,
            TRAIN_SIZE,
            N_EPOCHS,
            BATCH_SIZE,
            EVAL_INTERVAL,
            embedding,
            ansatz,
            N_LAYERS,
            LR,
            twirled_bool
        )
    
    plot_metrics_from_runs(['standard', 'geometric'], f'first_compare_geo_standard_{embedding}_{ansatz}')
    plot_metrics_from_runs(['standard', 'geometric'],f'roc_curves_{embedding}_{ansatz}', 'roc', valid_data, {'layers':N_LAYERS})


