"""
For now just make a simple classification pipeline with some basic model.
Later will expand to compare basic and geometric models.
We'll stick to 2x2 images at first.
#TODO: figure out how to jit properly
"""

from utils import SymmetricDatasetJax
from src.geometric_classifier import GeometricClassifierJax
from src.losses import sigmoid_activation
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from src.utils import loss_dict, circuit_dict
from pqc_training.trainer import JaxTrainer
import jax
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt
from src.twirling import c4_on_4_qubits
import dill
jax.config.update("jax_traceback_filtering", "off")
jax.config.update('jax_platform_name', 'cpu')




def experiment_on_simple_data(n_data,
                              train_size, 
                              n_epochs,
                              batch_size,
                              eval_interval,
                              feature_map,
                              ansatz, 
                              n_layers,
                              lr,
                              twirled_bool):
    
   

    #TODO: FIGURE JIT OUT   
    #loss_fn = jax.jit(loss_fn)

    image_size = 2
    num_wires = image_size*image_size

    test_size = n_data-train_size
    data = SymmetricDatasetJax(n_data, image_size)

    model = GeometricClassifierJax(feature_map, ansatz, num_wires, twirled_bool, c4_on_4_qubits)
    model_fn = model.prediction_circuit

    #this is made up of three functions; opt_init, opt_update, get_params
    optimiser  = optimizers.adam(lr)
    loss_fn = loss_dict['bce_loss']

    

    save_dir = '/Users/marcinjastrzebski/Desktop/ACADEMIA/THIRD_YEAR/Geometric_classifier/models_save_dir/'
    if twirled_bool:
        folder_name = 'geometric'
    else:
        folder_name = 'standard'
    save_dir+=folder_name

    circuit_properties = {'ansatz_fn': circuit_dict[ansatz],
                      'total_wires': num_wires,
                      'num_layers': n_layers,
                      'loss_fn': loss_fn,
                      # note input size in trainer is what the ansatz acts on, not ideal naming overall here
                      'input_size': num_wires,
                      #TODO: does this need to be passed twice????
                      'layers': n_layers,
                      'batch_size': batch_size}
    
    np.random.seed(1)
    init_params = np.random.uniform(0, 1, (n_layers, 2*num_wires))

    trainer = JaxTrainer(init_params,
                     train_size,
                     test_size,
                     epochs = n_epochs,
                     batch_size=batch_size,
                     callbacks=[],
                     eval_interval=eval_interval,
                     save_dir=save_dir)
    
    params, history, info = trainer.train(train_data=data,
                                      model_fn=model_fn,
                                      loss_fn=loss_fn,
                                      optimiser_fn=optimiser,
                                      circuit_properties=circuit_properties
                                      )

def plot_metrics_from_runs(run_names, figname, which_plot = 'loss', valid_data=None, config=None):

    """
    Plot losses and roc curves.
    """
    models_dir = '/Users/marcinjastrzebski/Desktop/ACADEMIA/THIRD_YEAR/Geometric_classifier/models_save_dir/'

    fig, ax = plt.subplots(1,1)

    for run_name in run_names:
        run_dir = models_dir + run_name
        if which_plot == 'loss':
            x_axis = np.load(run_dir+'/train_loss_intervlas.npy')[0]
            y_axis = np.load(run_dir + '/train_losses.npy')[0]
            additional_info = ''
        elif which_plot == 'roc':
            assert (valid_data is not None) and (config is not None)

            with open(run_dir + '/model.pkl', 'rb') as file:
                model = dill.load(file)
            with open(run_dir + '/params.pkl', 'rb') as file:
                params = dill.load(file)

            preds = []
            for data_point in valid_data[0]:
                pred = sigmoid_activation(model(params, data_point, config))
                preds.append(pred)

            x_axis, y_axis, _ = roc_curve(valid_data[1], preds, drop_intermediate=False)
            auc = roc_auc_score(valid_data[1], preds)
            additional_info = 'AUC: '+format(auc, '.3f')

        ax.plot(x_axis, y_axis, label = f'{run_name} '+additional_info)
        ax.legend()
        plt.savefig(figname, dpi = 300)


"""
N_DATA = 100
TRAIN_SIZE = 80
N_EPOCHS = 10
BATCH_SIZE = 10
EVAL_INTERVAL = 2

FEATURE_MAP= 'rx_embedding'
ANSATZ = 'SimpleAnsatz0'
LOSS_FN = loss_dict['bce_loss']
LR = 0.001
N_LAYERS= 1
TWIRLED_BOOL = True

experiment_on_simple_data(N_DATA, 
                        TRAIN_SIZE,
                        N_EPOCHS,
                        BATCH_SIZE,
                        EVAL_INTERVAL,
                        FEATURE_MAP,
                        ANSATZ,
                        N_LAYERS,
                        LR,
                        TWIRLED_BOOL,
                        )
"""
