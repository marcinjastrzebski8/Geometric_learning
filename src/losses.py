from jax import numpy as jnp
from jax import jit
# TODO: FIGURE OUT THE CORRECT STRUCTURE - THIS PARAMS, FEATURES, ENCODER, PROPERTIES FORMAT IS NECESSARY IM PRETTY SURE


def sigmoid_activation(score):
    return 1/(1+jnp.exp(-score))

def binary_cross_entropy_loss(params, features, encoder, properties):
    """
    Binary cross entropy with sigmoid applied to the raw output of model.
    """

    # ASSUME MODEL SPITS OUT PREDICTION FOR CLASS '1'
    encoder_outputs = jnp.array([encoder(params, feat, properties)
                                 for feat in features])
    loss = 0
    for output, data_point in zip(encoder_outputs, features):
        label = data_point[1]
        output = sigmoid_activation(output)
        loss_component = label * jnp.log(output) + (1-label)*jnp.log(1-output)

        loss -= loss_component
    loss = loss/len(features)
    return loss


@jit
def binary_cross_entropy_loss_jit(outputs, targets):
    """
    Version which is compatible with jit.
    """
    loss = 0
    outputs_len = outputs.shape[0]
    for point_id in range(outputs.shape[0]):
        label = targets[point_id]
        output = outputs[point_id]
        output = 1/(1+jnp.exp(-output))
        loss_component = label * jnp.log(output) + (1-label)*jnp.log(1-output)

        loss -= loss_component
    loss = loss/outputs_len
    return loss



def least_squares_loss(params, features, encoder, properties):
    encoder_outputs = jnp.array(encoder(params, feat, properties) for feat in features)
    loss = 0
    for output, data_point in zip(encoder_outputs, features):
        label = data_point[1]
        loss_component = []
    
    #NOT FINISHED