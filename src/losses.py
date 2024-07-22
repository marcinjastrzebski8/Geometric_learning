import numpy as np
import pennylane as qml
import pennylane.numpy as qnp


def binary_cross_entropy_loss(params, features, encoder, properties):
    # TODO: FIGURE OUT THE CORRECT STRUCTURE - THIS PARAMS, FEATURES, ENCODER, PROPERTIES FORMAT IS NECESSARY IM PRETTY SURE

    # ASSUME MODEL SPITS OUT PREDICTION FOR CLASS '1'
    encoder_outputs = qnp.array([encoder(params, feat, properties)
                                 for feat in features])
    loss = 0
    for output, data_point in zip(encoder_outputs, features):
        label = data_point[1]

        loss_component = label * qnp.log(output) + (1-label)*qnp.log(1-output)

        loss -= loss_component
    loss = loss/len(features)
    return loss
