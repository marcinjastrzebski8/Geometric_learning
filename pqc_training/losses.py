import pennylane as qml
import pennylane.numpy as qnp


def measurement_loss(params, features, encoder):
    probs = 0
    for i in range(features.shape[0]):
        p = encoder(params, features[i, :])
        probs += p[0]
    loss = 1 - (probs / features.shape[0])
    return loss


""""
def fidelity_loss(
    params,
    features,
    encoder,
    properties,
):
    probs = 0
    for i in range(features.shape[0]):
        p = encoder.apply(params, features[i, :])
        probs += 1 - p
    loss = probs / features.shape[0]
    return loss
"""


def fidelity_loss(
    params,
    features,
    encoder,
    properties,
):
    probs = 0
    for i in range(features.shape[0]):
        p = encoder(params, features[i, :], properties)
        probs += 1 - p
    loss = probs / features.shape[0]
    return loss


def qfidelity_loss(
    params,
    features,
    encoder,
    properties,
):
    return (
        qnp.sum((1 - encoder(params, feat, properties)) for feat in features)
        / features.shape[0]
    )


def bures_distance_loss(params, features, encoder):
    probs = 0
    for i in range(features.shape[-1]):
        p = encoder(params, features[i, :])
        d = qnp.sqrt(2 * (1 - qnp.sqrt(p)))
        probs += d
    loss = probs / features.shape[0]
    return loss


def bures_anlge_loss(params, features, encoder):
    probs = 0
    for i in range(features.shape[0]):
        p = encoder(params, features[i, :])
        probs += qnp.arccos(qnp.sqrt(p))
    loss = probs / features.shape[0]
    return loss


def mutual_info_loss(params, features, encoder, properties):
    probs = 0
    dim_a = properties["latent_size"]
    dim_b = properties["trash_size"]
    norm_factor = 1 / (qnp.log(dim_a) + qnp.log(dim_b))
    for i in range(features.shape[0]):
        p = encoder(params, features[i, :], properties)
        probs += norm_factor * p[0] + properties["c"] * p[1][0]
    loss = probs / features.shape[0]
    return loss


def qib_loss(params, features):
    pass


loss_dict = {
    "fidelity_loss": qfidelity_loss,
}
