"""
Circuits which are use to encode data into a quantum circuit.
They can but don't need to have trainable parameters.
"""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation


def epr_state(features, qubit_ids):
    """
    NOTE: this is not an embedding but in the first phase of the diffusion
    model project I just want to be able to learn a single state
    features argument unused but is there for compatibility
    """
    qml.Hadamard(wires=qubit_ids[0])
    for qubit_id in qubit_ids[:-1]:
        qml.CNOT([qubit_id, qubit_id+1])


def rx_embedding(features, qubit_ids):
    for qubit_id in qubit_ids:
        qml.RX(features[0], wires=qubit_id)


def rx_w_ent_embedding(features, qubit_ids):
    for qubit_id in qubit_ids:
        qml.RX(features[0], wires=qubit_id)
    for qubit_id in qubit_ids[:-1]:
        qml.CNOT([qubit_id, qubit_id+1])
