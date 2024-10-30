"""
Circuits which are use to encode data into a quantum circuit.
They can but don't need to have trainable parameters.
"""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from typing import Any


class RXEmbedding(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, params, wires=None, config=None):
        super().__init__(params, wires)

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        op_list = []
        wires = qml.wires.Wires(wires)
        for qubit_id, wire in enumerate(wires):
            op_list.append(qml.RX(params[0][qubit_id], wires=wire))
        return op_list


class RXEmbeddingWEnt(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, params, wires=None, config=None):
        super().__init__(params, wires)

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperaprameters):
        op_list = []
        wires = qml.wires.Wires(wires)
        for qubit_id, wire in enumerate(wires):
            op_list.append(qml.RX(params[0][qubit_id], wires=wire))
        for qubit_id, wire in enumerate(wires[:-1]):
            op_list.append(qml.CNOT(wires=[qubit_id, qubit_id+1]))
        return op_list


class RotEmbeddingWEnt(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, params, wires=None, config=None):
        self._hyperparameters = {"embedding_pauli": config['embedding_pauli']}
        super().__init__(params, wires)

    @staticmethod
    def compute_decomposition(params: Any, wires: Any | None = None, **hyperparameters: Any):
        op_list = []
        wires = qml.wires.Wires(wires)
        for qubit_id, wire in enumerate(wires):
            op_list.append(hyperparameters['embedding_pauli'](
                params[:, qubit_id], wires=wire))
        for qubit_id, wire in enumerate(wires[:-1]):
            op_list.append(qml.CNOT(wires=[qubit_id, qubit_id+1]))
        return op_list


class RotEmbedding(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, params, wires=None, config=None):
        self._hyperparameters = {"embedding_pauli": config['embedding_pauli']}
        super().__init__(params, wires)

    @staticmethod
    def compute_decomposition(params: Any, wires: Any | None = None, **hyperparameters: Any) -> list:
        op_list = []
        wires = qml.wires.Wires(wires)
        for qubit_id, wire in enumerate(wires):
            op_list.append(hyperparameters['embedding_pauli'](
                params[:, qubit_id], wires=wire))
        return op_list
