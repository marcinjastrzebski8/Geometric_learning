"""
Circuits which are used to create a quantum learning model.
They do not depend on any data input.
"""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane import numpy as qnp
from .twirling import twirl_an_ansatz, c4_on_4_qubits


class SomeAnsatz(Operation):
    """
    Copying Ansatz1 from Callum
    """
    num_wires = AnyWires
    grad_method = None

    def __init__(self, params, wires=None, config=None):
        self._hyperparameters = {"layers": config['layers']}

        super().__init__(params, wires)

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        """
        NOTE: Callum took layers and weights as parameters,
        I want to be compatible with parent class
        """
        op_list = []
        wires = qml.wires.Wires(wires)
        for l in range(hyperparameters['layers']):
            for i, wire in enumerate(wires):
                op_list.append(qml.RY(params[0][l][i], wires=wire))
            for i, wire in enumerate(wires):
                if i == len(wires)-1:
                    op_list.append(qml.PauliRot(
                        params[0][l][len(wires)+i], pauli_word='ZZ', wires=[wire, wires[0]]))
                else:
                    op_list.append(qml.PauliRot(
                        params[0][l][len(wires)+i], pauli_word='ZZ', wires=[wires[i], wires[i+1]]))
        return op_list

    # remember: static method is one you can call on the class itself without instantiating
    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires


class SomeAnsatzTwirled(SomeAnsatz):
    """
    NOTE: when this is called during training, does it get re-instantiated every time?
    If so this can become a bottleneck maybe?
    Solution would be to save the resulting ansatz op_list and store it somewhere.
    """
    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        return twirl_an_ansatz(SomeAnsatz.compute_decomposition(*params, wires=wires, **hyperparameters), group_actions=c4_on_4_qubits, wires=wires)
