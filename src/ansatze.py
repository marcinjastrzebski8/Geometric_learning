"""
Circuits which are used to create a quantum learning model.
They do not depend on any data input.
"""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires


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
                        params[0][l][len(wires)+i], pauli_word='ZZ', wires=[wire, wires[i+1]]))

        return op_list

    # remember: static method is one you can call on the class itself without instantiating
    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires


class SimpleAnsatz0(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, params, wires=None, config=None):
        self._hyperparameters = {"layers": config["layers"]}

        super().__init__(params, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        op_list = []
        wires = qml.wires.Wires(wires)
        for l in range(hyperparameters['layers']):
            for i, wire in enumerate(wires):
                op_list.append(qml.RY(params[0][l][i], wires=wire))

        return op_list


class SimpleAnsatz1(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, params, wires=None, config=None):
        self._hyperparameters = {"layers": config["layers"]}

        super().__init__(params, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        op_list = []
        wires = qml.wires.Wires(wires)
        for l in range(hyperparameters['layers']):
            for i, wire in enumerate(wires):
                op_list.append(qml.Hadamard(wires=wire))
                op_list.append(qml.RY(params[0][l][i], wires=wire))
                if i ==len(wires)-1:
                    op_list.append(qml.PauliRot(
                        params[0][l][len(wires)+i], pauli_word='YY', wires=[wire, wires[0]]))
                else:
                    op_list.append(qml.PauliRot(
                        params[0][l][len(wires)+i], pauli_word='YY', wires=[wire, wires[i+1]]))

        return op_list


class HardcodedTwirledSimpleAnsatz0(Operation):
    """
    SimpleAnsatz0 which is twirled manually.
    The twirl happens with the adjoint {I, SWAP} rep of C2.
    This can be compared to my automatic twirled circuit.
    Works on 2 qubits only.
    """
    num_wires = 2
    grad_method = None

    def __init__(self, params, wires=None, config=None):
        self._hyperparameters = {"layers": config["layers"]}

        super().__init__(params, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(*params, **hyperparameters):
        op_list = []
        wires = [0, 1]
        for l in range(hyperparameters['layers']):
            # for each Y gate, now the Y gate is applied to all qubits
            # with half the parameter
            for i, wire in enumerate(wires):
                op_list.append(qml.RY(params[0][l][i]/2, wires=0))
                op_list.append(qml.RY(params[0][l][i]/2, wires=1))
        return op_list
