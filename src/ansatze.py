"""
Circuits which are used to create a quantum learning model.
They do not depend on any data input.
"""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires


class SimpleAnsatz0(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, params, wires=None, config=None):
        self._hyperparameters = {'n_layers': config['n_layers']}
        super().__init__(params, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        op_list = []
        # the unpacking adds an extra dimension which is spurious
        params = params[0]
        wires = qml.wires.Wires(wires)
        for l in range(hyperparameters['n_layers']):
            for i, wire in enumerate(wires):
                op_list.append(qml.RY(params[l][i], wires=wire))

        return op_list


class SimpleAnsatz1(Operation):
    """
    I think this is stolen from Callum. It's IQP-esque but I don't think it's based on anything rigorous.
    """
    num_wires = AnyWires
    grad_method = None

    def __init__(self, params, wires=None, config=None):
        self._hyperparameters = {'n_layers': config['n_layers']}
        super().__init__(params, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        op_list = []
        wires = qml.wires.Wires(wires)
        n_layers = hyperparameters['n_layers']
        # the unpacking adds an extra dimension which is spurious
        params = params[0]
        for l in range(n_layers):
            for i, wire in enumerate(wires):
                op_list.append(qml.Hadamard(wires=wire))
                op_list.append(qml.RY(params[l][i], wires=wire))
                if i == len(wires)-1:
                    op_list.append(qml.PauliRot(
                        params[l][len(wires)+i], pauli_word='YY', wires=[wire, wires[0]]))
                else:
                    op_list.append(qml.PauliRot(
                        params[l][len(wires)+i], pauli_word='YY', wires=[wire, wires[i+1]]))

        return op_list


class MatchCallumAnsatz(Operation):
    """
    Seems slightly different to any other simple ansatz I've got so far and want to match Callum exactly.
    """

    def __init__(self, params, wires=None, config=None):
        self._hyperparameters = {'n_layers': config['n_layers'],
        'ansatz_block': config['ansatz_block']}
        super().__init__(params, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        # the unpacking adds an extra dimension which is spurious
        params = params[0]
        op_list = []
        wires = qml.wires.Wires(wires)
        for j in range(hyperparameters['n_layers']):
            for k, wire in enumerate(wires):
                for a_i, a in enumerate(hyperparameters['ansatz_block']):
                    op_list.append(a(params[j][k][a_i], wires=wire))
                if k != len(wires)-1:
                    qml.CNOT(wires=[k, k+1])

        return op_list


class GeneralCascadingAnsatz(Operation):
    """
    SimpleAnsatz1 generalised to general pauli-word rotations.
    Also got rid of the hadamard.
    """
    num_wires = AnyWires
    grad_method = None

    def __init__(self, params, wires=None, config=None):
        self._hyperparameters = {'n_layers': config['n_layers'],
                                 'single_qubit_pauli': config['single_qubit_pauli'],
                                 'two_qubit_pauli': config['two_qubit_pauli']}
        super().__init__(params, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        """
        single_qubit_pauli is a qml single qubit rotation gate (e.g. qml.RX)
        two_qubit_pauli is a pauli word, e.g. 'ZZ'
        """

        op_list = []
        wires = qml.wires.Wires(wires)
        for l in range(hyperparameters['n_layers']):
            for i, wire in enumerate(wires):
                op_list.append(hyperparameters['single_qubit_pauli'](
                    params[0][l][i], wires=wire))
                if i == len(wires)-1:
                    op_list.append(qml.PauliRot(
                        params[0][l][len(wires)+i], pauli_word=hyperparameters['two_qubit_pauli'], wires=[wire, wires[0]]))
                else:
                    op_list.append(qml.PauliRot(
                        params[0][l][len(wires)+i], pauli_word=hyperparameters['two_qubit_pauli'], wires=[wire, wires[i+1]]))

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
        self._hyperparameters = {'n_layers': config['n_layers']}
        super().__init__(params, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(*params, **hyperparameters):
        op_list = []
        wires = [0, 1]
        for l in range(hyperparameters['n_layers']):
            # for each Y gate, now the Y gate is applied to all qubits
            # with half the parameter
            for i, wire in enumerate(wires):
                op_list.append(qml.RY(params[0][l][i]/2, wires=0))
                op_list.append(qml.RY(params[0][l][i]/2, wires=1))
        return op_list


class GeometricAnsatzConstructor(Operation):
    """
    Construct an equivariant ansatz from a set of equivarant gates.
    Each 'instruction' corresponds to a single gate before twirling.
    After twirling usually becomes multiple gates with a shared parameter.

    Assumes an n_layers-repeated structure of :
    -/- 1-local gates -/- 2-local gates -/-
    """

    def __init__(self, params, wires, config):
        # NOTE: these are being implicitly passed to compute_decomposition
        self._hyperparameters = {"group_equiv_1local_gate": config['group_equiv_1local_gate'],
                                 'group_equiv_2local_gate': config['group_equiv_2local_gate'],
                                 'gate_1local_instructions': config['gate_1local_instructions'],
                                 'gate_2local_instructions': config['gate_2local_instructions'],
                                 'n_layers': config['n_layers']
                                 }

        super().__init__(params, wires)

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparams):
        """
        Allows to create any ansatz of the (1local-2local) x n_layers shape
        """
        # TODO: CLEAN THIS UP I WASNT SURE IF I CAN DEFINE VARIABLES WITHOUT THEM BEING ADDED TO THE QUEUE
        group_equiv_1local_gate = hyperparams['group_equiv_1local_gate']
        group_equiv_2local_gate = hyperparams['group_equiv_2local_gate']
        gate_1local_instructions = hyperparams['gate_1local_instructions']
        gate_2local_instructions = hyperparams['gate_2local_instructions']
        n_layers = hyperparams['n_layers']
        n_1local_gates = len(gate_1local_instructions)
        n_2local_gates = len(gate_2local_instructions)
        assert np.shape(*params) == (n_layers,
                                     len(hyperparams['gate_1local_instructions'])+len(hyperparams['gate_2local_instructions']))

        op_list = []
        for layer_id in range(n_layers):
            for gate_id, gate_1local_instruction in enumerate(hyperparams['gate_1local_instructions']):
                op_list.append(hyperparams['group_equiv_1local_gate'](
                    params[0][layer_id][gate_id], config=gate_1local_instruction))

            for gate_id, gate_2local_instruction in enumerate(hyperparams['gate_2local_instructions']):
                op_list.append(hyperparams['group_equiv_2local_gate'](params[0][layer_id][len(
                    hyperparams['gate_1local_instructions'])+gate_id], config=gate_2local_instruction))

        return op_list
