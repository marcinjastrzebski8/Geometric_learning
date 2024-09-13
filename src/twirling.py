"""
Collection of functions related to twirling of ansatze.
Probably won't be very general at first. Maybe never.
Goal is to be able to have
- at first a working twirled version of a simple ansatz
- later a working twirled version of a more useful ansatz

Twirling will be wrt to D4 [prolly C4 at first] on nxn images represented on n*n qubits
with n=2 at first.
"""

from typing import List
import pennylane as qml
from pennylane.operation import Operation

import jax
from pennylane.operation import Operator

# NOTE: would be nice to generalise this


def c4_rep_on_qubits(image_size=2):
    """
    Representation of the c4 group on qubits where the qubits represent pixels of an image.
    """
    # NOTE: rotation happens clockwise
    # NOTE: order of operations follows matrix notation
    #   qml.prod also follows matrix notation - as expected
    if image_size == 2:
        rot_90 = ((0, 2), (0, 3), (0, 1))
        rot_180 = ((0, 3), (1, 2))
        rot_270 = ((0, 1), (0, 3), (0, 2))
    elif image_size == 3:
        rot_90 = ((1, 3), (1, 7), (1, 5), (0, 6), (0, 8), (0, 2))
        rot_180 = ((5, 3), (1, 7), (2, 6), (0, 8))
        rot_270 = ((5, 7), (5, 3), (1, 5), (2, 8), (2, 6), (0, 2))
    else:
        raise NotImplementedError(
            'Image sizes other than 2, 3 are not implemented yet.')
    return [qml.I(wires=range(int(image_size*image_size)))] + [qml.prod(*[qml.SWAP(wires=indexes)
                                                                          for indexes in group_element]) for group_element in (rot_90, rot_180, rot_270)]


def some_simple_group(n_wires=4):
    return [qml.I(wires=range(n_wires)), qml.SWAP(wires=[0, 1])]


def twirl(operator, group_actions):
    # NOTE: 25/07 CHANGING TO ADJOINT REPRESENTATION
    # THERE'S SOMETHING I DONT UNDERSTAND ABOUT CHOOSING THE REPRESENTATION I WANT
    # FIG.9 I THINK HAS ANSWERS
    twirl_contributions = []
    for group_action in group_actions:
        twirl_contributions.append(
            qml.prod(qml.prod(group_action, operator), group_action.adjoint()))
    twirled_op = qml.s_prod(1/len(group_actions),
                            qml.sum(*twirl_contributions))
    return twirled_op


def twirl_an_ansatz(ansatz, group_actions, qubit_size):
    """
    Returns list of operations [twirled ansatz]
    """
    op_list = []
    for gate in ansatz:
        if gate.has_generator:
            twirled_generator = twirl(gate.generator(), group_actions)
            twirled_gate_matrix = jax.scipy.linalg.expm(
                1.0j*gate.parameters[0]*twirled_generator.matrix())
        else:
            twirled_gate_matrix = twirl(gate, group_actions).matrix()

        twirled_gate = qml.QubitUnitary(
            twirled_gate_matrix, wires=range(qubit_size))

        op_list.append(twirled_gate)
    return op_list


class C4On9QEquivGate1Local(Operation):
    """
    All single-qubit Paulis twirled by C4 on 9 qubits are of this type.
    Shape depends on where the original gate acted.
    Assuming qubit-to-pixel map as such:
    -------
    |0|1|2|
    |3|4|5|
    |6|7|8|
    -------
    """

    def __init__(self, params, wires=None, config=None):
        self._hyperparameters = {'gate_placement': config['gate_placement'],
                                 'gate': config['gate']}
        super().__init__(params, wires)

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters) -> List[Operator]:
        gate_placement = hyperparameters['gate_placement']
        gate = hyperparameters['gate']
        if gate_placement == 'corner':
            gate_qubits = [0, 2, 6, 8]
        elif gate_placement == 'side':
            gate_qubits = [1, 3, 5, 7]
        elif gate_placement == 'centre':
            gate_qubits = [4]
        else:
            raise ValueError(f'Incorrect gate_placement passed: \
                             {gate_placement}')

        op_list = []
        for wire in gate_qubits:
            op_list.append(gate(params[0]/4, wires=wire))
        return op_list


class C4On9QEquivGate2Local(Operation):
    """
    Two-qubit Paulis twirled by C4 on 9 qubits are of this type.
    Shape depends on where the original gates acted.
    Excludes global gates which cannot be decomposed into local gates
    (gates whose generators have non-commuting elements).

    Assuming qubit-to-pixel map as such:
    -------
    |0|1|2|
    |3|4|5|
    |6|7|8|
    -------
    NOTE: atm there's nothing checking that the gates passed are paulis
        I wonder if the considerations work for other two-qubit gates like cnots - should check
        This was made for gates with generators in mind - not sure what happens to gates wo generators
    """

    def __init__(self, params, wires=None, config=None):
        self._hyperparameters = {'gate_placement': config['gate_placement'],
                                 'pauli_word': config['pauli_word']}
        super().__init__(params, wires)

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters) -> List[Operator]:
        gate_placement = hyperparameters['gate_placement']
        pauli_word = hyperparameters['pauli_word']

        if gate_placement == 'corner_centre':
            gate_qubits = [[0, 4], [2, 4], [6, 4], [8, 4]]
        elif gate_placement == 'side_centre':
            gate_qubits = [[1, 4], [5, 4], [7, 4], [3, 4]]
        elif gate_placement == 'ring_neighbours_corner':
            gate_qubits = [[0, 1], [2, 5], [8, 7], [6, 3]]
        elif gate_placement == 'ring_neighbours_side':
            gate_qubits = [[1, 2], [5, 8], [7, 6], [3, 0]]
        elif gate_placement == 'ring_second_neighbours_corner':
            gate_qubits = [[0, 2], [2, 8], [8, 6], [6, 0]]
            if pauli_word not in ['ZZ', 'YY', 'XX']:
                raise ValueError(
                    f'This gate combination results in twirled non-local gates which are not supported, you passed {gate_placement}, {pauli_word}')
        elif gate_placement == 'ring_second_neighbours_side':
            gate_qubits = [[1, 5], [5, 7], [7, 3], [3, 1]]
            if pauli_word not in ['ZZ', 'YY', 'XX']:
                raise ValueError(
                    f'This gate combination results in twirled non-local gates which are not supported, you passed {gate_placement}, {pauli_word}')
        elif gate_placement == 'ring_third_neighbours_corner':
            gate_qubits = [[0, 5], [2, 7], [8, 3], [6, 1]]
        elif gate_placement == 'ring_third_neighbours_side':
            gate_qubits = [[1, 8], [5, 6], [7, 0], [3, 2]]
        # NOTE: the following two opitons could be simplified to double the param on the two pairs of qubits
        # leaving as is for consistency
        elif gate_placement == 'ring_fourth_neighbours_corner':
            gate_qubits = [[0, 8], [2, 6], [8, 0], [6, 2]]
            if pauli_word not in ['ZZ', 'YY', 'XX']:
                raise ValueError(
                    f'This gate combination results in twirled non-local gates which are not supported, you passed {gate_placement}, {pauli_word}')
        elif gate_placement == 'ring_fourth_neighbours_side':
            gate_qubits = [[1, 7], [5, 3], [7, 1], [3, 5]]
            if pauli_word not in ['ZZ', 'YY', 'XX']:
                raise ValueError(
                    f'This gate combination results in twirled non-local gates which are not supported, you passed {gate_placement}, {pauli_word}')
        else:
            raise ValueError(f'Incorrect gate_placement passed: \
                             {gate_placement}')

        op_list = []
        for wire_pair in gate_qubits:
            gate = qml.PauliRot(
                params[0]/4, pauli_word=pauli_word, wires=wire_pair)
            op_list.append(gate)
        return op_list
