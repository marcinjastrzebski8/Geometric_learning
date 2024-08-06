"""
Collection of functions related to twirling of ansatze.
Probably won't be very general at first. Maybe never.
Goal is to be able to have
- at first a working twirled version of a simple ansatz
- later a working twirled version of a more useful ansatz

Twirling will be wrt to D4 [prolly C4 at first] on nxn images represented on n*n qubits
with n=2 at first.

TODO: SEE IF CAN REWRITE OMITTING MATRICES, WORK ONLY WITH QML.OP OBJECTS
I THINK I NEED TO SWITCH TO WORKING WITH OPERATORS, LINALG EXP FUNCTION BREAKS WITH
"""

import pennylane as qml
from typing import Sequence
from dataclasses import dataclass
import numpy as np
import scipy
from jax import numpy as jnp
import scipy.linalg
import jax

# TODO: hopefully retire this


def swaps_as_matrices(index_pairs: Sequence[Sequence]):
    """
    Assumes 4 qubits.
    Returns a matrix representing a sequence of swaps.

    index_pairs should be gate-ordered, not matrix-ordered, meaning what's on the left comes first
    """
    matrix = np.eye(2**4)
    for index_pair in index_pairs[::-1]:
        matrix @= swap_on_4_qubits(index_pair)
    return matrix

# TODO: this I think with Op objects?


def swap_on_4_qubits(qubit_ids: Sequence):
    """
    Defines swap operators (between 2 qubits) in 4-qubit hilbert space.
    This could be generalised but seems like a lot of work.

    NOTE:Maybe this would be easier done in qiskit?
    """
    two_qubit_swap = qml.SWAP(wires=[0, 1]).matrix()
    iden = np.eye(2)
    swap_01 = qml.math.kron(qml.math.kron(two_qubit_swap, iden), iden)
    swap_12 = qml.math.kron(qml.math.kron(iden, two_qubit_swap), iden)
    swap_23 = qml.math.kron(qml.math.kron(iden, iden), two_qubit_swap)
    if set(qubit_ids) == {0, 1}:
        swap_matrix = swap_01
    elif set(qubit_ids) == {1, 2}:
        swap_matrix = swap_12
    elif set(qubit_ids) == {2, 3}:
        swap_matrix = swap_23
    elif set(qubit_ids) == {0, 2}:
        swap_matrix = swap_01@swap_12@swap_01
    elif set(qubit_ids) == {0, 3}:
        swap_matrix = swap_01@swap_12@swap_23@swap_12@swap_01
    elif set(qubit_ids) == {1, 3}:
        swap_matrix = swap_12@swap_23@swap_12
    return swap_matrix

# TODO: this hopefully to be retired


def put_generator_into_4_qubit_space(generator, qubit_ids):
    """
    A very non-general function, I want to find a better way of handling the dimensionality of the qml matrices.
    NOTE: im assuming here the generator.ops is a 1-element list. This might break [WHEN?].
    Same assumption for elements of the ops[0] list.

    Looking at it now it might be easy to generalise actually?
    """
    operator_in_correct_space = [qml.I(wires=[i]) for i in range(4)]
    print(generator, qubit_ids)
    if len(qubit_ids) > 1:
        # assuming ops has 1 element which is a product of ops
        generator_ops = generator.ops[0].overlapping_ops
        for generator_op, qubit_id in zip(generator_ops, qubit_ids):
            # assuming generator_op has 1 element
            operator_in_correct_space[qubit_id] = generator_op[0]
    else:
        # assuming ops has 1 element which is not a product
        generator_op = generator.ops[0]
        operator_in_correct_space[qubit_ids[0]] = generator_op

    print(operator_in_correct_space)
    return qml.prod(*operator_in_correct_space)


rot_90 = ((0, 1), (0, 3), (0, 2))
rot_180 = ((0, 3), (1, 2))
rot_270 = ((0, 2), (0, 3), (0, 1))

c4_on_4_qubits = [
    qml.I(wires=[0, 1, 2, 3]).matrix()] + [swaps_as_matrices(indexes) for indexes in (rot_90, rot_180, rot_270)]


def twirl(generator, group_actions):
    """
    Twirl the generator of a qnn's layer with a given group.
    The result is a new generator which can form an equivariant network [wrt to the group].
    Works for discrete groups only.
    GROUP ACTIONS AND GENERATOR NEED TO BE MATRIX REPRESENTAIONS OF THE OPERATORS
    """
    group_size = len(group_actions)
    new_generator = np.sum(
        [group_action@generator for group_action in group_actions], axis=0)/group_size
    return new_generator


def twirl_an_ansatz(ansatz, group_actions, wires):
    """
    Take an ansatz and twirl each of it's gates, return twirled ansatz.
    Returns list of operations.
    """
    op_list = []
    for gate in ansatz:
        if gate.has_generator:
            generator_correct_dimensions = put_generator_into_4_qubit_space(
                gate.generator(), list(gate.wires)).matrix()
            twirled_generator = twirl(
                generator_correct_dimensions, group_actions)
            print(gate.parameters)
            # TODO: I think this is not a smart [or even possible way]
            # to define this new operator.
            # I should be using the native Op class and its method.
            # You can take exponents
            unitary = scipy.linalg.expm(-1j *
                                        twirled_generator*gate.parameters[0])
            twirled_gate = qml.QubitUnitary(unitary, wires=wires)
        else:
            twirled_gate = twirl(gate, group_actions)
        op_list.append(twirled_gate)
    return op_list


def twirl_w_op(operator, group_actions):
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


def twirl_an_ansatz_w_op(ansatz, group_actions, try_with_qubitunitary=True):
    """
    Returns list of operations [ansatz, twirled]
    NOTE: TRYING TO DO IT USING QUBITUNITARY
    """
    op_list = []
    for gate in ansatz:
        if gate.has_generator:
            parameter_type = type(gate.parameters[0])
            # i reckon this line explicitly kills trainability[?]
            if parameter_type == jnp.tensor:
                param = np.array(gate.parameters[0])
            else:
                param = np.array(
                    gate.parameters[0]._value)
            twirled_generator = twirl_w_op(gate.generator(), group_actions)
            # NOTE: this not differentiable
            twirled_gate_op = qml.exp(
                op=twirled_generator, coeff=1j*param)
        else:
            twirled_gate_op = twirl_w_op(gate, group_actions)
        if try_with_qubitunitary:
            # note wires is hardcoded for 4 qubits
            twirled_gate = qml.QubitUnitary(
                twirled_gate_op.matrix(), wires=[0, 1, 2, 3])
        else:
            twirled_gate = twirled_gate_op
        op_list.append(twirled_gate)
    return op_list


def twirl_an_ansatz_w_op_try_again(ansatz, group_actions, params):
    """
    Returns list of operations [ansatz, twirled]
    NOTE: TRYING TO DO IT USING QUBITUNITARY
    NOTE: commenting out to avoid import errors
    op_list = []
    gate_is_paramed_list = [gate.has_generator for gate in ansatz]
    for gate in ansatz:
        if gate.has_generator:
            parameter_type = type(gate.parameters[0])
            # i reckon this line explicitly kills trainability[?]
            if parameter_type == qnp.tensor:
                param = qnp.array(gate.parameters[0], requires_grad=True)
            else:
                param = qnp.array(
                    gate.parameters[0]._value, requires_grad=True)
            twirled_generator = twirl_w_op(gate.generator(), group_actions)
            twirled_gate_matrix = scipy.linalg.expm(
                1.0j*param*twirled_generator.matrix())
        else:
            twirled_gate_matrix = twirl_w_op(gate, group_actions).matrix()

        # note wires is hardcoded for 4 qubits
        twirled_gate = qml.QubitUnitary(
            twirled_gate_matrix, wires=[0, 1, 2, 3])

        op_list.append(twirled_gate)
    return op_list
    """

c4_on_4_qubits_w_op = [qml.I(wires=[0, 1, 2, 3])] + [qml.prod(*[qml.SWAP(wires=indexes)
                                                                for indexes in group_element]) for group_element in (rot_90, rot_180, rot_270)]
some_simple_group = [qml.I(wires=[0, 1, 2, 3])] + [qml.SWAP(wires=[0, 1])]


def twirl_w_op_trainable(gate, param, group_actions):
    """
    NOTE: UNFINISHED
    NOTE: commenting out to avoid import errors
    Trying to see if backprop will happen if the trainable param is being explicitly passed to the function used.
    
    # twirl the generator
    twirled_generator = twirl_w_op(gate.generator(), group_actions)
    parameter_type = type(param)
    if parameter_type == qnp.tensor:
        param = qnp.array(param, requires_grad=True)
    else:
        param = qnp.array(
            param, requires_grad=True)
    twirled_gate_matrix = scipy.linalg.expm(1.0j *
                                            param*twirled_generator.matrix())
    """
def twirl_an_ansatz_w_op_jax(ansatz, group_actions):
    """
    Returns list of operations [ansatz, twirled]
    NOTE: TRYING TO DO IT USING QUBITUNITARY
    """
    op_list = []
    for gate in ansatz:
        if gate.has_generator:
            twirled_generator = twirl_w_op(gate.generator(), group_actions)
            twirled_gate_matrix = jax.scipy.linalg.expm(
                1.0j*gate.parameters[0]*twirled_generator.matrix())
        else:
            twirled_gate_matrix = twirl_w_op(gate, group_actions).matrix()

        # note wires is hardcoded for 4 qubits
        twirled_gate = qml.QubitUnitary(
            twirled_gate_matrix, wires=[0, 1, 2, 3])

        op_list.append(twirled_gate)
    return op_list