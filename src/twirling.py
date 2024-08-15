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

import jax

#TODO: this maybe made into a class... not sure yet
rot_90 = ((0, 1), (0, 3), (0, 2))
rot_180 = ((0, 3), (1, 2))
rot_270 = ((0, 2), (0, 3), (0, 1))

#would be nice to generalise this
def c4_on_4_qubits():
    return [qml.I(wires=[0, 1, 2, 3])] + [qml.prod(*[qml.SWAP(wires=indexes)
                                                                for indexes in group_element]) for group_element in (rot_90, rot_180, rot_270)]
def some_simple_group(n_wires = 4):
    return [qml.I(wires = range(n_wires)), qml.SWAP(wires = [0,1])]

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


def twirl_an_ansatz(ansatz, group_actions):
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

        # note wires is hardcoded for 4 qubits
        twirled_gate = qml.QubitUnitary(
            twirled_gate_matrix, wires=[0, 1, 2, 3])

        op_list.append(twirled_gate)
    return op_list