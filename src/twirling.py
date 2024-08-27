"""
Collection of functions related to twirling of ansatze.
Probably won't be very general at first. Maybe never.
Goal is to be able to have
- at first a working twirled version of a simple ansatz
- later a working twirled version of a more useful ansatz

Twirling will be wrt to D4 [prolly C4 at first] on nxn images represented on n*n qubits
with n=2 at first.
"""

import pennylane as qml

import jax

#NOTE: would be nice to generalise this
def c4_rep_on_qubits(image_size = 2):
    """
    Representation of the c4 group on qubits where the qubits represent pixels of an image.
    """
    #NOTE: rotation happens clockwise 
    #NOTE: order of operations follows matrix notation
    #   qml.prod also follows matrix notation - as expected
    if image_size == 2:
        rot_90 = ((0, 2), (0, 3), (0, 1))
        rot_180 = ((0, 3), (1, 2))
        rot_270 = ((0, 1), (0, 3), (0, 2))
    elif image_size == 3:
        rot_90 = ((1,3),(1,7),(1,5),(0,6),(0,8),(0,2))
        rot_180 = ((5,3),(1,7),(2,6),(0,8))
        rot_270 = ((5,7),(5,3),(1,5),(2,8),(2,6),(0,2))
    else:
        raise NotImplementedError('Image sizes other than 2, 3 are not implemented yet.')
    return [qml.I(wires=range(int(image_size*image_size)))] + [qml.prod(*[qml.SWAP(wires=indexes)
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