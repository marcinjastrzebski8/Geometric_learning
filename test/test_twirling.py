from pennylane import numpy as qnp
import pennylane as qml
from src.twirling import twirl, twirl_an_ansatz, some_simple_group, C4On9QEquivGate2Local, C4On9QEquivGate1Local, c4_rep_on_qubits
from src.ansatze import HardcodedTwirledSimpleAnsatz0, SimpleAnsatz0
import pytest
import itertools

import sys
import os
# this to be able to access what's in src
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

c4_group_gates_9_qubits = c4_rep_on_qubits(3)

c4_1_local_gate_placements = ['corner', 'side', 'centre']

c4_2_local_gate_placements = ['corner_centre',
                              'side_centre',
                              'ring_neighbours_corner',
                              'ring_neighbours_side',
                              'ring_second_neighbours_corner',
                              'ring_second_neighbours_side',
                              'ring_third_neighbours_corner',
                              'ring_third_neighbours_side',
                              'ring_fourth_neighbours_corner',
                              'ring_fourth_neighbours_side']

two_qubit_pauli_words = [combo[0] + combo[1]
                         for combo in itertools.combinations_with_replacement(['X', 'Y', 'Z'], r=2)]


@qml.qnode(device=qml.device('default.qubit', wires=9))
def c4_on_9_rotate_and_ansatz_1_local(gate_placement, gate, group_gate, order_ansatz_before=True):
    # set seed for testing
    qnp.random.seed(1)
    # fake encoding - random state
    for wire in range(9):
        qml.RX(qnp.random.uniform(0, 1, 1), wires=wire)
    if order_ansatz_before:
        # single gate from equivariant pool as ansatz
        C4On9QEquivGate1Local.compute_decomposition(
            0.1, gate=gate, gate_placement=gate_placement)

        # group symmetry gate
        group_gate.decomposition()
    else:
        # group symmetry gate
        group_gate.decomposition()

        # single gate from equivariant pool as ansatz
        C4On9QEquivGate1Local.compute_decomposition(
            0.1, gate=gate, gate_placement=gate_placement)
    # statvector for comparison
    return qml.state()


@qml.qnode(device=qml.device('default.qubit', wires=9))
def c4_on_9_rotate_and_ansatz_2_local(gate_placement, pauli_word, group_gate, order_ansatz_before=True):
    # set seed for testing
    qnp.random.seed(1)
    # fake encoding - random state
    for wire in range(9):
        qml.RX(qnp.random.uniform(0, 1, 1), wires=wire)
    if order_ansatz_before:
        # single gate from equivariant pool as ansatz
        C4On9QEquivGate2Local.compute_decomposition(
            0.1, pauli_word=pauli_word, gate_placement=gate_placement)

        # group symmetry gate
        group_gate.decomposition()
    else:
        # group symmetry gate
        group_gate.decomposition()

        # single gate from equivariant pool as ansatz
        C4On9QEquivGate2Local.compute_decomposition(
            0.1, pauli_word=pauli_word, gate_placement=gate_placement)
    # statvector for comparison
    return qml.state()


def z0_twirled_w_swap_group():
    """
    Z0 twirled with {I, SWAP}
    """
    return (qnp.array([[1, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, -1]]))


def hardcoded_twirled_ansatz():
    """
    Ansatz which has been figured out manually.
    NOTE: this ansatz can be defined for any number of qubits but the simple group only
    creates an invariant subspace on two qubits.
    """
    qnp.random.seed(1)
    random_params = qnp.random.rand(1, 2)
    config = {'n_layers': 1}
    ansatz = HardcodedTwirledSimpleAnsatz0(
        random_params, wires=[0, 1], config=config)
    circuit = ansatz.compute_decomposition(
        random_params, wires=[0, 1], **config)
    # combine the four matrices into a single matrix representing the full circuit
    matrix_0 = qnp.kron(circuit[0].matrix(), circuit[1].matrix())
    matrix_1 = qnp.kron(circuit[2].matrix(), circuit[3].matrix())
    # note reversed order for matrix notation
    full_matrix = qnp.kron(qnp.matmul(matrix_1, matrix_0),
                           qml.I(wires=[2, 3]).matrix())
    return full_matrix


@pytest.mark.parametrize("manual, op, group_actions", [
    (z0_twirled_w_swap_group(), qml.Z(wires=[0]), some_simple_group(2))
])
def test_twirl_returns_correct_matrix(manual, op, group_actions):
    """
    Tests if twirl computed with the twirling module agrees with calculation by hand.
    Tests single twirl operation.
    """
    automatic = twirl(op, group_actions).matrix()
    assert qml.math.allclose(manual, automatic)


def test_twirl_an_ansatz_returns_correct_matrix():
    """
    Tests if twirl computed with the twirling module agrees with calculation by hand.
    Tests a whole ansatz.
    """
    qnp.random.seed(1)
    random_params = qnp.random.rand(1, 2)
    config = {'n_layers': 1}

    untwirled = SimpleAnsatz0(random_params, wires=[
                              0, 1], config=config).compute_decomposition(random_params, wires=[0, 1], **config)
    autotwirled = twirl_an_ansatz(untwirled, some_simple_group(), 4)
    # combine the individual gates into full matrix (note reversed for matrix notation)
    full_circuit_matrix_autotwirled = qnp.matmul(
        autotwirled[1].matrix(), autotwirled[0].matrix())

    assert qml.math.allclose(
        full_circuit_matrix_autotwirled, hardcoded_twirled_ansatz())


@pytest.mark.parametrize('gate_placement', c4_1_local_gate_placements)
@pytest.mark.parametrize('group_gate', c4_group_gates_9_qubits)
@pytest.mark.parametrize('gate', [qml.RX, qml.RY, qml.RZ])
def test_C4On9QEquivGate1Local_gates_are_equivariant(gate_placement, gate, group_gate):
    state_ansatz_before_rot = c4_on_9_rotate_and_ansatz_1_local(
        gate_placement, gate, group_gate)

    state_rot_before_ansatz = c4_on_9_rotate_and_ansatz_1_local(
        gate_placement, gate, group_gate, False)

    assert qml.math.allclose(state_ansatz_before_rot, state_rot_before_ansatz)


@pytest.mark.parametrize('gate_placement', c4_2_local_gate_placements)
@pytest.mark.parametrize('group_gate', c4_group_gates_9_qubits)
@pytest.mark.parametrize('pauli_word', two_qubit_pauli_words)
def test_C4On9QEquivGate2Local_gates_are_equivariant(gate_placement, pauli_word, group_gate):
    # these combinations are not supported (really should test for the error raised but skipping)
    if (gate_placement in ['ring_second_neighbours_corner',
                           'ring_second_neighbours_side',
                           'ring_fourth_neighbours_side',
                           'ring_fourth_neighbours_corner']) and (pauli_word not in ['XX', 'YY', 'ZZ']):
        pass
    else:
        state_ansatz_before_rot = c4_on_9_rotate_and_ansatz_2_local(
            gate_placement, pauli_word, group_gate)
        state_rot_before_ansatz = c4_on_9_rotate_and_ansatz_2_local(
            gate_placement, pauli_word, group_gate)

        assert qml.math.allclose(
            state_ansatz_before_rot, state_rot_before_ansatz)
