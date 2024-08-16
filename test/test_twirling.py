import pytest

import sys
import os
#this to be able to access what's in src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ansatze import HardcodedTwirledSimpleAnsatz0, SimpleAnsatz0
from twirling import twirl, twirl_an_ansatz, some_simple_group

import pennylane as qml
from pennylane import numpy as qnp

def z0_twirled_w_swap_group():
    """
    Z0 twirled with {I, SWAP}
    """
    return(qnp.array([[1,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,-1]]))

def hardcoded_twirled_ansatz():
    """
    Ansatz which has been figured out manually.
    NOTE: this ansatz can be defined for any number of qubits but the simple group only
    creates an invariant subspace on two qubits.
    """
    qnp.random.seed(1)
    random_params = qnp.random.rand(1,2)
    config = {'layers':1}
    ansatz = HardcodedTwirledSimpleAnsatz0(random_params, wires=[0,1], config=config)
    circuit = ansatz.compute_decomposition(random_params, wires=[0,1], **config)
    #combine the four matrices into a single matrix representing the full circuit
    matrix_0 = qnp.kron(circuit[0].matrix(), circuit[1].matrix())
    matrix_1 = qnp.kron(circuit[2].matrix(), circuit[3].matrix())
    #note reversed order for matrix notation
    full_matrix = qnp.kron(qnp.matmul(matrix_1, matrix_0), qml.I(wires=[2,3]).matrix())
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
    assert qml.math.allclose(manual,automatic)

def test_twirl_an_ansatz_returns_correct_matrix():
    """
    Tests if twirl computed with the twirling module agrees with calculation by hand.
    Tests a whole ansatz.
    """
    qnp.random.seed(1)
    random_params = qnp.random.rand(1,2)
    config = {'layers':1}

    untwirled = SimpleAnsatz0(random_params, wires=[0,1], config=config).compute_decomposition(random_params,wires=[0,1], **config)
    autotwirled = twirl_an_ansatz(untwirled, some_simple_group())
    #combine the individual gates into full matrix (note reversed for matrix notation)
    full_circuit_matrix_autotwirled = qnp.matmul(autotwirled[1].matrix(), autotwirled[0].matrix())

    assert qml.math.allclose(full_circuit_matrix_autotwirled, hardcoded_twirled_ansatz())
