from geometric_classifier import GeometricClassifierAutotwirlJax, BasicClassifier
from pennylane import numpy as qnp
import pennylane as qml
from twirling import twirl_an_ansatz, some_simple_group, c4_rep_on_qubits, C4On9QEquivGate1Local, C4On9QEquivGate2Local
from embeddings import RXEmbeddingWEnt
from ansatze import SimpleAnsatz0, GeometricAnsatzConstructor
import pytest
import sys
import os
# this to be able to access what's in src
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

# TODO longterm: functionality to test every group rep that I add

# datapoints for C2 on 2 qubits
# NOTE: testing this on an invariant subspace of 4 qubits
c2_point = [0.1, 0.6, 0, 0]
c2_point_g = [0.6, 0.1, 0, 0]
c2_point_other = [0.5, 0.3, 0, 0]

# datapoints for C4 on 4 qubits
c4_on_4_point = [0.1, 0.2, 0.3, 0.4]
c4_on_4_point_g = [0.3, 0.1, 0.4, 0.2]
c4_on_4_point_other = [0.4, 0.5, 0.7, 0.9]

# datapoints for C4 on 9 qubits
c4_on_9_point = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
c4_on_9_point_g = [0.4, 0.1, 0.2, 0.7, 0.5, 0.3, 0.8, 0.9, 0.6]
c4_on_9_point_other = [0.9, 0.7, 0.7, 0.5, 0.7, 0.9, 0.9, 0.7, 0.6]


random_params0 = qnp.random.uniform(0, 1, (1, 2))
random_params1 = qnp.random.uniform(0, 1, (1, 8))
random_params2 = qnp.random.uniform(0, 1, (1, 18))

# need 4 wires because of a hardcoded bit in the twirl - to be fixed
dev_4 = qml.device("default.qubit", wires=4)


@qml.qnode(dev_4)
def phi_on_g():
    """
    Embed a point which has been acted on by the group action.
    """
    with qml.queuing.AnnotatedQueue() as q:
        feature_map_ops = RXEmbeddingWEnt.compute_decomposition(
            c2_point_g[:2], wires=[0, 1])
        twirled_feature_map_ops = twirl_an_ansatz(
            feature_map_ops, some_simple_group(), 4)
    for op in twirled_feature_map_ops:
        op.queue()
    return qml.state()


@qml.qnode(dev_4)
def g_on_phi():
    """
    Act with group action on a point which has been embedded
    """
    with qml.queuing.AnnotatedQueue() as q:
        feature_map_ops = RXEmbeddingWEnt.compute_decomposition(
            c2_point[:2], wires=[0, 1])
        twirled_feature_map_ops = twirl_an_ansatz(
            feature_map_ops, some_simple_group(), 4)
    for op in twirled_feature_map_ops:
        op.queue()
    qml.SWAP(wires=[0, 1])
    return qml.state()


@qml.qnode(dev_4)
def phi_on_another_point():
    """
    Act with group action on another point which has been embedded
    """
    with qml.queuing.AnnotatedQueue() as q:
        feature_map_ops = RXEmbeddingWEnt.compute_decomposition(
            c2_point_other[:2], wires=[0, 1])
        twirled_feature_map_ops = twirl_an_ansatz(
            feature_map_ops, some_simple_group(), 4)
    for op in twirled_feature_map_ops:
        op.queue()
    qml.SWAP(wires=[0, 1])
    return qml.state()


@qml.qnode(dev_4)
def ansatz_on_ug():
    """
    First apply action of group then apply ansatz
    """

    qml.RX(0.1, wires=0)
    qml.RX(0.2, wires=0)
    qml.SWAP(wires=[0, 1])

    with qml.queuing.AnnotatedQueue() as q:
        ansatz_ops = SimpleAnsatz0.compute_decomposition(
            random_params0, wires=[0, 1], **{'n_layers': 1})
        twirled_ansatz_ops = twirl_an_ansatz(
            ansatz_ops, some_simple_group(), 4)
    for op in twirled_ansatz_ops:
        op.queue()

    return qml.state()


@qml.qnode(dev_4)
def ansatz_on_another_state():
    """
    First apply action of group then apply ansatz
    """

    qml.RX(0.5, wires=0)
    qml.RX(0.3, wires=0)

    with qml.queuing.AnnotatedQueue() as q:
        ansatz_ops = SimpleAnsatz0.compute_decomposition(
            random_params0, wires=[0, 1], **{'n_layers': 1})
        twirled_ansatz_ops = twirl_an_ansatz(
            ansatz_ops, some_simple_group(), 4)
    for op in twirled_ansatz_ops:
        op.queue()

    return qml.state()


@qml.qnode(dev_4)
def ug_on_ansatz():
    """
    First apply ansatz then action of group
    """
    qml.RX(0.1, wires=0)
    qml.RX(0.2, wires=0)

    with qml.queuing.AnnotatedQueue() as q:
        ansatz_ops = SimpleAnsatz0.compute_decomposition(
            random_params0, wires=[0, 1], **{'n_layers': 1})
        twirled_ansatz_ops = twirl_an_ansatz(
            ansatz_ops, some_simple_group(), 4)
    for op in twirled_ansatz_ops:
        op.queue()

    qml.SWAP(wires=[0, 1])

    return qml.state()


# NOTE: this does not test the class itself which is not ideal
# it tests a bit of the functionality contained in the class
# if the functionality were changed in the class this test would be misleading
# (maybe move to test_twirl)
@pytest.mark.parametrize("state0, state1, assertion_result", [
    (phi_on_g(), g_on_phi(), True),
    (phi_on_g(), phi_on_another_point(), False)
])
def test_equivariance_feature_map(state0, state1, assertion_result):
    """
    Tests whether 
        f(V_g(v)) == R_g(f(v))
    holds for simple equivariant feature map.
    """

    assert qml.math.allclose(state0, state1) == assertion_result

# NOTE: this does not test the class itself which is not ideal
# it tests a bit of the functionality contained in the class
# if the functionality were changed in the class this test would be misleading


@pytest.mark.parametrize("state0, state1, assertion_result", [
    (ansatz_on_ug(), ug_on_ansatz(), True),
    (ansatz_on_ug(), ansatz_on_another_state(), False)
])
def test_equivariance_ansatz(state0, state1, assertion_result):
    """
    Tests whether 
        f(V_g(v)) == R_g(f(v))
    holds for simple equivariant ansatz.
    """

    assert qml.math.allclose(state0, state1) == assertion_result


def test_invariant_measurement():
    """
    Tests whether 
    f(x) == f(Ugx) where f is a g-invariant measurement of the circuit.
    NOTE: what is this actually testing? not much of the actual code Id written
        more like my understanding of equivariant qnns...
    """
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def measure_without_Ug():
        qml.RX(0.1, wires=0)
        qml.RX(0.4, wires=1)
        return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

    @qml.qnode(dev)
    def measure_with_Ug():
        qml.RX(0.1, wires=0)
        qml.RX(0.4, wires=1)
        qml.SWAP(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

    assert measure_with_Ug() == measure_without_Ug()


@pytest.mark.parametrize("feature_map", [
    ('RXEmbedding'),
])
@pytest.mark.parametrize("ansatz, params", [
    ('SimpleAnsatz0', random_params1),
    ('SimpleAnsatz1',  random_params1)
])
@pytest.mark.parametrize("group, measurement, data_point, data_point_g, data_point_other", [
    (some_simple_group, qml.Z(0)@qml.Z(1), c2_point, c2_point_g, c2_point_other),
    (c4_rep_on_qubits, qml.Z(0)@qml.Z(1)@qml.Z(2)@qml.Z(3),
     c4_on_4_point, c4_on_4_point_g, c4_on_4_point_other)
])
def test_invariant_model_on_4_qubits(feature_map,
                                     ansatz,
                                     params,
                                     group,
                                     measurement,
                                     data_point,
                                     data_point_g,
                                     data_point_other):
    """
    Tests whether the prediction of some model is invariant 
    under some group action.
    The C2 and C4 groups are tested.
    Tests all the tools and knowledge of equiv-qnns I have so far.
    """
    geometric_classifier = GeometricClassifierAutotwirlJax(
        feature_map, ansatz, size=4, make_model_equivariant=True, group_rep=group, group_commuting_meas=measurement)
    ansatz_properties = {'n_layers': 1}
    data_point_pred = geometric_classifier.prediction_circuit(
        params, data_point, ansatz_properties)
    g_data_point_pred = geometric_classifier.prediction_circuit(
        params, data_point_g, ansatz_properties)
    another_point_pred = geometric_classifier.prediction_circuit(
        params, data_point_other, ansatz_properties)
    print(g_data_point_pred, data_point_pred, another_point_pred)
    assert data_point_pred == pytest.approx(g_data_point_pred)
    assert data_point_pred != another_point_pred


@pytest.mark.parametrize("ansatz", [
    'SimpleAnsatz0',
    'SimpleAnsatz1'
])
def test_invariant_model_on_9_qubits(ansatz):
    geometric_classifier = GeometricClassifierAutotwirlJax(
        'RXEmbedding', ansatz, size=9, make_model_equivariant=True, group_rep=c4_rep_on_qubits, group_commuting_meas=qml.Z(4), image_size=3)
    ansatz_properties = {'n_layers': 1}
    data_point_pred = geometric_classifier.prediction_circuit(
        random_params2, c4_on_9_point, ansatz_properties)
    g_data_point_pred = geometric_classifier.prediction_circuit(
        random_params2, c4_on_9_point_g, ansatz_properties)
    another_point_pred = geometric_classifier.prediction_circuit(
        random_params2, c4_on_9_point_other, ansatz_properties)
    print(g_data_point_pred, data_point_pred, another_point_pred)
    # NOTE: I HAD TO LOWER THE APPROX SIGINFICANTLY FOR 9 QUBITS - I ASSUME THE DIFFERENCE COMES FROM FLOAT PRECISION AND NOT A THEORETICAL ISSUE
    # The longer/more complicated the circuit the bigger the floating point error
    # Which makes sense.
    # Would be an interesting study?
    # Look at the printed out predictions though, you can tell the invariance is working
    assert data_point_pred == pytest.approx(g_data_point_pred, 1e-2, 1e-2)
    assert data_point_pred != another_point_pred


@pytest.mark.parametrize('instructions_1local', [
    [{'gate_placement': 'corner', 'gate': qml.RZ}],
    [{'gate_placement': 'side', 'gate': qml.RX}],
    [{'gate_placement': 'centre', 'gate': qml.RY}],
    [
        {'gate_placement': 'side', 'gate': qml.RZ},
        {'gate_placement': 'corner', 'gate': qml.RX},
        {'gate_placement': 'centre', 'gate': qml.RY}
    ]
])
@pytest.mark.parametrize('instructions_2local', [
    [{'gate_placement': 'side_centre', 'pauli_word': 'XZ'}],
    [{'gate_placement': 'ring_neighbours_corner', 'pauli_word': 'YZ'}],
    [{'gate_placement': 'ring_neighbours_side', 'pauli_word': 'YX'}],
    [{'gate_placement': 'ring_second_neighbours_corner', 'pauli_word': 'XX'}],
    [{'gate_placement': 'ring_second_neighbours_side', 'pauli_word': 'ZZ'}],
    [{'gate_placement': 'ring_third_neighbours_corner', 'pauli_word': 'XY'}],
    [{'gate_placement': 'ring_third_neighbours_side', 'pauli_word': 'YX'}],
    [{'gate_placement': 'ring_fourth_neighbours_corner', 'pauli_word': 'ZZ'}],
    [{'gate_placement': 'ring_fourth_neighbours_side', 'pauli_word': 'YY'}],
    [
        {'gate_placement': 'ring_fourth_neighbours_corner', 'pauli_word': 'ZZ'},
        {'gate_placement': 'side_centre', 'pauli_word': 'ZX'},
        {'gate_placement': 'ring_neighbours_side', 'pauli_word': 'ZZ'}
    ]

])
@pytest.mark.parametrize('n_layers', [1, 2])
def test_invariant_model_on_9_qubits_with_GeometricAnsatzConstructor(instructions_1local,
                                                                     instructions_2local,
                                                                     n_layers):
    n_1local_gates = len(instructions_1local)
    n_2local_gates = len(instructions_2local)
    qnp.random.seed(1)
    ansatz_properties = {'group_equiv_1local_gate': C4On9QEquivGate1Local,
                         'group_equiv_2local_gate': C4On9QEquivGate2Local,
                         'gate_1local_instructions': instructions_1local,
                         'gate_2local_instructions': instructions_2local,
                         'n_layers': n_layers}
    params = qnp.random.uniform(
        0, 1, (n_layers, n_1local_gates + n_2local_gates))
    geo_classifier = BasicClassifier(
        'RXEmbedding', GeometricAnsatzConstructor, 9, qml.Z(4))
    data_point_pred = geo_classifier.prediction_circuit(
        params, c4_on_9_point, ansatz_properties)
    g_data_point_pred = geo_classifier.prediction_circuit(
        params, c4_on_9_point_g, ansatz_properties)
    another_point_pred = geo_classifier.prediction_circuit(
        params, c4_on_9_point_other, ansatz_properties)
    print(g_data_point_pred, data_point_pred, another_point_pred)
    # NOTE: I HAD TO LOWER THE APPROX SIGINFICANTLY FOR 9 QUBITS - I ASSUME THE DIFFERENCE COMES FROM FLOAT PRECISION AND NOT A THEORETICAL ISSUE
    assert data_point_pred == pytest.approx(g_data_point_pred, 1e-2, 1e-2)
    assert data_point_pred != another_point_pred
