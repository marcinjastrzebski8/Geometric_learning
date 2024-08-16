import pytest
import sys
import os
#this to be able to access what's in src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ansatze import SimpleAnsatz0
from embeddings import RXEmbeddingWEnt
from twirling import twirl_an_ansatz, some_simple_group, c4_on_4_qubits
import pennylane as qml
from pennylane import numpy as qnp
from geometric_classifier import GeometricClassifierJax

#TODO longterm: functionality to test every group rep that I add

#datapoints for C2 on 2 qubits
c2_point = [0.1,0.6,0,0]
c2_point_g = [0.6,0.1,0,0]
c2_point_other = [0.5,0.3,0,0]

#datapoints for C4 on 4 qubits
c4_point = [0.1,0.2,0.3,0.4]
c4_point_g = [0.3, 0.1, 0.4, 0.2]
c4_point_other = [0.4,0.1,0.2,0.3]


random_params0 = qnp.random.uniform(0,1,(1,2))
random_params1 = qnp.random.uniform(0,1,(1,4))

#need 4 wires because of a hardcoded bit in the twirl - to be fixed
dev = qml.device("default.qubit", wires = 4)

@qml.qnode(dev)
def phi_on_g():
    """
    Embed a point which has been acted on by the group action.
    """
    with qml.queuing.AnnotatedQueue() as q:
        feature_map_ops = RXEmbeddingWEnt.compute_decomposition(c2_point_g[:2], wires = [0,1])
        twirled_feature_map_ops = twirl_an_ansatz(feature_map_ops, some_simple_group())
    for op in twirled_feature_map_ops:
        op.queue()
    return qml.state()

@qml.qnode(dev)
def g_on_phi():
    """
    Act with group action on a point which has been embedded
    """
    with qml.queuing.AnnotatedQueue() as q:
        feature_map_ops = RXEmbeddingWEnt.compute_decomposition(c2_point[:2], wires = [0,1])
        twirled_feature_map_ops = twirl_an_ansatz(feature_map_ops, some_simple_group())
    for op in twirled_feature_map_ops:
        op.queue()
    qml.SWAP(wires = [0,1])
    return qml.state()

@qml.qnode(dev)
def phi_on_another_point():
    """
    Act with group action on another point which has been embedded
    """
    with qml.queuing.AnnotatedQueue() as q:
        feature_map_ops = RXEmbeddingWEnt.compute_decomposition(c2_point_other[:2], wires = [0,1])
        twirled_feature_map_ops = twirl_an_ansatz(feature_map_ops, some_simple_group())
    for op in twirled_feature_map_ops:
        op.queue()
    qml.SWAP(wires = [0,1])
    return qml.state()


@qml.qnode(dev)
def ansatz_on_ug():
    """
    First apply action of group then apply ansatz
    """
    
    qml.RX(0.1, wires = 0)
    qml.RX(0.2, wires = 0)
    qml.SWAP(wires = [0,1])

    with qml.queuing.AnnotatedQueue() as q:
        ansatz_ops = SimpleAnsatz0.compute_decomposition(
            random_params0, wires = [0,1], **{'layers':1})
        twirled_ansatz_ops = twirl_an_ansatz(
            ansatz_ops, some_simple_group())
    for op in twirled_ansatz_ops:
        op.queue()

    return qml.state()

@qml.qnode(dev)
def ansatz_on_another_state():
    """
    First apply action of group then apply ansatz
    """
    
    qml.RX(0.5, wires = 0)
    qml.RX(0.3, wires = 0)

    with qml.queuing.AnnotatedQueue() as q:
        ansatz_ops = SimpleAnsatz0.compute_decomposition(
            random_params0, wires = [0,1], **{'layers':1})
        twirled_ansatz_ops = twirl_an_ansatz(
            ansatz_ops, some_simple_group())
    for op in twirled_ansatz_ops:
        op.queue()

    return qml.state()

@qml.qnode(dev)
def ug_on_ansatz():
    """
    First apply ansatz then action of group
    """
    qml.RX(0.1, wires = 0)
    qml.RX(0.2, wires = 0)

    with qml.queuing.AnnotatedQueue() as q:
        ansatz_ops = SimpleAnsatz0.compute_decomposition(
            random_params0, wires = [0,1], **{'layers':1})
        twirled_ansatz_ops = twirl_an_ansatz(
            ansatz_ops, some_simple_group())
    for op in twirled_ansatz_ops:
        op.queue()

    qml.SWAP(wires = [0,1])

    return qml.state()


#NOTE: this does not test the class itself which is not ideal
#it tests a bit of the functionality contained in the class
#if the functionality were changed in the class this test would be misleading
#(maybe move to test_twirl)
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

#NOTE: this does not test the class itself which is not ideal
#it tests a bit of the functionality contained in the class
#if the functionality were changed in the class this test would be misleading
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
    dev = qml.device("default.qubit", wires = 4)

    @qml.qnode(dev)
    def measure_without_Ug():
        qml.RX(0.1, wires = 0)
        qml.RX(0.4, wires = 1)
        return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

    @qml.qnode(dev)
    def measure_with_Ug():
        qml.RX(0.1, wires = 0)
        qml.RX(0.4, wires = 1)
        qml.SWAP(wires = [0,1])
        return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

    assert measure_with_Ug() == measure_without_Ug()


#TODO: understand why these tests pass even without a commuting measurement operator
@pytest.mark.parametrize("feature_map", [
    ('RXEmbedding'), ('RXEmbeddingWEnt')
])
@pytest.mark.parametrize("ansatz, params", [
    ('SimpleAnsatz0', qnp.concatenate([random_params0,qnp.array([[0,0]])],axis=1)),
    ('SimpleAnsatz1',  qnp.concatenate([random_params1,qnp.array([[0,0,0,0]])],axis=1))
])
@pytest.mark.parametrize("group, data_point, data_point_g, data_point_other", [
    (some_simple_group, c2_point, c2_point_g, c2_point_other),
    (c4_on_4_qubits, c4_point, c4_point_g, c4_point_other)
])
def test_invariant_model(feature_map, 
                         ansatz, 
                         params, 
                         group, 
                         data_point,
                         data_point_g,
                         data_point_other):
    """
    Tests whether the prediction of some model is invariant 
    under some group action.
    Tests all the tools and knowledge of equiv-qnns I have so far.
    """
    geometric_classifier = GeometricClassifierJax(feature_map, ansatz, size = 4, make_model_equivariant=True, group_rep = group)
    ansatz_properties = {'layers':1}
    data_point_pred = geometric_classifier.prediction_circuit(params, data_point, ansatz_properties)
    g_data_point_pred = geometric_classifier.prediction_circuit(params, data_point_g, ansatz_properties)
    another_point_pred = geometric_classifier.prediction_circuit(params, data_point_other, ansatz_properties)
    print(g_data_point_pred, data_point_pred, another_point_pred)
    assert data_point_pred == pytest.approx(g_data_point_pred)
    assert data_point_pred !=another_point_pred
