"""
Simple model which can be made to respect the D_4 symmetry.
"""
from src.utils import circuit_dict
from src.twirling import twirl_an_ansatz_w_op, c4_on_4_qubits_w_op, some_simple_group, twirl_an_ansatz_w_op_try_again, twirl_an_ansatz_w_op_jax
import pennylane as qml


class QuanumClassifier():

    def __init__(self, feature_map: str, ansatz: str, size: int):
        self.feature_map = circuit_dict[feature_map]
        self.ansatz = circuit_dict[ansatz]
        self.size = size
        self.device = qml.device('default.qubit', wires=self.size)

    def prediction_circuit(self, params, features, properties):
        @qml.qnode(self.device, wires=self.size)
        def qnode(params, features):
            self.feature_map(
                features, list(range(self.size)))
            self.ansatz(params, list(range(self.size)), properties)
            return qml.expval(qml.Z(0))
        return qnode(params, features)


class GeometricClassifier():
    def __init__(self, feature_map: str, ansatz: str, size: int, make_ansatz_equivariant=True):
        self.feature_map = circuit_dict[feature_map]
        self.ansatz = circuit_dict[ansatz]
        self.size = size
        self.device = qml.device('default.qubit', wires=self.size)
        self.make_ansatz_equivariant = make_ansatz_equivariant

    def prediction_circuit(self, params, features, properties):
        @qml.qnode(self.device, wires=self.size)
        def qnode(params, features):
            self.feature_map(
                features, list(range(self.size)))

            if self.make_ansatz_equivariant:
                with qml.queuing.AnnotatedQueue() as q:
                    ansatz_ops = self.ansatz.compute_decomposition(
                        *params, wires=list(range(self.size)), **properties)
                    # NOTE so far twirling only with one hardcoded group
                    # NOTE: USING SIMPLE GROUP FOR DEBUGGING
                    twirled_ansatz_ops = twirl_an_ansatz_w_op_jax(
                        ansatz_ops, some_simple_group)
                for op in twirled_ansatz_ops:
                    op.queue()
            else:
                self.ansatz(params, list(range(self.size)), properties)
            return qml.expval(qml.Z(0))
        return qnode(params, features)

class GeometricClassifierJax():
    """
    TODO: either migrate completely to jax or write functionality to choose
    """
    def __init__(self, feature_map: str, ansatz: str, size: int, make_ansatz_equivariant=True):
        self.feature_map = circuit_dict[feature_map]
        self.ansatz = circuit_dict[ansatz]
        self.size = size
        self.device = qml.device('default.qubit.jax', wires=self.size)
        self.make_ansatz_equivariant = make_ansatz_equivariant

    def prediction_circuit(self, params, features, properties):
        @qml.qnode(self.device, wires=self.size, interface = 'jax')
        def qnode(params, features):
            self.feature_map(
                features, list(range(self.size)))

            if self.make_ansatz_equivariant:
                with qml.queuing.AnnotatedQueue() as q:
                    ansatz_ops = self.ansatz.compute_decomposition(
                        *params, wires=list(range(self.size)), **properties)
                    # NOTE so far twirling only with one hardcoded group
                    # NOTE: USING SIMPLE GROUP FOR DEBUGGING
                    twirled_ansatz_ops = twirl_an_ansatz_w_op_jax(
                        ansatz_ops, some_simple_group)
                for op in twirled_ansatz_ops:
                    op.queue()
            else:
                self.ansatz(params, list(range(self.size)), properties)
            return qml.expval(qml.Z(0))
        return qnode(params, features)