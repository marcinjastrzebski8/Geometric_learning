"""
Simple model which can be made to respect the D_4 symmetry.
"""
from src.utils import circuit_dict
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
