"""
Simple model which can be made to respect the D_4 symmetry.
"""
from src.utils import circuit_dict
from src.twirling import twirl_an_ansatz
import pennylane as qml
from typing import Sequence


# NOTE: when the Operation class is called via its init, it uses compute_decomposition implicitly
class GeometricClassifierAutotwirlJax():
    """
    Generic classifier which uses jax. Has the option to twirl the ansatz.
    Note that tiwrled ansatz isn't enough for the whole model to be equivariant/invariant.

    This version of the classifier uses my first attempt at twirling which is very general
    (twirl any ansatz pass) but the implementation is flawed - it computes the twirl every single forward pass
    (and i guess every backwards pass). This makes it already unusable for models with 9 qubits (e.g. representing 3x3 images).
    """

    def __init__(self, feature_map: str, ansatz: str, size: int, make_model_equivariant=True, group_rep=None, group_commuting_meas=None, **group_args):
        self.feature_map = circuit_dict[feature_map]
        self.ansatz = circuit_dict[ansatz]
        self.size = size
        self.device = qml.device('default.qubit.jax', wires=self.size)
        self.make_model_equivariant = make_model_equivariant
        self.group_commuting_meas = group_commuting_meas
        try:
            self.group_rep = group_rep(**group_args)
        except TypeError as exc:
            raise TypeError(
                'group rep needs to not be None if make_model_equivariant is True') from exc

    def prediction_circuit(self, params, features, properties):
        @qml.qnode(self.device, interface='jax')
        def qnode(params, features):
            self.feature_map(features, list(range(self.size)), properties)
            if self.make_model_equivariant:
                # twirl the ansatz
                with qml.queuing.AnnotatedQueue() as q:
                    ansatz_ops = self.ansatz.compute_decomposition(
                        *[params], wires=list(range(self.size)), **properties)
                    twirled_ansatz_ops = twirl_an_ansatz(
                        ansatz_ops, self.group_rep, self.size)
                for op in twirled_ansatz_ops:
                    op.queue()
            else:
                self.ansatz(params, list(range(self.size)), properties)
            meas_op = qml.Z(0)
            if self.group_commuting_meas is not None:
                meas_op = self.group_commuting_meas
            return qml.expval(meas_op)
        return qnode(params, features)


class BasicClassifier():
    """
    Classifier with a basic
    |0> -/- feature_map -/- ansatz -/- measurement
    structure.
    """

    def __init__(self, feature_map: str, ansatz, size: int, measurement=None, interface='jax'):
        self.feature_map = circuit_dict[feature_map]
        self.ansatz = ansatz
        self.size = size
        self.device = qml.device('default.qubit.jax', wires=self.size)
        self.measurement = measurement
        self.interface = interface

    def prediction_circuit(self, params, features, properties):

        @qml.qnode(self.device, interface=self.interface)
        def qnode(params, features):
            # feature map
            self.feature_map(features, list(range(self.size)), properties)

            # ansatz
            self.ansatz(params, list(range(self.size)), properties)

            return qml.expval(self.measurement)
        return qnode(params, features)


class BasicClassifierTorch():
    """
    Same as BasicClassifier but few minor changes to be compatible with pytorch/pennylane interface.
    """

    def __init__(self, feature_map: str, ansatz, size: int, measurement=qml.PauliZ(0)):
        self.feature_map = circuit_dict[feature_map]
        self.ansatz = ansatz
        self.size = size
        # device changed from jax
        self.device = qml.device('default.qubit', wires=self.size)
        self.measurement = measurement

    def prediction_circuit(self, properties):

        @qml.qnode(self.device)
        def qnode(inputs, params):
            """
            first parameter to qnode needs to be called 'inputs' for compatibility with torch
            """
            # feature map
            self.feature_map(inputs, list(range(self.size)), properties)

            # ansatz
            self.ansatz(params, list(range(self.size)), properties)
            return qml.expval(self.measurement)
        # output changed to just the function, not the function call
        return qnode
