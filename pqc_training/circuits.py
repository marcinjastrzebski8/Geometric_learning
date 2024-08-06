import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane import numpy as qnp
from pennylane import broadcast
import numpy as np


def rx_embedding(features, input_size):
    for i, wire in enumerate(input_size):
        qml.RX(features[i], wires=wire)


def full_rx_embedding(features, start, end):
    count = 0
    for i in range(start, end):
        qml.RX(features[count], wires=i)
        count += 1


def particle_2_embedding(features, input_size):
    for i, idx_q in zip(range(0, 2 * input_size, 2), range(input_size)):
        qml.Rot(qnp.pi / 2, features[i], features[i + 1], wires=idx_q)
    qml.broadcast(qml.CNOT, wires=range(input_size), pattern="chain")
    for i, idx_q in zip(range(0, 2 * input_size, 2), range(input_size)):
        qml.Rot(features[i], features[i + 1], 0, wires=idx_q)


def particle_2_embedding(features, input_size):
    for i, wire in enumerate(input_size):
        qml.Rot(qnp.pi / 2, features[2 * i], features[(2 * i) + 1], wires=wire)
    qml.broadcast(qml.CNOT, wires=input_size, pattern="chain")
    for i, wire in enumerate(input_size):
        qml.Rot(features[2 * i], features[(2 * i) + 1], 0, wires=wire)


def full_particle_2_embedding(features, start, end):
    n = end - start
    for i, idx_q in zip(range(0, 2 * n, 2), range(start, end)):
        qml.Rot(qnp.pi / 2, features[i], features[i + 1], wires=idx_q)
    qml.broadcast(qml.CNOT, wires=range(start, end), pattern="chain")
    for i, idx_q in zip(range(0, 2 * n, 2), range(start, end)):
        qml.Rot(features[i], features[i + 1], 0, wires=idx_q)


########################################


class Ansatz1(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {"layers": config["layers"]}

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers):
        op_list = []
        wires = qml.wires.Wires(wires)
        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(qml.RY(weights[l, i], wires=wires[i]))

        return op_list


class Ansatz2(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {"layers": config["layers"]}

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers):
        op_list = []
        wires = qml.wires.Wires(wires)
        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(qml.RY(weights[l, i], wires=wires[i]))
            for j in range(len(wires) - 1):
                op_list.append(qml.CNOT(wires=[wires[j], wires[j + 1]]))

        return op_list


class Ansatz2Reversed(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {"layers": config["layers"]}

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers):
        op_list = []
        wires = qml.wires.Wires(wires)
        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(qml.RY(weights[l, i], wires=wires[i]))
            for j in range(len(wires) - 1):
                op_list.append(qml.CNOT(wires=[wires[j + 1], wires[j]]))

        return op_list


class AnsatzRing(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {"layers": config["layers"]}

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers):
        op_list = []
        wires = qml.wires.Wires(wires)
        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(qml.RY(weights[l, i], wires=wires[i]))

            for j in range(len(wires)):
                if j == len(wires) - 1:
                    op_list.append(qml.CNOT(wires=[wires[j], wires[0]]))
                else:
                    op_list.append(qml.CNOT(wires=[wires[j], wires[j + 1]]))

        return op_list


class AnsatzRingReversed(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {"layers": config["layers"]}

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers):
        op_list = []
        wires = qml.wires.Wires(wires)
        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(qml.RY(weights[l, i], wires=wires[i]))

            for j in range(len(wires)):
                if j == len(wires) - 1:
                    op_list.append(qml.CNOT(wires=[wires[0], wires[j]]))
                else:
                    op_list.append(qml.CNOT(wires=[wires[j + 1], wires[j]]))

        return op_list


class AnsatzSkip(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {"layers": config["layers"]}

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers):
        op_list = []
        wires = qml.wires.Wires(wires)
        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(qml.RY(weights[l, i], wires=wires[i]))
            for j in range(len(wires) - 2):
                op_list.append(qml.CNOT(wires=[wires[j], wires[j + 2]]))

        return op_list


class AnsatzSkipReversed(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {"layers": config["layers"]}

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers):
        op_list = []
        wires = qml.wires.Wires(wires)
        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(qml.RY(weights[l, i], wires=wires[i]))
            for j in range(len(wires) - 2):
                op_list.append(qml.CNOT(wires=[wires[j + 2], wires[j]]))

        return op_list


class Ansatz3(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires=None, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(weights, wires):
        op_list = []
        layers = weights.shape[0]
        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(qml.RY(weights[l, i, 0], wires=i))
                op_list.append(qml.RZ(weights[l, i, 1], wires=i))
            for i in range(len(wires) - 1):
                op_list.append(qml.CNOT(wires=[i, i + 1]))

        return op_list


class ExpressAnsatz(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return 2 * layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires):
        op_list = []
        layers = weights.shape[0]
        wires = qml.wires.Wires(wires)
        operation_list = [qml.RX, qml.RY, qml.RZ]

        for i in range(len(wires)):
            for l in range(int(layers / 2) - 1, -1, -1):
                idx = l % 3
                op = operation_list[idx]
                op_list.append(op(weights[l, i], wires=wires[i]))

        for i in range(len(wires) - 1):
            op_list.append(qml.CNOT(wires=[wires[i], wires[i + 1]]))

        for i in range(len(wires)):
            for l in range(int(layers / 2)):
                idx = l % 3
                op = operation_list[idx]
                op_list.append(op(weights[l, i], wires=wires[i]))

        return op_list


class EntanglingAnsatz(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "cnot_count": config["cnot_count"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires):
        return n_wires, 3

    @staticmethod
    def compute_decomposition(weights, wires, cnot_count):
        op_list = []

        wires = qml.wires.Wires(wires)
        operation_list = [qml.RX, qml.RY, qml.RZ]
        counter = 0

        for j in range(3):
            for i in range(len(wires)):
                op = operation_list[j]
                op_list.append(op(weights[i, j], wires=[i]))
            for k in range(len(wires) - 1):
                if counter < cnot_count:
                    qml.CNOT(wires=[wires[k], wires[k + 1]])
                    counter += 1
                else:
                    break


class OriginalAnsatz(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None, id=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
        }

        super().__init__(weights, wires, id=id)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(qml.RY(weights[l, i], wires=wires[i]))

            for j in range(len(wires)):
                for k in range(j + 1, len(wires)):
                    op_list.append(qml.CNOT(wires=[wires[j], wires[k]]))
        return op_list


class OriginalAnsatzReversed(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(qml.RY(weights[l, i], wires=wires[i]))

            for j in range(len(wires)):
                for k in range(j + 1, len(wires)):
                    op_list.append(qml.CNOT(wires=[wires[k], wires[j]]))

        return op_list


class StructureAnsatz(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, latent_size, trash_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(qml.RY(weights[l, i], wires=wires[i]))

            for j in range(trash_size):
                for k in range(latent_size):
                    op_list.append(qml.CNOT(wires=[wires[j + latent_size], wires[k]]))
        return op_list


class FancyAnsatz(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, latent_size, trash_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for i in range(latent_size + trash_size - 1, latent_size - 1, -1):
                trash_control_qubit = [i]
                for j in range(latent_size):
                    control_qubits = trash_control_qubit + list(
                        range(j + 1, latent_size)
                    )
                    target_qubit = [j]
                    wire_set = control_qubits + target_qubit
                    op_list.append(qml.MultiControlledX(wires=wire_set))

            counter = 0
            for k in range(latent_size + trash_size - 1, latent_size - 1, -1):
                for c in range(latent_size):
                    p_idx = (counter * latent_size) + c
                    op_list.append(
                        qml.CRot(
                            weights[p_idx, 0],
                            weights[p_idx, 1],
                            weights[p_idx, 2],
                            wires=[c, k],
                        )
                    )
                counter += 1
        return op_list


class EncoderAnsatz(Operation):
    # this has not been constructed yet
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, latent_size, trash_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(qml.RY(weights[l, i], wires=wires[i]))

            for j in range(trash_size):
                for k in range(latent_size):
                    op_list.append(qml.CNOT(wires=[wires[k], wires[j + latent_size]]))
        return op_list


class StructureAnsatzReversed(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, latent_size, trash_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(qml.RY(weights[l, i], wires=wires[i]))

            for j in range(trash_size):
                for k in range(latent_size):
                    op_list.append(qml.CNOT(wires=[wires[k], wires[j + latent_size]]))
        return op_list


class FixedRandomEntanglementAnsatz(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 3

    @staticmethod
    def compute_decomposition(weights, wires, layers):
        op_list = []
        wires = qml.wires.Wires(wires)

        cnot_number = np.random.randint(1, 5, size=1)[0]
        cnot_idxs = []
        for i in range(cnot_number):
            idxs = np.random.choice(range(len(wires)), 2, replace=False)
            cnot_idxs.append(idxs)

        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(
                    qml.U3(
                        weights[l, i, 0],
                        weights[l, i, 1],
                        weights[l, i, 2],
                        wires=wires[i],
                    )
                )

            for i in cnot_idxs:
                op_list.append(qml.CNOT(wires=[wires[i[0]], wires[i[1]]]))

        return op_list


class RandomEntanglementAnsatz(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for i in range(len(wires)):
                op_list.append(qml.RY(weights[l, i], wires=wires[i]))

            cnot_number = np.random.randint(1, 10, size=1)[0]
            for i in range(cnot_number):
                idxs = np.random.choice(range(4), 2, replace=False)
                op_list.append(qml.CNOT(wires=[wires[idxs[0]], wires[idxs[1]]]))

        return op_list


class MultiLayerAnsatz(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None, id=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        # maybe layers can be inferred from dim in future
        self._hyperparameters = {
            "base_ansatz": config["base_ansatz"],
            "layers": config["layers"],
            "wire_list": config["wire_list"],
        }

        super().__init__(weights, wires, id=id)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, base_ansatz, layers, wire_list):
        op_list = []
        wires = qml.wires.Wires(wires)

        for _ in range(len(wire_list)):
            block_wires = range(wire_list)
            base_ansatz(weights, wires=wires[block_wires], config={"layers": layers})

        return op_list


class ReUploaderAnsatz(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, features, wires, config=None, id=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        # maybe layers can be inferred from dim in future
        self._hyperparameters = {"layers": config["layers"], "features": features}

        super().__init__(weights, wires, id=id)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2, 3

    @staticmethod
    def compute_decomposition(weights, wires, layers, features):
        # take not the weights will contain w.x too
        op_list = []
        wires = qml.wires.Wires(wires)
        for l in range(layers):
            for i in range(len(wires)):
                feat_idx = 3 * i
                theta = weights[l, i, 0, 0] + (weights[l, i, 1, 0] * features[feat_idx])
                phi = weights[l, i, 0, 1] + (
                    weights[l, i, 1, 1] * features[feat_idx + 1]
                )
                delta = weights[l, i, 0, 2] + (
                    weights[l, i, 1, 2] * features[feat_idx + 2]
                )
                qml.U3(theta, phi, delta)

        return op_list


class NewAnsatz(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 3

    @staticmethod
    def compute_decomposition(weights, wires, layers, latent_size, trash_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for t in range(trash_size):
                for i in range(latent_size):
                    control_idx = len(wires) - 1 - t
                    target_idx = i
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))

            for j in range(len(wires)):
                op_list.append(
                    qml.Rot(
                        weights[l, i, 0],
                        weights[l, i, 1],
                        weights[l, i, 2],
                        wires=wires[i],
                    )
                )

            for t in range(trash_size):
                target_idx = len(wires) - 1 - t
                control_idx = t
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
        return op_list


class NewAnsatzV0(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j], wires=wires[j]))

        return op_list


class NewAnsatzV2(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 3

    @staticmethod
    def compute_decomposition(weights, wires, layers, latent_size, trash_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

            for j in range(len(wires)):
                op_list.append(
                    qml.Rot(
                        weights[l, j, 0],
                        weights[l, j, 1],
                        weights[l, j, 2],
                        wires=wires[j],
                    )
                )
            counter = 0
            for t in range(trash_size):
                target_idx = len(wires) - 1 - t
                control_idx = counter
                if target_idx == control_idx:
                    counter = 0
                    control_idx = counter
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))
        return op_list


class NewAnsatzV3(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, latent_size, trash_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j], wires=wires[j]))

            counter = 0
            for t in range(trash_size):
                target_idx = len(wires) - 1 - t
                control_idx = counter
                if target_idx == control_idx:
                    counter = 0
                    control_idx = counter
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))
        return op_list


class NewAnsatzV4(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        weights[l, j, 0] * features[j] + weights[l, j, 1],
                        wires=wires[j],
                    )
                )

            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

            counter = 0
            for t in range(trash_size):
                target_idx = len(wires) - 1 - t
                control_idx = counter
                if target_idx == control_idx:
                    counter = 0
                    control_idx = counter
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))
        return op_list


class NewAnsatzV5(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        weights[l, j, 0] * features[j] + weights[l, j, 1],
                        wires=wires[j],
                    )
                )

            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

        counter = 0
        for t in range(trash_size):
            target_idx = len(wires) - 1 - t
            control_idx = counter
            if target_idx == control_idx:
                counter = 0
                control_idx = counter
            op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))
        return op_list


class NewAnsatzV6(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j], wires=wires[j]))

            counter = 0
            for t in range(trash_size):
                target_idx = len(wires) - 1 - t
                control_idx = counter
                if target_idx < control_idx:
                    counter = 0
                    control_idx = counter
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))
                counter += 1
        return op_list


class NewAnsatzV7(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j], wires=wires[j]))

            used_qubits = []
            for t in range(trash_size):
                target_idx = len(wires) - 1 - t
                control_idx = t
                if control_idx not in used_qubits:
                    op_list.append(
                        qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                    )

        return op_list


class NewAnsatzV8(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j], wires=wires[j]))

            used_qubits = []
            for t in range(trash_size):
                target_idx = len(wires) - 1 - t
                control_idx = t
                if (control_idx not in used_qubits) & (control_idx < target_idx):
                    op_list.append(
                        qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                    )

        return op_list


class NewAnsatzV9(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))
                spread_counter += 1

            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j], wires=wires[j]))

            counter = 0
            for t in range(trash_size):
                target_idx = len(wires) - 1 - t
                control_idx = counter
                if target_idx < control_idx:
                    counter = 0
                    control_idx = counter
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))
                counter += 1
        return op_list


class NewAnsatzV10(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j], wires=wires[j]))

            used_qubits = []
            for t in range(trash_size):
                target_idx = len(wires) - 1 - t
                control_idx = t
                if control_idx not in used_qubits:
                    op_list.append(
                        qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                    )

        return op_list


class NewAnsatzV11(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j], wires=wires[j]))

            used_qubits = []
            for t in range(trash_size):
                target_idx = len(wires) - 1 - t
                control_idx = t
                if (control_idx not in used_qubits) & (control_idx < target_idx):
                    op_list.append(
                        qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                    )

        return op_list


class NewAnsatzV12(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j], wires=wires[j]))

            if l == layers - 1:
                counter = 0
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = counter
                    if target_idx < control_idx:
                        counter = 0
                        control_idx = counter
                    op_list.append(
                        qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                    )
                    counter += 1

        return op_list


class NewAnsatzV13(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j, 0], wires=wires[j]))
                op_list.append(qml.RX(weights[l, j, 1], wires=wires[j]))

            if l == layers - 1:
                counter = 0
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = counter
                    if target_idx < control_idx:
                        counter = 0
                        control_idx = counter
                    op_list.append(
                        qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                    )
                    counter += 1

        return op_list


class NewAnsatzV14(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 3

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

            for j in range(len(wires)):
                op_list.append(
                    qml.Rot(
                        weights[l, j, 0],
                        weights[l, j, 1],
                        weights[l, j, 2],
                        wires=wires[j],
                    )
                )

            if l == layers - 1:
                counter = 0
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = counter
                    if target_idx < control_idx:
                        counter = 0
                        control_idx = counter
                    op_list.append(
                        qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                    )
                    counter += 1

        return op_list


class NewAnsatzV15(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j], wires=wires[j]))

            if l == layers - 1:
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if control_idx not in used_qubits:
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

        return op_list


class NewAnsatzV16(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j, 0], wires=wires[j]))
                op_list.append(qml.RX(weights[l, j, 1], wires=wires[j]))

            if l == layers - 1:
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if control_idx not in used_qubits:
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

        return op_list


class NewAnsatzV17(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 3

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

            for j in range(len(wires)):
                op_list.append(
                    qml.Rot(
                        weights[l, j, 0],
                        weights[l, j, 1],
                        weights[l, j, 2],
                        wires=wires[j],
                    )
                )

            if l == layers - 1:
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if control_idx not in used_qubits:
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

        return op_list


class NewAnsatzV18(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j], wires=wires[j]))

            if l == layers - 1:
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if (control_idx not in used_qubits) & (control_idx < target_idx):
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

        return op_list


class NewAnsatzV19(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j, 0], wires=wires[j]))
                op_list.append(qml.RX(weights[l, j, 1], wires=wires[j]))

            if l == layers - 1:
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if (control_idx not in used_qubits) & (control_idx < target_idx):
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

        return op_list


class NewAnsatzV20(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 3

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            spread_counter = 0
            for t in range(trash_size):
                control_idx = len(wires) - 1 - t
                if spread_counter >= latent_size:
                    spread_counter = 0
                target_idx = spread_counter
                op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                spread_counter += 1

            for j in range(len(wires)):
                op_list.append(
                    qml.Rot(
                        weights[l, j, 0],
                        weights[l, j, 1],
                        weights[l, j, 2],
                        wires=wires[j],
                    )
                )

            if l == layers - 1:
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if (control_idx not in used_qubits) & (control_idx < target_idx):
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

        return op_list


class NewAnsatzV21(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        spread_counter = 0
        for t in range(trash_size):
            control_idx = len(wires) - 1 - t
            if spread_counter >= latent_size:
                spread_counter = 0
            target_idx = spread_counter
            op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
            spread_counter += 1

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j], wires=wires[j]))

            if (l % 2 == 0) | (l == layers - 1):
                counter = 0
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = counter
                    if target_idx < control_idx:
                        counter = 0
                        control_idx = counter
                    op_list.append(
                        qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                    )
                    counter += 1

            elif l % 2 != 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class NewAnsatzV22(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        spread_counter = 0
        for t in range(trash_size):
            control_idx = len(wires) - 1 - t
            if spread_counter >= latent_size:
                spread_counter = 0
            target_idx = spread_counter
            op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
            spread_counter += 1

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j, 0], wires=wires[j]))
                op_list.append(qml.RX(weights[l, j, 1], wires=wires[j]))

            if (l % 2 == 0) | (l == layers - 1):
                counter = 0
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = counter
                    if target_idx < control_idx:
                        counter = 0
                        control_idx = counter
                    op_list.append(
                        qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                    )
                    counter += 1

            elif l % 2 != 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class NewAnsatzV23(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 3

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        spread_counter = 0
        for t in range(trash_size):
            control_idx = len(wires) - 1 - t
            if spread_counter >= latent_size:
                spread_counter = 0
            target_idx = spread_counter
            op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
            spread_counter += 1

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.Rot(
                        weights[l, j, 0],
                        weights[l, j, 1],
                        weights[l, j, 2],
                        wires=wires[j],
                    )
                )

            if (l % 2 == 0) | (l == layers - 1):
                counter = 0
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = counter
                    if target_idx < control_idx:
                        counter = 0
                        control_idx = counter
                    op_list.append(
                        qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                    )
                    counter += 1

            elif l % 2 != 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class NewAnsatzV24(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        spread_counter = 0
        for t in range(trash_size):
            control_idx = len(wires) - 1 - t
            if spread_counter >= latent_size:
                spread_counter = 0
            target_idx = spread_counter
            op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
            spread_counter += 1

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j], wires=wires[j]))

            if (l % 2 == 0) | (l == layers - 1):
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if control_idx not in used_qubits:
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )
            elif l % 2 != 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class NewAnsatzV25(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        spread_counter = 0
        for t in range(trash_size):
            control_idx = len(wires) - 1 - t
            if spread_counter >= latent_size:
                spread_counter = 0
            target_idx = spread_counter
            op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
            spread_counter += 1

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j, 0], wires=wires[j]))
                op_list.append(qml.RX(weights[l, j, 1], wires=wires[j]))

            if (l % 2 == 0) | (l == layers - 1):
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if control_idx not in used_qubits:
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )
            elif l % 2 != 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class NewAnsatzV26(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 3

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        spread_counter = 0
        for t in range(trash_size):
            control_idx = len(wires) - 1 - t
            if spread_counter >= latent_size:
                spread_counter = 0
            target_idx = spread_counter
            op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
            spread_counter += 1

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.Rot(
                        weights[l, j, 0],
                        weights[l, j, 1],
                        weights[l, j, 2],
                        wires=wires[j],
                    )
                )

            if (l % 2 == 0) | (l == layers - 1):
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if control_idx not in used_qubits:
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )
            elif l % 2 != 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class NewAnsatzV27(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        spread_counter = 0
        for t in range(trash_size):
            control_idx = len(wires) - 1 - t
            if spread_counter >= latent_size:
                spread_counter = 0
            target_idx = spread_counter
            op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
            spread_counter += 1

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j], wires=wires[j]))

            if (l % 2 == 0) | (l == layers - 1):
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if (control_idx not in used_qubits) & (control_idx < target_idx):
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

            elif l % 2 != 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class NewAnsatzV28(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        spread_counter = 0
        for t in range(trash_size):
            control_idx = len(wires) - 1 - t
            if spread_counter >= latent_size:
                spread_counter = 0
            target_idx = spread_counter
            op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
            spread_counter += 1

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(qml.RY(weights[l, j, 0], wires=wires[j]))
                op_list.append(qml.RX(weights[l, j, 1], wires=wires[j]))

            if (l % 2 == 0) | (l == layers - 1):
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if (control_idx not in used_qubits) & (control_idx < target_idx):
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

            elif l % 2 != 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class NewAnsatzV29(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "trash_size": config["trash_size"],
            "latent_size": config["latent_size"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 3

    @staticmethod
    def compute_decomposition(weights, wires, layers, trash_size, latent_size):
        op_list = []
        wires = qml.wires.Wires(wires)

        spread_counter = 0
        for t in range(trash_size):
            control_idx = len(wires) - 1 - t
            if spread_counter >= latent_size:
                spread_counter = 0
            target_idx = spread_counter
            op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
            spread_counter += 1

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.Rot(
                        weights[l, j, 0],
                        weights[l, j, 1],
                        weights[l, j, 2],
                        wires=wires[j],
                    )
                )

            if (l % 2 == 0) | (l == layers - 1):
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if (control_idx not in used_qubits) & (control_idx < target_idx):
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

            elif l % 2 != 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class ReuploaderAnsatzV1(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        weights[l, j, 0] * features[j] + weights[l, j, 1],
                        wires=wires[j],
                    )
                )
            if l != layers - 1:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        used_qubits = []
        for t in range(trash_size):
            target_idx = len(wires) - 1 - t
            control_idx = t
            if (control_idx not in used_qubits) & (control_idx < target_idx):
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))

        return op_list


class ReuploaderAnsatzV2(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        weights[l, j, 0] * features[j] + weights[l, j, 1],
                        wires=wires[j],
                    )
                )

            if (l % 2 != 0) | (l == layers - 1):
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if (control_idx not in used_qubits) & (control_idx < target_idx):
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

            elif l % 2 == 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class ReuploaderAnsatzV3(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        weights[l, j] * features[j],
                        wires=wires[j],
                    )
                )
            if l != layers - 1:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        used_qubits = []
        for t in range(trash_size):
            target_idx = len(wires) - 1 - t
            control_idx = t
            if (control_idx not in used_qubits) & (control_idx < target_idx):
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))

        return op_list


class ReuploaderAnsatzV4(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        weights[l, j] * features[j],
                        wires=wires[j],
                    )
                )

            if (l % 2 != 0) | (l == layers - 1):
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if (control_idx not in used_qubits) & (control_idx < target_idx):
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

            elif l % 2 == 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class ReuploaderAnsatzV5(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        features[j],
                        wires=wires[j],
                    )
                )

                op_list.append(
                    qml.RY(
                        weights[l, j],
                        wires=wires[j],
                    )
                )

            if l != layers - 1:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        used_qubits = []
        for t in range(trash_size):
            target_idx = len(wires) - 1 - t
            control_idx = t
            if (control_idx not in used_qubits) & (control_idx < target_idx):
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))

        return op_list


class ReuploaderAnsatzV6(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        features[j],
                        wires=wires[j],
                    )
                )
                op_list.append(
                    qml.RY(
                        weights[l, j],
                        wires=wires[j],
                    )
                )
            if (l % 2 != 0) | (l == layers - 1):
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if (control_idx not in used_qubits) & (control_idx < target_idx):
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

            elif l % 2 == 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class ReuploaderAnsatzV7(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        weights[l, j, 0] * features[2 * j],
                        wires=wires[j],
                    )
                )
                op_list.append(
                    qml.RZ(
                        weights[l, j, 1] * features[(2 * j) + 1],
                        wires=wires[j],
                    )
                )

            if l != layers - 1:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        used_qubits = []
        for t in range(trash_size):
            target_idx = len(wires) - 1 - t
            control_idx = t
            if (control_idx not in used_qubits) & (control_idx < target_idx):
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))

        return op_list


class ReuploaderAnsatzV8(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 4

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        (weights[l, j, 0] * features[2 * j]) + weights[l, j, 2],
                        wires=wires[j],
                    )
                )
                op_list.append(
                    qml.RZ(
                        (weights[l, j, 1] * features[(2 * j) + 1]) + weights[l, j, 3],
                        wires=wires[j],
                    )
                )

            if l != layers - 1:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        used_qubits = []
        for t in range(trash_size):
            target_idx = len(wires) - 1 - t
            control_idx = t
            if (control_idx not in used_qubits) & (control_idx < target_idx):
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))

        return op_list


class ReuploaderAnsatzV8(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 4

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        (weights[l, j, 0] * features[2 * j]) + weights[l, j, 2],
                        wires=wires[j],
                    )
                )
                op_list.append(
                    qml.RZ(
                        (weights[l, j, 1] * features[(2 * j) + 1]) + weights[l, j, 3],
                        wires=wires[j],
                    )
                )

            if l != layers - 1:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        used_qubits = []
        for t in range(trash_size):
            target_idx = len(wires) - 1 - t
            control_idx = t
            if (control_idx not in used_qubits) & (control_idx < target_idx):
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))

        return op_list


class ReuploaderAnsatzV9(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        features[2 * j],
                        wires=wires[j],
                    )
                )
                op_list.append(
                    qml.RY(
                        weights[l, j, 0],
                        wires=wires[j],
                    )
                )

                op_list.append(
                    qml.RZ(
                        features[(2 * j) + 1],
                        wires=wires[j],
                    )
                )

                op_list.append(
                    qml.RZ(
                        weights[l, j, 1],
                        wires=wires[j],
                    )
                )

            if l != layers - 1:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        used_qubits = []
        for t in range(trash_size):
            target_idx = len(wires) - 1 - t
            control_idx = t
            if (control_idx not in used_qubits) & (control_idx < target_idx):
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))

        return op_list


class ReuploaderAnsatzV10(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 4

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        (weights[l, j, 0] * features[2 * j]) + weights[l, j, 2],
                        wires=wires[j],
                    )
                )
                op_list.append(
                    qml.RZ(
                        (weights[l, j, 1] * features[(2 * j) + 1]) + weights[l, j, 3],
                        wires=wires[j],
                    )
                )

            if (l % 2 != 0) | (l == layers - 1):
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if (control_idx not in used_qubits) & (control_idx < target_idx):
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

            elif l % 2 == 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class ReuploaderAnsatzV11(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        (weights[l, j, 0] * features[2 * j]),
                        wires=wires[j],
                    )
                )
                op_list.append(
                    qml.RZ(
                        (weights[l, j, 1] * features[(2 * j) + 1]),
                        wires=wires[j],
                    )
                )

            if l != layers - 1:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        used_qubits = []
        for t in range(trash_size):
            target_idx = len(wires) - 1 - t
            control_idx = t
            if (control_idx not in used_qubits) & (control_idx < target_idx):
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))

        return op_list


class ReuploaderAnsatzV12(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        (weights[l, j, 0] * features[2 * j]),
                        wires=wires[j],
                    )
                )
                op_list.append(
                    qml.RZ(
                        (weights[l, j, 1] * features[(2 * j) + 1]),
                        wires=wires[j],
                    )
                )

            if (l % 2 != 0) | (l == layers - 1):
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if (control_idx not in used_qubits) & (control_idx < target_idx):
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

            elif l % 2 == 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class ReuploaderAnsatzV13(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        features[2 * j],
                        wires=wires[j],
                    )
                )
                op_list.append(
                    qml.RY(
                        weights[l, j, 0],
                        wires=wires[j],
                    )
                )

                op_list.append(
                    qml.RZ(
                        features[(2 * j) + 1],
                        wires=wires[j],
                    )
                )

                op_list.append(
                    qml.RZ(
                        weights[l, j, 1],
                        wires=wires[j],
                    )
                )

            if (l % 2 != 0) | (l == layers - 1):
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if (control_idx not in used_qubits) & (control_idx < target_idx):
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

            elif l % 2 == 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class ReuploaderAnsatzV14(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        (weights[l, j, 0] * features[2 * j])
                        + (weights[l, j, 1] * features[(2 * j) + 1]),
                        wires=wires[j],
                    )
                )

            if (l % 2 != 0) | (l == layers - 1):
                used_qubits = []
                for t in range(trash_size):
                    target_idx = len(wires) - 1 - t
                    control_idx = t
                    if (control_idx not in used_qubits) & (control_idx < target_idx):
                        op_list.append(
                            qml.CNOT(wires=[wires[control_idx], wires[target_idx]])
                        )

            elif l % 2 == 0:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        return op_list


class ReuploaderAnsatzV15(Operation):
    """This one includes fewer CNOT gates in the information sharing"""

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, config=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        self._hyperparameters = {
            "layers": config["layers"],
            "latent_size": config["latent_size"],
            "trash_size": config["trash_size"],
            "features": config["features"],
        }

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires, 2

    @staticmethod
    def compute_decomposition(
        weights, wires, layers, latent_size, trash_size, features
    ):
        op_list = []
        wires = qml.wires.Wires(wires)

        for l in range(layers):
            for j in range(len(wires)):
                op_list.append(
                    qml.RY(
                        (weights[l, j, 0] * features[2 * j])
                        + (weights[l, j, 1] * features[(2 * j) + 1]),
                        wires=wires[j],
                    )
                )

            if l != layers - 1:
                spread_counter = 0
                for t in range(trash_size):
                    control_idx = len(wires) - 1 - t
                    if spread_counter >= latent_size:
                        spread_counter = 0
                    target_idx = spread_counter
                    op_list.append(qml.CNOT(wires=[control_idx, target_idx]))
                    spread_counter += 1

        used_qubits = []
        for t in range(trash_size):
            target_idx = len(wires) - 1 - t
            control_idx = t
            if (control_idx not in used_qubits) & (control_idx < target_idx):
                op_list.append(qml.CNOT(wires=[wires[control_idx], wires[target_idx]]))

        return op_list


"""
class EntanglingAnsatz(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, layers, wires=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        super().__init__(weights, layers, wires)

    @staticmethod
    def shape(n_wires):
        return n_wires, 3

    @staticmethod
    def compute_decomposition(weights, layers, wires):
        op_list = []

        wires = qml.wires.Wires(wires)
        layer_count = 0
        for i in range(len(wires)):
            qml.RY(weights[i, 0], wires=wires[i])

        if layers == 1:
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[i, i + 1])

        if layers == 2:
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[i, i + 1])

            for i in range(len(wires) - 2):
                qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[i, i + 2])

        if layers == 3:
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[i, i + 1])

            for i in range(len(wires) - 2):
                qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[i, i + 2])

            for i in range(len(wires) - 3):
                qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[i, i + 3])

        return op_list
"""

""""
class ExpressAnsatz(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires=None):
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)
        shape = qml.math.shape(weights)

        super().__init__(weights, wires)

    @staticmethod
    def shape(n_wires, layers):
        return layers, n_wires

    @staticmethod
    def compute_decomposition(weights, wires):
        op_list = []
        layers = weights.shape[0]
        wires = qml.wires.Wires(wires)
        operation_list = [qml.RX, qml.RY, qml.RZ]

        for i in range(len(wires)):
            for l in range(int(layers)):
                idx = l % 3
                op = operation_list[idx]
                op(weights[l, i], wires=wires[i])

        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])

        return op_list
"""

#########################################


def swap_test(latent_size, trash_size):
    aux_qubit = latent_size + 2 * trash_size
    qml.Hadamard(wires=aux_qubit)
    for i in range(trash_size):
        qml.CSWAP(wires=[aux_qubit, latent_size + i, latent_size + trash_size + i])
    qml.Hadamard(wires=aux_qubit)


def full_swap_test(input_size):
    aux_qubit = 2 * input_size
    qml.Hadamard(wires=aux_qubit)
    for i in range(input_size):
        qml.CSWAP(wires=[aux_qubit, i, input_size + i])
    qml.Hadamard(wires=aux_qubit)


#######################################

"""
class FidelityCirc:
    def __init__(self, embedding, ansatz, input_size, latent_size, draw=True):
        self.weight_shape = ansatz.shape(input_size)
        print("weight shape", self.weight_shape)
        trash_size = input_size - latent_size
        total_wires = input_size + trash_size + 1
        dev = qml.device("default.qubit", wires=total_wires)

        def fid(params, features):
            embedding(features, input_size)
            ansatz(params, range(input_size))
            swap_test(latent_size, trash_size)
            return qml.expval(qml.operation.Tensor(*[qml.PauliZ(total_wires - 1)]))

        self.c = qml.QNode(fid, dev)

        if draw:
            drawer = qml.draw(self.c)
            p = qnp.random.uniform(low=0, high=2 * qnp.pi, size=self.weight_shape)
            x = qnp.random.uniform(low=0, high=2 * qnp.pi, size=input_size)
            print(drawer(p, x))

    def __call__(self, x):
        p = get_param("p", self.weight_shape)
        return self.c(p, x)
"""

########################################


class QAEBuilder:
    def __init__(
        self,
        loss,
        embedding,
        ansatz,
        input_size,
        total_wires,
        latent_size,
        trash_size,
        layers,
    ):
        self.circuit_architecture = loss.split("_")[0]
        self.embedding = embedding
        self.ansatz = ansatz
        self.input_size = input_size
        self.total_wires = total_wires
        self.latent_size = latent_size
        self.layers = layers
        self.trash_size = trash_size

    def fidelity_circuit(
        self,
        params,
        features,
    ):
        self.embedding(features, self.input_size)
        self.ansatz(params, self.input_size, self.layers)
        swap_test(self.latent_size, self.trash_size)
        return qml.expval(qml.operation.Tensor(*[qml.PauliZ(self.total_wires - 1)]))

    def mutual_information_circuit(
        self, embedding, ansatz, features, input_size, layers, latent_size, params
    ):
        embedding(features, input_size)
        ansatz(params, input_size, layers)
        return qml.mutual_info(
            wires0=range(latent_size), wires1=range(latent_size, input_size)
        ), qml.probs(wires=range(latent_size, input_size))

    def get_circuit(self):
        if self.circuit_architecture == "fidelity":
            circuit = self.fidelity_circuit
        elif self.circuit_architecture == "mutual":
            circuit = self.mutual_information_circuit
        return circuit

    def get_dev(self):
        if self.circuit_architecture == "fidelity":
            dev = qml.device("default.qubit", wires=self.total_wires)
        elif self.circuit_architecture == "mutual":
            dev = qml.device("default.qubit", wires=self.input_size)
        return dev


def fidelity_circuit(params, features, config):
    config["embedding_fn"](features, range(config["input_size"]))
    config["ansatz_fn"](params, wires=range(config["input_size"]), config=config)
    swap_test(config["latent_size"], config["trash_size"])
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(config["total_wires"] - 1)]))


def q_autoencoder(params, features, config):
    config["embedding_fn"](features, 0, config["input_size"])
    config["ansatz_fn"](params, wires=range(config["input_size"]), config=config)

    for r in range(config["latent_size"], config["input_size"]):
        qml.measure(r, reset=True)

    qml.adjoint(
        config["ansatz_fn"](params, wires=range(config["input_size"]), config=config)
    )

    config["embedding_fn"](features, config["input_size"], 2 * config["input_size"])
    full_swap_test(config["input_size"])
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(2 * config["input_size"])]))


def reuploader(params, features, config):
    config["features"] = features
    config["ansatz_fn"](params, wires=range(config["input_size"]), config=config)
    swap_test(config["latent_size"], config["trash_size"])
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(config["total_wires"] - 1)]))


def compact_reuploader(params, features, config):
    to_pad = features.shape[0] % 3
    padding = 3 - features.shape[0] % 3
    if to_pad != 0:
        features = np.pad(features, (0, padding), "constant")
    ReUploaderAnsatz(params, features, wires=range(config["input_size"]), config=config)
    swap_test(config["latent_size"], config["trash_size"])
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(config["total_wires"] - 1)]))


model_dict = {
    "fidelity_circuit": fidelity_circuit,
    "reuploader_circuit": reuploader,
}

ansatz_dict = {
    "ansatz_1": Ansatz1,
    "ansatz_2": Ansatz2,
    "ansatz_2_reversed": Ansatz2Reversed,
    "ansatz_ring": AnsatzRing,
    "ansatz_ring_reversed": AnsatzRingReversed,
    "ansatz_skip": AnsatzSkip,
    "ansatz_skip_reversed": AnsatzSkipReversed,
    "ansatz_3": Ansatz3,
    "original_ansatz": OriginalAnsatz,
    "original_ansatz_reversed": OriginalAnsatzReversed,
    "structure_ansatz": StructureAnsatz,
    "structure_ansatz_reversed": StructureAnsatzReversed,
    "random_entanglement_ansatz": RandomEntanglementAnsatz,
    "fixed_random_entanglement_ansatz": FixedRandomEntanglementAnsatz,
    "fancy_ansatz": FancyAnsatz,
    "new_ansatz": NewAnsatz,
    "new_ansatzv0": NewAnsatzV0,
    "new_ansatzv2": NewAnsatzV2,
    "new_ansatzv3": NewAnsatzV3,
    "new_ansatzv4": NewAnsatzV4,
    "new_ansatzv5": NewAnsatzV5,
    "new_ansatzv6": NewAnsatzV6,
    "new_ansatzv7": NewAnsatzV7,
    "new_ansatzv8": NewAnsatzV8,
    "new_ansatzv9": NewAnsatzV9,
    "new_ansatzv10": NewAnsatzV10,
    "new_ansatzv11": NewAnsatzV11,
    "new_ansatzv12": NewAnsatzV12,
    "new_ansatzv13": NewAnsatzV13,
    "new_ansatzv14": NewAnsatzV14,
    "new_ansatzv15": NewAnsatzV15,
    "new_ansatzv16": NewAnsatzV16,
    "new_ansatzv17": NewAnsatzV17,
    "new_ansatzv18": NewAnsatzV18,
    "new_ansatzv19": NewAnsatzV19,
    "new_ansatzv20": NewAnsatzV20,
    "new_ansatzv21": NewAnsatzV21,
    "new_ansatzv22": NewAnsatzV22,
    "new_ansatzv23": NewAnsatzV23,
    "new_ansatzv24": NewAnsatzV24,
    "new_ansatzv25": NewAnsatzV25,
    "new_ansatzv26": NewAnsatzV26,
    "new_ansatzv27": NewAnsatzV27,
    "new_ansatzv28": NewAnsatzV28,
    "new_ansatzv29": NewAnsatzV29,
    "reuploader_v1": ReuploaderAnsatzV1,
    "reuploader_v2": ReuploaderAnsatzV2,
    "reuploader_v3": ReuploaderAnsatzV3,
    "reuploader_v4": ReuploaderAnsatzV4,
    "reuploader_v5": ReuploaderAnsatzV5,
    "reuploader_v6": ReuploaderAnsatzV6,
    "reuploader_v7": ReuploaderAnsatzV7,
    "reuploader_v8": ReuploaderAnsatzV8,
    "reuploader_v9": ReuploaderAnsatzV9,
    "reuploader_v10": ReuploaderAnsatzV10,
    "reuploader_v11": ReuploaderAnsatzV11,
    "reuploader_v12": ReuploaderAnsatzV12,
    "reuploader_v13": ReuploaderAnsatzV13,
    "reuploader_v14": ReuploaderAnsatzV14,
    "reuploader_v15": ReuploaderAnsatzV15,
}

embedding_dict = {
    "rx_embedding": rx_embedding,
    "particle_embedding": particle_2_embedding,
    "full_rx_embedding": full_rx_embedding,
    "full_particle_embedding": full_particle_2_embedding,
}
