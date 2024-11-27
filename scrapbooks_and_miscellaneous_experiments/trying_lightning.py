import pennylane as qml


dev = qml.device("lightning.qubit", wires=3)


@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    return qml.state()


print(circuit())
