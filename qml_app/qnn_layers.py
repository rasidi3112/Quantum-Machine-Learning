from typing import Dict, Tuple

import pennylane as qml  # type: ignore
from pennylane import numpy as np  # type: ignore


def feature_map_template(x, wires, layers: int = 2):
    n_qubits = len(wires)
    qml.templates.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')  # type: ignore

    for _ in range(layers):

        for i in range(len(wires) - 1):
            qml.IsingXX(np.pi / 2, wires=[wires[i], wires[i + 1]])
        qml.IsingXX(np.pi / 2, wires=[wires[-1], wires[0]])  

        for i in range(len(wires) - 1):
            qml.IsingZZ(np.pi / 2, wires=[wires[i], wires[i + 1]])
        qml.IsingZZ(np.pi / 2, wires=[wires[-1], wires[0]])


def build_kernel_qnode(
    n_qubits: int,
    feature_layers: int,
    shots: int | None,
    use_complex_device: bool = False,
):
    """Bangun QNode kernel fidelity."""
    dev_kwargs = {"wires": n_qubits, "shots": shots}
    if use_complex_device:
        dev_kwargs["c_dtype"] = complex

    device = qml.device("default.qubit", **dev_kwargs)

    @qml.qnode(device, interface="autograd")
    def kernel_circuit(x, y):
        feature_map_template(x, wires=device.wires, layers=feature_layers)
        qml.adjoint(feature_map_template)(y, wires=device.wires, layers=feature_layers)
        
        return qml.probs(wires=device.wires)

    return kernel_circuit


def build_variational_circuit(
    n_qubits: int,
    feature_layers: int,
    variational_layers: int,
    shots: int | None,
    use_complex_device: bool = False,
) -> Tuple[qml.QNode, Dict[str, Tuple[int, ...]]]:
    dev_kwargs = {"wires": n_qubits, "shots": shots}
    if use_complex_device:
        dev_kwargs["c_dtype"] = complex

    device = qml.device("default.qubit", **dev_kwargs)

    weight_shapes = {"weights": (variational_layers, n_qubits, 3)}

    @qml.qnode(device, interface="torch")
    def circuit(inputs, weights):
        feature_map_template(inputs, wires=device.wires, layers=feature_layers)
        qml.StronglyEntanglingLayers(weights, wires=device.wires)
        return [qml.expval(qml.PauliZ(w)) for w in device.wires]

    return circuit, weight_shapes
