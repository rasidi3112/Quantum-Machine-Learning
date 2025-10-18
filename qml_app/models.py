from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np  # type: ignore
import pennylane as qml  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from pennylane import numpy as pnp  # type: ignore
from sklearn.metrics import accuracy_score, f1_score  # type: ignore
from sklearn.svm import SVC  # type: ignore

from qml_app.qnn_layers import build_kernel_qnode, build_variational_circuit


@dataclass
class KernelTrainingResult:
    accuracy: float
    f1: float


class QuantumKernelClassifier:
    def __init__(
        self,
        n_qubits: int,
        feature_layers: int,
        shots: Optional[int],
        c_regularization: float,
        use_complex_device: bool = False,
    ):
        self.n_qubits = n_qubits
        self.feature_layers = feature_layers
        self.shots = shots
        self.c_regularization = c_regularization
        self.use_complex_device = use_complex_device

        self._kernel_circuit = build_kernel_qnode(
            n_qubits=n_qubits,
            feature_layers=feature_layers,
            shots=shots,
            use_complex_device=use_complex_device,
        )
        self._svc: Optional[SVC] = None
        self._train_features: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> KernelTrainingResult:
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        kernel_matrix = qml.kernels.kernel_matrix(features, features, self._kernel_circuit)
        if kernel_matrix.ndim == 3:
            kernel_matrix = kernel_matrix.mean(axis=0)
            

        svc = SVC(kernel="precomputed", C=self.c_regularization, probability=True)
        svc.fit(kernel_matrix, labels)

        preds = svc.predict(kernel_matrix)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)

        self._svc = svc
        self._train_features = features

        return KernelTrainingResult(accuracy=acc, f1=f1)

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self._svc is not None and self._train_features is not None, "Model belum dilatih."

        kernel_block = qml.kernels.kernel_matrix(
            np.asarray(features, dtype=np.float64),
            self._train_features,
            self._kernel_circuit,
        )
        if kernel_block.ndim == 3:
            kernel_matrix = kernel_matrix  # jangan squeeze
 

        return self._svc.predict(kernel_block)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        assert self._svc is not None and self._train_features is not None, "Model belum dilatih."

        kernel_block = qml.kernels.kernel_matrix(
            np.asarray(features, dtype=np.float64),
            self._train_features,
            self._kernel_circuit,
        )
        if kernel_block.ndim == 3:
            kernel_block = kernel_block


        return self._svc.predict_proba(kernel_block)


class HybridVariationalClassifier(nn.Module):
    def __init__(
        self,
        n_qubits: int,
        feature_layers: int,
        variational_layers: int,
        shots: Optional[int],
        hidden_dim: int,
        dropout: float,
        use_complex_device: bool = False,
    ):
        super().__init__()

        circuit, weight_shapes = build_variational_circuit(
            n_qubits=n_qubits,
            feature_layers=feature_layers,
            variational_layers=variational_layers,
            shots=shots,
            use_complex_device=use_complex_device,
        )
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_features = self.quantum_layer(x)
        logits = self.post_net(q_features)
        return logits.squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return (probs >= 0.5).long()
