from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import seaborn as sns # type: ignore
import torch # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from qml_app.models import HybridVariationalClassifier, QuantumKernelClassifier
from qml_app.utils.logging_utils import init_logger # type: ignore


def evaluate_vqc(
    model: HybridVariationalClassifier,
    dataloader,
    device: torch.device,
    artifacts_dir: Path,
) -> Dict[str, float]:
    logger = init_logger()
    model.eval()
    model.to(device)

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            probs = model.predict_proba(features).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    y_true = np.concatenate(all_targets).astype(int)
    y_prob = np.concatenate(all_probs).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = _compute_metrics(y_true, y_pred, y_prob)
    _store_metrics(metrics, artifacts_dir / "vqc_metrics.json")
    _plot_curves(y_true, y_prob, artifacts_dir, prefix="vqc")
    logger.info(f"Evaluasi VQC: {metrics}")
    return metrics


def evaluate_kernel(
    model: QuantumKernelClassifier,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    artifacts_dir: Path,
) -> Dict[str, float]:
    logger = init_logger()
    probs = model.predict_proba(test_features)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = _compute_metrics(test_labels.astype(int), preds, probs)
    _store_metrics(metrics, artifacts_dir / "kernel_metrics.json")
    _plot_curves(test_labels, probs, artifacts_dir, prefix="kernel")
    logger.info(f"Evaluasi Kernel QML: {metrics}")
    return metrics


def _compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def _store_metrics(metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


def _plot_curves(y_true, y_prob, artifacts_dir: Path, prefix: str) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"{prefix.upper()} ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({prefix.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / f"{prefix}_roc_curve.png", dpi=200)
    plt.close()

    cm = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({prefix.upper()})")
    plt.tight_layout()
    plt.savefig(artifacts_dir / f"{prefix}_confusion_matrix.png", dpi=200)
    plt.close()