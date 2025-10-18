from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np # type: ignore
import torch # type: ignore
import typer # type: ignore
from rich import print # type: ignore

from qml_app.config import AppConfig
from qml_app.data import QuantumDataModule
from qml_app.evaluation import evaluate_kernel, evaluate_vqc
from qml_app.models import HybridVariationalClassifier, QuantumKernelClassifier
from qml_app.training import resolve_device, train_variational_model
from qml_app.utils.config_utils import build_app_config, load_yaml_config # type: ignore
from qml_app.utils.logging_utils import init_logger # type: ignore
from qml_app.utils.seed import set_global_seed # type: ignore

app = typer.Typer(add_completion=False, help="Advanced Quantum Machine Learning CLI")


def load_config(config_path: Path) -> AppConfig:
    cfg_dict = load_yaml_config(config_path)
    return build_app_config(cfg_dict)


@app.command()
def train(
    model: str = typer.Option("vqc", "--model", "-m", help="Model yang dilatih (vqc atau kernel)"),
    config: Path = typer.Option("config/default.yaml", "--config", "-c", exists=True, help="Path file konfigurasi."),
    seed: Optional[int] = typer.Option(None, help="Seed random global."),
):
    cfg = load_config(config)
    logger = init_logger()

    if seed is not None:
        set_global_seed(seed)
        logger.info(f"Seed global disetel ke {seed}")

    data_module = QuantumDataModule(cfg.data)
    data_module.prepare_data()

    artifacts_dir = Path(cfg.evaluation.save_path) / model
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if model.lower() == "vqc":
        vqc_model = HybridVariationalClassifier(
            n_qubits=cfg.data.n_qubits,
            feature_layers=cfg.model.feature_layers,
            variational_layers=cfg.model.variational_layers,
            shots=cfg.model.shots,
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.model.dropout,
            use_complex_device=cfg.model.use_complex_device,
        )
        dataloaders = data_module.get_dataloaders(cfg.training.batch_size)
        history = train_variational_model(
            model=vqc_model,
            dataloaders=dataloaders,
            config=cfg.training,
            artifacts_dir=artifacts_dir,
        )
        print("[bold green]Training VQC selesai.[/bold green]")
        print(f"History tersimpan di {artifacts_dir / 'training_history.json'}")

    elif model.lower() == "kernel":
        qsvc = QuantumKernelClassifier(
            n_qubits=cfg.data.n_qubits,
            feature_layers=cfg.model.feature_layers,
            shots=cfg.model.shots,
            c_regularization=cfg.model.kernel_regularization,
            use_complex_device=cfg.model.use_complex_device,
        )
        x_train, y_train, *_ = data_module.get_numpy_splits()
        result = qsvc.fit(np.asarray(x_train), np.asarray(y_train))
        print(
            "[bold green]Training Quantum Kernel SVM selesai.[/bold green] "
            f"(acc={result.accuracy:.4f}, f1={result.f1:.4f})"
        )
        torch.save(qsvc, artifacts_dir / "kernel_model.pt")
        print(f"Model kernel tersimpan di {artifacts_dir / 'kernel_model.pt'}")

    else:
        raise typer.BadParameter("Model harus 'vqc' atau 'kernel'.")


@app.command()
def evaluate(
    model: str = typer.Option("vqc", "--model", "-m"),
    config: Path = typer.Option("config/default.yaml", "--config", "-c", exists=True),
    weight_path: Optional[Path] = typer.Option(None, help="Path bobot/model khusus."),
):
    cfg = load_config(config)
    logger = init_logger()
    data_module = QuantumDataModule(cfg.data)
    data_module.prepare_data()

    artifacts_dir = Path(cfg.evaluation.save_path) / model
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if model.lower() == "vqc":
        model_weights = weight_path or artifacts_dir / "vqc_weights.pt"
        vqc_model = HybridVariationalClassifier(
            n_qubits=cfg.data.n_qubits,
            feature_layers=cfg.model.feature_layers,
            variational_layers=cfg.model.variational_layers,
            shots=cfg.model.shots,
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.model.dropout,
            use_complex_device=cfg.model.use_complex_device,
        )
        vqc_model.load_state_dict(torch.load(model_weights, map_location="cpu"))
        device = resolve_device(cfg.training.device)
        dataloaders = data_module.get_dataloaders(cfg.training.batch_size, shuffle=False)
        metrics = evaluate_vqc(
            model=vqc_model,
            dataloader=dataloaders["test"],
            device=device,
            artifacts_dir=artifacts_dir,
        )
        print(f"[bold cyan]Hasil evaluasi VQC:[/bold cyan] {metrics}")

    elif model.lower() == "kernel":
        model_path = weight_path or artifacts_dir / "kernel_model.pt"
        if model_path.exists():
            qsvc: QuantumKernelClassifier = torch.load(model_path)
        else:
            raise FileNotFoundError(f"Tidak menemukan model kernel di {model_path}")

        *_ignored, x_test, y_test = data_module.get_numpy_splits()
        metrics = evaluate_kernel(
            model=qsvc,
            test_features=np.asarray(x_test),
            test_labels=np.asarray(y_test),
            artifacts_dir=artifacts_dir,
        )
        print(f"[bold cyan]Hasil evaluasi Kernel-QML:[/bold cyan] {metrics}")

    else:
        raise typer.BadParameter("Model harus 'vqc' atau 'kernel'.")


if __name__ == "__main__":
    app()