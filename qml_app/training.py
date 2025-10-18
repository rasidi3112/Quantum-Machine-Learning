from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn # type: ignore
from torch.utils.data import DataLoader # type: ignore

from qml_app.config import TrainingConfig
from qml_app.models import HybridVariationalClassifier
from qml_app.utils.logging_utils import init_logger # type: ignore



def resolve_device(preference: str) -> torch.device:
    preference = preference.lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA tidak tersedia.")
    if preference == "mps":
        if torch.backends.mps.is_available():
            # ⚠️ QML (PennyLane) belum mendukung MPS dengan baik, jadi kita pakai CPU
            print("⚠️ MPS terdeteksi tapi belum stabil untuk QML — menggunakan CPU sebagai gantinya.")
            return torch.device("cpu")
        raise RuntimeError("MPS tidak tersedia.")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    # ⚠️ default ke CPU, bukan MPS 
    return torch.device("cpu")




def train_variational_model(
    model: HybridVariationalClassifier,
    dataloaders: Dict[str, DataLoader],
    config: TrainingConfig,
    artifacts_dir: Path,
) -> Dict[str, List[float]]:
    logger = init_logger()
    device = resolve_device(config.device)
    logger.info(f"Menggunakan device: [bold]{device}[/bold]")

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=max(2, config.patience // 2)
)


    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeRemainingColumn(),
    )

    with progress:
        train_task = progress.add_task("Training VQC", total=config.epochs)

        for epoch in range(1, config.epochs + 1):
            train_loss, train_correct, train_count = _run_epoch(
                model,
                dataloaders["train"],
                criterion,
                optimizer,
                device,
                train_mode=True,
                grad_clip=config.gradient_clip,
            )

            val_loss, val_correct, val_count = _run_epoch(
                model,
                dataloaders["val"],
                criterion,
                optimizer,
                device,
                train_mode=False,
            )

            train_acc = train_correct / train_count
            val_acc = val_correct / val_count

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            scheduler.step(val_loss)

            logger.info(
                f"Epoch {epoch:03d} | "
                f"Loss (train/val): {train_loss:.4f}/{val_loss:.4f} | "
                f"Acc (train/val): {train_acc:.4f}/{val_acc:.4f}"
            )

            progress.update(train_task, advance=1)

            if val_loss + config.min_delta < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    logger.warning("Early stopping dipicu.")
                    break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    history_path = artifacts_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    torch.save(model.state_dict(), artifacts_dir / "vqc_weights.pt")
    logger.info(f"Model tersimpan di {artifacts_dir / 'vqc_weights.pt'}")
    return history


def _run_epoch(
    model: HybridVariationalClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_mode: bool,
    grad_clip: float | None = None,
):
    if train_mode:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (features, targets) in enumerate(loader, start=1):
        features = features.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(features)
            loss = criterion(logits, targets)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        epoch_loss += loss.item() * features.size(0)

        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds.squeeze() == targets.long()).sum().item()
        total += features.size(0)

    epoch_loss /= total
    return epoch_loss, correct, total