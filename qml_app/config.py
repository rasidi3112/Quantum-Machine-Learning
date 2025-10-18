from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class DataConfig:
    dataset: Literal["moons", "circles", "breast_cancer"] = "moons"
    n_samples: int = 800
    noise: float = 0.2
    test_size: float = 0.2
    val_size: float = 0.15
    random_state: int = 42
    feature_scaler: Literal["standard", "minmax", "robust"] = "standard"
    feature_expansion: Literal["polynomial", "none"] = "polynomial"
    expansion_degree: int = 2
    n_qubits: int = 4


@dataclass
class ModelConfig:
    shots: Optional[int] = 1024
    feature_layers: int = 2
    variational_layers: int = 3
    hidden_dim: int = 32
    dropout: float = 0.1
    kernel_regularization: float = 1.0
    use_complex_device: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 0.005
    weight_decay: float = 5e-4
    gradient_clip: float = 1.0
    patience: int = 7
    min_delta: float = 1e-3
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    log_interval: int = 10


@dataclass
class EvaluationConfig:
    save_path: Path = Path("artifacts")
    roc_curve_points: int = 200
    confusion_matrix: bool = True
    report_average: Literal["weighted", "micro", "macro", "binary"] = "weighted"


@dataclass
class AppConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig