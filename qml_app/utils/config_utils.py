from pathlib import Path
from typing import Any, Dict

import yaml # type: ignore

from qml_app.config import AppConfig, DataConfig, EvaluationConfig, ModelConfig, TrainingConfig # type: ignore


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_app_config(cfg_dict: Dict[str, Any]) -> AppConfig:
    data_cfg = DataConfig(**cfg_dict.get("data", {}))
    model_cfg = ModelConfig(**cfg_dict.get("model", {}))
    training_cfg = TrainingConfig(**cfg_dict.get("training", {}))
    evaluation_cfg = EvaluationConfig(**cfg_dict.get("evaluation", {}))
    return AppConfig(data=data_cfg, model=model_cfg, training=training_cfg, evaluation=evaluation_cfg)