from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np # type: ignore
import torch # type: ignore
from sklearn.datasets import load_breast_cancer, make_circles, make_moons # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, RobustScaler, StandardScaler # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore

from qml_app.config import DataConfig


class QuantumDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


@dataclass
class ProcessedData:
    features: np.ndarray
    labels: np.ndarray


class QuantumDataModule:
    def __init__(self, config: DataConfig):
        self.cfg = config

        self.scaler = self._init_scaler(config.feature_scaler)
        self.expander: Optional[PolynomialFeatures] = None
        self.reducer: Optional[PCA] = None

        self.train_data: Optional[ProcessedData] = None
        self.val_data: Optional[ProcessedData] = None
        self.test_data: Optional[ProcessedData] = None

    def prepare_data(self) -> None:
        features, labels = self._create_raw_dataset()
        x_train, x_temp, y_train, y_temp = train_test_split(
            features,
            labels,
            test_size=self.cfg.test_size + self.cfg.val_size,
            random_state=self.cfg.random_state,
            stratify=labels,
        )

        relative_val_size = self.cfg.val_size / (self.cfg.test_size + self.cfg.val_size)
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=1 - relative_val_size,
            random_state=self.cfg.random_state,
            stratify=y_temp,
        )

        x_train = self._quantum_feature_pipeline_fit(x_train)
        x_val = self._quantum_feature_pipeline_transform(x_val)
        x_test = self._quantum_feature_pipeline_transform(x_test)

        self.train_data = ProcessedData(features=x_train, labels=y_train)
        self.val_data = ProcessedData(features=x_val, labels=y_val)
        self.test_data = ProcessedData(features=x_test, labels=y_test)

    def get_dataloaders(self, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> Dict[str, DataLoader]:
        assert self.train_data and self.val_data and self.test_data, "Data belum disiapkan."

        train_loader = DataLoader(
            QuantumDataset(self.train_data.features, self.train_data.labels),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            QuantumDataset(self.val_data.features, self.val_data.labels),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            QuantumDataset(self.test_data.features, self.test_data.labels),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return {"train": train_loader, "val": val_loader, "test": test_loader}

    def get_numpy_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.train_data and self.val_data and self.test_data, "Data belum disiapkan."
        return (
            self.train_data.features,
            self.train_data.labels,
            self.val_data.features,
            self.val_data.labels,
            self.test_data.features,
            self.test_data.labels,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _create_raw_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.cfg.dataset == "moons":
            features, labels = make_moons(
                n_samples=self.cfg.n_samples,
                noise=self.cfg.noise,
                random_state=self.cfg.random_state,
            )
        elif self.cfg.dataset == "circles":
            features, labels = make_circles(
                n_samples=self.cfg.n_samples,
                noise=self.cfg.noise,
                factor=0.5,
                random_state=self.cfg.random_state,
            )
        elif self.cfg.dataset == "breast_cancer":
            data = load_breast_cancer()
            features, labels = data.data, data.target
        else:
            raise ValueError(f"Dataset {self.cfg.dataset} belum didukung.")

        return features.astype(np.float64), labels.astype(np.int64)

    def _init_scaler(self, name: str):
        if name == "standard":
            return StandardScaler()
        if name == "minmax":
            return MinMaxScaler()
        if name == "robust":
            return RobustScaler()
        raise ValueError(f"Scaler {name} tidak dikenal.")

    def _quantum_feature_pipeline_fit(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.fit_transform(features)

        expanded = self._maybe_expand_features(scaled, fit=True)
        aligned = self._align_with_qubits(expanded, fit=True)
        return aligned

    def _quantum_feature_pipeline_transform(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)

        expanded = self._maybe_expand_features(scaled, fit=False)
        aligned = self._align_with_qubits(expanded, fit=False)
        return aligned

    def _maybe_expand_features(self, features: np.ndarray, fit: bool) -> np.ndarray:
        if self.cfg.feature_expansion == "none":
            return features
        if self.cfg.feature_expansion == "polynomial":
            if fit or self.expander is None:
                self.expander = PolynomialFeatures(
                    degree=self.cfg.expansion_degree,
                    include_bias=False,
                )
                return self.expander.fit_transform(features)
            return self.expander.transform(features)
        raise ValueError(f"Metode ekspansi {self.cfg.feature_expansion} tidak didukung.")

    def _align_with_qubits(self, features: np.ndarray, fit: bool) -> np.ndarray:
        current_dim = features.shape[1]
        target_dim = self.cfg.n_qubits
        if current_dim == target_dim:
            return features

        if current_dim > target_dim:
            if fit or self.reducer is None:
                self.reducer = PCA(n_components=target_dim, random_state=self.cfg.random_state)
                return self.reducer.fit_transform(features)
            return self.reducer.transform(features)


        padding = np.zeros((features.shape[0], target_dim - current_dim), dtype=features.dtype)
        return np.hstack([features, padding])