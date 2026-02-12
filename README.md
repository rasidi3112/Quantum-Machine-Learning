# Quantum Machine Learning: Hybrid VQC vs. Quantum Kernel SVM

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.15-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.30-lightblue?logo=pytorch&logoColor=white)](https://pennylane.ai/)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.43-purple?logo=qiskit&logoColor=white)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)


<p align="center">
  <img src="assets/images/qml_hero_banner.png" alt="Quantum Machine Learning" width="600">
</p>

---

# Overview

This repository hosts an Advanced Quantum Machine Learning (QML) project designed for rigorous comparative analysis and reproducible research. It implements and benchmarks two leading NISQ-era paradigms:

**Hybrid Variational Quantum Classifier (VQC):** A trainable quantum-classical hybrid model optimized end-to-end using PyTorch and PennyLane.

**Quantum Kernel Support Vector Machine (QSVM):** A kernel-based model utilizing quantum feature maps from PennyLane and Qiskit, interfaced with a classical SVM from scikit-learn.

**Status:** Active Research & Development / Reproducible Benchmarking Framework  
**Primary Goal:** To provide a statistically grounded, open-source comparison of VQC and QSVM stability and performance on binary classification tasks, emphasizing reproducibility and cross-platform execution.

---

## Technology Stack & Framework Components

| Component                   | Library/Framework                | Description                                  |
|-----------------------------|----------------------------------|----------------------------------------------|
| **VQC Training/Evaluation** | PyTorch, PennyLane               | Backpropagation and quantum differentiation  |
| **QSVM Training/Evaluation**| PennyLane, Qiskit, Scikit-learn  | Quantum kernel computation & classification  |
| **CLI Interface & Logging** | Typer, Rich                      | Clean command execution and formatted logs   |
| **Configuration Management**| PyYAML                           | Configurable experiment parameters           |
| **Numerical Processing**    | NumPy                            | Data preprocessing                           |

---

## Methodology & Model Architecture

<p align="center">
  <img src="assets/images/qml_architecture_diagram.png" alt="Architecture Diagram" width="500">
</p>

### Hybrid Variational Quantum Classifier (VQC)

<p align="center">
  <img src="assets/images/vqc_concept.png" alt="VQC Concept" width="400">
</p>

- **Architecture**: `HybridVariationalClassifier` (in `models.py`) combines a quantum layer implemented with `pennylane.QNode` and classical layers via `torch.nn.Module`.
- **Ansatz**: `build_variational_circuit` (in `qnn_layers.py`) implements a configurable Hardware-Efficient Ansatz.
- **Optimization**: `train_variational_model` (in `training.py`) uses `torch.optim.Adam`.
- **Execution**: `evaluate_vqc` (in `evaluation.py`) computes final metrics on the test set.

---

### Quantum Kernel Support Vector Machine (QSVM)

<p align="center">
  <img src="assets/images/qsvm_concept.png" alt="QSVM Concept" width="400">
</p>

- **Architecture**: `QuantumKernelClassifier` (in `models.py`) integrates `qml.kernels` with `sklearn.svm.SVC`.
- **Feature Map**: `build_kernel_qnode` (in `qnn_layers.py`) defines the `ZZFeatureMap`.
- **Execution**: `evaluate_kernel` (in `evaluation.py`) computes final metrics on the test set.

---

### Main Workflow (in `main.py`)

<p align="center">
  <img src="assets/images/qml_training_workflow.png" alt="Training Workflow" width="600">
</p>

- Provides `train` and `evaluate` commands for both models (`vqc`, `kernel`).
- Integrates Typer for the CLI and `seed.py` for global seed configuration.

### VQC vs QSVM Comparison

<p align="center">
  <img src="assets/images/vqc_vs_qsvm_comparison.png" alt="VQC vs QSVM" width="600">
</p>


---


## Contact & License

- **Author:** Ahmad Rasidi 
- **Email:** rasidi.basit@gmail.com  
- **GitHub:** [https://github.com/rasidi3112](https://github.com/rasidi3112)  

**License:** MIT License  

---

## Project Structure

```plaintex
qml_app/
├─ config/
│  └─ default.yaml
├─ qml_app/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ evaluation.py
│  ├─ main.py
│  ├─ models.py
│  ├─ qnn_layers.py
│  ├─ training.py
│  └─ utils/
│     ├─ __init__.py
│     ├─ config_utils.py
│     ├─ logging_utils.py
│     └─ seed.py
└─ requirements.txt

```

---


## How To Run

### 1. Clone the Repository

```bash
git clone https://github.com/rasidi3112/Quantum-Machine-Learning.git
cd Quantum-Machine-Learning
```
  
### 2. Create and Activate Virtual Environment

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train Models

**a. Hybrid Variational Quantum Classifier (VQC)**
```bash
python -m qml_app.main train --model vqc --config config/default.yaml
```

**b. Quantum Kernel SVM (QSVM)**
```bash
python -m qml_app.main train --model kernel --config config/default.yaml
```

> **Tip:** Modify `config/default.yaml` to change datasets, qubits, layers, batch size, etc.
  

### 5. Evaluate Models

```bash
python -m qml_app.main evaluate --model vqc --config config/default.yaml
python -m qml_app.main evaluate --model kernel --config config/default.yaml
```

<p align="center">
  <img src="assets/images/evaluation_metrics.png" alt="Evaluation Metrics" width="500">
</p>

Evaluation results, including metrics and confusion matrices, are saved in `artifacts/`.

### 6. Additional Notes

**Device Selection:**
- Apple M1/M2 → `device: mps`
- NVIDIA GPU → `device: cuda`
- CPU-only → `device: cpu`

**Shots:**
- `shots=null` for analytic/simulated mode (fast, ideal for CPU)
- `shots=1024` or higher for realistic sampling on quantum hardware

**Artifacts:** Check `artifacts/` for trained models, metrics, ROC curves, and confusion matrices.

> **Note:** Adjust `--device` flag in `config/default.yaml` for CPU or GPU.


### 7. Run Additional Scripts

**Kernel folder:**
```bash
# Convert or preprocess data with convert.py
python3 artifacts/kernel/convert.py

# Run custom scripts
python3 artifacts/kernel/script.py
```

**VQC folder:**
```bash
# Convert PyTorch model to JSON format
python3 artifacts/vqc/convert_pt_to_json.py
```

**Generate boxplots for VQC and QSVM results:**
```bash
python generate_boxplots.py
```




