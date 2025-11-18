# Quantum Machine Learning: Hybrid VQC vs. Quantum Kernel SVM

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.15-orange?logo=pytorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-0.30-lightblue?logo=pytorch&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-0.43-purple?logo=qiskit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

# Overview

This repository hosts an Advanced Quantum Machine Learning (QML) project designed for rigorous comparative analysis and reproducible research. It implements and benchmarks two leading NISQ-era paradigms:

**Hybrid Variational Quantum Classifier (VQC):** A trainable quantum-classical hybrid model optimized end-to-end using PyTorch and PennyLane.

**Quantum Kernel Support Vector Machine (QSVM):** A kernel-based model utilizing quantum feature maps from PennyLane and Qiskit, interfaced with a classical SVM from scikit-learn.

**Status:** Active Research & Development / Reproducible Benchmarking Framework  
**Primary Goal:** To provide a statistically grounded, open-source comparison of VQC and QSVM stability and performance on binary classification tasks, emphasizing reproducibility and cross-platform execution.

---

## Key Highlights & Features

**Statistically Robust Benchmarking:** Conducted six independent runs (random seeds 1001–1006) to quantify model variance and expose instability, moving beyond single-run demonstrations.

**Reproducible Pipeline:** Enforces deterministic seeding, YAML-driven configuration (config/default.yaml), and automatic artifact logging (artifacts/) for full result reproducibility.

**Cross-Platform Execution:** Seamlessly executes on CPU, Apple Silicon (MPS, with fallback), and CUDA, ensuring consistent results across diverse hardware.

**Modular Architecture:** Features a clean separation of concerns (data, models, training, evaluation) for easy modification and extension.

**Comprehensive Evaluation:** Reports accuracy, F1-score, and ROC-AUC, with automatic generation of confusion matrices and ROC curves.

**Open Science Commitment:** Fully open-source, documented, and designed for independent verification and extension by the research community.

---

## Empirical Findings (Based on 6-Run Study)

- VQC demonstrates high performance with variance (Mean Test Accuracy: ~81.1 percent, F1 ~0.80).
- QSVM exhibits extreme instability (Mean F1: ~0.22), failing completely in 4 out of 6 runs (F1 = 0.00).
- A classical SVM baseline consistently outperforms both QML models.

---

## Technology Stack
## Framework Components

| Komponen                     | Library/Framework                | Catatan                                      |
|-----------------------------|----------------------------------|----------------------------------------------|
| **Pelatihan/Evaluasi VQC**  | PyTorch, PennyLane               | Backpropagation dan diferensiasi kuantum     |
| **Pelatihan/Evaluasi QSVM** | PennyLane, Qiskit, Scikit-learn  | Quantum kernel computation & classification  |
| **Antarmuka CLI & Logging** | Typer, Rich                      | Eksekusi perintah yang bersih dan log berformat |
| **Manajemen Konfigurasi**   | PyYAML                           | Parameter eksperimen yang dapat dikonfigurasi |
| **Pemrosesan Numerik**      | NumPy                            | Pra-pemrosesan data                           |

---

## Methodology

## Model Architecture Overview

### Hybrid Variational Quantum Classifier (VQC)

- **Architecture**: `HybridVariationalClassifier` (in `models.py`) combines a quantum layer implemented with `pennylane.QNode` and classical layers via `torch.nn.Module`.
- **Ansatz**: `build_variational_circuit` (in `qnn_layers.py`) implements a configurable Hardware-Efficient Ansatz.
- **Optimization**: `train_variational_model` (in `training.py`) uses `torch.optim.Adam`.
- **Execution**: `evaluate_vqc` (in `evaluation.py`) computes final metrics on the test set.

---

### Quantum Kernel Support Vector Machine (QSVM)

- **Architecture**: `QuantumKernelClassifier` (in `models.py`) integrates `qml.kernels` with `sklearn.svm.SVC`.
- **Feature Map**: `build_kernel_qnode` (in `qnn_layers.py`) defines the `ZZFeatureMap`.
- **Execution**: `evaluate_kernel` (in `evaluation.py`) computes final metrics on the test set.

---

### Main Workflow (in `main.py`)

- Provides `train` and `evaluate` commands for both models (`vqc`, `kernel`).
- Integrates Typer for the CLI and `seed.py` for global seed configuration.


---


## Contact & License

- **Author:** Ahmad Rasidi 
- **Email:** rasidi.basit@gmail.com  
- **GitHub:** [https://github.com/rasidi3112](https://github.com/rasidi3112)  

**License:** MIT License  

**Disclaimer:** This repository is intended for experimentation and practical exploration of Quantum Machine Learning concepts.

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


---


How To Run
1. Clone the Repository
  git clone https://github.com/rasidi3112/Quantum-Machine-Learning.git
  cd Quantum-Machine-Learning
  
2. Create and Activate Virtual Environment
  # macOS / Linux
  python -m venv .venv
  source .venv/bin/activate
  
  # Windows
  python -m venv .venv
  .venv\Scripts\activate

3. Install Dependencies
  pip install --upgrade pip
  pip install -r requirements.txt

4. Train Models
  a. Hybrid Variational Quantum Classifier (VQC)
      Run :
      python -m qml_app.main train --model vqc --config config/default.yaml

  b. Quantum Kernel SVM (QSVM)
      Run :
      python -m qml_app.main train --model kernel --config config/default.yaml

        Tip: Modify config/default.yaml to change datasets, qubits, layers, batch size, etc.
  

5. Evaluate Models
      Run :
      python -m qml_app.main evaluate --model vqc --config config/default.yaml
      python -m qml_app.main evaluate --model kernel --config config/default.yaml

  Evaluation results, including metrics and confusion matrices, are saved in artifacts/.

6. Additional Notes
    Device Selection:
      - Apple M1/M2 → device: mps
      - NVIDIA GPU → device: cuda
      - CPU-only → device: cpu
    Shots:
    - nshots=null for analytic/simulated mode (fast, ideal for CPU)
    - shots=1024 or higher for realistic sampling on quantum hardware
    Artifacts: Check artifacts/ for trained models, metrics, ROC curves, and confusion matrices

# Note: Adjust --device flag in config/default.yaml for CPU or GPU.


7. Run Additional Scripts
Kernel folder 
# Convert or preprocess data with convert.py
python3 artifacts/kernel/convert.py

# Run custom scripts
python3 artifacts/kernel/script.py

VQC folder
# Convert PyTorch model to JSON format
python3 artifacts/vqc/convert_pt_to_json.py






