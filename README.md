# Quantum Machine Learning: Hybrid VQC vs. Quantum Kernel SVM

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.15-orange?logo=pytorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-0.30-lightblue?logo=pytorch&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-0.43-purple?logo=qiskit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This repository contains a **Quantum Machine Learning (QML) project** for learning purposes and binary classification experiments.  
It demonstrates a comparison between:

- **Hybrid Variational Quantum Classifier (VQC)** — hybrid quantum-classical model trained with **PyTorch + PennyLane**.
- **Quantum Kernel Support Vector Machine (QSVM)** — quantum kernel model leveraging **PennyLane + scikit-learn**.
  
 **Status**: Experimental Quantum Machine Learning Project  
 **Key Highlight**: Demonstrates hybrid quantum-classical ML workflows with configurable pipelines and modular design.

---

## Key Features & Highlights

- **Hybrid VQC:** Layered hardware-efficient ansatz, gradient-based optimization via Adam.  
- **Quantum Kernel SVM:** Feature mapping to quantum Hilbert space, kernel-based classical SVM.  
- **Modular architecture:** YAML-config driven, easy to modify datasets and experiments.  
- **Automatic artifacts:** Trained models, metrics, and confusion matrices stored in `artifacts/`.  
- **CLI & Logging:** Powered by **Typer** and **Rich** for clean command-line execution and logging.  
- **Customizable experiments:** Shots, feature layers, variational layers, and early stopping can be tuned.

---

## Technology Stack

| Component                  | Library/Framework       | Notes                             |
|----------------------------|-----------------------|----------------------------------|
| Quantum ML (VQC)           | PennyLane             | Hybrid quantum-classical model   |
| Quantum ML (QSVM)          | PennyLane + Qiskit    | Quantum kernel computation       |
| Classical ML / SVM         | Scikit-learn          | QSVM classifier                  |
| Deep Learning / Optimizer  | PyTorch               | Gradient-based VQC training      |
| CLI & Logging              | Typer, Rich           | Command-line interface & logging |
| Config Management          | PyYAML                | YAML configuration files         |
| Numerical Computation      | NumPy                 | Data preprocessing               |

---

## Methodology

### Hybrid Variational Quantum Classifier (VQC)
- **Ansatz:** Layered Hardware-Efficient Circuit  
- **Feature Encoding:** Classical features mapped to qubits using rotation gates  
- **Optimizer:** Adam  
- **Framework:** PennyLane + PyTorch  
- **Training:** Early stopping, configurable shots, batch-based gradient updates  

### Quantum Kernel SVM (QSVM)
- **Quantum Feature Map:** ZZFeatureMap  
- **Kernel:** Quantum state fidelity  
- **Classifier:** Classical SVM (Scikit-learn)  
- **Framework:** PennyLane + Scikit-learn  

---

## Results and Analysis

| Model                   | Accuracy | F1 Score |
|-------------------------|---------|----------|
| Hybrid VQC              | 70%     | 0.70     |
| Quantum Kernel SVM (QSVM)| 75%     | 0.75     |
| Classical SVM (baseline)| 80%     | 0.81     |

**Note:** QSVM shows slightly better accuracy in this simulation, while VQC allows hands-on experience with **hybrid quantum-classical pipelines**.

![VQC Confusion Matrix](artifacts/vqc_confusion_matrix.png)

> Confusion Matrix example stored in `artifacts/`.

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
