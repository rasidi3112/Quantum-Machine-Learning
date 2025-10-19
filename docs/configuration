
Deep Documentation – default.yaml Configuration
This document provides a comprehensive explanation of the default.yaml configuration file used in the Quantum Machine Learning project. It includes dataset, model, training, and evaluation settings, as well as guidelines for running the project on laptops other than MacBook M1.

1.Data Configuration (data)
---
data:
  dataset: moons           # options: moons, circles, breast_cancer
  n_samples: 300           # Reduced from 900 → 300 to avoid heavy computation in QML kernels
  noise: 0.25
  test_size: 0.2
  val_size: 0.15
  random_state: 123
  feature_scaler: standard # options: standard, minmax, robust
  feature_expansion: polynomial  # options: polynomial, none
  expansion_degree: 3
  n_qubits: 4              # fixed at 4 for stable simulation on M1
---

Parameter Details:
- dataset: Dataset selection. Synthetic datasets (moons, circles) are lightweight for quick experiments; breast_cancer requires more memory.
- n_samples: Controls the number of samples; reduced for efficient QML kernel computation.
- noise: Noise level for synthetic datasets.
- test_size & val_size: Splitting proportions for test and validation sets.
- random_state: Seed for reproducibility.
- feature_scaler: Method for feature scaling; choose based on dataset characteristics.
- feature_expansion & expansion_degree: Polynomial expansion increases feature dimensionality; higher degrees require more computational resources.
- n_qubits: Number of qubits in the quantum circuit; fixed to 4 for simulation efficiency.

Notes for Other Laptops:
Increase n_samples or expansion_degree only if CPU/GPU and RAM can handle higher complexity.
Adjust n_qubits if a larger simulator or quantum device is available.

2. Model Configuration (model)
---
model:
  shots: null              # Changed from 1024 → null for analytic mode (faster simulation)
  feature_layers: 2
  variational_layers: 3
  hidden_dim: 48
  dropout: 0.15
  kernel_regularization: 1.0
  use_complex_device: false  # Keep false; M1 does not require complex device
--- 
Parameter Details:
- shots: Number of circuit measurements. null enables analytic mode → faster. On other simulators or GPUs, shots=1024 can be used for realistic sampling.
- feature_layers & variational_layers: Depth of feature mapping and variational layers in the quantum circuit. More layers increase expressive power but require more resources.
- hidden_dim & dropout: Classical layer dimension and dropout for regularization.
- kernel_regularization: Regularization coefficient for the quantum kernel.
- use_complex_device: Whether to use a complex quantum device. False for M1; True can be tested on other platforms.

3. Training Configuration (training)
---
training:
  epochs: 40               # Reduced from 60 → 40 to save time; stable performance
  batch_size: 16           # Reduced from 32 → 16 to reduce memory load
  learning_rate: 0.004
  weight_decay: 0.0008
  gradient_clip: 1.0
  patience: 8
  min_delta: 0.001
  device: mps              # Use Metal Performance Shaders (M1 GPU)
  log_interval: 10
---
Parameter Details:
- epochs & batch_size: Reduced for faster training and lower memory consumption on M1. Increase on stronger hardware.
- learning_rate & weight_decay: Optimizer hyperparameters.
- gradient_clip: Prevents exploding gradients; keeps training stable.
- patience & min_delta: Early stopping criteria to prevent overfitting.
- device: Training device. Options:
          M1/M2 /Apple silicon : mps
          NVIDIA GPU : cuda
          CPU-only : cpu
- log_interval: Interval (in batches) for printing training logs.

4. Evaluation Configuration (evaluation)
---
evaluation:
  save_path: "artifacts"
  roc_curve_points: 200
  confusion_matrix: true
  report_average: "weighted"
---
Parameter Details:
- save_path: Directory to save model checkpoints, metrics, and artifacts.
- roc_curve_points: Number of points for ROC curve generation; can be reduced for low-performance machines.
- confusion_matrix: Generate confusion matrix.
- report_average: Weighted average for classification metrics (weighted, macro, micro)

5. MacBook M1 Specific Notes
- device: mps utilizes Apple’s Metal GPU for acceleration.
- n_samples=300 and shots=null ensure quick kernel evaluation and stable memory usage.
- feature_layers=2 and variational_layers=3 provide a moderate balance between model capacity and computational cost.

Running on Other Laptops:
Replace device with cuda (GPU) or cpu.
Increase shots to 1024–2048 if realistic sampling is required.
Adjust batch_size and epochs based on available RAM and CPU/GPU performance.


6. Professional Recommendations
    a. CPU-only Laptops: Use smaller batch sizes (8–16) and moderate epochs (20–40).
    b.CUDA-enabled GPU Laptops: Increase shots, batch_size, and circuit layers for higher accuracy.
    c.Memory Management: Polynomial feature expansion (expansion_degree>3) increases memory footprint; monitor RAM usage.
    d.Simulation Mode: shots=null provides analytic results and speeds up simulations; ideal for initial experiments or CPU-only environments.
