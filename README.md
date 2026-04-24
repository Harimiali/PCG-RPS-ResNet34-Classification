# Reconstructed Phase Space-Based Deep Learning for PCG Classification

This project implements a **nonlinear dynamical deep learning framework** for classifying phonocardiogram (PCG) signals using **Reconstructed Phase Space (RPS)** representations combined with a **ResNet-34 convolutional neural network**.

The method transforms 1D heart sound signals into enriched **2D RGB tensors**, capturing nonlinear dynamics and enabling robust classification of **normal vs abnormal cardiac activity**.

---

## Research Context

This implementation is developed as part of **PhD research work**.

> This code is part of the doctoral research of **Mr. Aghil Kashir**, focusing on advanced signal processing and deep learning methods for biomedical signal analysis.

---

## Overview

Heart sound signals are inherently **nonlinear and nonstationary**, making traditional feature extraction approaches insufficient.

This project addresses this challenge by integrating:

- Nonlinear dynamical analysis (**RPS**)
- Dimensionality reduction (**PCA**, optional)
- Spatial feature enhancement (**Gradient + Laplacian**)
- Deep learning (**ResNet-34**)

---

## Pipeline

The complete processing pipeline:

```

Raw PCG Signal
↓
Preprocessing (Band-pass + Spectral Subtraction)
↓
Reconstructed Phase Space (RPS)
↓
PCA (optional)
↓
2D Tensor (224×224 histogram)
↓
RGB Tensor (Original + Gradient + Laplacian)
↓
ResNet-34 Classifier
↓
Normal / Abnormal Classification

```

---

## Project Structure

```

├── main.py                    # Main execution (train / evaluate / full)
├── requirements.txt
├── README.md
│
├── src/
│   ├── **init**.py
│   ├── preprocessing.py        # Signal preprocessing
│   ├── rps.py                 # Phase space reconstruction + PCA
│   ├── tensor_generation.py   # 2D & RGB tensor creation
│   ├── model.py               # ResNet-34 architecture
│   ├── train.py               # Training pipeline
│   └── evaluate.py            # Evaluation pipeline
│
├── data/                      # Dataset (not included)
│   ├── train/
│   │   ├── normal/
│   │   └── abnormal/
│   └── validation/
│       ├── normal/
│       └── abnormal/
│
└── results/                   # Outputs (models, logs, metrics)

````

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/PCG-RPS-ResNet34-Classification.git
cd PCG-RPS-ResNet34-Classification

pip install -r requirements.txt
````

---

## Dataset

This project uses the **PhysioNet/CinC Challenge PCG dataset**.

* Sampling rate: **2000 Hz**
* Classes: **Normal / Abnormal**
* Format: `.wav`

> Dataset is not included. Please download from PhysioNet.

---

## Usage

### Run full pipeline (train + evaluate)

```bash
python main.py --mode full
```

---

### Train only

```bash
python main.py --mode train
```

---

### Evaluate only

```bash
python main.py --mode evaluate --model_path results/run_xxx/models/best_model.keras
```

---

## Ablation Study

You can disable individual components to evaluate their impact:

```bash
# Without PCA
python main.py --mode full --no_pca

# Without RGB
python main.py --mode full --no_rgb

# Without noise reduction
python main.py --mode full --no_noise_reduction

# Disable Sobel channel
python main.py --mode full --no_sobel

# Disable Laplacian channel
python main.py --mode full --no_laplacian
```

---

## Evaluation Metrics

The system evaluates:

* Accuracy
* Precision
* Recall (Sensitivity)
* Specificity
* **F1 Score (primary metric)**
* AUC (optional)

Outputs are saved in:

```
results/
└── run_xxx/
    ├── models/
    ├── evaluation/
    │   ├── metrics.json
    │   ├── predictions.csv
    │   ├── confusion_matrix.png
    │   └── classification_report.txt
```

---

## GPU Support

The model automatically uses GPU if available.

Check GPU:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## Key Features

✔ Nonlinear feature extraction using RPS
✔ Flexible and modular architecture
✔ Full ablation capability
✔ GPU-compatible training
✔ Research-grade evaluation pipeline
✔ Robust to noise and signal variability

---

## Key Insight

Unlike traditional approaches, this method:

> Learns directly from **nonlinear dynamics of heart signals**, rather than relying only on linear or time-frequency features.

---

## Notes

* Always run from project root:

```bash
python main.py --mode full
```

* Ensure correct dataset structure

* Required file:

```
src/__init__.py
```

---

## Future Work

* Attention mechanisms in RPS space
* Lightweight architectures for real-time systems
* Adaptive noise reduction
* Multimodal biomedical signal fusion

---

## Citation

If you use this work, please cite:

```
Reconstructed Phase Space-Based Deep Learning Framework for Detecting Cardiac Abnormalities from PCG Signals,
2026, Discovered Artificial Intelligence
```

---

## Contributors

* **Mr. Aghil Kashir** — PhD Researcher, and other authors of the paper

---

## License

This project is intended for **research and academic purposes**.

