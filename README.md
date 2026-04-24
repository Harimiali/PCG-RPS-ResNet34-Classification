

# Reconstructed Phase Space-Based Deep Learning for PCG Classification

This project implements a **nonlinear dynamical framework** for classifying phonocardiogram (PCG) signals using **Reconstructed Phase Space (RPS)** representations combined with **deep convolutional neural networks (ResNet-34)**.

The method transforms 1D heart sound signals into enriched 2D RGB tensors that capture nonlinear dynamics, enabling accurate classification of **normal vs abnormal cardiac activity**.

---

## Overview

Heart sound signals are inherently **nonlinear and nonstationary**, making traditional signal processing methods insufficient.

This project addresses this challenge by integrating:

- Nonlinear dynamical analysis (RPS)
- Dimensionality reduction (PCA)
- Spatial feature enhancement (Gradient + Laplacian)
- Deep learning (ResNet-34)

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
PCA (Dimensionality Reduction)
↓
2D Tensor (224×224)
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

├── src/
│   ├── preprocessing.py        # Signal denoising
│   ├── rps.py                 # Phase space reconstruction
│   ├── tensor_generation.py   # 2D and RGB tensor creation
│   ├── model.py               # ResNet-34 architecture
│   ├── train.py               # Training pipeline
│   └── evaluate.py            # Metrics calculation
│
├── data/                      # Dataset (not included)
├── results/                   # Outputs and logs
├── main.py                    # Entry point
├── requirements.txt
└── README.md

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

* Sampling rate: 2000 Hz
* Classes: Normal / Abnormal
* Format: `.wav`

>  Dataset is not included. Please download from PhysioNet.

---

## Training

```bash
python main.py
```

Or use:

```python
from src.train import train_model
train_model()
```

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score

The model focuses on **F1-score** due to class imbalance.

---

## Key Features

✔ Nonlinear feature extraction using RPS
✔ Multi-channel representation (RGB tensor)
✔ Deep feature learning via ResNet-34
✔ Robust to noise and variability
✔ Modular and extendable pipeline

---

## Ablation Study (Optional)

You can test:

* Without PCA
* Without RGB enhancement
* Without preprocessing

To evaluate contribution of each stage.

---

## Key Insight

Unlike traditional approaches, this method:

> Learns directly from **nonlinear dynamics of heart signals**, not just linear time-frequency features.

---

## Future Work

* Attention-based region selection in RPS
* Lightweight models for real-time deployment
* Improved noise suppression
* Extension to other biomedical signals

---

## Citation

If you use this code, please cite:

```
Reconstructed Phase Space-Based Deep Learning Framework for Detecting Cardiac Abnormalities from PCG Signals, 2026, Discovered Artificial Intelligence
```

