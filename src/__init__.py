"""
src package

This file makes the 'src' directory a Python package
and provides clean access to main modules.

--------------------------------------------------
USAGE EXAMPLES:

from src import preprocess_signal
from src import rps_pca_pipeline
from src import tensor_generation_pipeline
from src import build_model

--------------------------------------------------
"""

# Preprocessing
from .preprocessing import preprocess_signal

# RPS + PCA
from .rps import RPSConfig, rps_pca_pipeline

# Tensor generation
from .tensor_generation import TensorConfig, tensor_generation_pipeline

# Model
from .model import ModelConfig, build_model

# Training & Evaluation (optional exposure)
from .train import TrainConfig, train_pipeline
from .evaluate import EvaluationConfig, evaluate_pipeline


__all__ = [
    # Preprocessing
    "preprocess_signal",

    # RPS
    "RPSConfig",
    "rps_pca_pipeline",

    # Tensor
    "TensorConfig",
    "tensor_generation_pipeline",

    # Model
    "ModelConfig",
    "build_model",

    # Training
    "TrainConfig",
    "train_pipeline",

    # Evaluation
    "EvaluationConfig",
    "evaluate_pipeline",
]
