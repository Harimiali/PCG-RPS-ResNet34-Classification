"""
evaluate.py

Complete evaluation pipeline for PCG classification.

This file:
- Loads a trained model
- Loads WAV files from normal/abnormal folders
- Applies the same preprocessing + RPS + PCA + tensor generation pipeline
- Predicts normal/abnormal labels
- Calculates accuracy, precision, recall, F1-score
- Saves confusion matrix, metrics, and predictions

--------------------------------------------------
INSTALLATION:

pip install numpy scipy scikit-learn opencv-python tensorflow matplotlib pandas

--------------------------------------------------
EXPECTED DATA STRUCTURE:

data/
└── validation/
    ├── normal/
    │   ├── sample1.wav
    │   └── sample2.wav
    └── abnormal/
        ├── sample3.wav
        └── sample4.wav

--------------------------------------------------
NORMAL USE:

from src.evaluate import EvaluationConfig, evaluate_pipeline

config = EvaluationConfig(
    data_dir="data/validation",
    model_path="results/models/best_model.keras",
    output_dir="results/evaluation"
)

metrics = evaluate_pipeline(config)

--------------------------------------------------
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from src.train import (
    TrainConfig,
    list_wav_files,
    process_single_file
)

from src.model import enable_gpu_memory_growth


@dataclass
class EvaluationConfig:
    """
    Evaluation configuration.
    """

    # Paths
    data_dir: str = "data/validation"
    model_path: str = "results/models/best_model.keras"
    output_dir: str = "results/evaluation"

    # Prediction
    threshold: float = 0.5
    batch_size: int = 32

    # Audio and preprocessing settings
    target_sampling_rate: int = 2000
    max_signal_duration_sec: Optional[float] = None

    lowcut: float = 15.0
    highcut: float = 800.0
    filter_order: int = 3
    use_spectral_subtraction: bool = True
    noise_reduction_factor: float = 0.5
    normalize_signal: bool = True

    # RPS + PCA settings
    use_rps: bool = True
    rps_dimension: int = 4
    rps_delay: int = 3
    use_pca: bool = True
    pca_components: int = 2
    standardize_before_pca: bool = True

    # Tensor settings
    image_size: int = 224
    use_rgb: bool = True
    use_sobel: bool = True
    use_laplacian: bool = True
    log_transform: bool = True

    # Saving options
    save_predictions_csv: bool = True
    save_metrics_json: bool = True
    save_confusion_matrix_image: bool = True
    save_classification_report: bool = True


def create_evaluation_dirs(output_dir: str) -> None:
    """
    Create output directory for evaluation results.
    """

    os.makedirs(output_dir, exist_ok=True)


def save_evaluation_config(config: EvaluationConfig) -> None:
    """
    Save evaluation configuration as JSON.
    """

    path = os.path.join(config.output_dir, "evaluation_config.json")

    with open(path, "w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=4)


def convert_eval_config_to_train_config(
    config: EvaluationConfig
) -> TrainConfig:
    """
    Convert evaluation config to TrainConfig so we can reuse
    the exact same processing function from train.py.
    """

    train_config = TrainConfig(
        target_sampling_rate=config.target_sampling_rate,
        max_signal_duration_sec=config.max_signal_duration_sec,

        lowcut=config.lowcut,
        highcut=config.highcut,
        filter_order=config.filter_order,
        use_spectral_subtraction=config.use_spectral_subtraction,
        noise_reduction_factor=config.noise_reduction_factor,
        normalize_signal=config.normalize_signal,

        use_rps=config.use_rps,
        rps_dimension=config.rps_dimension,
        rps_delay=config.rps_delay,
        use_pca=config.use_pca,
        pca_components=config.pca_components,
        standardize_before_pca=config.standardize_before_pca,

        image_size=config.image_size,
        use_rgb=config.use_rgb,
        use_sobel=config.use_sobel,
        use_laplacian=config.use_laplacian,
        log_transform=config.log_transform,

        batch_size=config.batch_size
    )

    return train_config


def load_evaluation_data(
    config: EvaluationConfig
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and process evaluation WAV files.
    """

    file_paths, labels = list_wav_files(config.data_dir)

    processing_config = convert_eval_config_to_train_config(config)

    x_data = []
    y_data = []
    valid_file_paths = []

    total_files = len(file_paths)

    for index, (file_path, label) in enumerate(zip(file_paths, labels), start=1):
        try:
            tensor = process_single_file(file_path, processing_config)

            x_data.append(tensor)
            y_data.append(label)
            valid_file_paths.append(file_path)

            if index % 25 == 0 or index == total_files:
                print(f"Processed {index}/{total_files} files")

        except Exception as error:
            print(f"Skipping file: {file_path}")
            print(f"Error: {error}")

    if len(x_data) == 0:
        raise RuntimeError("No valid evaluation samples were processed.")

    x_array = np.stack(x_data, axis=0).astype(np.float32)
    y_array = np.asarray(y_data, dtype=np.int32)

    return x_array, y_array, valid_file_paths


def load_trained_model(model_path: str) -> tf.keras.Model:
    """
    Load trained Keras model.
    """

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    return model


def predict_labels(
    model: tf.keras.Model,
    x_data: np.ndarray,
    threshold: float = 0.5,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict probabilities and binary labels.

    Returns:
        y_prob: probability of abnormal class
        y_pred: predicted class, 0 normal, 1 abnormal
    """

    if threshold <= 0 or threshold >= 1:
        raise ValueError("threshold must be between 0 and 1.")

    y_prob = model.predict(
        x_data,
        batch_size=batch_size,
        verbose=1
    )

    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 2 and y_prob.shape[1] == 1:
        y_prob = y_prob[:, 0]

    elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]

    else:
        y_prob = y_prob.reshape(-1)

    y_pred = (y_prob >= threshold).astype(np.int32)

    return y_prob.astype(np.float32), y_pred


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics.

    Abnormal class is positive class = 1.
    """

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )
    recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = recall

    auc_value = None

    if y_prob is not None:
        try:
            auc_value = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_value = None

    metrics_dict = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "auc": None if auc_value is None else float(auc_value),

        "confusion_matrix": {
            "tn_normal_predicted_normal": int(tn),
            "fp_normal_predicted_abnormal": int(fp),
            "fn_abnormal_predicted_normal": int(fn),
            "tp_abnormal_predicted_abnormal": int(tp)
        },

        "support": {
            "normal": int(np.sum(y_true == 0)),
            "abnormal": int(np.sum(y_true == 1)),
            "total": int(len(y_true))
        }
    }

    return metrics_dict


def save_metrics(
    metrics_dict: Dict[str, Any],
    config: EvaluationConfig
) -> None:
    """
    Save metrics to JSON.
    """

    path = os.path.join(config.output_dir, "metrics.json")

    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics_dict, file, indent=4)


def save_predictions(
    file_paths: List[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    config: EvaluationConfig
) -> None:
    """
    Save prediction results as CSV.
    """

    records = []

    for file_path, true_label, probability, predicted_label in zip(
        file_paths,
        y_true,
        y_prob,
        y_pred
    ):
        records.append({
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "true_label": int(true_label),
            "true_class": "abnormal" if true_label == 1 else "normal",
            "predicted_label": int(predicted_label),
            "predicted_class": "abnormal" if predicted_label == 1 else "normal",
            "abnormal_probability": float(probability),
            "correct": bool(true_label == predicted_label)
        })

    dataframe = pd.DataFrame(records)

    path = os.path.join(config.output_dir, "predictions.csv")
    dataframe.to_csv(path, index=False)


def save_classification_report_file(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: EvaluationConfig
) -> None:
    """
    Save detailed classification report.
    """

    report = classification_report(
        y_true,
        y_pred,
        target_names=["normal", "abnormal"],
        digits=4,
        zero_division=0
    )

    path = os.path.join(config.output_dir, "classification_report.txt")

    with open(path, "w", encoding="utf-8") as file:
        file.write(report)


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: EvaluationConfig
) -> None:
    """
    Save confusion matrix image.
    """

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Normal", "Abnormal"])
    ax.set_yticklabels(["Normal", "Abnormal"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(image)
    fig.tight_layout()

    path = os.path.join(config.output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=300)
    plt.close(fig)


def print_metrics(metrics_dict: Dict[str, Any]) -> None:
    """
    Print metrics in a clean format.
    """

    print("\nEvaluation Results")
    print("-" * 40)
    print(f"Accuracy:     {metrics_dict['accuracy']:.4f}")
    print(f"Precision:    {metrics_dict['precision']:.4f}")
    print(f"Recall:       {metrics_dict['recall_sensitivity']:.4f}")
    print(f"Specificity:  {metrics_dict['specificity']:.4f}")
    print(f"F1 Score:     {metrics_dict['f1_score']:.4f}")

    if metrics_dict["auc"] is not None:
        print(f"AUC:          {metrics_dict['auc']:.4f}")

    print("\nConfusion Matrix")
    print("-" * 40)
    cm = metrics_dict["confusion_matrix"]
    print(f"TN: {cm['tn_normal_predicted_normal']}")
    print(f"FP: {cm['fp_normal_predicted_abnormal']}")
    print(f"FN: {cm['fn_abnormal_predicted_normal']}")
    print(f"TP: {cm['tp_abnormal_predicted_abnormal']}")


def evaluate_pipeline(
    config: Optional[EvaluationConfig] = None
) -> Dict[str, Any]:
    """
    Full evaluation pipeline.
    """

    if config is None:
        config = EvaluationConfig()

    enable_gpu_memory_growth()
    create_evaluation_dirs(config.output_dir)
    save_evaluation_config(config)

    print("Loading trained model...")
    model = load_trained_model(config.model_path)

    print("Loading and processing evaluation data...")
    x_data, y_true, file_paths = load_evaluation_data(config)

    print("Evaluation tensor shape:", x_data.shape)

    print("Predicting...")
    y_prob, y_pred = predict_labels(
        model=model,
        x_data=x_data,
        threshold=config.threshold,
        batch_size=config.batch_size
    )

    metrics_dict = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob
    )

    print_metrics(metrics_dict)

    if config.save_metrics_json:
        save_metrics(metrics_dict, config)

    if config.save_predictions_csv:
        save_predictions(
            file_paths=file_paths,
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            config=config
        )

    if config.save_classification_report:
        save_classification_report_file(
            y_true=y_true,
            y_pred=y_pred,
            config=config
        )

    if config.save_confusion_matrix_image:
        plot_and_save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            config=config
        )

    print("\nEvaluation completed.")
    print(f"Results saved to: {config.output_dir}")

    return metrics_dict


if __name__ == "__main__":

    config = EvaluationConfig(
        data_dir="data/validation",
        model_path="results/models/best_model.keras",
        output_dir="results/evaluation",

        threshold=0.5,
        batch_size=32,

        target_sampling_rate=2000,

        lowcut=15.0,
        highcut=800.0,
        filter_order=3,
        use_spectral_subtraction=True,
        noise_reduction_factor=0.5,
        normalize_signal=True,

        use_rps=True,
        rps_dimension=4,
        rps_delay=3,
        use_pca=True,
        pca_components=2,
        standardize_before_pca=True,

        image_size=224,
        use_rgb=True,
        use_sobel=True,
        use_laplacian=True,
        log_transform=True
    )

    evaluate_pipeline(config)
