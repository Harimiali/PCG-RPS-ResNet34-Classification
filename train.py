"""
train.py

Complete training pipeline for PCG classification using:

Raw PCG WAV files
    -> preprocessing
    -> RPS + PCA
    -> 224x224x3 RGB tensor
    -> ResNet-34 classifier
    -> training and model saving

--------------------------------------------------
INSTALLATION:

pip install numpy scipy scikit-learn opencv-python tensorflow

--------------------------------------------------
EXPECTED DATA STRUCTURE:

data/
├── train/
│   ├── normal/
│   │   ├── sample1.wav
│   │   └── sample2.wav
│   └── abnormal/
│       ├── sample3.wav
│       └── sample4.wav
│
└── validation/
    ├── normal/
    └── abnormal/

--------------------------------------------------
NORMAL USE:

from src.train import TrainConfig, train_pipeline

config = TrainConfig(
    train_dir="data/train",
    validation_dir="data/validation",
    output_dir="results",
    epochs=50,
    batch_size=32
)

history, model = train_pipeline(config)

--------------------------------------------------
"""

import os
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.utils.class_weight import compute_class_weight

from src.preprocessing import preprocess_signal
from src.rps import RPSConfig, rps_pca_pipeline
from src.tensor_generation import TensorConfig, tensor_generation_pipeline
from src.model import ModelConfig, build_model, enable_gpu_memory_growth


@dataclass
class TrainConfig:
    """
    Main training configuration.
    """

    # Dataset paths
    train_dir: str = "data/train"
    validation_dir: str = "data/validation"
    output_dir: str = "results"

    # Audio
    target_sampling_rate: int = 2000
    max_signal_duration_sec: Optional[float] = None

    # Preprocessing
    lowcut: float = 15.0
    highcut: float = 800.0
    filter_order: int = 3
    use_spectral_subtraction: bool = True
    noise_reduction_factor: float = 0.5
    normalize_signal: bool = True

    # RPS + PCA
    use_rps: bool = True
    rps_dimension: int = 4
    rps_delay: int = 3
    use_pca: bool = True
    pca_components: int = 2
    standardize_before_pca: bool = True

    # Tensor generation
    image_size: int = 224
    use_rgb: bool = True
    use_sobel: bool = True
    use_laplacian: bool = True
    log_transform: bool = True

    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    shuffle_buffer_size: int = 512
    use_class_weights: bool = True

    # Model
    dropout_rate: float = 0.3
    fc_units: int = 1000
    use_l2_regularization: bool = False
    l2_factor: float = 1e-4

    # Runtime
    seed: int = 42
    cache_dataset: bool = False
    prefetch: bool = True
    save_best_only: bool = True


def set_global_seed(seed: int) -> None:
    """
    Set seeds for reproducible training.
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_output_dirs(output_dir: str) -> Dict[str, str]:
    """
    Create output directories.
    """

    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, "models")
    log_dir = os.path.join(output_dir, "logs")
    history_dir = os.path.join(output_dir, "history")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    return {
        "model_dir": model_dir,
        "log_dir": log_dir,
        "history_dir": history_dir
    }


def list_wav_files(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    List WAV files from normal and abnormal folders.

    Labels:
        normal   -> 0
        abnormal -> 1
    """

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    class_map = {
        "normal": 0,
        "abnormal": 1
    }

    file_paths = []
    labels = []

    for class_name, label in class_map.items():
        class_dir = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected folder not found: {class_dir}"
            )

        for filename in os.listdir(class_dir):
            if filename.lower().endswith(".wav"):
                file_paths.append(os.path.join(class_dir, filename))
                labels.append(label)

    if len(file_paths) == 0:
        raise ValueError(f"No WAV files found in {data_dir}")

    return file_paths, labels


def read_wav_file(file_path: str) -> Tuple[int, np.ndarray]:
    """
    Read WAV file and convert to mono float32 signal.
    """

    sampling_rate, signal = wavfile.read(file_path)

    signal = np.asarray(signal)

    if signal.ndim == 2:
        signal = np.mean(signal, axis=1)

    if np.issubdtype(signal.dtype, np.integer):
        max_value = np.iinfo(signal.dtype).max
        signal = signal.astype(np.float32) / max_value
    else:
        signal = signal.astype(np.float32)

    return sampling_rate, signal


def resample_signal(
    signal: np.ndarray,
    original_sampling_rate: int,
    target_sampling_rate: int
) -> np.ndarray:
    """
    Resample signal to target sampling rate.
    """

    if original_sampling_rate == target_sampling_rate:
        return signal.astype(np.float32)

    gcd = np.gcd(original_sampling_rate, target_sampling_rate)

    up = target_sampling_rate // gcd
    down = original_sampling_rate // gcd

    resampled = resample_poly(signal, up, down)

    return resampled.astype(np.float32)


def trim_or_keep_signal(
    signal: np.ndarray,
    sampling_rate: int,
    max_duration_sec: Optional[float]
) -> np.ndarray:
    """
    Optionally limit signal length.
    """

    if max_duration_sec is None:
        return signal

    max_samples = int(max_duration_sec * sampling_rate)

    if max_samples <= 0:
        raise ValueError("max_signal_duration_sec must be positive.")

    return signal[:max_samples]


def process_single_file(
    file_path: str,
    config: TrainConfig
) -> np.ndarray:
    """
    Convert one WAV file to final CNN tensor.
    """

    original_fs, raw_signal = read_wav_file(file_path)

    signal = resample_signal(
        signal=raw_signal,
        original_sampling_rate=original_fs,
        target_sampling_rate=config.target_sampling_rate
    )

    signal = trim_or_keep_signal(
        signal=signal,
        sampling_rate=config.target_sampling_rate,
        max_duration_sec=config.max_signal_duration_sec
    )

    signal = preprocess_signal(
        signal=signal,
        sampling_rate=config.target_sampling_rate,
        lowcut=config.lowcut,
        highcut=config.highcut,
        filter_order=config.filter_order,
        apply_spectral_subtraction=config.use_spectral_subtraction,
        noise_reduction_factor=config.noise_reduction_factor,
        normalize=config.normalize_signal
    )

    rps_config = RPSConfig(
        use_rps=config.use_rps,
        dimension=config.rps_dimension,
        delay=config.rps_delay,
        use_pca=config.use_pca,
        pca_components=config.pca_components,
        standardize_before_pca=config.standardize_before_pca,
        return_pca_model=False,
        return_scaler=False
    )

    rps_result = rps_pca_pipeline(signal, rps_config)

    tensor_config = TensorConfig(
        image_size=config.image_size,
        normalize_tensor=True,
        log_transform=config.log_transform,
        use_rgb=config.use_rgb,
        use_sobel=config.use_sobel,
        use_laplacian=config.use_laplacian,
        output_dtype="float32"
    )

    tensor_result = tensor_generation_pipeline(
        data=rps_result["final_data"],
        config=tensor_config
    )

    final_tensor = tensor_result["final_tensor"]

    if final_tensor.shape != (config.image_size, config.image_size, 3):
        raise ValueError(
            f"Invalid tensor shape {final_tensor.shape}. "
            f"Expected {(config.image_size, config.image_size, 3)}."
        )

    return final_tensor.astype(np.float32)


def load_dataset_to_memory(
    file_paths: List[str],
    labels: List[int],
    config: TrainConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and process all files into memory.

    This is simple and reliable for medium-size datasets.
    """

    x_data = []
    y_data = []

    total_files = len(file_paths)

    for index, (file_path, label) in enumerate(zip(file_paths, labels), start=1):
        try:
            tensor = process_single_file(file_path, config)
            x_data.append(tensor)
            y_data.append(label)

            if index % 25 == 0 or index == total_files:
                print(f"Processed {index}/{total_files} files")

        except Exception as error:
            print(f"Skipping file due to error: {file_path}")
            print(f"Error: {error}")

    if len(x_data) == 0:
        raise RuntimeError("No valid samples were processed.")

    x_array = np.stack(x_data, axis=0).astype(np.float32)
    y_array = np.asarray(y_data, dtype=np.float32)

    return x_array, y_array


def create_tf_dataset(
    x_data: np.ndarray,
    y_data: np.ndarray,
    batch_size: int,
    shuffle: bool,
    shuffle_buffer_size: int,
    cache: bool,
    prefetch: bool
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset.
    """

    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(shuffle_buffer_size, len(x_data)),
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(batch_size)

    if cache:
        dataset = dataset.cache()

    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def compute_weights(labels: np.ndarray) -> Optional[Dict[int, float]]:
    """
    Compute class weights for imbalanced data.
    """

    unique_classes = np.unique(labels.astype(int))

    if len(unique_classes) < 2:
        return None

    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels.astype(int)
    )

    return {
        int(class_id): float(weight)
        for class_id, weight in zip(unique_classes, weights)
    }


def create_callbacks(
    output_dirs: Dict[str, str],
    config: TrainConfig
) -> List[tf.keras.callbacks.Callback]:
    """
    Create training callbacks.
    """

    best_model_path = os.path.join(
        output_dirs["model_dir"],
        "best_model.keras"
    )

    final_checkpoint_path = os.path.join(
        output_dirs["model_dir"],
        "checkpoint_epoch_{epoch:03d}.keras"
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path if config.save_best_only else final_checkpoint_path,
            monitor="val_loss",
            save_best_only=config.save_best_only,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(output_dirs["log_dir"], "training_log.csv"),
            append=False
        )
    ]

    return callbacks


def save_config(config: TrainConfig, output_dir: str) -> None:
    """
    Save training configuration.
    """

    config_path = os.path.join(output_dir, "train_config.json")

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=4)


def save_history(
    history: tf.keras.callbacks.History,
    history_dir: str
) -> None:
    """
    Save training history as JSON.
    """

    history_path = os.path.join(history_dir, "history.json")

    serializable_history = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(serializable_history, file, indent=4)


def build_training_model(config: TrainConfig) -> tf.keras.Model:
    """
    Build paper-aligned ResNet-34 model.
    """

    model_config = ModelConfig(
        input_shape=(config.image_size, config.image_size, 3),
        fc_units=config.fc_units,
        dropout_rate=config.dropout_rate,
        learning_rate=config.learning_rate,
        trainable=True,
        binary_output=True,
        compile_model=True,
        use_l2_regularization=config.use_l2_regularization,
        l2_factor=config.l2_factor,
        use_fc_batchnorm=True
    )

    model = build_model(model_config)

    return model


def train_pipeline(
    config: Optional[TrainConfig] = None
) -> Tuple[tf.keras.callbacks.History, tf.keras.Model]:
    """
    Full training pipeline.
    """

    if config is None:
        config = TrainConfig()

    set_global_seed(config.seed)
    enable_gpu_memory_growth()

    output_dirs = create_output_dirs(config.output_dir)
    save_config(config, config.output_dir)

    print("Listing training files...")
    train_files, train_labels = list_wav_files(config.train_dir)

    print("Listing validation files...")
    validation_files, validation_labels = list_wav_files(config.validation_dir)

    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(validation_files)}")

    print("\nProcessing training data...")
    x_train, y_train = load_dataset_to_memory(
        train_files,
        train_labels,
        config
    )

    print("\nProcessing validation data...")
    x_validation, y_validation = load_dataset_to_memory(
        validation_files,
        validation_labels,
        config
    )

    print("\nTraining tensor shape:", x_train.shape)
    print("Validation tensor shape:", x_validation.shape)

    train_dataset = create_tf_dataset(
        x_data=x_train,
        y_data=y_train,
        batch_size=config.batch_size,
        shuffle=True,
        shuffle_buffer_size=config.shuffle_buffer_size,
        cache=config.cache_dataset,
        prefetch=config.prefetch
    )

    validation_dataset = create_tf_dataset(
        x_data=x_validation,
        y_data=y_validation,
        batch_size=config.batch_size,
        shuffle=False,
        shuffle_buffer_size=config.shuffle_buffer_size,
        cache=config.cache_dataset,
        prefetch=config.prefetch
    )

    class_weights = None

    if config.use_class_weights:
        class_weights = compute_weights(y_train)
        print("Class weights:", class_weights)

    print("\nBuilding model...")
    model = build_training_model(config)
    model.summary()

    callbacks = create_callbacks(output_dirs, config)

    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    final_model_path = os.path.join(
        output_dirs["model_dir"],
        "final_model.keras"
    )

    model.save(final_model_path)

    save_history(history, output_dirs["history_dir"])

    print("\nTraining completed.")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {os.path.join(output_dirs['model_dir'], 'best_model.keras')}")

    return history, model


if __name__ == "__main__":

    config = TrainConfig(
        train_dir="data/train",
        validation_dir="data/validation",
        output_dir="results",

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

        image_size=224,
        use_rgb=True,
        use_sobel=True,
        use_laplacian=True,

        epochs=50,
        batch_size=32,
        learning_rate=0.001,

        dropout_rate=0.3,
        fc_units=1000,

        use_class_weights=True,
        cache_dataset=False,
        prefetch=True
    )

    train_pipeline(config)
