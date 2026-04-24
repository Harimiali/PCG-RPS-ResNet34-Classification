"""
rps.py

Reconstructed Phase Space (RPS) and PCA utilities for PCG signal classification.

Includes:
- Time-delay embedding (RPS)
- Optional PCA dimensionality reduction
- Fully configurable pipeline

--------------------------------------------------
INSTALLATION (run in terminal):

pip install numpy scikit-learn

--------------------------------------------------
USAGE EXAMPLE:

from src.rps import RPSConfig, rps_pca_pipeline

config = RPSConfig(
    use_rps=True,
    dimension=4,
    delay=3,
    use_pca=True,
    pca_components=2
)

result = rps_pca_pipeline(preprocessed_signal, config)

final_data = result["final_data"]

--------------------------------------------------
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class RPSConfig:
    """
    Configuration for RPS + PCA pipeline.
    """

    # RPS parameters
    use_rps: bool = True
    dimension: int = 4
    delay: int = 3

    # PCA parameters
    use_pca: bool = True
    pca_components: int = 2
    standardize_before_pca: bool = True

    # Output control
    return_pca_model: bool = False
    return_scaler: bool = False


def validate_signal(signal: np.ndarray) -> np.ndarray:
    """
    Validate and prepare input signal.
    """

    if signal is None:
        raise ValueError("Input signal is None.")

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 1:
        raise ValueError("Input signal must be 1D.")

    if len(signal) == 0:
        raise ValueError("Input signal is empty.")

    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError("Signal contains invalid values.")

    return signal


def reconstruct_phase_space(
    signal: np.ndarray,
    dimension: int = 4,
    delay: int = 3
) -> np.ndarray:
    """
    Perform time-delay embedding to construct RPS.
    """

    signal = validate_signal(signal)

    if dimension < 2:
        raise ValueError("dimension must be >= 2.")

    if delay < 1:
        raise ValueError("delay must be >= 1.")

    required_length = (dimension - 1) * delay + 1

    if len(signal) < required_length:
        raise ValueError("Signal too short for given dimension and delay.")

    num_vectors = len(signal) - (dimension - 1) * delay

    rps_matrix = np.empty((num_vectors, dimension), dtype=np.float32)

    for i in range(dimension):
        start = i * delay
        rps_matrix[:, i] = signal[start:start + num_vectors]

    return rps_matrix


def apply_pca(
    data: np.ndarray,
    n_components: int = 2,
    standardize: bool = True,
    return_model: bool = False,
    return_scaler: bool = False
) -> Tuple[np.ndarray, Optional[PCA], Optional[StandardScaler]]:
    """
    Apply PCA to reduce dimensionality.
    """

    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError("Input must be 2D.")

    if n_components > data.shape[1]:
        raise ValueError("PCA components exceed feature dimension.")

    scaler = None

    if standardize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    pca_model = PCA(n_components=n_components)
    reduced_data = pca_model.fit_transform(data)

    if not return_model:
        pca_model = None

    if not return_scaler:
        scaler = None

    return reduced_data.astype(np.float32), pca_model, scaler


def rps_pca_pipeline(
    signal: np.ndarray,
    config: Optional[RPSConfig] = None
) -> Dict[str, Any]:
    """
    Full configurable RPS + PCA pipeline.
    """

    if config is None:
        config = RPSConfig()

    signal = validate_signal(signal)

    output = {
        "input_signal": signal,
        "rps_data": None,
        "pca_data": None,
        "final_data": None,
        "pca_model": None,
        "scaler": None,
        "config": config
    }

    # RPS stage
    if config.use_rps:
        rps_data = reconstruct_phase_space(
            signal,
            config.dimension,
            config.delay
        )
    else:
        rps_data = signal.reshape(-1, 1)

    output["rps_data"] = rps_data

    # PCA stage
    if config.use_pca:
        pca_data, pca_model, scaler = apply_pca(
            rps_data,
            n_components=config.pca_components,
            standardize=config.standardize_before_pca,
            return_model=config.return_pca_model,
            return_scaler=config.return_scaler
        )

        output["pca_data"] = pca_data
        output["pca_model"] = pca_model
        output["scaler"] = scaler
        output["final_data"] = pca_data

    else:
        output["final_data"] = rps_data

    return output


def get_pca_explained_variance(pca_model: PCA) -> np.ndarray:
    """
    Return explained variance ratio of PCA.
    """

    if pca_model is None:
        raise ValueError("PCA model is None.")

    return pca_model.explained_variance_ratio_


# --------------------------------------------------
# SIMPLE TEST (optional)
# --------------------------------------------------
if __name__ == "__main__":

    fs = 2000
    t = np.linspace(0, 3, fs * 3)

    signal = (
        0.6 * np.sin(2 * np.pi * 80 * t) +
        0.3 * np.sin(2 * np.pi * 150 * t) +
        0.05 * np.random.randn(len(t))
    ).astype(np.float32)

    config = RPSConfig()

    result = rps_pca_pipeline(signal, config)

    print("RPS shape:", result["rps_data"].shape)
    print("Final shape:", result["final_data"].shape)
