"""
tensor_generation.py

Utilities for converting 2D RPS/PCA data into CNN-ready tensors.

Includes:
- 2D histogram tensor generation
- Tensor normalization
- Sobel gradient channel
- Laplacian channel
- RGB tensor construction

--------------------------------------------------
INSTALLATION (run in terminal):

pip install numpy opencv-python

--------------------------------------------------
USAGE EXAMPLE:

from src.tensor_generation import TensorConfig, tensor_generation_pipeline

config = TensorConfig(
    image_size=224,
    use_rgb=True,
    use_sobel=True,
    use_laplacian=True
)

result = tensor_generation_pipeline(pca_data, config)

rgb_tensor = result["final_tensor"]

--------------------------------------------------
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import cv2


@dataclass
class TensorConfig:
    """
    Configuration for 2D tensor and RGB tensor generation.
    """

    image_size: int = 224

    # 2D tensor control
    normalize_tensor: bool = True
    log_transform: bool = True

    # RGB control
    use_rgb: bool = True
    use_sobel: bool = True
    use_laplacian: bool = True

    # Output format
    output_dtype: str = "float32"


def validate_2d_data(data: np.ndarray) -> np.ndarray:
    """
    Validate input 2D data.

    Expected shape:
        (N, 2)

    Parameters
    ----------
    data : np.ndarray
        2D RPS/PCA data.

    Returns
    -------
    np.ndarray
        Validated float32 data.
    """

    if data is None:
        raise ValueError("Input data is None.")

    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError("Input data must be 2D.")

    if data.shape[1] != 2:
        raise ValueError("Input data must have exactly 2 columns.")

    if data.shape[0] == 0:
        raise ValueError("Input data is empty.")

    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values.")

    return data


def normalize_array(array: np.ndarray) -> np.ndarray:
    """
    Normalize an array to the range [0, 1].
    """

    array = np.asarray(array, dtype=np.float32)

    min_value = np.min(array)
    max_value = np.max(array)

    if max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.float32)

    return ((array - min_value) / (max_value - min_value)).astype(np.float32)


def create_2d_tensor(
    data: np.ndarray,
    image_size: int = 224,
    normalize: bool = True,
    log_transform: bool = True
) -> np.ndarray:
    """
    Convert 2D RPS/PCA points into a 2D histogram tensor.

    Parameters
    ----------
    data : np.ndarray
        Input 2D points with shape (N, 2).
    image_size : int
        Output tensor size. Default is 224.
    normalize : bool
        Whether to normalize output to [0, 1].
    log_transform : bool
        Whether to apply log(1 + x) to reduce extreme values.

    Returns
    -------
    np.ndarray
        2D tensor with shape (image_size, image_size).
    """

    data = validate_2d_data(data)

    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    x = data[:, 0]
    y = data[:, 1]

    tensor, _, _ = np.histogram2d(
        x,
        y,
        bins=image_size
    )

    tensor = tensor.astype(np.float32)

    # Transpose to match image coordinate convention
    tensor = tensor.T

    if log_transform:
        tensor = np.log1p(tensor)

    if normalize:
        tensor = normalize_array(tensor)

    return tensor.astype(np.float32)


def create_sobel_channel(tensor: np.ndarray) -> np.ndarray:
    """
    Create Sobel gradient magnitude channel from a 2D tensor.
    """

    tensor = np.asarray(tensor, dtype=np.float32)

    sobel_x = cv2.Sobel(tensor, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(tensor, cv2.CV_32F, 0, 1, ksize=3)

    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    return normalize_array(sobel_magnitude)


def create_laplacian_channel(tensor: np.ndarray) -> np.ndarray:
    """
    Create Laplacian channel from a 2D tensor.
    """

    tensor = np.asarray(tensor, dtype=np.float32)

    laplacian = cv2.Laplacian(tensor, cv2.CV_32F, ksize=3)

    laplacian = np.abs(laplacian)

    return normalize_array(laplacian)


def create_rgb_tensor(
    tensor: np.ndarray,
    use_sobel: bool = True,
    use_laplacian: bool = True
) -> np.ndarray:
    """
    Create RGB tensor from the 2D tensor.

    Channel 1: original 2D tensor
    Channel 2: Sobel gradient
    Channel 3: Laplacian

    If Sobel or Laplacian is disabled, the original tensor is repeated
    in that channel to keep output shape compatible with CNN input.

    Parameters
    ----------
    tensor : np.ndarray
        Input 2D tensor.
    use_sobel : bool
        Whether to use Sobel as second channel.
    use_laplacian : bool
        Whether to use Laplacian as third channel.

    Returns
    -------
    np.ndarray
        RGB tensor with shape (H, W, 3).
    """

    tensor = np.asarray(tensor, dtype=np.float32)

    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D.")

    base_channel = normalize_array(tensor)

    if use_sobel:
        sobel_channel = create_sobel_channel(base_channel)
    else:
        sobel_channel = base_channel.copy()

    if use_laplacian:
        laplacian_channel = create_laplacian_channel(base_channel)
    else:
        laplacian_channel = base_channel.copy()

    rgb_tensor = np.stack(
        [base_channel, sobel_channel, laplacian_channel],
        axis=-1
    )

    return rgb_tensor.astype(np.float32)


def tensor_generation_pipeline(
    data: np.ndarray,
    config: Optional[TensorConfig] = None
) -> Dict[str, Any]:
    """
    Complete configurable tensor generation pipeline.

    Parameters
    ----------
    data : np.ndarray
        2D PCA/RPS data with shape (N, 2).
    config : TensorConfig
        Tensor generation configuration.

    Returns
    -------
    dict
        Dictionary containing generated tensors and metadata.
    """

    if config is None:
        config = TensorConfig()

    tensor_2d = create_2d_tensor(
        data=data,
        image_size=config.image_size,
        normalize=config.normalize_tensor,
        log_transform=config.log_transform
    )

    if config.use_rgb:
        final_tensor = create_rgb_tensor(
            tensor=tensor_2d,
            use_sobel=config.use_sobel,
            use_laplacian=config.use_laplacian
        )
    else:
        final_tensor = tensor_2d[..., np.newaxis]

    if config.output_dtype == "float32":
        final_tensor = final_tensor.astype(np.float32)
    elif config.output_dtype == "uint8":
        final_tensor = (normalize_array(final_tensor) * 255).astype(np.uint8)
    else:
        raise ValueError("output_dtype must be 'float32' or 'uint8'.")

    output = {
        "tensor_2d": tensor_2d,
        "final_tensor": final_tensor,
        "config": config,
        "metadata": {
            "input_shape": data.shape,
            "tensor_2d_shape": tensor_2d.shape,
            "final_tensor_shape": final_tensor.shape,
            "image_size": config.image_size,
            "use_rgb": config.use_rgb,
            "use_sobel": config.use_sobel,
            "use_laplacian": config.use_laplacian,
            "output_dtype": config.output_dtype
        }
    }

    return output


# --------------------------------------------------
# SIMPLE TEST (optional)
# --------------------------------------------------
if __name__ == "__main__":

    # Example synthetic 2D RPS/PCA data
    np.random.seed(42)

    x = np.random.randn(5000)
    y = 0.5 * x + 0.2 * np.random.randn(5000)

    data_2d = np.column_stack((x, y)).astype(np.float32)

    config = TensorConfig(
        image_size=224,
        normalize_tensor=True,
        log_transform=True,
        use_rgb=True,
        use_sobel=True,
        use_laplacian=True,
        output_dtype="float32"
    )

    result = tensor_generation_pipeline(data_2d, config)

    print("2D tensor shape:", result["tensor_2d"].shape)
    print("Final tensor shape:", result["final_tensor"].shape)
    print("Final tensor dtype:", result["final_tensor"].dtype)
