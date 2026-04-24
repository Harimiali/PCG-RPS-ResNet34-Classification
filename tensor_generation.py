"""
tensor_generation.py

Utilities for converting RPS/PCA point data into CNN-ready tensors.

Includes:
- 2D histogram tensor generation
- Tensor normalization
- Sobel gradient channel
- Laplacian channel
- RGB tensor construction
- Grayscale ablation mode compatible with ResNet input

--------------------------------------------------
INSTALLATION:

pip install numpy opencv-python

--------------------------------------------------
NORMAL USE:

from src.tensor_generation import TensorConfig, tensor_generation_pipeline

config = TensorConfig(
    image_size=224,
    use_rgb=True,
    use_sobel=True,
    use_laplacian=True
)

result = tensor_generation_pipeline(pca_data, config)

final_tensor = result["final_tensor"]

Output shape:
    224 × 224 × 3

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

    # RGB / ablation control
    use_rgb: bool = True
    use_sobel: bool = True
    use_laplacian: bool = True

    # Output format
    output_dtype: str = "float32"


def validate_point_data(data: np.ndarray) -> np.ndarray:
    """
    Validate input point data.

    Expected input:
        (N, 2) or (N, D)

    If more than 2 columns are provided, only the first two columns are used.
    This keeps the system compatible when PCA is disabled.
    """

    if data is None:
        raise ValueError("Input data is None.")

    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError("Input data must be 2D.")

    if data.shape[0] == 0:
        raise ValueError("Input data is empty.")

    if data.shape[1] < 2:
        raise ValueError(
            "Input data must have at least 2 columns. "
            "For CNN tensor generation, 2D coordinates are required."
        )

    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values.")

    return data[:, :2]


def normalize_array(array: np.ndarray) -> np.ndarray:
    """
    Normalize array to range [0, 1].
    """

    array = np.asarray(array, dtype=np.float32)

    min_value = np.min(array)
    max_value = np.max(array)

    if max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.float32)

    normalized = (array - min_value) / (max_value - min_value)

    return normalized.astype(np.float32)


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
    data:
        Point data with shape (N, 2) or (N, D).
    image_size:
        Output tensor size.
    normalize:
        Normalize tensor to [0, 1].
    log_transform:
        Apply log(1 + x) to reduce dominance of high-density cells.

    Returns
    -------
    tensor:
        2D tensor with shape (image_size, image_size).
    """

    data = validate_point_data(data)

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

    # Match image coordinate convention
    tensor = tensor.T

    if log_transform:
        tensor = np.log1p(tensor)

    if normalize:
        tensor = normalize_array(tensor)

    return tensor.astype(np.float32)


def create_sobel_channel(tensor: np.ndarray) -> np.ndarray:
    """
    Create Sobel gradient magnitude channel.
    """

    tensor = np.asarray(tensor, dtype=np.float32)

    if tensor.ndim != 2:
        raise ValueError("Sobel input tensor must be 2D.")

    sobel_x = cv2.Sobel(
        tensor,
        cv2.CV_32F,
        dx=1,
        dy=0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        tensor,
        cv2.CV_32F,
        dx=0,
        dy=1,
        ksize=3
    )

    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    return normalize_array(sobel_magnitude)


def create_laplacian_channel(tensor: np.ndarray) -> np.ndarray:
    """
    Create Laplacian channel.
    """

    tensor = np.asarray(tensor, dtype=np.float32)

    if tensor.ndim != 2:
        raise ValueError("Laplacian input tensor must be 2D.")

    laplacian = cv2.Laplacian(
        tensor,
        cv2.CV_32F,
        ksize=3
    )

    laplacian = np.abs(laplacian)

    return normalize_array(laplacian)


def create_rgb_tensor(
    tensor: np.ndarray,
    use_sobel: bool = True,
    use_laplacian: bool = True
) -> np.ndarray:
    """
    Create RGB tensor.

    Channel 1:
        Original 2D tensor

    Channel 2:
        Sobel gradient channel if enabled;
        otherwise original tensor

    Channel 3:
        Laplacian channel if enabled;
        otherwise original tensor

    Returns
    -------
    rgb_tensor:
        Tensor with shape (H, W, 3).
    """

    tensor = np.asarray(tensor, dtype=np.float32)

    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D.")

    base_channel = normalize_array(tensor)

    if use_sobel:
        second_channel = create_sobel_channel(base_channel)
    else:
        second_channel = base_channel.copy()

    if use_laplacian:
        third_channel = create_laplacian_channel(base_channel)
    else:
        third_channel = base_channel.copy()

    rgb_tensor = np.stack(
        [
            base_channel,
            second_channel,
            third_channel
        ],
        axis=-1
    )

    return rgb_tensor.astype(np.float32)


def create_grayscale_compatible_tensor(tensor: np.ndarray) -> np.ndarray:
    """
    Create grayscale ablation tensor while keeping ResNet-compatible shape.

    Instead of returning:
        H × W × 1

    This returns:
        H × W × 3

    by repeating the same grayscale channel three times.
    """

    tensor = np.asarray(tensor, dtype=np.float32)

    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D.")

    base_channel = normalize_array(tensor)

    grayscale_tensor = np.stack(
        [
            base_channel,
            base_channel,
            base_channel
        ],
        axis=-1
    )

    return grayscale_tensor.astype(np.float32)


def convert_dtype(
    tensor: np.ndarray,
    output_dtype: str = "float32"
) -> np.ndarray:
    """
    Convert output tensor dtype.
    """

    if output_dtype == "float32":
        return tensor.astype(np.float32)

    if output_dtype == "uint8":
        return (normalize_array(tensor) * 255).astype(np.uint8)

    raise ValueError("output_dtype must be 'float32' or 'uint8'.")


def tensor_generation_pipeline(
    data: np.ndarray,
    config: Optional[TensorConfig] = None
) -> Dict[str, Any]:
    """
    Complete configurable tensor generation pipeline.

    This function always returns a CNN-compatible tensor:

        image_size × image_size × 3

    This is important because ResNet-34 expects 3 input channels.
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
        final_tensor = create_grayscale_compatible_tensor(tensor_2d)

    final_tensor = convert_dtype(
        tensor=final_tensor,
        output_dtype=config.output_dtype
    )

    expected_shape = (
        config.image_size,
        config.image_size,
        3
    )

    if final_tensor.shape != expected_shape:
        raise ValueError(
            f"Final tensor shape is {final_tensor.shape}, "
            f"but expected {expected_shape}."
        )

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
# SIMPLE TEST
# --------------------------------------------------
if __name__ == "__main__":

    np.random.seed(42)

    # Example 2D data, like PCA output
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

    # Test ablation mode: no RGB
    config_no_rgb = TensorConfig(
        image_size=224,
        use_rgb=False
    )

    result_no_rgb = tensor_generation_pipeline(data_2d, config_no_rgb)

    print("No-RGB final tensor shape:", result_no_rgb["final_tensor"].shape)
