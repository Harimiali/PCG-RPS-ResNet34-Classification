"""
model.py

ResNet-34 model for RPS-PCA-RGB based PCG classification.

This implementation follows the model description used in the paper:

Input:
    224 × 224 × 3 RGB tensor

Architecture:
    Conv1           → 7×7, stride 2, 64 filters
    Max Pooling     → 3×3, stride 2
    Conv2_x         → 3 residual blocks, 64 filters
    Conv3_x         → 4 residual blocks, 128 filters
    Conv4_x         → 6 residual blocks, 256 filters
    Conv5_x         → 3 residual blocks, 512 filters
    Global Avg Pool → 1×1×512
    FC Layer        → 1000 units
    Output          → Binary classification

--------------------------------------------------
INSTALLATION:

pip install tensorflow

--------------------------------------------------
NORMAL USE:

from src.model import ModelConfig, build_model

config = ModelConfig(
    input_shape=(224, 224, 3),
    learning_rate=0.001,
    dropout_rate=0.3,
    trainable=True
)

model = build_model(config)
model.summary()

--------------------------------------------------
GPU CHECK:

import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))

--------------------------------------------------
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics


@dataclass
class ModelConfig:
    """
    Configuration for ResNet-34 PCG classifier.
    """

    input_shape: Tuple[int, int, int] = (224, 224, 3)

    # Paper-based classifier structure
    fc_units: int = 1000
    dropout_rate: float = 0.3

    # Training parameters from paper
    learning_rate: float = 0.001
    trainable: bool = True

    # Output control
    binary_output: bool = True
    compile_model: bool = True

    # Optional regularization
    use_l2_regularization: bool = False
    l2_factor: float = 1e-4

    # Optional batch normalization after FC layer
    use_fc_batchnorm: bool = True


def get_regularizer(config: ModelConfig):
    """
    Return L2 regularizer if enabled.
    """

    if config.use_l2_regularization:
        return tf.keras.regularizers.l2(config.l2_factor)

    return None


def basic_residual_block(
    x: tf.Tensor,
    filters: int,
    stride: int,
    config: ModelConfig,
    block_name: str
) -> tf.Tensor:
    """
    Basic residual block used in ResNet-34.

    Structure:
        Conv 3×3
        BatchNorm
        ReLU
        Conv 3×3
        BatchNorm
        Shortcut connection
        ReLU

    If the spatial size or number of channels changes,
    a 1×1 convolution is applied to the shortcut path.
    """

    shortcut = x
    regularizer = get_regularizer(config)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
        name=f"{block_name}_conv1"
    )(x)

    x = layers.BatchNormalization(
        name=f"{block_name}_bn1"
    )(x)

    x = layers.Activation(
        "relu",
        name=f"{block_name}_relu1"
    )(x)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
        name=f"{block_name}_conv2"
    )(x)

    x = layers.BatchNormalization(
        name=f"{block_name}_bn2"
    )(x)

    shortcut_channels = int(shortcut.shape[-1])

    if stride != 1 or shortcut_channels != filters:
        shortcut = layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizer,
            name=f"{block_name}_shortcut_conv"
        )(shortcut)

        shortcut = layers.BatchNormalization(
            name=f"{block_name}_shortcut_bn"
        )(shortcut)

    x = layers.Add(
        name=f"{block_name}_add"
    )([x, shortcut])

    x = layers.Activation(
        "relu",
        name=f"{block_name}_relu_out"
    )(x)

    return x


def make_resnet_stage(
    x: tf.Tensor,
    filters: int,
    num_blocks: int,
    first_stride: int,
    config: ModelConfig,
    stage_name: str
) -> tf.Tensor:
    """
    Build one ResNet stage.

    ResNet-34 stages:
        conv2_x: 3 blocks, 64 filters, stride 1
        conv3_x: 4 blocks, 128 filters, stride 2
        conv4_x: 6 blocks, 256 filters, stride 2
        conv5_x: 3 blocks, 512 filters, stride 2
    """

    x = basic_residual_block(
        x=x,
        filters=filters,
        stride=first_stride,
        config=config,
        block_name=f"{stage_name}_block1"
    )

    for block_index in range(2, num_blocks + 1):
        x = basic_residual_block(
            x=x,
            filters=filters,
            stride=1,
            config=config,
            block_name=f"{stage_name}_block{block_index}"
        )

    return x


def build_resnet34_backbone(config: ModelConfig) -> tf.keras.Model:
    """
    Build ResNet-34 feature extractor exactly according to the paper table.
    """

    regularizer = get_regularizer(config)

    inputs = layers.Input(
        shape=config.input_shape,
        name="input_224x224x3"
    )

    # --------------------------------------------------
    # Conv1
    # Output: 112×112×64
    # Kernel: 7×7
    # Stride: 2
    # --------------------------------------------------
    x = layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=2,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizer,
        name="conv1_7x7_s2"
    )(inputs)

    x = layers.BatchNormalization(
        name="conv1_bn"
    )(x)

    x = layers.Activation(
        "relu",
        name="conv1_relu"
    )(x)

    # --------------------------------------------------
    # Max Pooling
    # Output: 56×56×64
    # Kernel: 3×3
    # Stride: 2
    # --------------------------------------------------
    x = layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding="same",
        name="maxpool_3x3_s2"
    )(x)

    # --------------------------------------------------
    # Conv2_x
    # Output: 56×56×64
    # Number of residual blocks: 3
    # --------------------------------------------------
    x = make_resnet_stage(
        x=x,
        filters=64,
        num_blocks=3,
        first_stride=1,
        config=config,
        stage_name="conv2_x"
    )

    # --------------------------------------------------
    # Conv3_x
    # Output: 28×28×128
    # Number of residual blocks: 4
    # --------------------------------------------------
    x = make_resnet_stage(
        x=x,
        filters=128,
        num_blocks=4,
        first_stride=2,
        config=config,
        stage_name="conv3_x"
    )

    # --------------------------------------------------
    # Conv4_x
    # Output: 14×14×256
    # Number of residual blocks: 6
    # --------------------------------------------------
    x = make_resnet_stage(
        x=x,
        filters=256,
        num_blocks=6,
        first_stride=2,
        config=config,
        stage_name="conv4_x"
    )

    # --------------------------------------------------
    # Conv5_x
    # Output: 7×7×512
    # Number of residual blocks: 3
    # --------------------------------------------------
    x = make_resnet_stage(
        x=x,
        filters=512,
        num_blocks=3,
        first_stride=2,
        config=config,
        stage_name="conv5_x"
    )

    backbone = models.Model(
        inputs=inputs,
        outputs=x,
        name="ResNet34_Backbone_Paper_Aligned"
    )

    backbone.trainable = config.trainable

    return backbone


def build_resnet34_pcg_classifier(
    config: Optional[ModelConfig] = None
) -> tf.keras.Model:
    """
    Build the full ResNet-34 PCG classification model.

    The paper table includes:
        Global Average Pooling
        Fully Connected Layer: 1000 units

    Since the task is binary classification, this implementation adds:
        Final sigmoid output layer

    This keeps the 1000-unit FC layer from the paper while making
    the model suitable for normal/abnormal classification.
    """

    if config is None:
        config = ModelConfig()

    backbone = build_resnet34_backbone(config)

    inputs = layers.Input(
        shape=config.input_shape,
        name="pcg_rgb_tensor_input"
    )

    x = backbone(inputs)

    # --------------------------------------------------
    # Global Average Pooling
    # Output: 1×1×512
    # --------------------------------------------------
    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    # --------------------------------------------------
    # Fully Connected Layer
    # Paper table: 1000 units
    # --------------------------------------------------
    x = layers.Dense(
        units=config.fc_units,
        activation=None,
        kernel_regularizer=get_regularizer(config),
        name="fc_1000"
    )(x)

    if config.use_fc_batchnorm:
        x = layers.BatchNormalization(
            name="fc_1000_bn"
        )(x)

    x = layers.Activation(
        "relu",
        name="fc_1000_relu"
    )(x)

    x = layers.Dropout(
        rate=config.dropout_rate,
        name="fc_dropout"
    )(x)

    # --------------------------------------------------
    # Binary classification output
    # Positive class: Abnormal
    # Negative class: Normal
    # --------------------------------------------------
    if config.binary_output:
        outputs = layers.Dense(
            units=1,
            activation="sigmoid",
            name="abnormal_probability"
        )(x)
    else:
        outputs = layers.Dense(
            units=2,
            activation="softmax",
            name="normal_abnormal_softmax"
        )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="RPS_PCA_RGB_ResNet34_PCG_Classifier"
    )

    return model


def compile_model(
    model: tf.keras.Model,
    config: ModelConfig
) -> tf.keras.Model:
    """
    Compile the model using paper-based settings:

        Optimizer: Adam
        Learning rate: 0.001
        Loss: Binary cross-entropy
        Metrics: Accuracy, Precision, Recall
    """

    optimizer = optimizers.Adam(
        learning_rate=config.learning_rate
    )

    if config.binary_output:
        loss_function = losses.BinaryCrossentropy(
            name="binary_crossentropy"
        )

        metric_list = [
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc")
        ]

    else:
        loss_function = losses.CategoricalCrossentropy(
            name="categorical_crossentropy"
        )

        metric_list = [
            metrics.CategoricalAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc")
        ]

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metric_list
    )

    return model


def build_model(
    config: Optional[ModelConfig] = None
) -> tf.keras.Model:
    """
    Main function for building the complete model.
    """

    if config is None:
        config = ModelConfig()

    model = build_resnet34_pcg_classifier(config)

    if config.compile_model:
        model = compile_model(model, config)

    return model


def enable_gpu_memory_growth() -> None:
    """
    Enable TensorFlow GPU memory growth.

    This prevents TensorFlow from occupying all GPU memory at once.
    """

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass


def get_model_information(
    model: tf.keras.Model
) -> Dict[str, Any]:
    """
    Return useful model information.
    """

    trainable_params = int(
        sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    )

    non_trainable_params = int(
        sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    )

    total_params = trainable_params + non_trainable_params

    return {
        "model_name": model.name,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params
    }


# --------------------------------------------------
# SIMPLE TEST
# --------------------------------------------------
if __name__ == "__main__":

    enable_gpu_memory_growth()

    config = ModelConfig(
        input_shape=(224, 224, 3),
        fc_units=1000,
        dropout_rate=0.3,
        learning_rate=0.001,
        trainable=True,
        binary_output=True,
        compile_model=True,
        use_l2_regularization=False,
        l2_factor=1e-4,
        use_fc_batchnorm=True
    )

    model = build_model(config)

    model.summary()

    info = get_model_information(model)

    print("\nModel Information:")
    for key, value in info.items():
        print(f"{key}: {value}")

    print("\nAvailable GPUs:")
    print(tf.config.list_physical_devices("GPU"))
