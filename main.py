"""
main.py

Main entry point for the complete PCG classification project.

Pipeline:
    train       -> train the ResNet-34 model
    evaluate    -> evaluate a saved model
    full        -> train then evaluate

--------------------------------------------------
INSTALLATION:

pip install numpy scipy scikit-learn opencv-python tensorflow matplotlib pandas

--------------------------------------------------
RUN EXAMPLES:

1) Train and evaluate:

python main.py --mode full

2) Train only:

python main.py --mode train

3) Evaluate only:

python main.py --mode evaluate

4) Use custom paths:

python main.py --mode full \
    --train_dir data/train \
    --validation_dir data/validation \
    --output_dir results

--------------------------------------------------
EXPECTED DATA STRUCTURE:

data/
├── train/
│   ├── normal/
│   └── abnormal/
│
└── validation/
    ├── normal/
    └── abnormal/

--------------------------------------------------
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

from src.train import TrainConfig, train_pipeline
from src.evaluate import EvaluationConfig, evaluate_pipeline


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="PCG classification using RPS-PCA-RGB tensors and ResNet-34"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["train", "evaluate", "full"],
        help="Execution mode: train, evaluate, or full"
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        default="data/train",
        help="Path to training dataset directory"
    )

    parser.add_argument(
        "--validation_dir",
        type=str,
        default="data/validation",
        help="Path to validation/evaluation dataset directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save models, logs, and evaluation results"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved model for evaluation"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training/evaluation batch size"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer"
    )

    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=2000,
        help="Target sampling rate"
    )

    parser.add_argument(
        "--rps_dimension",
        type=int,
        default=4,
        help="RPS embedding dimension"
    )

    parser.add_argument(
        "--rps_delay",
        type=int,
        default=3,
        help="RPS time delay"
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Generated tensor image size"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for abnormal probability"
    )

    parser.add_argument(
        "--no_pca",
        action="store_true",
        help="Disable PCA stage"
    )

    parser.add_argument(
        "--no_rgb",
        action="store_true",
        help="Disable RGB tensor construction"
    )

    parser.add_argument(
        "--no_sobel",
        action="store_true",
        help="Disable Sobel channel"
    )

    parser.add_argument(
        "--no_laplacian",
        action="store_true",
        help="Disable Laplacian channel"
    )

    parser.add_argument(
        "--no_noise_reduction",
        action="store_true",
        help="Disable spectral subtraction"
    )

    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable class weighting"
    )

    parser.add_argument(
        "--max_duration",
        type=float,
        default=None,
        help="Optional maximum duration of each audio file in seconds"
    )

    return parser.parse_args()


def create_run_directory(output_dir: str) -> str:
    """
    Create a timestamped run directory.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def save_run_summary(run_dir: str, summary: Dict[str, Any]) -> None:
    """
    Save run summary as JSON.
    """

    path = os.path.join(run_dir, "run_summary.json")

    with open(path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)


def build_train_config(args: argparse.Namespace, run_dir: str) -> TrainConfig:
    """
    Build TrainConfig from command-line arguments.
    """

    return TrainConfig(
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        output_dir=run_dir,

        target_sampling_rate=args.sampling_rate,
        max_signal_duration_sec=args.max_duration,

        lowcut=15.0,
        highcut=800.0,
        filter_order=3,
        use_spectral_subtraction=not args.no_noise_reduction,
        noise_reduction_factor=0.5,
        normalize_signal=True,

        use_rps=True,
        rps_dimension=args.rps_dimension,
        rps_delay=args.rps_delay,
        use_pca=not args.no_pca,
        pca_components=2,
        standardize_before_pca=True,

        image_size=args.image_size,
        use_rgb=not args.no_rgb,
        use_sobel=not args.no_sobel,
        use_laplacian=not args.no_laplacian,
        log_transform=True,

        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_class_weights=not args.no_class_weights,

        dropout_rate=0.3,
        fc_units=1000,

        cache_dataset=False,
        prefetch=True
    )


def build_evaluation_config(
    args: argparse.Namespace,
    run_dir: str,
    model_path: str
) -> EvaluationConfig:
    """
    Build EvaluationConfig from command-line arguments.
    """

    evaluation_output_dir = os.path.join(run_dir, "evaluation")

    return EvaluationConfig(
        data_dir=args.validation_dir,
        model_path=model_path,
        output_dir=evaluation_output_dir,

        threshold=args.threshold,
        batch_size=args.batch_size,

        target_sampling_rate=args.sampling_rate,
        max_signal_duration_sec=args.max_duration,

        lowcut=15.0,
        highcut=800.0,
        filter_order=3,
        use_spectral_subtraction=not args.no_noise_reduction,
        noise_reduction_factor=0.5,
        normalize_signal=True,

        use_rps=True,
        rps_dimension=args.rps_dimension,
        rps_delay=args.rps_delay,
        use_pca=not args.no_pca,
        pca_components=2,
        standardize_before_pca=True,

        image_size=args.image_size,
        use_rgb=not args.no_rgb,
        use_sobel=not args.no_sobel,
        use_laplacian=not args.no_laplacian,
        log_transform=True,

        save_predictions_csv=True,
        save_metrics_json=True,
        save_confusion_matrix_image=True,
        save_classification_report=True
    )


def validate_paths(args: argparse.Namespace) -> None:
    """
    Validate required paths before running.
    """

    if args.mode in ["train", "full"]:
        if not os.path.isdir(args.train_dir):
            raise FileNotFoundError(f"Training directory not found: {args.train_dir}")

        if not os.path.isdir(args.validation_dir):
            raise FileNotFoundError(f"Validation directory not found: {args.validation_dir}")

    if args.mode == "evaluate":
        if not os.path.isdir(args.validation_dir):
            raise FileNotFoundError(f"Evaluation directory not found: {args.validation_dir}")

        if args.model_path is None:
            raise ValueError("For evaluate mode, please provide --model_path")

        if not os.path.isfile(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")


def print_run_header(args: argparse.Namespace, run_dir: str) -> None:
    """
    Print run information.
    """

    print("\n" + "=" * 60)
    print("PCG Classification using RPS-PCA-RGB and ResNet-34")
    print("=" * 60)
    print(f"Mode:              {args.mode}")
    print(f"Train directory:   {args.train_dir}")
    print(f"Validation dir:    {args.validation_dir}")
    print(f"Output directory:  {run_dir}")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Learning rate:     {args.learning_rate}")
    print(f"RPS dimension:     {args.rps_dimension}")
    print(f"RPS delay:         {args.rps_delay}")
    print(f"Use PCA:           {not args.no_pca}")
    print(f"Use RGB:           {not args.no_rgb}")
    print(f"Use Sobel:         {not args.no_sobel}")
    print(f"Use Laplacian:     {not args.no_laplacian}")
    print(f"Noise reduction:   {not args.no_noise_reduction}")
    print("=" * 60 + "\n")


def main() -> None:
    """
    Main project runner.
    """

    args = parse_arguments()

    try:
        validate_paths(args)

        run_dir = create_run_directory(args.output_dir)

        print_run_header(args, run_dir)

        summary = {
            "mode": args.mode,
            "run_dir": run_dir,
            "status": "started",
            "train_completed": False,
            "evaluation_completed": False,
            "model_path": None,
            "metrics": None
        }

        save_run_summary(run_dir, summary)

        model_path = args.model_path

        if args.mode in ["train", "full"]:
            print("Starting training stage...\n")

            train_config = build_train_config(args, run_dir)
            train_pipeline(train_config)

            model_path = os.path.join(
                run_dir,
                "models",
                "best_model.keras"
            )

            summary["train_completed"] = True
            summary["model_path"] = model_path
            save_run_summary(run_dir, summary)

            print("\nTraining stage completed.")
            print(f"Best model path: {model_path}\n")

        if args.mode in ["evaluate", "full"]:
            print("Starting evaluation stage...\n")

            if model_path is None:
                raise ValueError("Model path is missing for evaluation.")

            evaluation_config = build_evaluation_config(
                args=args,
                run_dir=run_dir,
                model_path=model_path
            )

            metrics = evaluate_pipeline(evaluation_config)

            summary["evaluation_completed"] = True
            summary["metrics"] = metrics
            save_run_summary(run_dir, summary)

            print("\nEvaluation stage completed.\n")

        summary["status"] = "completed"
        save_run_summary(run_dir, summary)

        print("=" * 60)
        print("Pipeline completed successfully.")
        print(f"Results saved in: {run_dir}")
        print("=" * 60)

    except Exception as error:
        print("\nERROR:")
        print(str(error))

        try:
            if "run_dir" in locals():
                error_summary = {
                    "status": "failed",
                    "error": str(error)
                }
                save_run_summary(run_dir, error_summary)
        except Exception:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()
