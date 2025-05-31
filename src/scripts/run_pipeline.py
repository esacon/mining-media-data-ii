import argparse
import sys
from pathlib import Path
from typing import Tuple

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import Settings, get_settings
from src.data_processing import DataPipeline
from src.models.model_pipeline import ModelPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Run churn prediction data and model pipeline"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config.yaml file",
    )

    # Directory overrides
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Override base data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override processed data output directory",
    )

    # Parameter overrides
    parser.add_argument(
        "--observation-days",
        type=int,
        help="Override observation period in days",
    )
    parser.add_argument(
        "--churn-days",
        type=int,
        help="Override churn prediction period in days",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Override logging level (INFO, DEBUG, etc.)",
    )

    # Step selection (mutually exclusive)
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument(
        "--prep-only",
        action="store_true",
        help="Run only data preparation step",
    )
    step_group.add_argument(
        "--create-only",
        action="store_true",
        help="Run only dataset creation step",
    )
    step_group.add_argument(
        "--features-only",
        action="store_true",
        help="Run only feature extraction step",
    )
    step_group.add_argument(
        "--models-only",
        action="store_true",
        help="Run only model training and evaluation step",
    )
    step_group.add_argument(
        "--data-only",
        action="store_true",
        help="Run only data processing pipeline (prep + create + features)",
    )

    return parser.parse_args()


def apply_overrides(settings, args: argparse.Namespace) -> None:
    """Apply command-line argument overrides to settings."""
    overrides = {
        "data_dir": args.data_dir,
        "processed_dir": args.output_dir,
        "observation_days": args.observation_days,
        "churn_period_days": args.churn_days,
        "log_level": args.log_level,
    }

    for attr, value in overrides.items():
        if value is not None:
            if attr in ["data_dir", "processed_dir"]:
                setattr(settings, attr, Path(value))
            else:
                setattr(settings, attr, value)


def run_pipeline_steps(
    args: argparse.Namespace,
    data_pipeline: DataPipeline = None,
    model_pipeline: ModelPipeline = None,
) -> None:
    """Run appropriate pipeline steps based on arguments."""
    if args.prep_only:
        data_pipeline.logger.info("Running data preparation only...")
        data_pipeline.run_preparation()
    elif args.create_only:
        data_pipeline.logger.info("Running dataset creation only...")
        data_pipeline.run_dataset_creation()
    elif args.features_only:
        data_pipeline.logger.info("Running feature extraction only...")
        data_pipeline.run_feature_extraction()
    elif args.models_only:
        model_pipeline.logger.info("Running model training and evaluation only...")
        model_pipeline.run_pipeline()
    elif args.data_only:
        data_pipeline.logger.info("Running data processing pipeline only...")
        data_pipeline.run_full_pipeline()
    else:
        data_pipeline.logger.info(
            "Running full pipeline (data processing + model training)..."
        )
        data_pipeline.run_full_pipeline()
        model_pipeline.logger.info("Starting model training and evaluation...")
        model_pipeline.run_pipeline()


def _initialize_pipelines(
    settings: Settings, args: argparse.Namespace
) -> Tuple[DataPipeline, ModelPipeline]:
    """Initialize pipelines based on what's needed."""
    data_pipeline = None
    model_pipeline = None

    if (
        args.prep_only
        or args.create_only
        or args.features_only
        or args.data_only
        or (
            not any(
                [
                    args.prep_only,
                    args.create_only,
                    args.features_only,
                    args.models_only,
                    args.data_only,
                ]
            )
        )
    ):
        data_pipeline = DataPipeline(settings)

    if args.models_only or (
        not any(
            [
                args.prep_only,
                args.create_only,
                args.features_only,
                args.models_only,
                args.data_only,
            ]
        )
    ):
        model_pipeline = ModelPipeline(settings)

    return data_pipeline, model_pipeline


def main() -> None:
    """Main function to load settings, initialize, and run the data and model pipelines."""
    try:
        args = parse_args()

        # Load and configure settings
        settings = get_settings(args.config)
        apply_overrides(settings, args)
        settings.__post_init__()

        data_pipeline, model_pipeline = _initialize_pipelines(settings, args)

        # Run appropriate pipeline steps
        run_pipeline_steps(args, data_pipeline, model_pipeline)

        print("\n🎉 Pipeline execution completed successfully!")

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
