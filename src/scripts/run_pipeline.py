import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import get_settings
from src.data_processing import DataPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pipeline execution."""
    parser = argparse.ArgumentParser(description="Run churn prediction data pipeline")

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


def run_pipeline_steps(pipeline: DataPipeline, args: argparse.Namespace) -> None:
    """Run appropriate pipeline steps based on arguments."""
    if args.prep_only:
        pipeline.logger.info("Running data preparation only...")
        pipeline.run_preparation()
    elif args.create_only:
        pipeline.logger.info("Running dataset creation only...")
        pipeline.run_dataset_creation()
    elif args.features_only:
        pipeline.logger.info("Running feature extraction only...")
        pipeline.run_feature_extraction()
    else:
        pipeline.logger.info("Running full pipeline...")
        pipeline.run_full_pipeline()


def main() -> None:
    """Main function to load settings, initialize, and run the data pipeline."""
    try:
        args = parse_args()

        # Load and configure settings
        settings = get_settings(args.config)
        apply_overrides(settings, args)
        settings.__post_init__()

        # Initialize and run pipeline
        pipeline = DataPipeline(settings)
        run_pipeline_steps(pipeline, args)

        print("\n🎉 Pipeline execution completed successfully!")

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
