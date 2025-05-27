import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import get_settings
from src.data_processing import DataPipeline


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for pipeline execution.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run churn prediction data pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom config.yaml file. Defaults to project root/config.yaml.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override the base data directory from the config file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the processed data output directory from the config file.",
    )

    parser.add_argument(
        "--observation-days",
        type=int,
        default=None,
        help="Override the observation period in days from the config file.",
    )

    parser.add_argument(
        "--churn-days",
        type=int,
        default=None,
        help="Override the churn prediction period in days from the config file.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Override the logging level (e.g., 'INFO', 'DEBUG') from the config file.",
    )

    parser.add_argument(
        "--prep-only",
        action="store_true",
        help="Run only the data preparation step (conversion and splitting).",
    )

    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Run only the dataset creation step (requires data preparation to be completed).",
    )

    return parser.parse_args()


def _apply_argument_overrides(settings, args: argparse.Namespace) -> None:
    """Apply command-line argument overrides to settings."""
    if args.data_dir:
        settings.data_dir = Path(args.data_dir)
    if args.output_dir:
        settings.processed_dir = Path(args.output_dir)
    if args.observation_days is not None:
        settings.observation_days = args.observation_days
    if args.churn_days is not None:
        settings.churn_period_days = args.churn_days
    if args.log_level:
        settings.log_level = args.log_level


def _run_pipeline_step(pipeline, args: argparse.Namespace) -> None:
    """Run the appropriate pipeline step based on command-line arguments."""
    if args.prep_only:
        pipeline.logger.info("Running only data preparation step...")
        pipeline.run_preparation()
    elif args.create_only:
        pipeline.logger.info("Running only dataset creation step...")
        pipeline.run_dataset_creation()
        pipeline.print_summary(pipeline.dataset_creator.create_all_datasets())
    else:
        pipeline.logger.info("Running full data pipeline...")
        pipeline.run_full_pipeline()


def main() -> None:
    """Main function to load settings, initialize, and run the data pipeline."""
    args = parse_args()
    settings = get_settings(args.config)

    _apply_argument_overrides(settings, args)
    settings.__post_init__()

    pipeline = DataPipeline(settings=settings)

    try:
        _run_pipeline_step(pipeline, args)
    except KeyboardInterrupt:
        pipeline.logger.info("Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        pipeline.logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
