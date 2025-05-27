import time
from typing import Dict, Any

from src.data_processing.data_preparation import DataPreparation
from src.data_processing.dataset_creation import DatasetCreator
from src.utils import LoggerMixin, setup_logger, save_json, format_duration
from src.config import Settings


class DataPipeline(LoggerMixin):
    """Orchestrates the complete data processing pipeline, including data preparation
    (conversion and splitting) and labeled dataset creation.
    """

    def __init__(
        self,
        settings: Settings
    ):
        """Initializes the DataPipeline with configurations.

        Args:
            settings (Settings): An instance of the Settings dataclass containing
                                 all necessary configurations for the pipeline.
        """
        self.settings = settings
        self.data_dir = self.settings.data_dir
        self.output_dir = self.settings.processed_dir
        self.observation_days = self.settings.observation_days
        self.churn_period_days = self.settings.churn_period_days

        log_file_path = None
        if self.settings.log_to_file:
            log_file_path = self.settings.logs_dir / \
                f"{self.__class__.__name__.lower()}.log"

        self._logger = setup_logger(
            name=self.__class__.__name__,
            level=self.settings.log_level,
            log_file=str(log_file_path) if log_file_path else None,
            format_string=self.settings.log_format
        )

        # Initialize data preparation and dataset creation components
        self.data_prep = DataPreparation(settings)
        self.dataset_creator = DatasetCreator(settings)

    def run_preparation(self) -> Dict[str, Any]:
        """Runs the data preparation step of the pipeline.

        This involves converting raw data (e.g., Game 1 CSV) and splitting
        the data into training and evaluation sets based on player IDs.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the preparation
                            and the execution time.
        """
        self.logger.info("=" * 50)
        self.logger.info("STEP 1: DATA PREPARATION")
        self.logger.info("=" * 50)

        start_time = time.time()
        prep_results = self.data_prep.prepare_all_data()
        prep_time = time.time() - start_time

        self.logger.info(f"Data preparation completed in {format_duration(prep_time)}")
        for game, files in prep_results.items():
            self.logger.info(f"  {game}: train={files['train']}, eval={files['eval']}")

        return {
            "results": prep_results,
            "execution_time": prep_time
        }

    def run_dataset_creation(self) -> Dict[str, Any]:
        """Runs the labeled dataset creation step of the pipeline.

        This involves processing player events to define observation and churn periods,
        and labeling players as churned or retained.

        Returns:
            Dict[str, Any]: A dictionary containing the names of the created datasets
                            and the execution time.
        """
        self.logger.info("\n" + "=" * 50)
        self.logger.info("STEP 2: DATASET CREATION WITH LABELS")
        self.logger.info("=" * 50)

        start_time = time.time()
        dataset_results = self.dataset_creator.create_all_datasets()
        creation_time = time.time() - start_time

        self.logger.info(
            f"Dataset creation completed in {format_duration(creation_time)}")

        return {
            "results": dataset_results,
            "execution_time": creation_time
        }

    def print_summary(self, dataset_results: Dict[str, Dict[str, str]]) -> None:
        """Prints a summary of the created datasets.

        Provides statistics like total players, churned/retained counts and rates,
        and average event counts for each generated dataset.

        Args:
            dataset_results (Dict[str, Dict[str, str]]): A dictionary mapping game names
                                                       to dataset types and their filenames.
        """
        self.logger.info("\n" + "=" * 50)
        self.logger.info("PIPELINE COMPLETE - SUMMARY")
        self.logger.info("=" * 50)

        for game, datasets in dataset_results.items():
            self.logger.info(f"\n{game.upper()} DATASETS:")
            for dataset_type, filename in datasets.items():
                self.logger.info(f"\n  {dataset_type} ({filename}):")
                summary = self.dataset_creator.get_dataset_summary(filename)

                self.logger.info(
                    f"    - Total players: {summary.get('total_players', 0):,}")
                self.logger.info(
                    f"    - Churned: {summary.get('churned_players', 0):,} ({summary.get('churn_rate', 0.0):.1%})")
                self.logger.info(
                    f"    - Retained: {summary.get('retained_players', 0):,} ({summary.get('retention_rate', 0.0):.1%})")
                self.logger.info(
                    f"    - Avg events in OP: {summary.get('op_events', {}).get('mean', 0.0):.1f}")
                self.logger.info(
                    f"    - Event range (OP): {summary.get('op_events', {}).get('min', 0)} - {summary.get('op_events', {}).get('max', 0)}")
                self.logger.info(
                    f"    - Avg events in CP: {summary.get('cp_events', {}).get('mean', 0.0):.1f}")
                self.logger.info(
                    f"    - Event range (CP): {summary.get('cp_events', {}).get('min', 0)} - {summary.get('cp_events', {}).get('max', 0)}")

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Runs the complete data processing pipeline from raw data to labeled datasets.

        Orchestrates the data preparation and dataset creation steps, logs progress,
        and provides a final summary of the generated datasets. All results and
        execution times are compiled and saved.

        Returns:
            Dict[str, Any]: A dictionary containing comprehensive results and metadata
                            about the pipeline execution.

        Raises:
            Exception: If any step of the pipeline fails.
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING CHURN PREDICTION DATA PIPELINE")
        self.logger.info("=" * 80)

        pipeline_start = time.time()

        try:
            prep_step = self.run_preparation()

            creation_step = self.run_dataset_creation()

            self.print_summary(creation_step["results"])

            total_time = time.time() - pipeline_start
            self.logger.info(
                f"\nTotal pipeline execution time: {format_duration(total_time)}")

            pipeline_results = {
                "preparation": prep_step,
                "dataset_creation": creation_step,
                "total_execution_time": total_time,
                "configuration": {
                    "observation_days": self.observation_days,
                    "churn_period_days": self.churn_period_days,
                    "data_dir": str(self.data_dir),
                    "output_dir": str(self.output_dir),
                    "log_level": self.settings.log_level,
                    "random_seed": self.settings.random_seed
                }
            }

            save_json(pipeline_results, self.output_dir / self.settings.pipeline_results)

            self.logger.info(
                f"\nPipeline results saved to: {self.output_dir}/{self.settings.pipeline_results}")
            self.logger.info("\nNext steps:")
            self.logger.info("  1. Run feature engineering on the labeled datasets")
            self.logger.info("  2. Train and evaluate churn prediction models")

            return pipeline_results

        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            raise
