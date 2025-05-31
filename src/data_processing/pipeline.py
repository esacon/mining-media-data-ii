import time
from typing import Any, Dict

from src.config import Settings
from src.data_processing.data_preparation import DataPreparation
from src.data_processing.dataset_creation import DatasetCreator
from src.data_processing.feature_engineering import FeatureEngineering
from src.utils import LoggerMixin, setup_logger


class DataPipeline(LoggerMixin):
    """Orchestrates the complete data processing pipeline.

    Args:
        settings (Settings): The settings for the pipeline.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._setup_logger()

        # Initialize components
        self.data_prep = DataPreparation(settings)
        self.dataset_creator = DatasetCreator(settings)
        self.feature_extractor = FeatureEngineering(settings)

    def _setup_logger(self):
        """Setup logger with optional file output."""
        log_file_path = None
        if self.settings.log_to_file:
            log_file_path = str(
                self.settings.logs_dir / f"{self.__class__.__name__.lower()}.log"
            )

        self._logger = setup_logger(
            name=self.__class__.__name__,
            level=self.settings.log_level,
            log_file=log_file_path,
            format_string=self.settings.log_format,
        )

    def _run_step(self, step_name: str, step_func) -> Dict[str, Any]:
        """Run a pipeline step and return results with timing."""
        self.logger.info(f"Running {step_name} step...")
        start_time = time.time()
        results = step_func()
        execution_time = time.time() - start_time
        self.logger.info(f"{step_name} completed in {execution_time:.2f}s")
        return {"results": results, "execution_time": execution_time}

    def run_preparation(self) -> Dict[str, Any]:
        """Run data preparation step.

        Args:
            None

        Returns:
            Dict[str, Any]: A dictionary containing the results of the preparation step.
        """
        return self._run_step("preparation", self.data_prep.prepare_all_data)

    def run_dataset_creation(self) -> Dict[str, Any]:
        """Run labeled dataset creation step.

        Args:
            None

        Returns:
            Dict[str, Any]: A dictionary containing the results of the dataset creation step.
        """
        return self._run_step(
            "dataset creation", self.dataset_creator.create_all_datasets
        )

    def run_feature_extraction(self) -> Dict[str, Any]:
        """Run feature extraction step.

        Args:
            None

        Returns:
            Dict[str, Any]: A dictionary containing the results of the feature extraction step.
        """
        return self._run_step(
            "feature extraction", self.feature_extractor.run_feature_extraction
        )

    def run_full_pipeline(self) -> None:
        """Run the complete data processing pipeline.

        Args:
            None

        Returns:
            None
        """
        try:
            self.logger.info("Starting full pipeline execution...")

            self.run_preparation()
            self.run_dataset_creation()
            self.run_feature_extraction()

            self.logger.info("Pipeline execution complete!")
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
