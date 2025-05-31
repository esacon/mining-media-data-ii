import time
from typing import Any, Dict

from src.config import Settings
from src.data_processing.data_preparation import DataPreparation
from src.data_processing.dataset_creation import DatasetCreator
from src.data_processing.feature_engineering import FeatureExtractor
from src.utils import LoggerMixin, setup_logger


class DataPipeline(LoggerMixin):
    """Orchestrates the complete data processing pipeline, including data preparation
    (conversion and splitting), labeled dataset creation, and feature extraction.
    """

    def __init__(self, settings: Settings):
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
            log_file_path = (
                self.settings.logs_dir / f"{self.__class__.__name__.lower()}.log"
            )

        self._logger = setup_logger(
            name=self.__class__.__name__,
            level=self.settings.log_level,
            log_file=str(log_file_path) if log_file_path else None,
            format_string=self.settings.log_format,
        )

        self.data_prep = DataPreparation(settings)
        self.dataset_creator = DatasetCreator(settings)
        self.feature_extractor = FeatureExtractor(settings)

    def run_preparation(self) -> Dict[str, Any]:
        """Runs the data preparation step of the pipeline.

        This involves converting raw data (e.g., Game 1 CSV) and splitting
        the data into training and evaluation sets based on player IDs.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the preparation
                            and the execution time.
        """
        start_time = time.time()
        prep_results = self.data_prep.prepare_all_data()
        prep_time = time.time() - start_time

        return {"results": prep_results, "execution_time": prep_time}

    def run_dataset_creation(self) -> Dict[str, Any]:
        """Runs the labeled dataset creation step of the pipeline.

        This involves processing player events to define observation and churn periods,
        and labeling players as churned or retained.

        Returns:
            Dict[str, Any]: A dictionary containing the names of the created datasets
                            and the execution time.
        """
        start_time = time.time()
        dataset_results = self.dataset_creator.create_all_datasets()
        creation_time = time.time() - start_time

        return {"results": dataset_results, "execution_time": creation_time}

    def run_feature_extraction(self) -> Dict[str, Any]:
        """Runs the feature extraction step of the pipeline.

        This involves extracting behavioral features from the labeled datasets,
        including common features from Kim et al. (2017) and game-specific features.

        Returns:
            Dict[str, Any]: A dictionary containing feature extraction results
                            and the execution time.
        """
        start_time = time.time()
        feature_results = self.feature_extractor.extract_all_features()
        extraction_time = time.time() - start_time

        return {"results": feature_results, "execution_time": extraction_time}

    def run_full_pipeline(self) -> None:
        """Runs the complete data processing pipeline from raw data to extracted features.

        Orchestrates the data preparation, dataset creation, and feature extraction steps.

        Raises:
            Exception: If any step of the pipeline fails.
        """
        try:
            self.run_preparation()
            self.run_dataset_creation()
            self.run_feature_extraction()
            self.logger.info("Pipeline execution complete.")

        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            raise
