from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class Settings:
    """Project settings configuration, managing various parameters and paths.

    Attributes:
        project_root (Path): The root directory of the project.
        data_dir (Path): Directory for raw input data. Defaults to 'src/data'.
        processed_dir (Path): Directory for processed data. Defaults to 'src/data/processed'.
        logs_dir (Path): Directory for log files. Defaults to 'logs'.
        results_dir (Path): Directory for storing analysis results. Defaults to 'results'.
        features_dir (Path): Directory for storing feature extraction results. Defaults to 'results/features'.
        observation_days (int): Number of days for the observation period in data processing.
        churn_period_days (int): Number of days for the churn period in data processing.
        train_ratio (float): Ratio for splitting data into training and evaluation sets.
        random_seed (int): Seed for reproducibility in random operations.
        game1_csv (str): Filename for raw data from game 1.
        game2_jsonl (str): Filename for raw data from game 2.
        log_level (str): Logging level (e.g., "INFO", "DEBUG").
        log_format (str): Format string for log messages.
        log_to_console (bool): Whether to output logs to the console.
        log_to_file (bool): Whether to output logs to a file.
        batch_size (int): Batch size for processing operations.
        progress_interval (int): Interval for reporting progress during processing.
        max_workers (Optional[int]): Maximum number of worker processes for parallel operations.
                                     None means no limit.
        common_features (List[str]): List of common features to extract from game data.
        game2_specific_features (List[str]): List of game 2 specific features.
        multicollinearity_threshold (float): Correlation threshold for feature removal.
        decision_tree_params (Dict[str, Any]): Parameters for Decision Tree classifier.
        logistic_regression_params (Dict[str, Any]): Parameters for Logistic Regression classifier.
        random_forest_params (Dict[str, Any]): Parameters for Random Forest classifier.
    """

    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Optional[Path] = None
    processed_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    features_dir: Optional[Path] = None

    observation_days: int = 5
    churn_period_days: int = 10
    train_ratio: float = 0.8
    random_seed: int = 42

    # Input files
    game1_csv: str = "dataset_game1/rawdata_game1.csv"
    game2_jsonl: str = "dataset_game2/playerLogs_game2_playerbasedlines.jsonl"

    # Intermediate files
    game1_converted: str = "game1_player_events.jsonl"
    game1_train: str = "game1_player_events_train.jsonl"
    game1_eval: str = "game1_player_events_eval.jsonl"
    game2_train: str = "playerLogs_game2_playerbasedlines_train.jsonl"
    game2_eval: str = "playerLogs_game2_playerbasedlines_eval.jsonl"

    # Final labeled datasets
    game1_ds1: str = "game1_DS1_labeled.jsonl"
    game1_ds2: str = "game1_DS2_labeled.jsonl"
    game2_ds1: str = "game2_DS1_labeled.jsonl"
    game2_ds2: str = "game2_DS2_labeled.jsonl"

    # Feature files
    game1_ds1_features: str = "game1_DS1_features.csv"
    game1_ds2_features: str = "game1_DS2_features.csv"
    game2_ds1_features: str = "game2_DS1_features.csv"
    game2_ds2_features: str = "game2_DS2_features.csv"

    # File suffixes
    train_suffix: str = "_train.jsonl"
    eval_suffix: str = "_eval.jsonl"
    labeled_suffix: str = "_labeled.jsonl"
    features_suffix: str = "_features.csv"

    # Feature engineering configuration
    common_features: List[str] = field(
        default_factory=lambda: [
            "playCount",  # Total number of plays in observation period
            "bestScore",  # Maximum score achieved
            "meanScore",  # Average score
            "worstScore",  # Minimum score achieved
            "sdScore",  # Standard deviation of scores
            "bestScoreIndex",  # Index of best score (normalized)
            "bestSubMeanCount",  # (Best - Mean) / Play count
            "bestSubMeanRatio",  # (Best - Mean) / Mean
            "activeDuration",  # Time between first and last play (hours)
            "consecutivePlayRatio",  # Ratio of consecutive plays
        ]
    )

    game2_specific_features: List[str] = field(
        default_factory=lambda: [
            "purchaseCount",  # Total number of vehicle purchases
            "highestPrice",  # Highest price among purchases
        ]
    )

    # Model configuration
    multicollinearity_threshold: float = 0.95

    decision_tree_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "random_state": 42,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini",
            "max_features": None,
        }
    )

    logistic_regression_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "random_state": 42,
            "solver": "saga",
            "penalty": "l2",
            "max_iter": 5000,
            "C": 0.1,
            "tol": 1e-3,
        }
    )

    random_forest_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "random_state": 42,
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "n_jobs": -1,
            "bootstrap": True,
            "oob_score": True,
        }
    )

    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_console: bool = True
    log_to_file: bool = True

    batch_size: int = 1000
    progress_interval: int = 1000
    max_workers: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize default paths and create directories."""
        self._set_default_paths()
        self._create_directories()
        self._process_model_params()

    def _process_model_params(self) -> None:
        """Process model parameters to handle special values like null/None."""
        # Process decision tree parameters
        if self.decision_tree_params.get("max_depth") == "null":
            self.decision_tree_params["max_depth"] = None
        if self.decision_tree_params.get("max_features") == "null":
            self.decision_tree_params["max_features"] = None

        # Process random forest parameters
        if self.random_forest_params.get("max_depth") == "null":
            self.random_forest_params["max_depth"] = None

        # Ensure random_state is consistent across all models
        if hasattr(self, "random_seed"):
            self.decision_tree_params["random_state"] = self.random_seed
            self.logistic_regression_params["random_state"] = self.random_seed
            self.random_forest_params["random_state"] = self.random_seed

    def _set_default_paths(self) -> None:
        """Set default paths if not explicitly provided."""
        if self.data_dir is None:
            self.data_dir = self.project_root / "src" / "data"
        if self.processed_dir is None:
            self.processed_dir = self.project_root / "src" / "data" / "processed"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        if self.results_dir is None:
            self.results_dir = self.project_root / "results"
        if self.features_dir is None:
            self.features_dir = self.project_root / "results" / "features"

    def _create_directories(self) -> None:
        """Create necessary directories."""
        for directory in [
            self.processed_dir,
            self.logs_dir,
            self.results_dir,
            self.features_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_config_section(self, config: dict, section: str, mappings: dict) -> None:
        """Load configuration section with field mappings."""
        if section not in config:
            return

        section_config = config[section]
        for config_key, attr_name in mappings.items():
            if config_key in section_config:
                value = section_config[config_key]
                # Handle path conversions
                if attr_name.endswith("_dir") and isinstance(value, str):
                    value = self.project_root / value
                setattr(self, attr_name, value)

    def _load_from_config(self, config: dict) -> None:
        """Load all configuration sections."""
        # Define all configuration mappings
        config_mappings = {
            "data_processing": {
                "observation_days": "observation_days",
                "churn_period_days": "churn_period_days",
                "train_ratio": "train_ratio",
                "random_seed": "random_seed",
            },
            "paths": {
                "data_dir": "data_dir",
                "processed_dir": "processed_dir",
                "logs_dir": "logs_dir",
                "results_dir": "results_dir",
                "features_dir": "features_dir",
            },
            "filenames": {
                # Input files
                "game1_csv": "game1_csv",
                "game2_jsonl": "game2_jsonl",
                # Intermediate files
                "game1_converted": "game1_converted",
                "game1_train": "game1_train",
                "game1_eval": "game1_eval",
                "game2_train": "game2_train",
                "game2_eval": "game2_eval",
                # Labeled datasets
                "game1_ds1": "game1_ds1",
                "game1_ds2": "game1_ds2",
                "game2_ds1": "game2_ds1",
                "game2_ds2": "game2_ds2",
                # Feature files
                "game1_ds1_features": "game1_ds1_features",
                "game1_ds2_features": "game1_ds2_features",
                "game2_ds1_features": "game2_ds1_features",
                "game2_ds2_features": "game2_ds2_features",
                # Suffixes
                "train_suffix": "train_suffix",
                "eval_suffix": "eval_suffix",
                "labeled_suffix": "labeled_suffix",
                "features_suffix": "features_suffix",
            },
            "feature_engineering": {
                "common_features": "common_features",
                "game2_specific_features": "game2_specific_features",
            },
            "logging": {
                "level": "log_level",
                "console": "log_to_console",
                "file": "log_to_file",
            },
            "performance": {
                "batch_size": "batch_size",
                "progress_interval": "progress_interval",
                "max_workers": "max_workers",
            },
        }

        # Load model configuration
        if "models" in config:
            models_config = config["models"]

            # Load multicollinearity threshold
            if "multicollinearity_threshold" in models_config:
                self.multicollinearity_threshold = models_config[
                    "multicollinearity_threshold"
                ]

            # Load individual model parameters
            if "decision_tree" in models_config:
                self.decision_tree_params.update(models_config["decision_tree"])

            if "logistic_regression" in models_config:
                self.logistic_regression_params.update(
                    models_config["logistic_regression"]
                )

            if "random_forest" in models_config:
                self.random_forest_params.update(models_config["random_forest"])

        # Load each section
        for section, mappings in config_mappings.items():
            self._load_config_section(config, section, mappings)

    @classmethod
    def from_yaml(cls, config_path: Optional[Union[str, Path]] = None) -> "Settings":
        """Create Settings instance from YAML configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        else:
            config_path = Path(config_path)

        settings = cls()

        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            settings._load_from_config(config)

        settings.__post_init__()
        return settings


# Global settings management
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[Union[str, Path]] = None) -> Settings:
    """Get global Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml(config_path)
    return _settings


def reload_settings(config_path: Optional[Union[str, Path]] = None) -> Settings:
    """Reload global Settings instance."""
    global _settings
    _settings = Settings.from_yaml(config_path)
    return _settings
