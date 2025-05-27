from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

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
    """

    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Optional[Path] = None
    processed_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    results_dir: Optional[Path] = None

    observation_days: int = 5
    churn_period_days: int = 10
    train_ratio: float = 0.8
    random_seed: int = 42

    # Input files
    game1_csv: str = "dataset_1_game1/rawdata_game1.csv"
    game2_jsonl: str = "dataset_2_game2/playerLogs_game2_playerbasedlines.jsonl"

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

    # Result files
    preparation_results: str = "preparation_results.json"
    dataset_creation_results: str = "dataset_creation_results.json"
    pipeline_results: str = "pipeline_results.json"

    # File suffixes
    train_suffix: str = "_train.jsonl"
    eval_suffix: str = "_eval.jsonl"
    labeled_suffix: str = "_labeled.jsonl"

    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_console: bool = True
    log_to_file: bool = True

    batch_size: int = 1000
    progress_interval: int = 1000
    max_workers: Optional[int] = None

    def __post_init__(self) -> None:
        """Initializes default paths and ensures that necessary directories exist.

        This method is automatically called after the dataclass is initialized.
        It sets default values for directory paths if they are not explicitly provided
        and then creates these directories if they do not already exist.
        """
        if self.data_dir is None:
            self.data_dir = self.project_root / "src" / "data"
        if self.processed_dir is None:
            self.processed_dir = self.project_root / "src" / "data" / "processed"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        if self.results_dir is None:
            self.results_dir = self.project_root / "results"

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_data_processing_config(self, config: dict) -> None:
        """Load data processing configuration from config dict."""
        if "data_processing" in config:
            dp = config["data_processing"]
            self.observation_days = dp.get("observation_days", self.observation_days)
            self.churn_period_days = dp.get("churn_period_days", self.churn_period_days)
            self.train_ratio = dp.get("train_ratio", self.train_ratio)
            self.random_seed = dp.get("random_seed", self.random_seed)

    def _load_paths_config(self, config: dict) -> None:
        """Load paths configuration from config dict."""
        if "paths" in config:
            paths = config["paths"]
            if "data_dir" in paths:
                self.data_dir = self.project_root / paths["data_dir"]
            if "processed_dir" in paths:
                self.processed_dir = self.project_root / paths["processed_dir"]
            if "logs_dir" in paths:
                self.logs_dir = self.project_root / paths["logs_dir"]
            if "results_dir" in paths:
                self.results_dir = self.project_root / paths["results_dir"]

    def _load_filenames_config(self, config: dict) -> None:
        """Load filenames configuration from config dict."""
        if "filenames" in config:
            filenames = config["filenames"]
            # Input files
            self.game1_csv = filenames.get("game1_csv", self.game1_csv)
            self.game2_jsonl = filenames.get("game2_jsonl", self.game2_jsonl)

            # Intermediate files
            self.game1_converted = filenames.get(
                "game1_converted", self.game1_converted
            )
            self.game1_train = filenames.get("game1_train", self.game1_train)
            self.game1_eval = filenames.get("game1_eval", self.game1_eval)
            self.game2_train = filenames.get("game2_train", self.game2_train)
            self.game2_eval = filenames.get("game2_eval", self.game2_eval)

            # Final labeled datasets
            self.game1_ds1 = filenames.get("game1_ds1", self.game1_ds1)
            self.game1_ds2 = filenames.get("game1_ds2", self.game1_ds2)
            self.game2_ds1 = filenames.get("game2_ds1", self.game2_ds1)
            self.game2_ds2 = filenames.get("game2_ds2", self.game2_ds2)

            # Result files
            self.preparation_results = filenames.get(
                "preparation_results", self.preparation_results
            )
            self.dataset_creation_results = filenames.get(
                "dataset_creation_results", self.dataset_creation_results
            )
            self.pipeline_results = filenames.get(
                "pipeline_results", self.pipeline_results
            )

            # File suffixes
            self.train_suffix = filenames.get("train_suffix", self.train_suffix)
            self.eval_suffix = filenames.get("eval_suffix", self.eval_suffix)
            self.labeled_suffix = filenames.get("labeled_suffix", self.labeled_suffix)

    def _load_logging_config(self, config: dict) -> None:
        """Load logging configuration from config dict."""
        if "logging" in config:
            log = config["logging"]
            self.log_level = log.get("level", self.log_level)
            self.log_to_console = log.get("console", self.log_to_console)
            self.log_to_file = log.get("file", self.log_to_file)

    def _load_performance_config(self, config: dict) -> None:
        """Load performance configuration from config dict."""
        if "performance" in config:
            perf = config["performance"]
            self.batch_size = perf.get("batch_size", self.batch_size)
            self.progress_interval = perf.get(
                "progress_interval", self.progress_interval
            )
            self.max_workers = perf.get("max_workers", self.max_workers)

    @classmethod
    def from_yaml(cls, config_path: Optional[Union[str, Path]] = None) -> "Settings":
        """Creates a Settings instance by loading configuration from a YAML file.

        If no configuration path is provided, it defaults to 'config.yaml' in the
        project root. Settings from the YAML file will override default values.

        Args:
            config_path (Optional[Union[str, Path]]): The path to the YAML configuration file.

        Returns:
            Settings: A configured Settings instance.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        else:
            config_path = Path(config_path)

        settings = cls()

        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            settings._load_data_processing_config(config)
            settings._load_paths_config(config)
            settings._load_filenames_config(config)
            settings._load_logging_config(config)
            settings._load_performance_config(config)

        settings.__post_init__()
        return settings


_settings: Optional[Settings] = None


def get_settings(config_path: Optional[Union[str, Path]] = None) -> Settings:
    """Retrieves the global Settings instance.

    If the settings have not been loaded yet, they will be loaded from the
    specified config path or the default 'config.yaml'. This ensures a
    singleton-like behavior for the project settings.

    Args:
        config_path (Optional[Union[str, Path]]): Path to the YAML configuration file.

    Returns:
        Settings: The global Settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml(config_path)
    return _settings


def reload_settings(config_path: Optional[Union[str, Path]] = None) -> Settings:
    """Reloads the global Settings instance from a configuration file.

    This function forces the settings to be reloaded, discarding any
    previously loaded configuration.

    Args:
        config_path (Optional[Union[str, Path]]): Path to the YAML configuration file.

    Returns:
        Settings: The reloaded global Settings instance.
    """
    global _settings
    _settings = Settings.from_yaml(config_path)
    return _settings
