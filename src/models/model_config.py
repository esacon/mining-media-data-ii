"""
Model configuration management for centralized model parameter handling.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import Settings
from src.utils import LoggerMixin


@dataclass
class ModelConfig:
    """Configuration class for a single model."""

    name: str
    params: Dict[str, Any]
    description: Optional[str] = None


class ModelConfigManager(LoggerMixin):
    """
    Centralized manager for model configurations.
    Provides validation, parameter management, and configuration utilities.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the model configuration manager.

        Args:
            settings (Settings): Project settings instance.
        """
        self.settings = settings
        self._model_configs = self._initialize_model_configs()
        self.logger.info("ModelConfigManager initialized with configured models")

    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """
        Initialize model configurations from settings.

        Returns:
            Dict[str, ModelConfig]: Dictionary of model configurations.
        """
        configs = {
            "DecisionTree": ModelConfig(
                name="DecisionTree",
                params=self.settings.decision_tree_params.copy(),
                description="Decision Tree classifier with configurable parameters",
            ),
            "LogisticRegression": ModelConfig(
                name="LogisticRegression",
                params=self.settings.logistic_regression_params.copy(),
                description="Logistic Regression with L1/L2 regularization",
            ),
            "RandomForest": ModelConfig(
                name="RandomForest",
                params=self.settings.random_forest_params.copy(),
                description="Random Forest ensemble classifier",
            ),
        }

        # Validate configurations
        for model_name, config in configs.items():
            self._validate_model_config(config)
            self.logger.debug(f"Validated configuration for {model_name}")

        return configs

    def _validate_model_config(self, config: ModelConfig) -> None:
        """
        Validate a model configuration.

        Args:
            config (ModelConfig): Model configuration to validate.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not config.name:
            raise ValueError("Model name cannot be empty")

        if not isinstance(config.params, dict):
            raise ValueError(f"Model parameters must be a dictionary for {config.name}")

        # Validate required parameters
        if "random_state" not in config.params:
            self.logger.warning(
                f"No random_state specified for {config.name}, reproducibility may be affected"
            )

        # Model-specific validations
        if config.name == "LogisticRegression":
            self._validate_logistic_regression_config(config)
        elif config.name == "RandomForest":
            self._validate_random_forest_config(config)
        elif config.name == "DecisionTree":
            self._validate_decision_tree_config(config)

    def _validate_logistic_regression_config(self, config: ModelConfig) -> None:
        """Validate Logistic Regression specific parameters."""
        params = config.params

        if "max_iter" in params and params["max_iter"] < 100:
            self.logger.warning(
                f"Low max_iter ({params['max_iter']}) for LogisticRegression may cause convergence issues"
            )

        if "C" in params and params["C"] <= 0:
            raise ValueError("Regularization parameter C must be positive")

    def _validate_random_forest_config(self, config: ModelConfig) -> None:
        """Validate Random Forest specific parameters."""
        params = config.params

        if "n_estimators" in params and params["n_estimators"] < 10:
            self.logger.warning(
                f"Low n_estimators ({params['n_estimators']}) for RandomForest may underperform"
            )

        if "max_features" in params and isinstance(params["max_features"], str):
            valid_options = ["sqrt", "log2", "auto"]
            if params["max_features"] not in valid_options:
                raise ValueError(
                    f"Invalid max_features: {params['max_features']}. Must be one of {valid_options}"
                )

    def _validate_decision_tree_config(self, config: ModelConfig) -> None:
        """Validate Decision Tree specific parameters."""
        params = config.params

        if (
            "max_depth" in params
            and params["max_depth"] is not None
            and params["max_depth"] < 1
        ):
            raise ValueError("max_depth must be None or positive integer")

    def get_model_config(self, model_name: str) -> ModelConfig:
        """
        Get configuration for a specific model.

        Args:
            model_name (str): Name of the model.

        Returns:
            ModelConfig: Model configuration.

        Raises:
            KeyError: If model not found.
        """
        if model_name not in self._model_configs:
            available = list(self._model_configs.keys())
            raise KeyError(
                f"Model '{model_name}' not found. Available models: {available}"
            )

        return self._model_configs[model_name]

    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific model.

        Args:
            model_name (str): Name of the model.

        Returns:
            Dict[str, Any]: Model parameters.
        """
        return self.get_model_config(model_name).params.copy()

    def update_model_params(self, model_name: str, new_params: Dict[str, Any]) -> None:
        """
        Update parameters for a specific model.

        Args:
            model_name (str): Name of the model.
            new_params (Dict[str, Any]): New parameters to update.
        """
        config = self.get_model_config(model_name)
        config.params.update(new_params)
        self._validate_model_config(config)
        self.logger.info(f"Updated parameters for {model_name}")

    def list_available_models(self) -> List[str]:
        """
        Get list of available model names.

        Returns:
            List[str]: List of available model names.
        """
        return list(self._model_configs.keys())

    def get_multicollinearity_threshold(self) -> float:
        """
        Get the multicollinearity threshold from settings.

        Returns:
            float: Multicollinearity threshold.
        """
        return self.settings.multicollinearity_threshold

    def log_configuration_summary(self) -> None:
        """Log a summary of all model configurations."""
        self.logger.info("Model Configuration Summary:")
        self.logger.info(
            f"Multicollinearity threshold: {self.get_multicollinearity_threshold()}"
        )

        model_names = list(self._model_configs.keys())
        self.logger.info(f"Configured models: {', '.join(model_names)}")
