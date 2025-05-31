from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd

from src.utils import LoggerMixin


class BaseClassifier(ABC, LoggerMixin):
    """
    Abstract base class for all classifiers.
    It defines the common interface for training, predicting, evaluating,
    saving, and loading models.
    """

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the BaseClassifier.

        Args:
            model_params (Dict[str, Any], optional): Parameters for the specific model.
                                                     Defaults to None.
        """
        self.model_params = model_params if model_params is not None else {}
        self.model = None

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the classifier using the provided training data.

        Args:
            X_train (pd.DataFrame): DataFrame containing the training features.
            y_train (pd.Series): Series containing the training labels.
        """
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Makes predictions on new data.

        Args:
            X_test (pd.DataFrame): DataFrame containing the test features.

        Returns:
            pd.Series: Series containing the predicted labels.
        """
        pass

    @abstractmethod
    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities for new data.

        Args:
            X_test (pd.DataFrame): DataFrame containing the test features.

        Returns:
            pd.DataFrame: DataFrame containing class probabilities for each sample.
                          Columns should correspond to class labels.
        """
        pass

    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates the model's performance on the test data.

        Args:
            X_test (pd.DataFrame): DataFrame containing the test features.
            y_test (pd.Series): Series containing the true test labels.

        Returns:
            Dict[str, float]: A dictionary of performance metrics (e.g., accuracy, precision).
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importances from the trained model.

        Returns:
            Union[pd.Series, None]: Feature importances with feature names as index,
                                   or None if not available.
        """
        pass

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Saves the trained model to a file using joblib.

        Args:
            path (Union[str, Path]): The path where the model should be saved.
        """
        if self.model:
            try:
                model_path = Path(path)
                if not model_path.is_absolute() and "results" not in str(model_path):
                    model_path = Path("results") / "models" / model_path.name
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.model, model_path)
                self.logger.info(f"Model saved to {model_path}")
            except Exception as e:
                self.logger.error(f"Error saving model to {path}: {e}")
        else:
            self.logger.warning("No model to save. Train the model first.")

    def load_model(self, path: Union[str, Path]) -> None:
        """
        Loads a trained model from a file using joblib.

        Args:
            path (Union[str, Path]): The path from where the model should be loaded.
        """
        try:
            model_path = Path(path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.model = joblib.load(model_path)
            self.logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model from {path}: {e}")
            self.model = None
