from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import pandas as pd


class BaseClassifier(ABC):
    """
    Abstract base class for all classifiers.
    It defines the common interface for training, predicting, evaluating,
    saving, and loading models.
    """

    def __init__(self, model_params: Dict[str, Any] = None, logger=None):
        """
        Initializes the BaseClassifier.

        Args:
            model_params (Dict[str, Any], optional): Parameters for the specific model.
                                                     Defaults to None.
            logger (optional): Logger instance for logging messages. Defaults to None.
        """
        self.model_params = model_params if model_params is not None else {}
        self.model = None  # The actual model instance (e.g., from scikit-learn)
        self.logger = logger

        if self.logger:
            self.logger.debug(
                f"BaseClassifier initialized for {self.__class__.__name__} with params: {self.model_params}"
            )

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

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Saves the trained model to a file using joblib.

        Args:
            path (Union[str, Path]): The path where the model should be saved.
        """
        if self.model:
            try:
                model_path = Path(path)
                model_path.parent.mkdir(
                    parents=True, exist_ok=True
                )  # Ensure directory exists
                joblib.dump(self.model, model_path)
                if self.logger:
                    self.logger.info(
                        f"Model for {self.__class__.__name__} saved to {model_path}"
                    )
                else:
                    print(f"Model for {self.__class__.__name__} saved to {model_path}")
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Error saving model for {self.__class__.__name__} to {path}: {e}",
                        exc_info=True,
                    )
                else:
                    print(
                        f"Error saving model for {self.__class__.__name__} to {path}: {e}"
                    )
        else:
            if self.logger:
                self.logger.warning(
                    f"No model to save for {self.__class__.__name__}. Train the model first."
                )
            else:
                print(
                    f"No model to save for {self.__class__.__name__}. Train the model first."
                )

    def load_model(self, path: Union[str, Path]) -> None:
        """
        Loads a trained model from a file using joblib.

        Args:
            path (Union[str, Path]): The path from where the model should be loaded.
        """
        try:
            model_path = Path(path)
            if not model_path.exists():
                error_msg = f"Model file not found at {model_path} for {self.__class__.__name__}."
                if self.logger:
                    self.logger.error(error_msg)
                else:
                    print(error_msg)
                raise FileNotFoundError(error_msg)

            self.model = joblib.load(model_path)
            if self.logger:
                self.logger.info(
                    f"Model for {self.__class__.__name__} loaded from {model_path}"
                )
            else:
                print(f"Model for {self.__class__.__name__} loaded from {model_path}")
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error loading model for {self.__class__.__name__} from {path}: {e}",
                    exc_info=True,
                )
            else:
                print(
                    f"Error loading model for {self.__class__.__name__} from {path}: {e}"
                )
            self.model = None  # Ensure model is None if loading fails
