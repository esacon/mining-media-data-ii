from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.config import Settings
from src.utils import LoggerMixin


class BaseClassifier(ABC, LoggerMixin):
    """
    Abstract base class for all classifiers.
    It defines the common interface for training, predicting, evaluating,
    saving, and loading models.
    """

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initializes the BaseClassifier.

        Args:
            model_params (Dict[str, Any], optional): Parameters for the specific model.
                                                     Defaults to None.
            settings (Settings, optional): Project settings for configuration.
                                         Defaults to None.
        """
        self.model_params = model_params if model_params is not None else {}
        self.settings = settings
        self.model = None
        self.feature_names_ = None
        self.is_fitted_ = False

    def _validate_features_basic(
        self, X: pd.DataFrame, operation: str
    ) -> Tuple[bool, str]:
        """
        Validates basic feature requirements (empty, non-numeric, NaN, infinite values).

        Args:
            X (pd.DataFrame): Feature DataFrame.
            operation (str): Type of operation.

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if X.empty:
            return False, f"Empty feature DataFrame provided for {operation}"

        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            return (
                False,
                f"Non-numeric columns found for {operation}: {non_numeric_cols}",
            )

        if X.isnull().any().any():
            nan_cols = X.columns[X.isnull().any()].tolist()
            return False, f"NaN values found in features for {operation}: {nan_cols}"

        if np.isinf(X.values).any():
            inf_cols = X.columns[np.isinf(X).any()].tolist()
            return (
                False,
                f"Infinite values found in features for {operation}: {inf_cols}",
            )

        return True, ""

    def _validate_target(
        self, X: pd.DataFrame, y: pd.Series, operation: str
    ) -> Tuple[bool, str]:
        """
        Validates target variable requirements.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target Series.
            operation (str): Type of operation.

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if len(X) != len(y):
            return (
                False,
                f"Feature and target length mismatch for {operation}: {len(X)} vs {len(y)}",
            )

        if y.isnull().any():
            return False, f"NaN values found in target for {operation}"

        unique_values = y.unique()
        if len(unique_values) != 2 or not all(val in [0, 1] for val in unique_values):
            return (
                False,
                f"Target must be binary (0, 1) for {operation}, got: {unique_values}",
            )

        return True, ""

    def _validate_feature_consistency(
        self, X: pd.DataFrame, operation: str
    ) -> Tuple[bool, str]:
        """
        Validates feature consistency for prediction/evaluation operations.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            operation (str): Type of operation.

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if (
            operation in ["prediction", "evaluation"]
            and self.feature_names_ is not None
        ):
            if list(X.columns) != self.feature_names_:
                missing_features = set(self.feature_names_) - set(X.columns)
                extra_features = set(X.columns) - set(self.feature_names_)
                error_msg = f"Feature mismatch for {operation}."
                if missing_features:
                    error_msg += f" Missing: {list(missing_features)}."
                if extra_features:
                    error_msg += f" Extra: {list(extra_features)}."
                return False, error_msg

        return True, ""

    def _validate_input_data(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        operation: str = "training",
    ) -> Tuple[bool, str]:
        """
        Validates input data for training or prediction.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series, optional): Target Series (required for training).
            operation (str): Type of operation ("training", "prediction", "evaluation").

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Validate basic feature requirements
        is_valid, error_msg = self._validate_features_basic(X, operation)
        if not is_valid:
            return is_valid, error_msg

        # Validate target if provided
        if y is not None:
            is_valid, error_msg = self._validate_target(X, y, operation)
            if not is_valid:
                return is_valid, error_msg

        # Validate feature consistency for prediction/evaluation
        is_valid, error_msg = self._validate_feature_consistency(X, operation)
        if not is_valid:
            return is_valid, error_msg

        return True, ""

    def _validate_and_train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Validates and trains the classifier.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
        """
        is_valid, error_msg = self._validate_input_data(X_train, y_train, "training")
        if not is_valid:
            raise ValueError(f"Training data validation failed: {error_msg}")

        self.feature_names_ = list(X_train.columns)

        self.train(X_train, y_train)
        self.is_fitted_ = True

    def _validate_and_predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Validates and makes predictions.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.Series: Predictions.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be trained before making predictions")

        is_valid, error_msg = self._validate_input_data(X_test, operation="prediction")
        if not is_valid:
            raise ValueError(f"Prediction data validation failed: {error_msg}")

        return self.predict(X_test)

    def _validate_and_predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and predicts class probabilities.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.DataFrame: Class probabilities.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be trained before making predictions")

        is_valid, error_msg = self._validate_input_data(X_test, operation="prediction")
        if not is_valid:
            raise ValueError(f"Prediction data validation failed: {error_msg}")

        return self.predict_proba(X_test)

    def _validate_and_evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Validates and evaluates the model's performance.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be trained before evaluation")

        is_valid, error_msg = self._validate_input_data(X_test, y_test, "evaluation")
        if not is_valid:
            raise ValueError(f"Evaluation data validation failed: {error_msg}")

        return self.evaluate(X_test, y_test)

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

                model_data = {
                    "model": self.model,
                    "feature_names": self.feature_names_,
                    "model_params": self.model_params,
                    "is_fitted": self.is_fitted_,
                }

                joblib.dump(model_data, model_path)
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

            model_data = joblib.load(model_path)

            if isinstance(model_data, dict):
                self.model = model_data.get("model")
                self.feature_names_ = model_data.get("feature_names")
                self.model_params = model_data.get("model_params", {})
                self.is_fitted_ = model_data.get("is_fitted", True)
            else:
                self.model = model_data
                self.feature_names_ = None
                self.is_fitted_ = True

            self.logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model from {path}: {e}")
            self.model = None
            self.is_fitted_ = False
