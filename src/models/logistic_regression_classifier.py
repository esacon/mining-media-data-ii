from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import Settings
from src.models.base_classifier import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    """
    Logistic Regression Classifier implementation.
    Inherits from BaseClassifier and uses scikit-learn's LogisticRegression.
    """

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initializes the LogisticRegressionClassifier.

        Args:
            model_params (Dict[str, Any], optional): Parameters for scikit-learn's
                                                     LogisticRegression. Defaults to None.
            settings (Settings, optional): Project settings for configuration.
                                         Defaults to None.
        """
        super().__init__(model_params, logger)
        if "solver" not in self.model_params:  # Default solver
            self.model_params["solver"] = "liblinear"
        self.model = SklearnLogisticRegression(**self.model_params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the Logistic Regression model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.

        Raises:
            ValueError: If training data validation fails.
            RuntimeError: If model training fails.
        """
        try:
            self.feature_names_ = list(X_train.columns)
            self.model.fit(X_train, y_train)
            self.is_fitted_ = True

            self.logger.info(
                f"Logistic Regression trained on {X_train.shape[0]} samples with {X_train.shape[1]} features"
            )

        except Exception as e:
            self.logger.error(f"Error training Logistic Regression model: {e}")
            self.is_fitted_ = False
            raise RuntimeError(f"Model training failed: {e}") from e

    def _prepare_features(self, X):
        """Helper method to prepare features for prediction."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return X

    def _validate_input(self, X):
        """Helper method to validate input data."""
        if X is None:
            raise ValueError("Input data cannot be None")
        if len(X.shape) != 2:
            raise ValueError("Input data must be 2-dimensional")

    def _align_features(self, X_test):
        """Helper method to align features with training data."""
        if hasattr(self.model, "feature_names_in_"):
            model_features = list(self.model.feature_names_in_)
            missing_cols = set(model_features) - set(X_test.columns)
            if missing_cols:
                error_msg = (
                    f"X_test is missing columns required by the model: {missing_cols}. "
                    f"Model was trained on: {model_features}"
                )
                if self.logger:
                    self.logger.error(error_msg)
                else:
                    print(error_msg)
                raise ValueError(error_msg)
            return X_test[model_features]
        elif self.logger:
            self.logger.warning(
                "Model does not have 'feature_names_in_'. "
                "Proceeding with X_test as is. This may lead to errors if features mismatch."
            )
        return X_test

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the trained Logistic Regression model.
        Ensures X_test columns match those seen during training.
        """
        if self.model is None or not hasattr(self.model, "coef_"):
            error_msg = "Model not trained yet or training failed. Call train() first."
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            raise ValueError(error_msg)

        try:
            X_test_aligned = self._align_features(X_test)
            X_test_aligned = self._prepare_features(X_test_aligned)
            predictions = self.model.predict(X_test_aligned)
            return pd.Series(
                predictions, index=X_test_aligned.index, name="predictions"
            )
        except ValueError as ve:
            if self.logger:
                self.logger.error(
                    f"ValueError during Logistic Regression prediction: {ve}",
                    exc_info=True,
                )
            else:
                print(f"ValueError during Logistic Regression prediction: {ve}")
            raise
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise

    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities using the trained Logistic Regression model.
        Ensures X_test columns match those seen during training.
        """
        if self.model is None or not hasattr(self.model, "coef_"):
            error_msg = "Model not trained yet or training failed. Call train() first."
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            raise ValueError(error_msg)

        try:
            X_test_aligned = self._align_features(X_test)
            X_test_aligned = self._prepare_features(X_test_aligned)
            probabilities = self.model.predict_proba(X_test_aligned)
            return pd.DataFrame(
                probabilities, index=X_test_aligned.index, columns=self.model.classes_
            )
        except ValueError as ve:
            if self.logger:
                self.logger.error(
                    f"ValueError during Logistic Regression probability prediction: {ve}",
                    exc_info=True,
                )
            else:
                print(
                    f"ValueError during Logistic Regression probability prediction: {ve}"
                )
            raise
        except Exception as e:
            self.logger.error(f"Error during probability prediction: {e}")
            raise

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates the Logistic Regression model.
        """
        predictions = self.predict(X_test)  # Handles X_test alignment
        metrics = {}
        try:
            metrics["accuracy"] = accuracy_score(y_test, predictions)
            metrics["precision"] = precision_score(y_test, predictions, zero_division=0)
            metrics["recall"] = recall_score(y_test, predictions, zero_division=0)
            metrics["f1_score"] = f1_score(y_test, predictions, zero_division=0)

        try:
            predictions = self.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
            }

            try:
                if len(y_test.unique()) > 1 and len(self.model.classes_) > 1:
                    y_pred_proba = self.predict_proba(
                        X_test
                    )  # Handles X_test alignment
                    if y_pred_proba.shape[1] > 1:
                        metrics["roc_auc"] = roc_auc_score(
                            y_test, y_pred_proba.iloc[:, 1]
                        )
                    else:
                        metrics["roc_auc"] = 0.5
                else:
                    metrics["roc_auc"] = float("nan")
            except Exception as roc_error:
                self.logger.warning(f"Could not calculate ROC AUC: {roc_error}")
                metrics["roc_auc"] = float("nan")

            return metrics

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error during Logistic Regression evaluation: {e}", exc_info=True
                )
            else:
                print(f"Error during Logistic Regression evaluation: {e}")
        return metrics
