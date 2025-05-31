from typing import Any, Dict, Optional

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

from .base_classifier import BaseClassifier


class DecisionTreeClassifier(BaseClassifier):
    """
    Decision Tree Classifier implementation.
    Inherits from BaseClassifier and uses scikit-learn's DecisionTreeClassifier.
    """

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the DecisionTreeClassifier.

        Args:
            model_params (Dict[str, Any], optional): Parameters for scikit-learn's
                                                     DecisionTreeClassifier. Defaults to None.
        """
        super().__init__(model_params)
        self.model = SklearnDecisionTreeClassifier(**self.model_params)

    def _validate_model_trained(self) -> None:
        """Validate that the model has been trained."""
        if (
            self.model is None
            or not hasattr(self.model, "tree_")
            or self.model.tree_ is None
        ):
            raise ValueError(
                "Model not trained yet or training failed. Call train() first."
            )

    def _align_test_features(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Align test features with training features."""
        if hasattr(self.model, "feature_names_in_"):
            model_features = list(self.model.feature_names_in_)
            missing_cols = set(model_features) - set(X_test.columns)
            if missing_cols:
                raise ValueError(
                    f"X_test is missing columns required by the model: {missing_cols}"
                )
            return X_test[model_features]
        return X_test

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the Decision Tree model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        """
        try:
            self.model.fit(X_train, y_train)
            self.logger.info("Decision Tree model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error training Decision Tree model: {e}")
            raise

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the trained Decision Tree model.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.Series: Predicted labels.

        Raises:
            ValueError: If the model has not been trained yet, or if X_test is missing required columns.
        """
        self._validate_model_trained()
        X_test_aligned = self._align_test_features(X_test)
        predictions = self.model.predict(X_test_aligned)
        return pd.Series(predictions, index=X_test_aligned.index, name="predictions")

    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities using the trained Decision Tree model.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.DataFrame: Class probabilities. Columns are class labels.

        Raises:
            ValueError: If the model has not been trained yet, or if X_test is missing required columns.
        """
        self._validate_model_trained()
        X_test_aligned = self._align_test_features(X_test)
        probabilities = self.model.predict_proba(X_test_aligned)
        return pd.DataFrame(
            probabilities, index=X_test_aligned.index, columns=self.model.classes_
        )

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates the Decision Tree model.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True test labels.

        Returns:
            Dict[str, float]: Dictionary of performance metrics.
        """
        try:
            predictions = self.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
            }

            if hasattr(self.model, "predict_proba"):
                if len(y_test.unique()) > 1 and len(self.model.classes_) > 1:
                    y_pred_proba = self.predict_proba(X_test)
                    if y_pred_proba.shape[1] > 1:
                        metrics["roc_auc"] = roc_auc_score(
                            y_test, y_pred_proba.iloc[:, 1]
                        )
                    else:
                        metrics["roc_auc"] = 0.0
                else:
                    metrics["roc_auc"] = float("nan")

            return metrics
        except Exception as e:
            self.logger.error(f"Error during Decision Tree evaluation: {e}")
            return {}

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importances from the trained Decision Tree model.

        Returns:
            pd.Series: Feature importances with feature names as index,
                      or None if model not trained.
        """
        self._validate_model_trained()
        try:
            if hasattr(self.model, "feature_importances_") and hasattr(
                self.model, "feature_names_in_"
            ):
                feature_names = self.model.feature_names_in_
                importances = self.model.feature_importances_
                importance_series = pd.Series(
                    importances, index=feature_names, name="importance"
                )
                return importance_series.sort_values(ascending=False)
            return None
        except Exception as e:
            self.logger.error(
                f"Error extracting Decision Tree feature importances: {e}"
            )
            return None
