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
        self.logger.debug(
            f"SklearnDecisionTreeClassifier initialized with params: {self.model_params}"
        )

    def _validate_model_trained(self) -> None:
        """Validate that the model has been trained."""
        if (
            self.model is None
            or not hasattr(self.model, "tree_")
            or self.model.tree_ is None
        ):
            error_msg = "Model not trained yet or training failed. Call train() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _align_test_features(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Align test features with training features."""
        if hasattr(self.model, "feature_names_in_"):
            model_features = list(self.model.feature_names_in_)
            missing_cols = set(model_features) - set(X_test.columns)
            if missing_cols:
                error_msg = (
                    f"X_test is missing columns required by the model: {missing_cols}. "
                    f"Model was trained on: {model_features}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            return X_test[model_features]
        else:
            self.logger.warning(
                "Model does not have 'feature_names_in_'. "
                "Proceeding with X_test as is. This may lead to errors if features mismatch."
            )
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
            self.logger.error(f"Error training Decision Tree model: {e}", exc_info=True)
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
        try:
            X_test_aligned = self._align_test_features(X_test)
            predictions = self.model.predict(X_test_aligned)
            return pd.Series(
                predictions, index=X_test_aligned.index, name="predictions"
            )
        except Exception as e:
            self.logger.error(
                f"Error during Decision Tree prediction: {e}", exc_info=True
            )
            raise

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
        try:
            X_test_aligned = self._align_test_features(X_test)
            probabilities = self.model.predict_proba(X_test_aligned)
            return pd.DataFrame(
                probabilities, index=X_test_aligned.index, columns=self.model.classes_
            )
        except Exception as e:
            self.logger.error(
                f"Error during Decision Tree probability prediction: {e}", exc_info=True
            )
            raise

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates the Decision Tree model.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True test labels.

        Returns:
            Dict[str, float]: Dictionary of performance metrics.
        """
        metrics = {}
        try:
            predictions = self.predict(X_test)
            metrics["accuracy"] = accuracy_score(y_test, predictions)
            metrics["precision"] = precision_score(y_test, predictions, zero_division=0)
            metrics["recall"] = recall_score(y_test, predictions, zero_division=0)
            metrics["f1_score"] = f1_score(y_test, predictions, zero_division=0)

            if hasattr(self.model, "predict_proba"):
                if len(y_test.unique()) > 1 and len(self.model.classes_) > 1:
                    y_pred_proba = self.predict_proba(X_test)
                    if y_pred_proba.shape[1] > 1:
                        metrics["roc_auc"] = roc_auc_score(
                            y_test, y_pred_proba.iloc[:, 1]
                        )
                    else:
                        metrics["roc_auc"] = 0.0
                        self.logger.warning(
                            "ROC AUC cannot be computed for single-class probability output."
                        )
                else:
                    metrics["roc_auc"] = float("nan")
                    self.logger.warning(
                        "ROC AUC score calculation skipped: requires multi-class labels and predictions."
                    )

            self.logger.info(f"Decision Tree evaluation metrics: {metrics}")

        except Exception as e:
            self.logger.error(
                f"Error during Decision Tree evaluation: {e}", exc_info=True
            )
        return metrics

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
                importance_series = importance_series.sort_values(ascending=False)
                self.logger.info(
                    f"Decision Tree feature importances extracted: {len(importance_series)} features"
                )
                return importance_series
            else:
                self.logger.warning(
                    "Feature importances not available - model may not be trained or missing feature names"
                )
                return None
        except Exception as e:
            self.logger.error(
                f"Error extracting Decision Tree feature importances: {e}",
                exc_info=True,
            )
            return None
