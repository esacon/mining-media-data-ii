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

from src.config import Settings
from src.models.base_classifier import BaseClassifier


class DecisionTreeClassifier(BaseClassifier):
    """
    Decision Tree Classifier implementation.
    Inherits from BaseClassifier and uses scikit-learn's DecisionTreeClassifier.
    """

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initializes the DecisionTreeClassifier.

        Args:
            model_params (Dict[str, Any], optional): Parameters for scikit-learn's
                                                     DecisionTreeClassifier. Defaults to None.
            settings (Settings, optional): Project settings for configuration.
                                         Defaults to None.
        """
        super().__init__(model_params, settings)
        default_params = {
            "random_state": 42,
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "criterion": "gini",
            "max_features": "sqrt",
            "class_weight": "balanced",
            "ccp_alpha": 0.01,
        }
        default_params.update(self.model_params)
        self.model_params = default_params
        self.model = SklearnDecisionTreeClassifier(**self.model_params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the Decision Tree model.

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
                f"Decision Tree trained on {X_train.shape[0]} samples with {X_train.shape[1]} features"
            )

        except Exception as e:
            self.logger.error(f"Error training Decision Tree model: {e}")
            self.is_fitted_ = False
            raise RuntimeError(f"Model training failed: {e}") from e

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the trained Decision Tree model.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.Series: Predicted labels.

        Raises:
            ValueError: If the model has not been trained yet or data validation fails.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be trained before making predictions")

        try:
            predictions = self.model.predict(X_test)
            return pd.Series(predictions, index=X_test.index, name="predictions")
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise

    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities using the trained Decision Tree model.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.DataFrame: Class probabilities with class labels as columns.

        Raises:
            ValueError: If the model has not been trained yet or data validation fails.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be trained before making predictions")

        try:
            probabilities = self.model.predict_proba(X_test)
            return pd.DataFrame(
                probabilities, index=X_test.index, columns=self.model.classes_
            )
        except Exception as e:
            self.logger.error(f"Error during probability prediction: {e}")
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
        if not self.is_fitted_:
            raise ValueError("Model must be trained before evaluation")

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
                    y_pred_proba = self.predict_proba(X_test)
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
            self.logger.error(f"Error during Decision Tree evaluation: {e}")
            return {
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1_score": float("nan"),
                "roc_auc": float("nan"),
            }

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importances from the trained Decision Tree model.

        Returns:
            pd.Series: Feature importances with feature names as index,
                      or None if model not trained or extraction fails.
        """
        if not self.is_fitted_:
            self.logger.warning(
                "Model not trained yet. Cannot extract feature importances."
            )
            return None

        try:
            if (
                hasattr(self.model, "feature_importances_")
                and self.feature_names_ is not None
            ):
                importances = self.model.feature_importances_
                importance_series = pd.Series(
                    importances, index=self.feature_names_, name="importance"
                )
                return importance_series.sort_values(ascending=False)
            else:
                self.logger.warning(
                    "Model feature importances not available for extraction."
                )
                return None

        except Exception as e:
            self.logger.error(
                f"Error extracting Decision Tree feature importances: {e}"
            )
            return None
