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
        Scikit-learn's fit method, when given a pandas DataFrame, will store
        feature names in `self.model.feature_names_in_`.

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
            if self.logger:
                self.logger.error(
                    f"Error training Decision Tree model: {e}", exc_info=True
                )
            else:
                print(f"Error training Decision Tree model: {e}")
            raise

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
        Makes predictions using the trained Decision Tree model.
        Ensures X_test columns match those seen during training if feature_names_in_ is available.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.Series: Predicted labels.

        Raises:
            ValueError: If the model has not been trained yet, or if X_test is missing required columns.
        """
        if (
            self.model is None
            or not hasattr(self.model, "tree_")
            or self.model.tree_ is None
        ):
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
                    f"ValueError during Decision Tree prediction: {ve}", exc_info=True
                )
            else:
                print(f"ValueError during Decision Tree prediction: {ve}")
            raise
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise

    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities using the trained Decision Tree model.
        Ensures X_test columns match those seen during training if feature_names_in_ is available.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.DataFrame: Class probabilities with class labels as columns.

        Raises:
            ValueError: If the model has not been trained yet, or if X_test is missing required columns.
        """
        if (
            self.model is None
            or not hasattr(self.model, "tree_")
            or self.model.tree_ is None
        ):
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
                    f"ValueError during Decision Tree probability prediction: {ve}",
                    exc_info=True,
                )
            else:
                print(f"ValueError during Decision Tree probability prediction: {ve}")
            raise
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
        # predict and predict_proba will handle X_test alignment
        predictions = self.predict(X_test)

        try:
            predictions = self.predict(X_test)

            if hasattr(self.model, "predict_proba"):
                if len(y_test.unique()) > 1 and len(self.model.classes_) > 1:
                    # Pass the original X_test, predict_proba will align it
                    y_pred_proba = self.predict_proba(X_test)
                    if y_pred_proba.shape[1] > 1:
                        metrics["roc_auc"] = roc_auc_score(
                            y_test, y_pred_proba.iloc[:, 1]
                        )
                    else:
                        metrics["roc_auc"] = 0.0
                        if self.logger:
                            self.logger.warning(
                                "ROC AUC cannot be computed for single-class probability output."
                            )
                else:
                    metrics["roc_auc"] = float("nan")
                    if self.logger:
                        self.logger.warning(
                            "ROC AUC score calculation skipped: requires multi-class labels and predictions."
                        )

            return metrics

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error during Decision Tree evaluation: {e}", exc_info=True
                )
            else:
                print(f"Error during Decision Tree evaluation: {e}")
        return metrics
