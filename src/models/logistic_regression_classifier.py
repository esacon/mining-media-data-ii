from typing import Any, Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .base_classifier import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    """
    Logistic Regression Classifier implementation.
    Inherits from BaseClassifier and uses scikit-learn's LogisticRegression.
    """

    def __init__(self, model_params: Dict[str, Any] = None, logger=None):
        """
        Initializes the LogisticRegressionClassifier.

        Args:
            model_params (Dict[str, Any], optional): Parameters for scikit-learn's
                                                     LogisticRegression. Defaults to None.
            logger (optional): Logger instance. Defaults to None.
        """
        super().__init__(model_params, logger)
        if "solver" not in self.model_params: # Default solver
            self.model_params["solver"] = "liblinear"
        self.model = SklearnLogisticRegression(**self.model_params)
        if self.logger:
            self.logger.debug(
                f"SklearnLogisticRegression initialized with params: {self.model_params}"
            )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the Logistic Regression model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        """
        try:
            self.model.fit(X_train, y_train)
            if self.logger:
                self.logger.info("Logistic Regression model trained successfully.")
            else:
                print("Logistic Regression model trained successfully.")
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error training Logistic Regression model: {e}", exc_info=True
                )
            else:
                print(f"Error training Logistic Regression model: {e}")
            raise

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the trained Logistic Regression model.
        Ensures X_test columns match those seen during training.
        """
        if self.model is None or not hasattr(self.model, "coef_"):
            error_msg = "Model not trained yet or training failed. Call train() first."
            if self.logger: self.logger.error(error_msg)
            else: print(error_msg)
            raise ValueError(error_msg)

        try:
            X_test_aligned = X_test
            if hasattr(self.model, "feature_names_in_"):
                model_features = list(self.model.feature_names_in_)
                missing_cols = set(model_features) - set(X_test.columns)
                if missing_cols:
                    error_msg = (
                        f"X_test is missing columns required by the model: {missing_cols}. "
                        f"Model was trained on: {model_features}"
                    )
                    if self.logger: self.logger.error(error_msg)
                    else: print(error_msg)
                    raise ValueError(error_msg)
                X_test_aligned = X_test[model_features]
            elif self.logger:
                 self.logger.warning(
                        "Model does not have 'feature_names_in_'. "
                        "Proceeding with X_test as is. This may lead to errors if features mismatch."
                    )

            predictions = self.model.predict(X_test_aligned)
            return pd.Series(predictions, index=X_test_aligned.index, name="predictions")
        except ValueError as ve:
            if self.logger: self.logger.error(f"ValueError during Logistic Regression prediction: {ve}", exc_info=True)
            else: print(f"ValueError during Logistic Regression prediction: {ve}")
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error during Logistic Regression prediction: {e}", exc_info=True
                )
            else:
                print(f"Error during Logistic Regression prediction: {e}")
            raise

    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities using the trained Logistic Regression model.
        Ensures X_test columns match those seen during training.
        """
        if self.model is None or not hasattr(self.model, "coef_"):
            error_msg = "Model not trained yet or training failed. Call train() first."
            if self.logger: self.logger.error(error_msg)
            else: print(error_msg)
            raise ValueError(error_msg)

        try:
            X_test_aligned = X_test
            if hasattr(self.model, "feature_names_in_"):
                model_features = list(self.model.feature_names_in_)
                missing_cols = set(model_features) - set(X_test.columns)
                if missing_cols:
                    error_msg = (
                        f"X_test is missing columns required by the model for predict_proba: {missing_cols}. "
                        f"Model was trained on: {model_features}"
                    )
                    if self.logger: self.logger.error(error_msg)
                    else: print(error_msg)
                    raise ValueError(error_msg)
                X_test_aligned = X_test[model_features]
            elif self.logger:
                 self.logger.warning(
                        "Model does not have 'feature_names_in_'. "
                        "Proceeding with X_test as is for predict_proba. This may lead to errors if features mismatch."
                    )
                    
            probabilities = self.model.predict_proba(X_test_aligned)
            return pd.DataFrame(
                probabilities, index=X_test_aligned.index, columns=self.model.classes_
            )
        except ValueError as ve:
            if self.logger: self.logger.error(f"ValueError during Logistic Regression probability prediction: {ve}", exc_info=True)
            else: print(f"ValueError during Logistic Regression probability prediction: {ve}")
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error during Logistic Regression probability prediction: {e}",
                    exc_info=True,
                )
            else:
                print(f"Error during Logistic Regression probability prediction: {e}")
            raise

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates the Logistic Regression model.
        """
        predictions = self.predict(X_test) # Handles X_test alignment
        metrics = {}
        try:
            metrics["accuracy"] = accuracy_score(y_test, predictions)
            metrics["precision"] = precision_score(y_test, predictions, zero_division=0)
            metrics["recall"] = recall_score(y_test, predictions, zero_division=0)
            metrics["f1_score"] = f1_score(y_test, predictions, zero_division=0)

            if hasattr(self.model, "predict_proba"):
                if len(y_test.unique()) > 1 and len(self.model.classes_) > 1:
                    y_pred_proba = self.predict_proba(X_test) # Handles X_test alignment
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

            if self.logger:
                self.logger.info(f"Logistic Regression evaluation metrics: {metrics}")
            else:
                print(f"Logistic Regression evaluation metrics: {metrics}")

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error during Logistic Regression evaluation: {e}", exc_info=True
                )
            else:
                print(f"Error during Logistic Regression evaluation: {e}")
        return metrics