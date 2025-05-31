from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .base_classifier import BaseClassifier


class RandomForestClassifier(BaseClassifier):
    """
    Random Forest Classifier implementation.
    Inherits from BaseClassifier and uses scikit-learn's RandomForestClassifier.
    """

    def __init__(self, model_params: Dict[str, Any] = None, logger=None):
        """
        Initializes the RandomForestClassifier.

        Args:
            model_params (Dict[str, Any], optional): Parameters for scikit-learn's
                                                     RandomForestClassifier. Defaults to None.
            logger (optional): Logger instance. Defaults to None.
        """
        super().__init__(model_params, logger)
        self.model = SklearnRandomForestClassifier(**self.model_params)
        if self.logger:
            self.logger.debug(
                f"SklearnRandomForestClassifier initialized with params: {self.model_params}"
            )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the Random Forest model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        """
        try:
            self.model.fit(X_train, y_train)
            if self.logger:
                self.logger.info("Random Forest model trained successfully.")
            else:
                print("Random Forest model trained successfully.")
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error training Random Forest model: {e}", exc_info=True
                )
            else:
                print(f"Error training Random Forest model: {e}")
            raise

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the trained Random Forest model.
        Ensures X_test columns match those seen during training.
        """
        if self.model is None or not hasattr(self.model, "estimators_"):
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
            if self.logger: self.logger.error(f"ValueError during Random Forest prediction: {ve}", exc_info=True)
            else: print(f"ValueError during Random Forest prediction: {ve}")
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error during Random Forest prediction: {e}", exc_info=True
                )
            else:
                print(f"Error during Random Forest prediction: {e}")
            raise

    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities using the trained Random Forest model.
        Ensures X_test columns match those seen during training.
        """
        if self.model is None or not hasattr(self.model, "estimators_"):
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
            if self.logger: self.logger.error(f"ValueError during Random Forest probability prediction: {ve}", exc_info=True)
            else: print(f"ValueError during Random Forest probability prediction: {ve}")
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error during Random Forest probability prediction: {e}",
                    exc_info=True,
                )
            else:
                print(f"Error during Random Forest probability prediction: {e}")
            raise

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates the Random Forest model.
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
                self.logger.info(f"Random Forest evaluation metrics: {metrics}")
            else:
                print(f"Random Forest evaluation metrics: {metrics}")

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error during Random Forest evaluation: {e}", exc_info=True
                )
            else:
                print(f"Error during Random Forest evaluation: {e}")
        return metrics