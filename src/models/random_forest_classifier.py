import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any

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
            self.logger.debug(f"SklearnRandomForestClassifier initialized with params: {self.model_params}")


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
                self.logger.error(f"Error training Random Forest model: {e}", exc_info=True)
            else:
                print(f"Error training Random Forest model: {e}")
            raise

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the trained Random Forest model.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.Series: Predicted labels.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None or not hasattr(self.model, 'estimators_'): # Check if model is fitted
            error_msg = "Model not trained yet or training failed. Call train() first."
            if self.logger: self.logger.error(error_msg)
            else: print(error_msg)
            raise ValueError(error_msg)
        
        try:
            predictions = self.model.predict(X_test)
            return pd.Series(predictions, index=X_test.index, name="predictions")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during Random Forest prediction: {e}", exc_info=True)
            else:
                print(f"Error during Random Forest prediction: {e}")
            raise

    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities using the trained Random Forest model.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.DataFrame: Class probabilities. Columns are class labels.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None or not hasattr(self.model, 'estimators_'): # Check if model is fitted
            error_msg = "Model not trained yet or training failed. Call train() first."
            if self.logger: self.logger.error(error_msg)
            else: print(error_msg)
            raise ValueError(error_msg)

        try:
            probabilities = self.model.predict_proba(X_test)
            return pd.DataFrame(probabilities, index=X_test.index, columns=self.model.classes_)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during Random Forest probability prediction: {e}", exc_info=True)
            else:
                print(f"Error during Random Forest probability prediction: {e}")
            raise

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates the Random Forest model.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True test labels.

        Returns:
            Dict[str, float]: Dictionary of performance metrics.
        """
        predictions = self.predict(X_test)
        metrics = {}
        try:
            metrics["accuracy"] = accuracy_score(y_test, predictions)
            metrics["precision"] = precision_score(y_test, predictions, zero_division=0)
            metrics["recall"] = recall_score(y_test, predictions, zero_division=0)
            metrics["f1_score"] = f1_score(y_test, predictions, zero_division=0)

            if hasattr(self.model, "predict_proba"):
                if len(y_test.unique()) > 1 and len(self.model.classes_) > 1:
                    y_pred_proba = self.predict_proba(X_test)
                    if y_pred_proba.shape[1] > 1:
                        metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba.iloc[:, 1])
                    else:
                        metrics["roc_auc"] = 0.0
                        if self.logger: self.logger.warning("ROC AUC cannot be computed for single-class probability output.")
                else:
                    metrics["roc_auc"] = float('nan')
                    if self.logger:
                        self.logger.warning("ROC AUC score calculation skipped: requires multi-class labels and predictions.")
            
            if self.logger:
                self.logger.info(f"Random Forest evaluation metrics: {metrics}")
            else:
                print(f"Random Forest evaluation metrics: {metrics}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during Random Forest evaluation: {e}", exc_info=True)
            else:
                print(f"Error during Random Forest evaluation: {e}")
        
        return metrics
