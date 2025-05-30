import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any

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
        # Ensure 'solver' is appropriate if not specified, e.g., 'liblinear' for small datasets
        if 'solver' not in self.model_params:
            self.model_params['solver'] = 'liblinear'
        self.model = SklearnLogisticRegression(**self.model_params)
        if self.logger:
            self.logger.debug(f"SklearnLogisticRegression initialized with params: {self.model_params}")

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
                self.logger.error(f"Error training Logistic Regression model: {e}", exc_info=True)
            else:
                print(f"Error training Logistic Regression model: {e}")
            raise

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the trained Logistic Regression model.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.Series: Predicted labels.
        
        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None or not hasattr(self.model, 'coef_'): # Check if model is fitted
            error_msg = "Model not trained yet or training failed. Call train() first."
            if self.logger: self.logger.error(error_msg)
            else: print(error_msg)
            raise ValueError(error_msg)
        
        try:
            predictions = self.model.predict(X_test)
            return pd.Series(predictions, index=X_test.index, name="predictions")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during Logistic Regression prediction: {e}", exc_info=True)
            else:
                print(f"Error during Logistic Regression prediction: {e}")
            raise

    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities using the trained Logistic Regression model.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.DataFrame: Class probabilities. Columns are class labels.
        
        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None or not hasattr(self.model, 'coef_'): # Check if model is fitted
            error_msg = "Model not trained yet or training failed. Call train() first."
            if self.logger: self.logger.error(error_msg)
            else: print(error_msg)
            raise ValueError(error_msg)
        
        try:
            probabilities = self.model.predict_proba(X_test)
            return pd.DataFrame(probabilities, index=X_test.index, columns=self.model.classes_)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during Logistic Regression probability prediction: {e}", exc_info=True)
            else:
                print(f"Error during Logistic Regression probability prediction: {e}")
            raise

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates the Logistic Regression model.

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
                self.logger.info(f"Logistic Regression evaluation metrics: {metrics}")
            else:
                print(f"Logistic Regression evaluation metrics: {metrics}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during Logistic Regression evaluation: {e}", exc_info=True)
            else:
                print(f"Error during Logistic Regression evaluation: {e}")
        
        return metrics
