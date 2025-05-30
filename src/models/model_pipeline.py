from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import Settings
from src.models.base_classifier import BaseClassifier
from src.models.decision_tree_classifier import DecisionTreeClassifier
from src.models.logistic_regression_classifier import LogisticRegressionClassifier
from src.models.model_config import ModelConfigManager
from src.models.random_forest_classifier import RandomForestClassifier
from src.utils import LoggerMixin, generate_performance_report, save_json


class ModelPipeline(LoggerMixin):
    """
    Orchestrates the complete model training and evaluation pipeline.
    """

    def __init__(self, settings: Settings):
        """
        Initializes the ModelPipeline.

        Args:
            settings (Settings): Configuration settings containing paths, model parameters, etc.
        """
        self.settings = settings
        self.config_manager = ModelConfigManager(settings)
        self.processed_dir = settings.processed_dir
        self.results_dir = settings.results_dir
        self.features_dir = self.results_dir / "features"
        self.models_output_dir = self.results_dir / "models"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("ModelPipeline initialized.")
        self.logger.info(f"Results directory: {self.results_dir}")
        self.logger.info(f"Models output directory: {self.models_output_dir}")

        # Log configuration summary
        self.config_manager.log_configuration_summary()

        self.all_game_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.all_feature_importance: Dict[str, Dict[str, Dict[str, float]]] = {}

    def _print_header(self, title: str):
        """
        Prints a formatted header to console and logs it.

        Args:
            title (str): The title to display.
        """
        header_str = f"\n{'=' * 70}\n===== {title.upper()} =====\n{'=' * 70}"
        print(header_str)
        self.logger.info(title)

    def _print_subheader(self, title: str):
        """
        Prints a formatted subheader to console and logs it.

        Args:
            title (str): The subheader to display.
        """
        subheader_str = f"\n{'-' * 50}\n--- {title} ---\n{'-' * 50}"
        print(subheader_str)
        self.logger.info(title)

    def _load_features(self, feature_file_path: Path) -> Optional[pd.DataFrame]:
        """
        Loads features from a CSV file.

        Args:
            feature_file_path (Path): The full path to the feature CSV file.

        Returns:
            Optional[pd.DataFrame]: The loaded DataFrame, or None if an error occurs.
        """
        try:
            df = pd.read_csv(feature_file_path)
            for col in df.columns:
                if df[col].dtype == "object" and col not in [
                    "player_id",
                    "op_start",
                    "op_end",
                    "cp_start",
                    "cp_end",
                ]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            self.logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            self.logger.error(f"Feature file not found at {feature_file_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading feature file {feature_file_path}: {e}")
            return None

    def _validate_and_convert_target(
        self, df: pd.DataFrame, target_column: str, game_id: str
    ) -> Optional[pd.Series]:
        """
        Validates and converts the target column to integer format (0 or 1).

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_column (str): The name of the target column.
            game_id (str): Identifier for the game (for logging/error messages).

        Returns:
            Optional[pd.Series]: The converted target Series, or None if conversion fails.
        """
        if target_column not in df.columns:
            self.logger.error(
                f"Target column '{target_column}' not found for {game_id}"
            )
            return None

        if df[target_column].isnull().any():
            nan_count = df[target_column].isnull().sum()
            self.logger.warning(
                f"Target column contains {nan_count} NaN values. Dropping these rows."
            )
            df.dropna(subset=[target_column], inplace=True)
            if df.empty:
                self.logger.error(
                    f"DataFrame empty after dropping NaN targets for {game_id}"
                )
                return None

        try:
            map_to_bool = {
                "True": True,
                "False": False,
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                1: True,
                0: False,
                1.0: True,
                0.0: False,
            }
            mapped_series = df[target_column].astype(str).map(map_to_bool)
            df[target_column] = mapped_series.combine_first(
                df[target_column].astype(bool)
            )

            if df[target_column].isnull().any():
                raise ValueError("Mapping to boolean resulted in NaNs.")
        except Exception as e:
            self.logger.error(f"Failed to convert target to boolean for {game_id}: {e}")
            return None

        return df[target_column].astype(int)

    def _detect_id_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Automatically detects ID-like columns that should not be used for modeling.

        Args:
            df (pd.DataFrame): DataFrame to analyze.

        Returns:
            List[str]: List of column names that appear to be ID columns.
        """
        id_columns = []

        for col in df.columns:
            if col in ["player_id", "op_start", "op_end", "cp_start", "cp_end"]:
                id_columns.append(col)
                continue

            if df[col].dtype == "object":
                unique_ratio = df[col].nunique() / len(df)
                avg_length = df[col].astype(str).str.len().mean()

                if unique_ratio > 0.8 and avg_length > 10:
                    id_columns.append(col)

        return id_columns

    def _clean_feature_columns(self, X: pd.DataFrame, game_id: str) -> pd.DataFrame:
        """
        Cleans and validates feature columns, converting to numeric and handling NaNs.

        Args:
            X (pd.DataFrame): DataFrame of features.
            game_id (str): Identifier for the game (for logging/error messages).

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        if X.empty:
            return X

        X_clean = X.copy()
        id_columns = self._detect_id_columns(X_clean)

        for col in X_clean.columns:
            if col in id_columns:
                continue

            if not pd.api.types.is_numeric_dtype(X_clean[col]):
                X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce")

        if X_clean.isnull().values.any():
            nan_counts = X_clean.isnull().sum()
            nan_cols = nan_counts[nan_counts > 0].index
            self.logger.warning(
                f"NaN values found in feature columns for {game_id}: {list(nan_cols)}"
            )

            for col in nan_cols:
                median_val = X_clean[col].median()

                if pd.isna(median_val):
                    self.logger.warning(
                        f"Column '{col}' has all NaN values for {game_id}. Filling with 0."
                    )
                    X_clean[col] = 0
                else:
                    X_clean.loc[:, col] = X_clean[col].fillna(median_val)

                    if X_clean[col].isnull().any():
                        X_clean.loc[:, col] = X_clean[col].fillna(0)

        return X_clean

    def _remove_constant_columns(
        self, X: pd.DataFrame, game_id: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Removes constant (zero-variance) columns from the DataFrame.

        Args:
            X (pd.DataFrame): DataFrame of features.
            game_id (str): Identifier for the game (for logging/error messages).

        Returns:
            Tuple[pd.DataFrame, List[str]]: Cleaned DataFrame and list of remaining feature names.
        """
        constant_cols = []
        for col in X.columns:
            unique_vals = X[col].nunique()
            if unique_vals <= 1:
                constant_cols.append(col)
            elif unique_vals == 2 and X[col].var() < 1e-10:
                constant_cols.append(col)

        if constant_cols:
            self.logger.warning(
                f"Constant/low-variance columns found for {game_id}: {constant_cols}. Dropping them."
            )
            X_clean = X.drop(columns=constant_cols)
        else:
            X_clean = X.copy()

        feature_columns = list(X_clean.columns)
        if not feature_columns:
            self.logger.error(f"No feature columns remaining for {game_id}")

        return X_clean, feature_columns

    def _handle_multicollinearity(
        self, X: pd.DataFrame, game_id: str, threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Detects and removes highly correlated features to reduce multicollinearity.

        Args:
            X (pd.DataFrame): DataFrame of features.
            game_id (str): Identifier for the game (for logging).
            threshold (float, optional): Correlation threshold above which features are considered highly correlated.
                                       If None, uses settings.multicollinearity_threshold.

        Returns:
            Tuple[pd.DataFrame, List[str]]: DataFrame with correlated features removed and list of removed features.
        """
        if threshold is None:
            threshold = self.config_manager.get_multicollinearity_threshold()

        if X.shape[1] < 2:
            return X, []

        correlation_matrix = X.corr().abs()
        highly_correlated_pairs = []
        features_to_remove = set()

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if corr_value > threshold:
                    feature_i = correlation_matrix.columns[i]
                    feature_j = correlation_matrix.columns[j]
                    highly_correlated_pairs.append((feature_i, feature_j, corr_value))

                    if feature_i > feature_j:
                        features_to_remove.add(feature_i)
                    else:
                        features_to_remove.add(feature_j)

        if highly_correlated_pairs:
            self.logger.warning(
                f"Highly correlated feature pairs (>{threshold}) found for {game_id}: {highly_correlated_pairs}"
            )

        if features_to_remove:
            features_to_remove = list(features_to_remove)
            self.logger.info(
                f"Removing {len(features_to_remove)} highly correlated features for {game_id}: {features_to_remove}"
            )
            X_clean = X.drop(columns=features_to_remove)
            return X_clean, features_to_remove

        return X, []

    def _validate_data_quality(
        self, X: pd.DataFrame, y: pd.Series, game_id: str
    ) -> bool:
        """
        Validates data quality before model training.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target Series.
            game_id (str): Identifier for the game.

        Returns:
            bool: True if data passes validation, False otherwise.
        """
        if X.empty or y.empty:
            self.logger.error(f"Empty dataset for {game_id}")
            return False

        if len(X) != len(y):
            self.logger.error(
                f"Shape mismatch for {game_id}: X has {len(X)} rows, y has {len(y)} rows"
            )
            return False

        if np.isinf(X.values).any():
            self.logger.error(f"Infinite values found in features for {game_id}")
            return False

        if X.isnull().values.any():
            self.logger.error(f"NaN values still present in features for {game_id}")
            return False

        if y.isnull().any():
            self.logger.error(f"NaN values found in target for {game_id}")
            return False

        class_counts = y.value_counts()
        min_samples = 10
        if (class_counts < min_samples).any():
            self.logger.warning(
                f"Classes with fewer than {min_samples} samples for {game_id}: {class_counts[class_counts < min_samples]}"
            )

        class_ratio = class_counts.min() / class_counts.max()
        if class_ratio < 0.01:
            self.logger.warning(
                f"Extremely imbalanced classes for {game_id} (ratio: {class_ratio:.4f}): {class_counts.to_dict()}"
            )

        self.logger.info(f"Data quality validation passed for {game_id}")
        return True

    def _prepare_data_for_model(
        self, df: pd.DataFrame, game_id: str, target_column: str = "churned"
    ) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
        """
        Prepares data for modeling by cleaning and validating features and target.

        Args:
            df (pd.DataFrame): The input DataFrame.
            game_id (str): Identifier for the game (e.g., "game1_train").
            target_column (str): The name of the target column. Defaults to "churned".

        Returns:
            Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]: A tuple of (features, target, feature_names),
                                                                  or None if preparation fails.
        """
        y = self._validate_and_convert_target(df, target_column, game_id)
        if y is None:
            return None

        id_columns = self._detect_id_columns(df)
        exclude_columns = [target_column] + id_columns
        feature_columns = [col for col in df.columns if col not in exclude_columns]

        X = df[feature_columns].copy()
        X = self._clean_feature_columns(X, game_id)

        X, feature_names = self._remove_constant_columns(X, game_id)
        if not feature_names:
            return None

        X, removed_features = self._handle_multicollinearity(X, game_id)
        feature_names = list(X.columns)

        if removed_features:
            self.logger.info(
                f"Final feature count after multicollinearity removal: {len(feature_names)}"
            )

        y = y.loc[X.index]

        if not self._validate_data_quality(X, y, game_id):
            return None

        return X, y, feature_names

    def _perform_data_inspection(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, game_id: str
    ):
        """
        Performs essential data inspection on scaled data.

        Args:
            X_train (pd.DataFrame): Scaled training features.
            X_test (pd.DataFrame): Scaled test features.
            game_id (str): Identifier for the game.
        """
        self._print_subheader(f"Data Inspection ({game_id}) PRE-TRAINING")

        self.logger.info(f"X_train (scaled) shape: {X_train.shape}")
        self.logger.info(f"X_test (scaled) shape: {X_test.shape}")

        train_nan_check = X_train.isnull().values.any()
        train_inf_check = np.isinf(X_train.values).any()
        test_nan_check = X_test.isnull().values.any()
        test_inf_check = np.isinf(X_test.values).any()

        self.logger.info(
            f"X_train (scaled) NaN/Inf check: NaNs: {train_nan_check}, Infs: {train_inf_check}"
        )
        self.logger.info(
            f"X_test (scaled) NaN/Inf check: NaNs: {test_nan_check}, Infs: {test_inf_check}"
        )

        # Check for low variance columns
        variances = X_train.var()
        low_variance_cols = variances[variances < 1e-6]
        if not low_variance_cols.empty:
            self.logger.warning(
                f"Columns in X_train with very low variance (<1e-6) AFTER scaling: {low_variance_cols.to_dict()}"
            )
        else:
            self.logger.info(
                "No columns with extremely low variance (<1e-6) in X_train found after scaling."
            )

        # Check for high correlations
        self.logger.info("Correlation Matrix of Scaled X_train:")
        correlation_matrix = X_train.corr()
        highly_correlated = []

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > 0.8:
                    highly_correlated.append(
                        (
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            correlation_matrix.iloc[i, j],
                        )
                    )

        if highly_correlated:
            self.logger.info(
                f"Feature pairs with correlation >0.8 in X_train: {highly_correlated}"
            )
        else:
            self.logger.info(
                "No highly correlated feature pairs (>0.8) found in X_train."
            )

        # Validate feature scaling
        train_means = X_train.mean()
        train_stds = X_train.std()

        mean_close_to_zero = np.allclose(train_means, 0, atol=1e-6)
        std_close_to_one = np.allclose(train_stds, 1, atol=1e-3)

        if not mean_close_to_zero:
            self.logger.warning(
                f"Training features not properly centered. Mean range: [{train_means.min():.6f}, {train_means.max():.6f}]"
            )
        if not std_close_to_one:
            self.logger.warning(
                f"Training features not properly scaled. Std range: [{train_stds.min():.6f}, {train_stds.max():.6f}]"
            )

        if mean_close_to_zero and std_close_to_one:
            self.logger.info(
                "Feature scaling validation passed: features are properly standardized"
            )

    def _display_game_summary_table(
        self, game_id: str, game_metrics: Dict[str, Dict[str, float]]
    ):
        """
        Displays a formatted table of evaluation metrics for a specific game.

        Args:
            game_id (str): The ID of the game.
            game_metrics (Dict[str, Dict[str, float]]): Metrics for each model in the game.
        """
        self._print_subheader(f"Evaluation Summary for {game_id}")

        headers = ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"]
        col_widths = [
            max(
                len(h),
                max((len(model_name) for model_name in game_metrics.keys()), default=0)
                + 2,
            )
            for h in headers
        ]

        header_str = (
            "|"
            + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers))
            + "|"
        )
        print(header_str)

        sep_str = "|" + "|".join("-" * (width + 2) for width in col_widths) + "|"
        print(sep_str)

        best_f1 = -1.0
        best_model_f1 = "N/A"

        for model_name, metrics in game_metrics.items():
            is_error_only = "error" in metrics and not any(
                k in metrics for k in headers[1:]
            )

            if is_error_only:
                error_msg_display = metrics.get("error", "Unknown Error")[
                    : sum(col_widths[1:]) + (len(col_widths[1:]) - 1) * 3 - 2
                ]
                row_str = f"| {model_name:<{col_widths[0]}} | {error_msg_display.center(sum(col_widths[1:]) + (len(col_widths[1:])-1)*3)} |"
            else:
                row_str = f"| {model_name:<{col_widths[0]}} |"
                row_str += (
                    f" {metrics.get('accuracy', float('nan')):<{col_widths[1]}.4f} |"
                )
                row_str += (
                    f" {metrics.get('precision', float('nan')):<{col_widths[2]}.4f} |"
                )
                row_str += (
                    f" {metrics.get('recall', float('nan')):<{col_widths[3]}.4f} |"
                )
                row_str += (
                    f" {metrics.get('f1_score', float('nan')):<{col_widths[4]}.4f} |"
                )
                row_str += (
                    f" {metrics.get('roc_auc', float('nan')):<{col_widths[5]}.4f} |"
                )

                current_f1 = metrics.get("f1_score", -1.0)
                if pd.notna(current_f1) and current_f1 > best_f1:
                    best_f1 = current_f1
                    best_model_f1 = model_name
            print(row_str)

        print(sep_str)
        if best_f1 != -1.0:
            print(
                f"Best model for {game_id} (by F1-score): {best_model_f1} ({best_f1:.4f})"
            )
            self.logger.info(
                f"Best model for {game_id} (by F1-score): {best_model_f1} ({best_f1:.4f})"
            )
        else:
            print(f"No valid F1-scores to determine best model for {game_id}.")
            self.logger.info(
                f"No valid F1-scores to determine best model for {game_id}."
            )

    def _display_feature_importance_summary(
        self, game_id: str, feature_importance_dict: Dict[str, Dict[str, float]]
    ):
        """
        Displays feature importance analysis for each model.

        Args:
            game_id (str): The ID of the game.
            feature_importance_dict (Dict[str, Dict[str, float]]): Dictionary of feature importances by model.
        """
        if not feature_importance_dict:
            self._print_subheader(
                f"Feature Importance Analysis for {game_id}: No data available"
            )
            return

        self._print_subheader(f"Feature Importance Analysis for {game_id}")

        for model_name, importance_dict in feature_importance_dict.items():
            if not importance_dict:
                continue

            print(f"\n{model_name} - Top 5 Most Important Features:")
            sorted_features = sorted(
                importance_dict.items(), key=lambda x: x[1], reverse=True
            )[:5]

            for i, (feature, importance) in enumerate(sorted_features, 1):
                print(f"  {i}. {feature}: {importance:.4f}")

    def _display_overall_best_models(self):
        """
        Displays overall comparison of best performing models across all games.
        Uses cached `self.all_game_metrics`.
        Following Kim et al. (2017), ranks primarily by AUC (ROC-AUC).
        """
        self._print_header("OVERALL MODEL PERFORMANCE COMPARISON")

        all_results = []
        for game_id, game_metrics in self.all_game_metrics.items():
            for model_name, metrics in game_metrics.items():
                if isinstance(metrics, dict) and "error" not in metrics:
                    result = {
                        "game": game_id,
                        "model": model_name,
                        "accuracy": metrics.get("accuracy", float("nan")),
                        "precision": metrics.get("precision", float("nan")),
                        "recall": metrics.get("recall", float("nan")),
                        "f1_score": metrics.get("f1_score", float("nan")),
                        "roc_auc": metrics.get("roc_auc", float("nan")),
                    }
                    all_results.append(result)

        if not all_results:
            self.logger.info("No valid results to compare.")
            print("No valid results to compare.")
            return

        metrics_to_check = ["roc_auc", "accuracy", "precision", "recall", "f1_score"]

        print("Best Performing Models by Metric:")
        print("-" * 60)

        for metric in metrics_to_check:
            valid_results = [r for r in all_results if not pd.isna(r[metric])]
            if valid_results:
                best_result = max(valid_results, key=lambda x: x[metric])
                print(
                    f"{metric.upper():>10}: {best_result['model']} on {best_result['game']} ({best_result[metric]:.4f})"
                )
            else:
                print(f"{metric.upper():>10}: No valid data")

        print(f"\n{'='*60}")
        print("OVERALL RANKING BY ROC-AUC (PRIMARY METRIC - Kim et al. 2017):")
        print(f"{'='*60}")

        # Rank by AUC
        auc_results = [r for r in all_results if not pd.isna(r["roc_auc"])]
        if auc_results:
            auc_results.sort(key=lambda x: x["roc_auc"], reverse=True)

            print(
                f"{'Rank':<6}{'Model':<18}{'Game':<8}{'ROC-AUC':<10}{'Accuracy':<10}{'F1-Score':<10}"
            )
            print("-" * 62)

            for i, result in enumerate(auc_results, 1):
                print(
                    f"{i:<6}{result['model']:<18}{result['game']:<8}{result['roc_auc']:<10.4f}"
                    f"{result['accuracy']:<10.4f}{result['f1_score']:<10.4f}"
                )

            best_overall = auc_results[0]
            self.logger.info(
                f"BEST OVERALL MODEL (by AUC): {best_overall['model']} on {best_overall['game']} (ROC-AUC: {best_overall['roc_auc']:.4f})"
            )
            print(
                f"\n🏆 BEST OVERALL MODEL (by AUC): {best_overall['model']} on {best_overall['game']} "
                f"(ROC-AUC: {best_overall['roc_auc']:.4f})"
            )
        else:
            self.logger.info("No valid AUC scores available for ranking.")
            print("No valid AUC scores available for ranking.")

    def _save_performance_report(self) -> None:
        """
        Saves a comprehensive performance report to a text file.
        """
        try:
            report_path = self.results_dir / "model_performance_report.txt"

            # Generate the report content using utility function
            report_content = generate_performance_report(
                self.all_game_metrics, self.all_feature_importance
            )

            # Write the report to file
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            self.logger.info(f"Performance report saved to: {report_path}")
            print(f"📄 Performance report saved: {report_path}")
        except Exception as e:
            self.logger.error(f"Error saving performance report: {e}")
            print(f"Error saving performance report: {e}")

    def _process_single_game(self, game_id: str) -> None:
        """
        Processes a single game: loads data, prepares features, trains models, and evaluates.

        Args:
            game_id (str): The ID of the game to process.
        """
        self._print_header(f"Processing Game: {game_id}")

        feature_file_ds1 = self.features_dir / f"{game_id}_DS1_features.csv"
        feature_file_ds2 = self.features_dir / f"{game_id}_DS2_features.csv"

        self.logger.info(f"Loading training data from: {feature_file_ds1}")
        self.logger.info(f"Loading evaluation data from: {feature_file_ds2}")

        df_train = self._load_features(feature_file_ds1)
        df_eval = self._load_features(feature_file_ds2)

        if not self._validate_data_loading(
            game_id, df_train, df_eval, feature_file_ds1, feature_file_ds2
        ):
            return

        prepared_data = self._prepare_game_data(game_id, df_train, df_eval)
        if prepared_data is None:
            return

        X_train_raw, y_train, X_eval_raw, y_eval, feature_names = prepared_data

        scaled_data = self._scale_features(
            game_id, X_train_raw, X_eval_raw, y_train, y_eval, feature_names
        )
        if scaled_data is None:
            return

        X_train, X_eval = scaled_data
        self._perform_data_inspection(X_train, X_eval, game_id)

        current_game_metrics, current_game_feature_importance = (
            self._train_and_evaluate_models(game_id, X_train, y_train, X_eval, y_eval)
        )

        self.all_game_metrics[game_id] = current_game_metrics
        self.all_feature_importance[game_id] = current_game_feature_importance

        self._display_game_summary_table(game_id, current_game_metrics)
        self._display_feature_importance_summary(
            game_id, current_game_feature_importance
        )

    def _validate_data_loading(
        self,
        game_id: str,
        df_train: Optional[pd.DataFrame],
        df_eval: Optional[pd.DataFrame],
        feature_file_ds1: Path,
        feature_file_ds2: Path,
    ) -> bool:
        """
        Validates that data was loaded successfully.

        Args:
            game_id (str): The ID of the game.
            df_train (Optional[pd.DataFrame]): Training data.
            df_eval (Optional[pd.DataFrame]): Evaluation data.
            feature_file_ds1 (Path): Path to training data file.
            feature_file_ds2 (Path): Path to evaluation data file.

        Returns:
            bool: True if data is valid, False otherwise.
        """
        if df_train is None or df_train.empty:
            self.logger.warning(
                f"Skipping {game_id} due to issues loading or empty training features."
            )
            self.all_game_metrics[game_id] = {
                "error": f"Failed to load training features from {feature_file_ds1}"
            }
            return False

        if df_eval is None or df_eval.empty:
            self.logger.warning(
                f"Skipping {game_id} due to issues loading or empty evaluation features."
            )
            self.all_game_metrics[game_id] = {
                "error": f"Failed to load evaluation features from {feature_file_ds2}"
            }
            return False

        return True

    def _prepare_game_data(
        self, game_id: str, df_train: pd.DataFrame, df_eval: pd.DataFrame
    ) -> Optional[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]]:
        """
        Prepares training and evaluation data for a game.

        Args:
            game_id (str): The ID of the game.
            df_train (pd.DataFrame): Training data.
            df_eval (pd.DataFrame): Evaluation data.

        Returns:
            Optional[Tuple]: Prepared data tuple or None if preparation fails.
        """
        prepared_train_data = self._prepare_data_for_model(df_train, f"{game_id}_train")
        if prepared_train_data is None:
            self.logger.warning(
                f"Skipping {game_id} due to issues preparing training data."
            )
            self.all_game_metrics[game_id] = {
                "error": "Training data preparation failed or no features remained"
            }
            return None

        X_train_raw, y_train, train_feature_names = prepared_train_data

        prepared_eval_data = self._prepare_data_for_model(df_eval, f"{game_id}_eval")
        if prepared_eval_data is None:
            self.logger.warning(
                f"Skipping {game_id} due to issues preparing evaluation data."
            )
            self.all_game_metrics[game_id] = {
                "error": "Evaluation data preparation failed or no features remained"
            }
            return None

        X_eval_raw, y_eval, eval_feature_names = prepared_eval_data

        common_features = list(set(train_feature_names) & set(eval_feature_names))
        if not common_features:
            self.logger.error(
                f"No common features between training and evaluation datasets for {game_id}."
            )
            self.all_game_metrics[game_id] = {
                "error": "No common features between DS1 and DS2"
            }
            return None

        X_train_raw = X_train_raw[common_features]
        X_eval_raw = X_eval_raw[common_features]
        feature_names = common_features

        self.logger.info(f"Data prepared for {game_id}:")
        self.logger.info(
            f"  Training: {X_train_raw.shape[0]} samples, {len(feature_names)} features"
        )
        self.logger.info(
            f"  Evaluation: {X_eval_raw.shape[0]} samples, {len(feature_names)} features"
        )

        if X_train_raw.empty or y_train.empty or X_eval_raw.empty or y_eval.empty:
            self.logger.error(
                f"Empty datasets after preparation for {game_id}. Skipping."
            )
            self.all_game_metrics[game_id] = {
                "error": "Empty datasets after preparation"
            }
            return None

        # Log class distribution
        train_churn_rate = y_train.value_counts(normalize=True).get(1, 0)
        eval_churn_rate = y_eval.value_counts(normalize=True).get(1, 0)
        self.logger.info(f"Class distribution in training set for {game_id}:")
        self.logger.info(
            f"churned\n1    {train_churn_rate:.2%}\n0     {1-train_churn_rate:.2%}\nName: proportion, dtype: object"
        )
        self.logger.info(f"Class distribution in evaluation set for {game_id}:")
        self.logger.info(
            f"churned\n1    {eval_churn_rate:.2%}\n0     {1-eval_churn_rate:.2%}\nName: proportion, dtype: object"
        )

        return X_train_raw, y_train, X_eval_raw, y_eval, feature_names

    def _scale_features(
        self,
        game_id: str,
        X_train_raw: pd.DataFrame,
        X_eval_raw: pd.DataFrame,
        y_train: pd.Series,
        y_eval: pd.Series,
        feature_names: List[str],
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Scales features using StandardScaler and validates the results.

        Args:
            game_id (str): The ID of the game.
            X_train_raw (pd.DataFrame): Raw training features.
            X_eval_raw (pd.DataFrame): Raw evaluation features.
            y_train (pd.Series): Training targets.
            y_eval (pd.Series): Evaluation targets.
            feature_names (List[str]): List of feature names.

        Returns:
            Optional[Tuple[pd.DataFrame, pd.DataFrame]]: Scaled training and evaluation features or None if scaling fails.
        """
        scaler = StandardScaler()
        X_train_scaled_np = scaler.fit_transform(X_train_raw)
        X_eval_scaled_np = scaler.transform(X_eval_raw)

        X_train = pd.DataFrame(
            X_train_scaled_np, columns=feature_names, index=X_train_raw.index
        )
        X_eval = pd.DataFrame(
            X_eval_scaled_np, columns=feature_names, index=X_eval_raw.index
        )

        if X_train.isnull().values.any() or np.isinf(X_train.values).any():
            self.logger.critical(
                f"NaNs or Infs found in training data after scaling for {game_id}"
            )
            self.all_game_metrics[game_id] = {
                "error": "NaNs/Infs in training data after scaling"
            }
            return None

        if X_eval.isnull().values.any() or np.isinf(X_eval.values).any():
            self.logger.critical(
                f"NaNs or Infs found in evaluation data after scaling for {game_id}"
            )
            self.all_game_metrics[game_id] = {
                "error": "NaNs/Infs in evaluation data after scaling"
            }
            return None

        self.logger.info(f"Scaled features for {game_id}: {X_train.shape[1]} features")
        return X_train, X_eval

    def _create_classifiers(self) -> Dict[str, BaseClassifier]:
        """
        Creates and returns a dictionary of classifier instances using settings-based configuration.

        Returns:
            Dict[str, BaseClassifier]: Dictionary of classifier instances.
        """
        classifiers = {}

        for model_name in self.config_manager.list_available_models():
            model_params = self.config_manager.get_model_params(model_name)

            if model_name == "DecisionTree":
                classifiers[model_name] = DecisionTreeClassifier(
                    model_params=model_params, settings=self.settings
                )
            elif model_name == "LogisticRegression":
                classifiers[model_name] = LogisticRegressionClassifier(
                    model_params=model_params, settings=self.settings
                )
            elif model_name == "RandomForest":
                classifiers[model_name] = RandomForestClassifier(
                    model_params=model_params, settings=self.settings
                )
            else:
                self.logger.warning(f"Unknown model type: {model_name}")

        self.logger.info(
            f"Created {len(classifiers)} classifiers: {list(classifiers.keys())}"
        )
        return classifiers

    def _train_and_evaluate_models(
        self,
        game_id: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Trains and evaluates all models for a given game.

        Args:
            game_id (str): The ID of the game.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_eval (pd.DataFrame): Evaluation features.
            y_eval (pd.Series): Evaluation targets.

        Returns:
            Tuple[Dict, Dict]: Metrics and feature importance dictionaries.
        """
        current_game_metrics = {}
        current_game_feature_importance = {}
        classifiers_to_run = self._create_classifiers()

        for model_name, classifier_instance in classifiers_to_run.items():
            self.logger.info(f"Training {model_name} for {game_id}...")
            try:
                classifier_instance._validate_and_train(X_train, y_train)

                metrics = classifier_instance._validate_and_evaluate(X_eval, y_eval)
                current_game_metrics[model_name] = metrics

                feature_importance = classifier_instance.get_feature_importance()
                if feature_importance is not None:
                    current_game_feature_importance[model_name] = (
                        feature_importance.to_dict()
                    )

                model_save_path = (
                    self.models_output_dir / f"{game_id}_{model_name}.joblib"
                )
                classifier_instance.save_model(model_save_path)

            except Exception as e:
                self.logger.error(
                    f"Failed to train/evaluate {model_name} for {game_id}: {e}"
                )
                current_game_metrics[model_name] = {"error": str(e)}

        return current_game_metrics, current_game_feature_importance

    def _save_results(self) -> None:
        """
        Saves the evaluation metrics and feature importance to JSON files.
        """
        metrics_file_path = self.results_dir / "model_evaluation_metrics.json"
        save_json(self.all_game_metrics, metrics_file_path)
        self.logger.info(f"Model evaluation metrics saved to: {metrics_file_path}")

        feature_importance_file_path = self.results_dir / "feature_importance.json"
        save_json(self.all_feature_importance, feature_importance_file_path)
        self.logger.info(
            f"Feature importance analysis saved to: {feature_importance_file_path}"
        )

    def run_pipeline(self) -> None:
        """
        Executes the complete model training and evaluation pipeline.
        """
        self._print_header("MODEL TRAINING & EVALUATION PIPELINE")
        self.logger.info("Starting model training and evaluation pipeline.")

        for game_id in ["game1", "game2"]:
            self._process_single_game(game_id)

        self._save_results()
        self._display_overall_best_models()
        self._save_performance_report()

        self._print_header("PIPELINE FINISHED")
        self.logger.info("Model training and evaluation pipeline finished.")
