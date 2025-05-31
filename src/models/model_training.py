import argparse

# import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Start: Python Path Modification ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent
if (project_root / "src").is_dir():
    sys.path.insert(0, str(project_root))
else:
    project_root = current_script_dir.parent.parent
    if (project_root / "src").is_dir():
        sys.path.insert(0, str(project_root))
    else:
        print(
            "ERROR: Could not automatically determine project root to add 'src' to PYTHONPATH. "
            "Please ensure 'src' is discoverable.",
            file=sys.stderr,
        )
# --- End: Python Path Modification ---

from src.config import Settings, get_settings
from src.models import (  # BaseClassifier,
    DecisionTreeClassifier,
    LogisticRegressionClassifier,
    RandomForestClassifier,
)
from src.utils import save_json, setup_logger

logger = setup_logger("ModelTraining", "INFO", log_to_console=False)


def print_header(title: str):
    """Print a header with the given title."""
    print("\n" + "=" * 70)
    print(f"===== {title.upper()} =====")
    print("=" * 70)


def print_subheader(title: str):
    """Print a subheader with the given title."""
    print("\n" + "-" * 50)
    print(f"--- {title} ---")
    print("-" * 50)


def load_features(feature_file_path: str) -> pd.DataFrame:
    """Load features from a CSV file."""
    print(f"\nAttempting to load features from: {feature_file_path}")
    logger.info(f"Loading features from: {feature_file_path}")

    try:
        df = pd.read_csv(feature_file_path)
        print(f"Successfully loaded {len(df)} records with {len(df.columns)} columns.")
        logger.info(
            f"Loaded {len(df)} records with {len(df.columns)} columns from {feature_file_path}."
        )
        return df
    except FileNotFoundError:
        print(f"ERROR: Feature file not found at {feature_file_path}")
        raise
    except Exception as e:
        print(f"ERROR: Could not load feature file {feature_file_path}. Reason: {e}")
        raise


def _convert_target_to_bool(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Convert target column to boolean values."""
    original_type = df[target_column].dtype
    df[target_column] = df[target_column].map(
        {1: True, 0: False, "1": True, "0": False, True: True, False: False}
    )
    if df[target_column].isnull().any():
        raise ValueError(
            f"Mapping to boolean resulted in NaNs. Original type: {original_type}"
        )
    return df


def _drop_game_specific_columns(X: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """Drop game-specific columns based on game ID."""
    if game_id == "game1":
        cols_to_drop = ["game1_specific_col1", "game1_specific_col2"]
    else:
        cols_to_drop = ["game2_specific_col1", "game2_specific_col2"]

    actual_cols_to_drop = [col for col in cols_to_drop if col in X.columns]
    if actual_cols_to_drop:
        print(f"  For {game_id}, dropped game-specific columns: {actual_cols_to_drop}")
        X = X.drop(columns=actual_cols_to_drop)
    return X


def _drop_base_columns(X: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Drop base columns that are not needed for modeling."""
    base_cols_to_drop = ["player_id", "timestamp", target_column]
    actual_base_cols_to_drop = [col for col in base_cols_to_drop if col in X.columns]
    if actual_base_cols_to_drop:
        print(
            f"  Dropped general ID/metadata/target columns: {actual_base_cols_to_drop}"
        )
        X = X.drop(columns=actual_base_cols_to_drop)
    return X


def _handle_numeric_conversion(X: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """Handle conversion of feature columns to numeric types and impute missing values."""
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(
                f"WARNING: Feature column '{col}' for {game_id} is non-numeric (dtype: {X[col].dtype}). Coercing to numeric."
            )
            X[col] = pd.to_numeric(X[col], errors="coerce")

    nan_cols = X.columns[X.isnull().any()]
    if len(nan_cols) > 0:
        print(
            f"WARNING: NaN values found in feature columns for {game_id}: {list(nan_cols)}. Imputing with column medians."
        )
        X = X.fillna(X.median())

    return X


def prepare_data_for_model(
    df: pd.DataFrame, game_id: str, target_column: str = "churned"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for modeling by handling target conversion and feature preprocessing."""
    print(f"\nPreparing data for modeling: {game_id}")
    logger.info(f"Preparing data for {game_id}...")

    if target_column not in df.columns:
        print(
            f"ERROR: Target column '{target_column}' not found in DataFrame for {game_id}."
        )
        raise ValueError(f"Target column '{target_column}' not found")

    if df[target_column].isnull().any():
        print(
            f"WARNING: Target column '{target_column}' contains {df[target_column].isnull().sum()} NaN values. Dropping these rows."
        )
        df = df.dropna(subset=[target_column])
        if len(df) == 0:
            print(f"ERROR: DataFrame empty after dropping NaN targets for {game_id}.")
            raise ValueError("No valid data after dropping NaN targets")

    try:
        df = _convert_target_to_bool(df, target_column)
        print(f"Target column '{target_column}' successfully converted to boolean.")
    except Exception as e:
        print(
            f"ERROR: Failed to convert target '{target_column}' to boolean for {game_id}. Error: {e}"
        )
        raise

    X = df.copy()
    y = X.pop(target_column)

    X = _drop_game_specific_columns(X, game_id)
    X = _drop_base_columns(X, target_column)
    X = _handle_numeric_conversion(X, game_id)

    if len(X.columns) == 0:
        print(
            f"ERROR: No feature columns remaining for {game_id} before numeric conversion/imputation."
        )
        raise ValueError("No feature columns remaining after preprocessing")

    feature_columns = X.columns.tolist()
    print(
        f"  Features selected for {game_id} ({len(feature_columns)}) PRE-CLEANING: {', '.join(feature_columns)}"
    )

    return X, y


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, model_name: str, game_id: str
) -> Any:
    """Train a model on the given data."""
    print(f"\n  Training {model_name} for {game_id}...")
    logger.info(
        f"Training {model_name} for {game_id} with X_train shape: {X_train.shape}"
    )

    if model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str, game_id: str
) -> Dict[str, float]:
    """Evaluate a trained model on test data."""
    print(f"\n  Evaluating {model_name} for {game_id}...")
    logger.info(f"Evaluating {model_name} for {game_id}")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
    }

    print(f"\n{model_name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


def plot_feature_importance(
    model: Any, feature_names: List[str], model_name: str, game_id: str
):
    """Plot feature importance for a trained model."""
    plt.figure(figsize=(10, 6))

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = model.coef_[0]

    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(
        range(len(importances)),
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right",
    )
    plt.title(f"Feature Importance - {model_name} (Game {game_id})")
    plt.tight_layout()
    plt.savefig(f"plots/feature_importance_{model_name}_game{game_id}.png")
    plt.close()


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], game_id: str):
    """Plot comparison of metrics across different models."""
    plt.figure(figsize=(12, 6))

    models = list(metrics_dict.keys())
    metrics = ["accuracy", "f1", "auc_roc"]

    x = np.arange(len(models))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [metrics_dict[model][metric] for model in models]
        plt.bar(x + i * width, values, width, label=metric)

    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.title(f"Model Performance Comparison (Game {game_id})")
    plt.xticks(x + width, models)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/metrics_comparison_game{game_id}.png")
    plt.close()


def display_game_summary_table(game_id: str, metrics_dict: Dict[str, Dict[str, float]]):
    """Display a summary table of metrics for a game."""
    print(f"\nMetrics Summary for Game {game_id}:")
    print("-" * 50)
    print(f"{'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'AUC-ROC':<10}")
    print("-" * 50)

    for model_name, metrics in metrics_dict.items():
        print(
            f"{model_name:<20} "
            f"{metrics['accuracy']:.4f}    "
            f"{metrics['f1']:.4f}    "
            f"{metrics['auc_roc']:.4f}"
        )
    print("-" * 50)


def _prepare_and_split_data(
    df_features: pd.DataFrame, game_id: str, settings: Settings
):
    """Prepare and split data for training."""
    prepared_data = prepare_data_for_model(df_features, game_id)
    if prepared_data is None:
        print(f"Skipping {game_id} due to issues preparing data or no features left.")
        return None, None, None, None

    X, y = prepared_data

    if X.empty or y.empty:
        print(
            f"ERROR: No data available for training/testing for {game_id} after preparation. Skipping."
        )
        return None, None, None, None

    try:
        stratify_option = (
            y if y.nunique() > 1 and len(y) >= (2 * max(2, y.nunique())) else None
        )
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=settings.random_seed,
            stratify=stratify_option,
        )
        return X_train_raw, X_test_raw, y_train, y_test
    except ValueError as e:
        print(
            f"Warning: Error during stratified train_test_split for {game_id}: {e}. Trying without stratification."
        )
        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=settings.random_seed
            )
            return X_train_raw, X_test_raw, y_train, y_test
        except Exception as split_e:
            print(
                f"ERROR: Train/test split failed for {game_id}: {split_e}. Skipping game."
            )
            return None, None, None, None


def _scale_features(
    X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame, X: pd.DataFrame
):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_raw_fs = X_train_raw[X.columns]
    X_test_raw_fs = X_test_raw[X.columns]

    X_train_scaled_np = scaler.fit_transform(X_train_raw_fs)
    X_test_scaled_np = scaler.transform(X_test_raw_fs)

    X_train = pd.DataFrame(
        X_train_scaled_np, columns=X.columns, index=X_train_raw_fs.index
    )
    X_test = pd.DataFrame(
        X_test_scaled_np, columns=X.columns, index=X_test_raw_fs.index
    )

    return X_train, X_test, scaler


def _validate_scaled_data(X_train: pd.DataFrame, X_test: pd.DataFrame, game_id: str):
    """Validate scaled data for NaNs and Infs."""
    if X_train.isnull().values.any() or np.isinf(X_train.values).any():
        print(
            f"CRITICAL ERROR: NaNs or Infs found in X_train AFTER scaling for {game_id}."
        )
        logger.error(
            f"CRITICAL: NaNs/Infs in X_train post-scaling for {game_id}. Columns: {X_train.columns[X_train.isnull().any(axis=0)]}"
        )
        return False

    if X_test.isnull().values.any() or np.isinf(X_test.values).any():
        print(
            f"CRITICAL ERROR: NaNs or Infs found in X_test AFTER scaling for {game_id}."
        )
        logger.error(
            f"CRITICAL: NaNs/Infs in X_test post-scaling for {game_id}. Columns: {X_test.columns[X_test.isnull().any(axis=0)]}"
        )
        return False

    return True


def _train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    game_id: str,
    settings: Settings,
):
    """Train and evaluate models for a game."""
    current_game_metrics = {}

    classifiers_to_run = {
        "DecisionTree": DecisionTreeClassifier(
            model_params={"random_state": settings.random_seed}, logger=logger
        ),
        "LogisticRegression": LogisticRegressionClassifier(
            model_params={
                "random_state": settings.random_seed,
                "solver": "saga",
                "penalty": "l2",
                "max_iter": 5000,
                "C": 0.1,
                "tol": 1e-3,
            },
            logger=logger,
        ),
        "RandomForest": RandomForestClassifier(
            model_params={
                "random_state": settings.random_seed,
                "n_estimators": 100,
            },
            logger=logger,
        ),
    }

    for model_name, classifier_instance in classifiers_to_run.items():
        metrics_result = evaluate_model(
            classifier_instance,
            X_test,
            y_test,
            model_name,
            game_id,
        )
        current_game_metrics[model_name] = metrics_result

    return current_game_metrics


def main_training_pipeline(settings: Settings):
    """Main training pipeline for all games and models."""
    print_header("MODEL TRAINING & EVALUATION PIPELINE")

    processed_dir = settings.processed_dir
    results_dir = settings.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    print("Using processed data from: {}".format(processed_dir))
    print("Saving model artifacts to: {}".format(processed_dir / "models"))
    print("Saving evaluation metrics JSON to: {}".format(results_dir))

    all_game_metrics_for_json = {}

    for game_id in ["game1", "game2"]:
        print_header("Processing Game: {}".format(game_id))

        feature_file_ds1 = processed_dir / "{}_DS1_features.jsonl".format(game_id)

        df_features = load_features(feature_file_ds1)
        if df_features is None or df_features.empty:
            print(
                "Skipping {} due to issues loading or empty features.".format(game_id)
            )
            all_game_metrics_for_json[game_id] = {
                "error": "Failed to load features from {}".format(feature_file_ds1)
            }
            continue

        # Prepare and split data
        X_train_raw, X_test_raw, y_train, y_test = _prepare_and_split_data(
            df_features, game_id, settings
        )
        if X_train_raw is None:
            all_game_metrics_for_json[game_id] = {
                "error": "Data preparation or split failed"
            }
            continue

        print("\n  Class distribution in y_train for {}:".format(game_id))
        print(y_train.value_counts(normalize=True).apply(lambda x: "{:.2%}".format(x)))

        print(
            "\n  Applying Feature Scaling (StandardScaler) for {} on {} features...".format(
                game_id, len(X_train_raw.columns)
            )
        )
        X_train, X_test, scaler = _scale_features(X_train_raw, X_test_raw, X_train_raw)
        print("  Feature scaling completed.")
        logger.info("Feature scaling applied for {}.".format(game_id))

        if not _validate_scaled_data(X_train, X_test, game_id):
            all_game_metrics_for_json[game_id] = {"error": "NaNs/Infs in scaled data"}
            continue

        print("\nData split and scaled for {}:".format(game_id))
        print(
            "  X_train shape: {}, y_train shape: {}".format(
                X_train.shape, y_train.shape
            )
        )
        print("  X_test shape: {}, y_test shape: {}".format(X_test.shape, y_test.shape))

        # Train and evaluate models
        current_game_metrics = _train_and_evaluate_models(
            X_train, X_test, y_train, y_test, game_id, settings
        )

        all_game_metrics_for_json[game_id] = current_game_metrics
        display_game_summary_table(game_id, current_game_metrics)

    metrics_file_path = results_dir / "model_evaluation_metrics.json"
    save_json(all_game_metrics_for_json, metrics_file_path)
    print_header("PIPELINE FINISHED")
    print(f"All model evaluation metrics (JSON) saved to: {metrics_file_path}")


def parse_model_training_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model training and evaluation pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom config.yaml file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_model_training_args()

    try:
        settings = get_settings(args.config)
        settings.__post_init__()
    except Exception as e:
        print(f"FATAL: Could not load settings. Error: {e}", file=sys.stderr)
        sys.exit(1)

    logger = setup_logger(
        "ModelTrainingScript",
        settings.log_level if hasattr(settings, "log_level") else "INFO",
        log_file=(
            settings.logs_dir / "model_training_script.log"
            if hasattr(settings, "logs_dir") and settings.logs_dir
            else None
        ),
        log_to_console=False,  # Make sure your setup_logger handles this argument
    )
    logger.info(
        f"Main ModelTrainingScript logger initialized. Log level: {logger.level}"
    )

    print("Starting model training pipeline with settings...")
    print(f"  Processed data directory: {settings.processed_dir}")
    if hasattr(settings, "results_dir"):
        print(f"  Results directory: {settings.results_dir}")
    if hasattr(settings, "logs_dir"):
        print(f"  Logs directory: {settings.logs_dir}")
    print(f"  Random seed: {settings.random_seed}")

    try:
        main_training_pipeline(settings)
    except Exception as e:
        print(f"FATAL ERROR in model training pipeline: {e}", file=sys.stderr)
        logger.error("Model training pipeline failed critically.", exc_info=True)
        sys.exit(1)

    print("\nModel training and evaluation complete.")
