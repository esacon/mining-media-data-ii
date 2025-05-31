import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Optional

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
            file=sys.stderr
        )
# --- End: Python Path Modification ---

from src.config import get_settings, Settings
from src.utils import setup_logger, save_json
from src.models import (
    DecisionTreeClassifier,
    LogisticRegressionClassifier,
    RandomForestClassifier,
    BaseClassifier
)

logger = setup_logger("ModelTraining", "INFO", log_to_console=False)

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"===== {title.upper()} =====")
    print("=" * 70)

def print_subheader(title: str):
    print("\n" + "-" * 50)
    print(f"--- {title} ---")
    print("-" * 50)

def load_features(feature_file_path: Path) -> Optional[pd.DataFrame]:
    print(f"\nAttempting to load features from: {feature_file_path}")
    logger.info(f"Loading features from: {feature_file_path}")
    records = []
    try:
        with open(feature_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)
        for col in df.columns:
            if df[col].dtype == 'object':
                if col not in ['player_id', 'op_start', 'op_end', 'cp_start', 'cp_end']:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception:
                         pass
        print(f"Successfully loaded {len(df)} records with {len(df.columns)} columns.")
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns from {feature_file_path}.")
        return df
    except FileNotFoundError:
        print(f"ERROR: Feature file not found at {feature_file_path}")
        return None
    except Exception as e:
        print(f"ERROR: Could not load feature file {feature_file_path}. Reason: {e}")
        return None

def prepare_data_for_model(
    df: pd.DataFrame,
    game_id: str,
    target_column: str = 'churned'
) -> Optional[tuple[pd.DataFrame, pd.Series, List[str]]]:
    print(f"\nPreparing data for modeling: {game_id}")
    logger.info(f"Preparing data for {game_id}...")

    if target_column not in df.columns:
        print(f"ERROR: Target column '{target_column}' not found in DataFrame for {game_id}.")
        return None
    
    if df[target_column].isnull().any():
        print(f"WARNING: Target column '{target_column}' contains {df[target_column].isnull().sum()} NaN values. Dropping these rows.")
        df = df.dropna(subset=[target_column])
        if df.empty:
            print(f"ERROR: DataFrame empty after dropping NaN targets for {game_id}.")
            return None

    if df[target_column].dtype != bool:
        try:
            map_to_bool = {'True': True, 'False': False, 'true': True, 'false': False, 
                           '1': True, '0': False, 1: True, 0: False, 1.0: True, 0.0: False}
            original_type = df[target_column].dtype
            mapped_series = df[target_column].map(map_to_bool)
            # Combine mapped with original converted to bool, prioritizing mapped
            df[target_column] = mapped_series.combine_first(df[target_column].astype(bool))

            if df[target_column].isnull().any():
                raise ValueError(f"Mapping to boolean resulted in NaNs. Original type: {original_type}")
            print(f"Target column '{target_column}' successfully converted to boolean.")
        except Exception as e:
            print(f"ERROR: Failed to convert target '{target_column}' to boolean for {game_id}. Error: {e}")
            return None
    y = df[target_column].astype(int)

    X = df.copy()
    
    features_to_drop_for_game1 = ['purchase_count', 'max_purchase']
    if game_id == 'game1':
        actual_cols_to_drop_game1 = [col for col in features_to_drop_for_game1 if col in X.columns]
        if actual_cols_to_drop_game1:
            X = X.drop(columns=actual_cols_to_drop_game1)
            print(f"  For {game_id}, dropped game-specific columns: {actual_cols_to_drop_game1}")
            logger.info(f"For {game_id}, dropped specific columns: {actual_cols_to_drop_game1}")

    base_features_to_drop = [
        'player_id', target_column,
        'op_start', 'op_end', 'cp_start', 'cp_end', 'op_event_count'
    ]
    actual_base_cols_to_drop = [col for col in base_features_to_drop if col in X.columns]
    if actual_base_cols_to_drop:
        X = X.drop(columns=actual_base_cols_to_drop)
        print(f"  Dropped general ID/metadata/target columns: {actual_base_cols_to_drop}")
        logger.info(f"Dropped general columns for {game_id}: {actual_base_cols_to_drop}")
    
    feature_columns = list(X.columns)
    if not feature_columns:
        print(f"ERROR: No feature columns remaining for {game_id} before numeric conversion/imputation.")
        return None
    print(f"  Features selected for {game_id} ({len(feature_columns)}) PRE-CLEANING: {', '.join(feature_columns)}")

    for col in feature_columns:
        if col not in X.columns: continue
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"WARNING: Feature column '{col}' for {game_id} is non-numeric (dtype: {X[col].dtype}). Coercing to numeric.")
            X[col] = pd.to_numeric(X[col], errors='coerce')

    if X.isnull().values.any():
        nan_counts = X.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0].index
        print(f"WARNING: NaN values found in feature columns for {game_id}: {list(nan_cols)}. Imputing with column medians.")
        for col in nan_cols:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            if X[col].isnull().any():
                print(f"WARNING: Column '{col}' still has NaNs for {game_id} after median. Filling with 0.")
                X[col].fillna(0, inplace=True)
    
    constant_cols_after_cleaning = [col for col in X.columns if X[col].nunique(dropna=False) == 1] # Check nunique with dropna=False
    if constant_cols_after_cleaning:
        print(f"WARNING: Constant columns found for {game_id} AFTER cleaning and BEFORE scaling: {constant_cols_after_cleaning}. Values: {X[constant_cols_after_cleaning].iloc[0].to_dict() if not X.empty else 'N/A'}")
        print("  These constant columns will cause issues with StandardScaler. Dropping them.")
        logger.warning(f"Dropping constant columns for {game_id} found after cleaning: {constant_cols_after_cleaning}")
        X = X.drop(columns=constant_cols_after_cleaning)
        feature_columns = list(X.columns) 
        if not feature_columns:
            print(f"ERROR: No feature columns remaining for {game_id} after dropping constant columns found post-cleaning.")
            return None
        print(f"  Features after dropping post-cleaning constant columns ({len(feature_columns)}): {', '.join(feature_columns)}")

    print(f"  Final prepared X shape for {game_id}: {X.shape}, y shape: {y.shape}")
    return X, y, feature_columns

def train_and_evaluate_classifier(
    classifier: BaseClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    game_id: str,
    output_dir: Path
) -> Optional[Dict[str, float]]:
    print(f"\n  Training {model_name} for {game_id}...")
    logger.info(f"Training {model_name} for {game_id} with X_train shape: {X_train.shape}")
    try:
        # --- MODIFIED: Detailed data inspection before Logistic Regression training ---
        if model_name == "LogisticRegression":
            print_subheader(f"Data Inspection for LogisticRegression ({game_id}) PRE-TRAINING")
            print("  X_train (scaled) sample:")
            print(X_train.head())
            print("\n  X_train (scaled) description:")
            print(X_train.describe().T) # Transpose for better readability with many columns
            print("\n  X_train (scaled) NaN/Inf check:")
            print(f"    NaNs present in X_train: {X_train.isnull().values.any()}")
            print(f"    Infs present in X_train: {np.isinf(X_train.values).any()}")
            variances = X_train.var()
            low_variance_cols = variances[variances < 1e-6] # Adjusted threshold
            if not low_variance_cols.empty:
                print(f"    WARNING: Columns in X_train with very low variance (<1e-6) AFTER scaling: \n{low_variance_cols}")
            else:
                print("    No columns with extremely low variance (<1e-6) in X_train found after scaling.")
            
            print("\n  X_test (scaled) NaN/Inf check:") # Also check X_test
            print(f"    NaNs present in X_test: {X_test.isnull().values.any()}")
            print(f"    Infs present in X_test: {np.isinf(X_test.values).any()}")

            print("\n  --- Correlation Matrix of Scaled X_train for LogisticRegression ---")
            correlation_matrix = X_train.corr()
            # print(correlation_matrix) # Can be very large, print selectively
            highly_correlated = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i): # Avoid duplicates and self-correlation
                    if abs(correlation_matrix.iloc[i, j]) > 0.95:
                        highly_correlated.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))
            if highly_correlated:
                print("  Highly correlated feature pairs in X_train (>0.95 absolute):")
                for pair_info in highly_correlated:
                    print(f"    {pair_info[0]} & {pair_info[1]}: {pair_info[2]:.4f}")
            else:
                print("  No highly correlated feature pairs (>0.95) found in X_train.")
            print("  -----------------------------------------------------------------")
        # --- END MODIFIED ---

        if np.isnan(X_train.values).any() or np.isinf(X_train.values).any():
            print(f"ERROR: NaNs or Infs found in X_train for {model_name}, {game_id} before training.")
            return {"error": "NaNs/Infs in training data"}

        classifier.train(X_train, y_train)
        print(f"  {model_name} training completed.")
        logger.info(f"{model_name} training completed for {game_id}.")
        
        print(f"  Evaluating {model_name} for {game_id}...")
        logger.info(f"Evaluating {model_name} for {game_id} with X_test shape: {X_test.shape}")
        if np.isnan(X_test.values).any() or np.isinf(X_test.values).any():
            print(f"ERROR: NaNs or Infs found in X_test for {model_name}, {game_id} before evaluation.")
            return {"error": "NaNs/Infs in test data"}

        metrics = classifier.evaluate(X_test, y_test)
        
        print(f"\n  --- Metrics for {model_name} on {game_id} ---")
        for metric_name, value in metrics.items():
            print(f"    {metric_name.replace('_', ' ').capitalize():<18}: {value:.4f}")
        print("  --------------------------------------")
        logger.info(f"Metrics for {model_name} on {game_id}: {metrics}")
        
        model_save_path = output_dir / "models" / f"{game_id}_{model_name}.joblib"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        classifier.save_model(model_save_path)
        print(f"  {model_name} model saved to: {model_save_path}")
        
        return metrics
    except Exception as e:
        print(f"ERROR: Failed to train or evaluate {model_name} for {game_id}. Reason: {e}")
        logger.error(f"Error with {model_name} for {game_id}: {e}", exc_info=True)
        if isinstance(e, dict) and "error" in e:
             return e
        return {"error": str(e)}

def display_game_summary_table(game_id: str, game_metrics: Dict[str, Dict[str, float]]):
    print_subheader(f"Evaluation Summary for {game_id}")
    
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"]
    col_widths = [max(20, len(h)) for h in headers] # Increased model name width slightly

    header_str = "|"
    for i, header in enumerate(headers):
        header_str += f" {header:<{col_widths[i]}} |"
    print(header_str)

    sep_str = "|"
    for width in col_widths:
        sep_str += "-" * (width + 2) + "|"
    print(sep_str)

    best_f1 = -1.0
    best_model_f1 = "N/A"

    for model_name, metrics in game_metrics.items():
        is_error_only = "error" in metrics and not any(k in metrics for k in headers[1:])

        if is_error_only:
            error_msg_display = metrics.get("error", "Unknown Error")[:sum(col_widths[1:]) + (len(col_widths[1:])-1)*3 - 2]
            row_str = f"| {model_name:<{col_widths[0]}} | {error_msg_display.center(sum(col_widths[1:]) + (len(col_widths[1:])-1)*3)} |"
        else:
            row_str = f"| {model_name:<{col_widths[0]}} |"
            row_str += f" {metrics.get('accuracy', float('nan')):<{col_widths[1]}.4f} |"
            row_str += f" {metrics.get('precision', float('nan')):<{col_widths[2]}.4f} |"
            row_str += f" {metrics.get('recall', float('nan')):<{col_widths[3]}.4f} |"
            row_str += f" {metrics.get('f1_score', float('nan')):<{col_widths[4]}.4f} |"
            row_str += f" {metrics.get('roc_auc', float('nan')):<{col_widths[5]}.4f} |"

            current_f1 = metrics.get('f1_score', -1.0)
            if pd.notna(current_f1) and current_f1 > best_f1:
                best_f1 = current_f1
                best_model_f1 = model_name
        print(row_str)
            
    print(sep_str)
    if best_f1 != -1.0:
        print(f"Best model for {game_id} (by F1-score): {best_model_f1} ({best_f1:.4f})")
    else:
        print(f"No valid F1-scores to determine best model for {game_id}.")


def main_training_pipeline(settings: Settings):
    print_header("MODEL TRAINING & EVALUATION PIPELINE")

    processed_dir = settings.processed_dir
    results_dir = settings.results_dir 
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using processed data from: {processed_dir}")
    print(f"Saving model artifacts to: {processed_dir / 'models'}")
    print(f"Saving evaluation metrics JSON to: {results_dir}")

    all_game_metrics_for_json = {}

    for game_id in ["game1", "game2"]:
        print_header(f"Processing Game: {game_id}")
        
        feature_file_ds1 = processed_dir / f"{game_id}_DS1_features.jsonl"
        
        df_features = load_features(feature_file_ds1)
        if df_features is None or df_features.empty:
            print(f"Skipping {game_id} due to issues loading or empty features.")
            all_game_metrics_for_json[game_id] = {"error": f"Failed to load features from {feature_file_ds1}"}
            continue
            
        prepared_data = prepare_data_for_model(df_features, game_id)
        if prepared_data is None:
            print(f"Skipping {game_id} due to issues preparing data or no features left.")
            all_game_metrics_for_json[game_id] = {"error": "Data preparation failed or no features remained"}
            continue
        
        X, y, feature_names = prepared_data

        if X.empty or y.empty:
            print(f"ERROR: No data available for training/testing for {game_id} after preparation. Skipping.")
            all_game_metrics_for_json[game_id] = {"error": "X or y is empty after preparation"}
            continue
        if not feature_names:
            print(f"ERROR: No feature names identified for {game_id}. Skipping.")
            all_game_metrics_for_json[game_id] = {"error": "No feature names for scaling/training"}
            continue

        try:
            stratify_option = y if y.nunique() > 1 and len(y) >= (2 * max(2, y.nunique())) else None # Ensure enough samples per class for stratify
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=settings.random_seed, stratify=stratify_option
            )
        except ValueError as e:
            print(f"Warning: Error during stratified train_test_split for {game_id}: {e}. Trying without stratification.")
            try:
                 X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=settings.random_seed
                )
            except Exception as split_e:
                print(f"ERROR: Train/test split failed for {game_id}: {split_e}. Skipping game.")
                all_game_metrics_for_json[game_id] = {"error": f"Train/test split failed: {split_e}"}
                continue
        
        print(f"\n  Class distribution in y_train for {game_id}:") # Check class balance
        print(y_train.value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))


        print(f"\n  Applying Feature Scaling (StandardScaler) for {game_id} on {len(feature_names)} features...")
        scaler = StandardScaler()
        
        X_train_raw_fs = X_train_raw[feature_names]
        X_test_raw_fs = X_test_raw[feature_names]

        X_train_scaled_np = scaler.fit_transform(X_train_raw_fs)
        X_test_scaled_np = scaler.transform(X_test_raw_fs)

        X_train = pd.DataFrame(X_train_scaled_np, columns=feature_names, index=X_train_raw_fs.index)
        X_test = pd.DataFrame(X_test_scaled_np, columns=feature_names, index=X_test_raw_fs.index)
        print(f"  Feature scaling completed.")
        logger.info(f"Feature scaling applied for {game_id}.")
        
        if X_train.isnull().values.any() or np.isinf(X_train.values).any():
            print(f"CRITICAL ERROR: NaNs or Infs found in X_train AFTER scaling for {game_id}.")
            logger.error(f"CRITICAL: NaNs/Infs in X_train post-scaling for {game_id}. Columns: {X_train.columns[X_train.isnull().any(axis=0)]}")
            all_game_metrics_for_json[game_id] = {"error": "NaNs/Infs in X_train after scaling"}
            continue
        if X_test.isnull().values.any() or np.isinf(X_test.values).any():
            print(f"CRITICAL ERROR: NaNs or Infs found in X_test AFTER scaling for {game_id}.") # Test set should also be clean
            logger.error(f"CRITICAL: NaNs/Infs in X_test post-scaling for {game_id}. Columns: {X_test.columns[X_test.isnull().any(axis=0)]}")
            all_game_metrics_for_json[game_id] = {"error": "NaNs/Infs in X_test after scaling"}
            # Continue to allow other models to run, but Logistic Regression might fail if data is bad
            # Or: continue # to skip the game entirely if test data is also corrupted

        print(f"\nData split and scaled for {game_id}:")
        print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        current_game_metrics = {}
        
        classifiers_to_run = {
            "DecisionTree": DecisionTreeClassifier(model_params={'random_state': settings.random_seed}, logger=logger),
            
            # --- MODIFIED: Using 'saga' solver for Logistic Regression ---
            "LogisticRegression": LogisticRegressionClassifier(
                model_params={
                    'random_state': settings.random_seed,
                    'solver': 'saga',        # Switched to 'saga'
                    'penalty': 'l2',        # Using L2 penalty with saga
                    'max_iter': 5000,       # Increased max_iter significantly for saga
                    'C': 0.1,               # Kept stronger regularization
                    'tol': 1e-3             # Saga might benefit from a slightly looser tolerance
                },
                logger=logger
            ),
            # --- You can revert to liblinear with L1 if saga also fails or is too slow ---
            # "LogisticRegression": LogisticRegressionClassifier(
            #     model_params={
            #         'random_state': settings.random_seed,
            #         'solver': 'liblinear',
            #         'penalty': 'l1',
            #         'max_iter': 1000,
            #         'C': 0.1 
            #     },
            #     logger=logger
            # ),

            "RandomForest": RandomForestClassifier(model_params={'random_state': settings.random_seed, 'n_estimators': 100}, logger=logger),
        }

        for model_name, classifier_instance in classifiers_to_run.items():
            metrics_result = train_and_evaluate_classifier(
                classifier_instance, X_train, y_train, X_test, y_test,
                model_name, game_id, settings.processed_dir 
            )
            current_game_metrics[model_name] = metrics_result
        
        all_game_metrics_for_json[game_id] = current_game_metrics
        display_game_summary_table(game_id, current_game_metrics)

    metrics_file_path = results_dir / "model_evaluation_metrics.json"
    save_json(all_game_metrics_for_json, metrics_file_path)
    print_header("PIPELINE FINISHED")
    print(f"All model evaluation metrics (JSON) saved to: {metrics_file_path}")


def parse_model_training_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model training and evaluation pipeline.")
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
        settings.log_level if hasattr(settings, 'log_level') else "INFO",
        log_file=settings.logs_dir / "model_training_script.log" if hasattr(settings, 'logs_dir') and settings.logs_dir else None,
        log_to_console=False # Make sure your setup_logger handles this argument
    )
    logger.info(f"Main ModelTrainingScript logger initialized. Log level: {logger.level}")

    print(f"Starting model training pipeline with settings...")
    print(f"  Processed data directory: {settings.processed_dir}")
    if hasattr(settings, 'results_dir'): print(f"  Results directory: {settings.results_dir}")
    if hasattr(settings, 'logs_dir'): print(f"  Logs directory: {settings.logs_dir}")
    print(f"  Random seed: {settings.random_seed}")

    try:
        main_training_pipeline(settings)
    except Exception as e:
        print(f"FATAL ERROR in model training pipeline: {e}", file=sys.stderr)
        logger.error("Model training pipeline failed critically.", exc_info=True)
        sys.exit(1)
    
    print("\nModel training and evaluation complete.")