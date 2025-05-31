import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns  # Remove unused import
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_dataset(game_number, dataset_type):
    """Load processed dataset for a specific game."""
    with open(
        f"results/features/game{game_number}_DS{1 if dataset_type == 'train' else 2}_features.csv",
        "r",
    ) as f:
        data = pd.read_csv(f)
    return data


def prepare_features_and_labels(data):
    """Convert features and labels into numpy arrays for training."""
    X = data.drop(columns=["player_id", "churned"]).values
    y = data["churned"].values
    return X, y


def train_models(X_train, y_train):
    """Train multiple classifiers on the data."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize models
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    }

    # Train models
    trained_models = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models, scaler


def evaluate_model(model, X_test, y_test, scaler=None, model_name=""):
    """Evaluate a trained model on test data."""
    # Scale features if necessary
    if model_name == "Logistic Regression" and scaler is not None:
        X_test = scaler.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
    }

    return metrics


def plot_feature_importance(model, feature_names, model_name, game_number):
    """Plot feature importance for a given model."""
    plt.figure(figsize=(10, 6))

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
    else:
        return

    indices = np.argsort(importances)[::-1]

    plt.title(f"Feature Importance - {model_name} (Game {game_number})")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(
        range(len(importances)), [feature_names[i] for i in indices], rotation=45
    )
    plt.tight_layout()

    # Create plots directory if it doesn't exist
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(
        f'plots/feature_importance_game{game_number}_{model_name.lower().replace(" ", "_")}.png'
    )
    plt.close()


def plot_metrics_comparison(metrics_dict, game_number):
    """Plot comparison of model metrics."""
    plt.figure(figsize=(12, 6))

    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    x = np.arange(len(metrics))
    width = 0.25

    for i, (model_name, model_metrics) in enumerate(metrics_dict.items()):
        values = [model_metrics[metric] for metric in metrics]
        plt.bar(x + i * width, values, width, label=model_name)

    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title(f"Model Performance Comparison (Game {game_number})")
    plt.xticks(x + width, metrics, rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"plots/metrics_comparison_game{game_number}.png")
    plt.close()


def main():
    # Process both games
    for game_number in [1, 2]:
        print(f"\nProcessing Game {game_number}...")

        # Load data
        train_data = load_dataset(game_number, "train")
        test_data = load_dataset(game_number, "test")

        # Prepare features and labels
        X_train, y_train = prepare_features_and_labels(train_data)
        X_test, y_test = prepare_features_and_labels(test_data)

        # Get feature names
        feature_names = train_data.columns[:-1].tolist()

        # Train models
        trained_models, scaler = train_models(X_train, y_train)

        # Evaluate models and store metrics
        metrics_dict = {}
        for model_name, model in trained_models.items():
            metrics = evaluate_model(
                model,
                X_test,
                y_test,
                scaler if model_name == "Logistic Regression" else None,
                model_name,
            )
            metrics_dict[model_name] = metrics

            # Plot feature importance
            plot_feature_importance(model, feature_names, model_name, game_number)

            # Print metrics
            print(f"\n{model_name} Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

        # Plot metrics comparison
        plot_metrics_comparison(metrics_dict, game_number)

        # Save metrics to file
        Path("results").mkdir(exist_ok=True)
        with open(f"results/metrics_game{game_number}.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)


if __name__ == "__main__":
    main()
