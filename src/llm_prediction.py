import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# some available models with their descriptions
AVAILABLE_MODELS = {
    "distilbert": {
        "name": "distilbert-base-uncased",
        "description": "Lightweight and fast, good for starting out",
    },
    "roberta": {
        "name": "roberta-base",
        "description": "Better performance than BERT, good balance of speed and accuracy",
    },
    "albert": {
        "name": "albert-base-v2",
        "description": "Memory efficient, faster inference",
    },
    "xlm-roberta": {
        "name": "xlm-roberta-base",
        "description": "Good for multilingual data",
    },
    "electra": {
        "name": "google/electra-small-discriminator",
        "description": "Efficient and effective, good for resource-constrained environments",
    },
    "deberta": {
        "name": "microsoft/deberta-v3-small",
        "description": "State-of-the-art performance, good for complex tasks",
    },
}


def load_dataset(game_number, dataset_type):
    """Load processed dataset."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path relative to the script location
    filename = os.path.join(
        script_dir, "data", "processed", f"game{game_number}_DS1_features.jsonl"
    )
    try:
        with open(filename, "r") as f:
            # Load JSONL file (one JSON object per line)
            data = []
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    data.append(json.loads(line))
            return data
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filename}")
        raise
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from file {filename}. Ensure it's a valid JSONL file."
        )
        raise


def create_prompt(player_data, game_number):
    """Create a prompt for the model based on player data."""
    # Extract features from the flat player data structure
    features = {
        "active_duration": player_data.get("active_duration", 0),
        "play_count": player_data.get("play_count", 0),
        "consecutive_play_ratio": player_data.get("consecutive_play_ratio", 0),
        "best_score": player_data.get("best_score", 0),
        "worst_score": player_data.get("worst_score", 0),
        "mean_score": player_data.get("mean_score", 0),
        "purchase_count": player_data.get("purchase_count", 0),
        "max_purchase": player_data.get("max_purchase", 0),
    }
    feature_str = ", ".join([f"{k}: {v}" for k, v in features.items()])

    prompt = f"""In a mobile racing game (Game {game_number}), a player has shown the following activity in their first 5 days of playing: {feature_str}. Will the player stop playing in the next 10 days?"""

    return prompt


class ChurnPredictor:
    def __init__(self, model_key="distilbert"):
        """Initialize the model and tokenizer."""
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_key} not found. Available models: {list(AVAILABLE_MODELS.keys())}"
            )

        model_info = AVAILABLE_MODELS[model_key]
        model_name = model_info["name"]
        print(f"Loading model {model_name} ({model_info['description']})...")

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,  # Binary classification: churn or not churn
                problem_type="single_label_classification",
            )
            self.model.to(self.device)

            # Set model to evaluation mode
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict(self, prompt):
        """Get prediction from the model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                probability = probabilities[0][1].item()  # Probability of churning

            return prediction, probability
        except Exception as e:
            print(f"Error getting prediction: {e}")
            return None, None


def evaluate_predictions(y_true, y_pred, y_pred_proba):
    """Calculate evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),  # Added zero_division
        "recall": recall_score(y_true, y_pred, zero_division=0),  # Added zero_division
        "f1": f1_score(y_true, y_pred, zero_division=0),  # Added zero_division
        "auc_roc": (
            roc_auc_score(y_true, y_pred_proba)
            if len(np.unique(y_true)) > 1
            else "N/A (only one class present)"
        ),  # Handle single class case for AUC
    }
    return metrics


def list_available_models():
    """Print information about available models."""
    print("\nAvailable Models:")
    print("-" * 80)
    for key, info in AVAILABLE_MODELS.items():
        print(f"{key}:")
        print(f"  Model: {info['name']}")
        print(f"  Description: {info['description']}")
    print("-" * 80)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Churn Prediction using Local Models")
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to use for prediction",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )

    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        list_available_models()
        return

    # Initialize predictor with selected model
    try:
        predictor = ChurnPredictor(model_key=args.model)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # Process both games
    for game_number in [1, 2]:
        print(f"\nProcessing Game {game_number}...")

        # Load test data
        try:
            test_data = load_dataset(
                game_number, "test"
            )  # "test" is passed but not used in new filename logic
        except Exception as e:
            print(f"Could not load data for Game {game_number}. Skipping. Error: {e}")
            continue  # Get predictions for each player
        predictions = []
        true_labels = []
        prediction_probas = []

        if not test_data:
            print(f"Loaded data for Game {game_number} is empty. Skipping.")
            continue

        # Check if there are any player records
        if len(test_data) == 0:
            print(
                f"No player data found for Game {game_number}. Skipping metrics calculation."
            )
            continue

        for player_data in tqdm(test_data):
            # Get prediction from model
            prompt = create_prompt(player_data, game_number)
            pred, prob = predictor.predict(prompt)

            if pred is not None and prob is not None:
                predictions.append(pred)
                try:
                    true_labels.append(int(player_data["churned"]))
                except KeyError:
                    print(
                        f"Warning: 'churned' key not found for player {player_data.get('player_id', 'unknown')} in game {game_number}. Skipping this player for metrics."
                    )
                    continue  # Skip if true label is missing
                except ValueError:
                    print(
                        f"Warning: 'churned' value for player {player_data.get('player_id', 'unknown')} in game {game_number} is not a valid integer. Skipping this player for metrics."
                    )
                    continue  # Skip if true label is not valid int

                prediction_probas.append(prob)
            else:
                print(
                    f"Warning: Could not get prediction for player {player_data.get('player_id', 'unknown')} in game {game_number}."
                )

        # Calculate metrics only if there are predictions
        if not predictions or not true_labels:
            print(
                f"No valid predictions or true labels to evaluate for Game {game_number}."
            )
            continue

        metrics = evaluate_predictions(
            np.array(true_labels), np.array(predictions), np.array(prediction_probas)
        )

        # Print metrics
        print(f"\nMetrics for {args.model.upper()} on Game {game_number}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")  # Save metrics
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(
            script_dir,
            "data",
            "processed",
            f"{args.model}_metrics_game{game_number}.json",
        )
        try:
            with open(output_file, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Metrics saved to {output_file}")
        except IOError:
            print(f"Error: Could not write metrics to {output_file}")


if __name__ == "__main__":
    main()
