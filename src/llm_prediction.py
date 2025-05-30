import json
import numpy as np
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import argparse

# some available models with their descriptions
AVAILABLE_MODELS = {
    "distilbert": {
        "name": "distilbert-base-uncased",
        "description": "Lightweight and fast, good for starting out"
    },
    "roberta": {
        "name": "roberta-base",
        "description": "Better performance than BERT, good balance of speed and accuracy"
    },
    "albert": {
        "name": "albert-base-v2",
        "description": "Memory efficient, faster inference"
    },
    "xlm-roberta": {
        "name": "xlm-roberta-base",
        "description": "Good for multilingual data"
    },
    "electra": {
        "name": "google/electra-small-discriminator",
        "description": "Efficient and effective, good for resource-constrained environments"
    },
    "deberta": {
        "name": "microsoft/deberta-v3-small",
        "description": "State-of-the-art performance, good for complex tasks"
    }
}

def load_dataset(game_number, dataset_type):
    """Load processed dataset."""
    with open(f'processed_data/game{game_number}_{dataset_type}.json', 'r') as f:
        return json.load(f)

def create_prompt(player_data, game_number):
    """Create a prompt for the model based on player data."""
    features = player_data['features']
    feature_str = ', '.join([f"{k}: {v}" for k, v in features.items()])
    
    prompt = f"""In a mobile racing game (Game {game_number}), a player has shown the following activity in their first 5 days of playing: {feature_str}. Will the player stop playing in the next 10 days?"""
    
    return prompt

class ChurnPredictor:
    def __init__(self, model_key="distilbert"):
        """Initialize the model and tokenizer."""
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Model {model_key} not found. Available models: {list(AVAILABLE_MODELS.keys())}")
        
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
                problem_type="single_label_classification"
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
                padding=True
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
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_pred_proba)
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
    parser = argparse.ArgumentParser(description='Churn Prediction using Local Models')
    parser.add_argument('--model', type=str, default='distilbert',
                      choices=list(AVAILABLE_MODELS.keys()),
                      help='Model to use for prediction')
    parser.add_argument('--list-models', action='store_true',
                      help='List available models and exit')
    
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
        test_data = load_dataset(game_number, 'test')
        
        # Get predictions for each player
        predictions = []
        true_labels = []
        prediction_probas = []
        
        for player_id in tqdm(test_data['features'].keys()):
            # Create player data dictionary
            player_data = {
                'features': test_data['features'][player_id],
                'windows': test_data['windows'][player_id]
            }
            
            # Get prediction from model
            prompt = create_prompt(player_data, game_number)
            pred, prob = predictor.predict(prompt)
            
            if pred is not None:
                predictions.append(pred)
                true_labels.append(int(test_data['windows'][player_id]['churned']))
                prediction_probas.append(prob)
        
        # Calculate metrics
        metrics = evaluate_predictions(
            np.array(true_labels),
            np.array(predictions),
            np.array(prediction_probas)
        )
        
        # Print metrics
        print(f"\nMetrics for {args.model.upper()} on Game {game_number}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save metrics
        output_file = f'processed_data/{args.model}_metrics_game{game_number}.json'
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {output_file}")

if __name__ == "__main__":
    main() 
