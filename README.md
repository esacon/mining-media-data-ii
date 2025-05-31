# Telemetry-Based Churn Prediction in Mobile Racing Games

A complete machine learning pipeline for predicting player churn in freemium mobile racing games using telemetry data and behavioral features from Kim et al. (2017).

## 🎯 Overview

This project analyzes player behavior patterns to predict which players will stop playing (churn) within a 10-day period after observing their activity for 5 days. It implements a complete end-to-end pipeline from raw telemetry data to trained ML models with comprehensive evaluation.

---

## 🚀 Quick Start

### Installation
```bash
# Clone and navigate
git clone git@github.com:esacon/mining-media-data-ii.git
cd mining-media-data-ii/project-1

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline
```bash
# Complete pipeline
python src/scripts/run_pipeline.py

```

---

## 📁 Project Structure

```
project-1/
├── src/
│   ├── config/settings.py           # Configuration management
│   ├── data_processing/             # Data pipeline
│   │   ├── data_preparation.py      # Convert & split data
│   │   ├── dataset_creation.py      # Create labeled datasets
│   │   ├── feature_engineering.py   # Extract features
│   │   └── pipeline.py              # Orchestrate everything
│   ├── models/                      # ML pipeline
│   │   ├── base_classifier.py       # Abstract base classifier
│   │   ├── decision_tree_classifier.py     # Decision Tree implementation
│   │   ├── logistic_regression_classifier.py # Logistic Regression implementation
│   │   ├── random_forest_classifier.py     # Random Forest implementation
│   │   ├── model_config.py          # Model configuration management
│   │   └── model_pipeline.py        # ML pipeline orchestrator
│   ├── scripts/
│   │   ├── run_pipeline.py          # Complete pipeline runner
│   └── utils/                       # Helper functions
├── results/
│   ├── features/                    # Generated feature files
│   ├── models/                      # Trained model files (.joblib)
│   ├── feature_importance.json      # Feature importance rankings
│   └── model_evaluation_metrics.json # Model performance metrics
├── config.yaml                      # Main configuration
└── src/data/                        # Raw datasets
```

---

## 🔧 Configuration

### Simple Configuration (`config.yaml`)
```yaml
data_processing:
  observation_days: 5      # How long to observe players
  churn_period_days: 10    # How long to wait for churn
  train_ratio: 0.8         # 80% train, 20% evaluation

models:
  decision_tree:
    max_depth: 10
    min_samples_split: 20
    random_state: 42
  random_forest:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  logistic_regression:
    max_iter: 1000
    random_state: 42

filenames:
  game1_csv: "dataset_game1/rawdata_game1.csv"
  game2_jsonl: "dataset_game2/playerLogs_game2_playerbasedlines.jsonl"
```

### Command Line Overrides
```bash
# Experiment with different parameters
python src/scripts/run_pipeline.py --observation-days 7 --churn-days 14

# Run individual steps
python src/scripts/run_pipeline.py --prep-only
python src/scripts/run_pipeline.py --create-only
python src/scripts/run_pipeline.py --features-only
```

---

## 📊 Data Pipeline

### Step 1: Data Preparation
- Converts Game 1 CSV to standardized JSONL format
- Creates 80/20 train/evaluation splits for both games
- Maintains player-based splitting for proper evaluation

### Step 2: Dataset Creation
- Defines observation period (5 days from first play)
- Defines churn prediction period (10 days after observation)
- Labels players as churned (no activity) or retained
- Generates DS1 (training) and DS2 (evaluation) datasets

### Step 3: Feature Engineering
- Extracts 10 common behavioral features from Kim et al. (2017)
- Adds game-specific features (e.g., purchase behavior for Game 2)
- Validates and cleans all extracted features
- Generates ML-ready CSV files

---

## 🎮 Game Data

### Game 1: "Dodge the Mud"
- **Source**: CSV format with device IDs, scores, and timestamps
- **Features**: Play patterns and scoring behavior
- **Players**: ~1,500 unique devices
- **Best Model**: Logistic Regression (AUC: 79.2%, following Kim et al. 2017)

### Game 2: Racing Game with Purchases
- **Source**: JSONL format with detailed event logs
- **Features**: Play patterns, scoring, and purchase behavior
- **Players**: ~27,000 unique players
- **Best Model**: Random Forest (AUC: 73.4%, following Kim et al. 2017)

---

## 📈 Feature Importance Analysis

### Top Features by Game

#### Game 1 (All Models)
1. **activeDuration** - Time span between first and last play
2. **meanScore** / **worstScore** - Scoring patterns
3. **bestSubMeanRatio** - Performance improvement patterns

#### Game 2 (All Models)
1. **activeDuration** - Most predictive across all models
2. **consecutivePlayRatio** - Play consistency patterns
3. **purchaseCount** - Monetization behavior
4. **playCount** - Activity level

### Common Features (Both Games)
Following Kim et al. (2017) methodology:

1. **playCount** - Total plays during observation
2. **bestScore** - Maximum score achieved
3. **meanScore** - Average score
4. **worstScore** - Minimum score
5. **sdScore** - Standard deviation of scores
6. **bestScoreIndex** - Position of best score (normalized)
7. **bestSubMeanCount** - (Best - Mean) / Play count
8. **bestSubMeanRatio** - (Best - Mean) / Mean
9. **activeDuration** - Time span between first and last play
10. **consecutivePlayRatio** - Ratio of consecutive plays

### Game 2 Specific Features
11. **purchaseCount** - Total in-game purchases
12. **highestPrice** - Most expensive purchase

---

## 📁 Generated Files

### Labeled Datasets
- `game1_DS1_labeled.jsonl` - Game 1 training data
- `game1_DS2_labeled.jsonl` - Game 1 evaluation data
- `game2_DS1_labeled.jsonl` - Game 2 training data
- `game2_DS2_labeled.jsonl` - Game 2 evaluation data

### Feature Files (ML-Ready)
- `results/features/game1_DS1_features.csv` - Game 1 training features
- `results/features/game1_DS2_features.csv` - Game 1 evaluation features
- `results/features/game2_DS1_features.csv` - Game 2 training features
- `results/features/game2_DS2_features.csv` - Game 2 evaluation features

### Trained Models
- `results/models/game1_DecisionTree.joblib` - Game 1 Decision Tree
- `results/models/game1_RandomForest.joblib` - Game 1 Random Forest
- `results/models/game1_LogisticRegression.joblib` - Game 1 Logistic Regression
- `results/models/game2_DecisionTree.joblib` - Game 2 Decision Tree
- `results/models/game2_RandomForest.joblib` - Game 2 Random Forest
- `results/models/game2_LogisticRegression.joblib` - Game 2 Logistic Regression

### Results & Analysis
- `results/model_evaluation_metrics.json` - Complete performance metrics
- `results/feature_importance.json` - Feature importance rankings

---

## 🔬 Research Context

### Key Research Reference
*Kim, S., Choi, D., Lee, E., & Rhee, W. (2017). Churn prediction of mobile and online casual games using play log data. PLoS one, 12(7), e0180735.*

---

## 🛠️ Development

### Code Quality
```bash
# Format code
make format

# Run linting
make lint

# Run all checks
make all
```

### Testing
```bash
# Quick test - run pipeline with sample data
python src/scripts/run_pipeline.py --prep-only
```

---

## 📋 Assignment Information

- **Course**: Mining Media Data II, Summer Semester 25
- **Assignment**: Assignment-1 (50 Pts. + 15 Bonus Pts.)
- **Deadline**: 03.06.2025 at 11:59 am Germany time
- **Instructors**: Prof. Dr. Rafet Sifa, Dr. Lorenz Sparrenberg, Maren Pielka

---

For detailed technical documentation, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).
