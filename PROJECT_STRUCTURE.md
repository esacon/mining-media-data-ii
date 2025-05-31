# Project Structure

This document outlines the streamlined structure of the churn prediction project, designed for simplicity and maintainability.

## Directory Structure

```
project-1/
├── src/                              # Source code
│   ├── config/                       # Configuration management
│   │   └── settings.py               # Centralized project settings
│   ├── utils/                        # Utility functions
│   │   ├── logging_utils.py          # Logging setup
│   │   ├── file_utils.py             # File operations
│   │   ├── time_utils.py             # Time/timestamp utilities
│   │   └── data_utils.py             # Data processing utilities
│   ├── data_processing/              # Main data pipeline
│   │   ├── data_preparation.py       # CSV→JSONL conversion & splitting
│   │   ├── dataset_creation.py       # Observation/churn labeling
│   │   ├── feature_engineering.py    # Kim et al. (2017) features
│   │   └── pipeline.py               # Pipeline orchestrator
│   ├── scripts/                      # Executable scripts
│   │   ├── run_pipeline.py           # Main pipeline runner
│   │   └── inspect_datasets.py       # Dataset inspection
│   └── data/                         # Data directory
│       ├── dataset_game1/            # Game 1 raw data
│       ├── dataset_game2/            # Game 2 raw data
│       └── processed/                # Generated datasets
├── results/features/                 # Extracted feature CSV files
├── logs/                             # Log files
├── config.yaml                       # Main configuration
└── requirements.txt                  # Dependencies
```

## Core Components

### Data Processing Pipeline (`src/data_processing/`)

The pipeline follows a simple 3-step process:

#### 1. **Data Preparation** (`data_preparation.py`)
- **Purpose**: Convert and split raw data
- **Key Features**:
  - Converts Game 1 CSV to JSONL format
  - Creates 80/20 train/eval splits for both games
  - Unified path handling for all file operations

#### 2. **Dataset Creation** (`dataset_creation.py`)
- **Purpose**: Create labeled datasets with observation/churn periods
- **Key Features**:
  - Defines 5-day observation period per player
  - Defines 10-day churn prediction period
  - Labels players as churned/not churned
  - Generates DS1 (train) and DS2 (eval) datasets

#### 3. **Feature Engineering** (`feature_engineering.py`)
- **Purpose**: Extract behavioral features from observation periods
- **Key Features**:
  - Implements 10 common features from Kim et al. (2017)
  - Adds game-specific features for Game 2
  - Validates and cleans extracted features
  - Generates ML-ready CSV files

#### 4. **Pipeline Orchestrator** (`pipeline.py`)
- **Purpose**: Coordinates the complete pipeline
- **Key Features**:
  - Manages step execution with timing
  - Handles logging and error reporting
  - Supports step-by-step or full pipeline execution

### Configuration (`src/config/`)

Centralized configuration management with YAML support:
- **Single source of truth** for all settings
- **Flexible overrides** via command line
- **Automatic directory creation**
- **Type-safe configuration** with validation

### Utilities (`src/utils/`)

Reusable functions supporting the pipeline:
- **Logging**: Consistent logging across all modules
- **File Operations**: Reading, writing, and path management
- **Time Handling**: Timestamp conversion and boundary calculations
- **Data Processing**: JSONL operations and statistics

## Quick Start

### Basic Usage
```bash
# Run complete pipeline
python src/scripts/run_pipeline.py

# Run with custom parameters
python src/scripts/run_pipeline.py --observation-days 7 --churn-days 14

# Run individual steps
python src/scripts/run_pipeline.py --prep-only
python src/scripts/run_pipeline.py --create-only
python src/scripts/run_pipeline.py --features-only
```

### Programmatic Usage
```python
from src.config import get_settings
from src.data_processing import DataPipeline

# Load settings and run pipeline
settings = get_settings()
pipeline = DataPipeline(settings)
pipeline.run_full_pipeline()
```

## Configuration System

### YAML Configuration (`config.yaml`)
```yaml
data_processing:
  observation_days: 5          # Observation period
  churn_period_days: 10        # Churn prediction period
  train_ratio: 0.8             # Train/eval split

paths:
  data_dir: "src/data"
  processed_dir: "src/data/processed"
  results_dir: "results"

filenames:
  # All filenames centrally managed
  game1_csv: "dataset_game1/rawdata_game1.csv"
  game2_jsonl: "dataset_game2/playerLogs_game2_playerbasedlines.jsonl"
  # ... (see config.yaml for complete list)

logging:
  level: "INFO"
  console: true
  file: true
```

### Command Line Overrides
```bash
# Override any configuration parameter
python src/scripts/run_pipeline.py \
  --data-dir /custom/data \
  --observation-days 7 \
  --churn-days 14 \
  --log-level DEBUG
```

## Features Extracted

### Common Features (Both Games)
Based on Kim et al. (2017) methodology:

1. **playCount** - Total plays in observation period
2. **bestScore** - Maximum score achieved
3. **meanScore** - Average score
4. **worstScore** - Minimum score
5. **sdScore** - Standard deviation of scores
6. **bestScoreIndex** - Position of best score (normalized)
7. **bestSubMeanCount** - (Best - Mean) / Play count
8. **bestSubMeanRatio** - (Best - Mean) / Mean
9. **activeDuration** - Time between first and last play (hours)
10. **consecutivePlayRatio** - Ratio of consecutive plays

### Game 2 Specific Features
11. **purchaseCount** - Total vehicle purchases
12. **highestPrice** - Highest purchase price

## Output Files

The pipeline generates the following files:

### Intermediate Files
- `game1_player_events.jsonl` - Converted Game 1 data
- `*_train.jsonl` / `*_eval.jsonl` - Train/eval splits

### Labeled Datasets
- `game1_DS1_labeled.jsonl` - Game 1 training dataset
- `game1_DS2_labeled.jsonl` - Game 1 evaluation dataset
- `game2_DS1_labeled.jsonl` - Game 2 training dataset
- `game2_DS2_labeled.jsonl` - Game 2 evaluation dataset

### Feature Files
- `results/features/game1_DS1_features.csv` - Game 1 training features
- `results/features/game1_DS2_features.csv` - Game 1 evaluation features
- `results/features/game2_DS1_features.csv` - Game 2 training features
- `results/features/game2_DS2_features.csv` - Game 2 evaluation features
