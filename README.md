## MMD Assignment 1: Telemetry-Based Churn Prediction in Mobile Racing Games

## 1. Overview & Goal

This project implements churn prediction for two freemium mobile racing games using telemetry data. We analyze player behavior patterns to predict which players will stop playing (churn) within a 10-day period after observing their activity for 5 days.

The project emphasizes efficient data processing, modular design, and follows the methodology from the Kim et al. (2017) research on mobile game churn prediction.

---

## 2. Project Structure

The project is organized into a modular, scalable structure:

```
project-1/
├── src/                              # Source code
│   ├── config/                       # Configuration management
│   ├── utils/                        # Reusable utilities
│   ├── data_processing/              # Data pipeline modules
│   ├── scripts/                      # Executable scripts
│   ├── data/                         # Data directory
│   ├── models/                       # Model implementations
│   └── tests/                        # Unit tests
├── logs/                             # Log files
├── results/                          # Analysis results
└── requirements.txt                  # Dependencies
```

For detailed structure documentation, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

---

## 3. Quick Start

### Installation

1. **Clone and navigate to the project:**
   ```bash
   git clone git@github.com:esacon/mining-media-data-ii.git
   cd mining-media-data-ii/project-1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Development Setup

For development work, use the automated setup script:

```bash
./scripts/setup_dev.sh
```

This script will:
- Create a virtual environment
- Install all dependencies
- Set up pre-commit hooks
- Run initial code formatting and linting

### Code Quality Tools

The project includes comprehensive code quality tools:

#### Formatting and Linting
- **Black**: Code formatter with 88-character line length
- **isort**: Import sorter compatible with Black
- **flake8**: Linter for code quality and style

#### Available Commands
```bash
# Format code (black + isort)
make format

# Check formatting without changes
make check

# Run linter
make lint

# Run tests
make test

# Run all checks
make all

# Clean cache files
make clean
```

#### Pre-commit Hooks
Pre-commit hooks automatically run formatting and linting before each commit:
```bash
# Install hooks (done automatically by setup script)
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Running the Data Pipeline

1. **Run the complete pipeline:**
   ```bash
   python src/scripts/run_pipeline.py
   ```

2. **Inspect the results:**
   ```bash
   python src/scripts/inspect_datasets.py
   ```

3. **Run with custom parameters:**
   ```bash
   python src/scripts/run_pipeline.py --observation-days 7 --churn-days 14
   ```

### Command Line Options

The pipeline script supports various options:
- `--config`: Path to custom config.yaml file
- `--data-dir`: Override input data directory from config
- `--output-dir`: Override output directory from config
- `--observation-days`: Override observation period in days from config
- `--churn-days`: Override churn prediction period in days from config
- `--prep-only`: Run only data preparation
- `--create-only`: Run only dataset creation

---

## 4. Assignment Tasks Checklist

### Data Preparation ✅

- [x] Download telemetry datasets for two mobile racing games
- [x] Structure Game 1 dataset into JSONL format (one JSON object per player per line)
- [x] Verify Game 2 dataset is already in correct JSONL format
- [x] Create 80/20 player-based train/evaluation split for both games

### Dataset Creation ✅

- [x] Create DS1 (training dataset) from the 80% training split
- [x] Define a 5-day Observation Period (OP) for each player based on their log times
- [x] Define a 10-day Churn Prediction Period (CP) immediately following the OP
- [x] Label players as "churned" (no activity in CP) or "not churned" based on event occurrence
- [x] Create DS2 (evaluation dataset) from the 20% evaluation split using the same OP, CP, and labeling process

### Feature Engineering

- [ ] Extract behavioral features from the 5-day observation periods (OP) in DS1
- [ ] Implement common features as described in Kim et al. (2017) (e.g., playCount, activeDuration, bestScore, meanScore)
- [ ] Add game-specific features (e.g., purchase data for Game 2, if applicable and identifiable in logs)
- [ ] Apply the same feature extraction steps to DS2

### Model Training & Evaluation

- [ ] Train a Decision Tree classifier (required)
- [ ] Train at least two additional distinct classifiers (e.g., Random Forest, Logistic Regression, Gradient Boosting)
- [ ] Evaluate all trained models on DS2
- [ ] Report performance metrics (e.g., AUC, Accuracy, Precision, Recall, F1-score)
- [ ] Analyze and report important features for churn prediction and identify the best-performing classifier

### Bonus: LLM Integration

- [ ] Choose and set up a Large Language Model (LLM)
- [ ] Create prompts for each user in DS2 for zero-shot churn prediction, including relevant user information
- [ ] Evaluate the LLM's churn prediction performance on DS2
- [ ] Compare LLM results with the traditional classifiers

### Documentation & Submission

- [x] Document the data processing pipeline and methodology
- [ ] Document feature engineering and model training processes
- [ ] Ensure all results and model comparisons are clearly presented
- [ ] Prepare for a potential presentation/discussion of the results

---

## 5. Dataset Description

This project utilizes telemetry data from the first two freemium mobile racing games described in Kim et al. (2017).

### Game 1: "Dodge the Mud"
- **Original Format**: CSV
- **Target Format**: JSONL
- **Key Fields**: `device.id` (player ID), `time` (end time of the play), `score` (score of the play)
- **Processing**: Raw data grouped by `device.id` and transformed into JSONL format

### Game 2: Undisclosed Casual Racing Game
- **Format**: JSONL
- **Key Fields**: `uid` (unique user ID), `time` (end time of the event), `event` (type of record), `score` (score of the play), `purchase.price` (for purchase events)
- **Processing**: Utilize existing JSONL structure

**Data Source URL**: `https://tinyurl.com/pchurndatasets`

---

## 6. Key Technologies & Libraries

### Core Libraries
- **Data Handling**: `polars` (primary), `pandas`, `json`
- **Machine Learning**: `scikit-learn` (for classifiers, metrics, preprocessing)
- **Time Processing**: `datetime`, custom time utilities
- **LLM Integration (Bonus)**: `transformers`, `openai`

### Performance Optimization
- **Fast I/O**: `pyarrow` (Parquet format), `orjson` (faster JSON parsing)
- **Memory Management**: `memory_profiler`

### Development Tools
- **Code Quality**: `black`, `isort`, `flake8`
- **Testing**: `pytest`
- **Notebooks**: `jupyter`, `jupyterlab`

---

## 7. Configuration

The project uses a YAML-based configuration system for easy parameter management and centralized filename configuration. The main configuration file is `config.yaml` in the project root.

### Default Configuration

```yaml
# Data Processing Parameters
data_processing:
  observation_days: 5          # Days for observation period
  churn_period_days: 10        # Days for churn prediction period
  train_ratio: 0.8             # Train/eval split ratio
  random_seed: 42              # Random seed for reproducibility

# Directory Paths (relative to project root)
paths:
  data_dir: "src/data"
  processed_dir: "src/data/processed"
  logs_dir: "logs"
  results_dir: "results"

# File Names Configuration
filenames:
  # Input files
  game1_csv: "dataset_game1/rawdata_game1.csv"
  game2_jsonl: "dataset_game2/playerLogs_game2_playerbasedlines.jsonl"
  
  # Intermediate files (after conversion and splitting)
  game1_converted: "game1_player_events.jsonl"
  game1_train: "game1_player_events_train.jsonl"
  game1_eval: "game1_player_events_eval.jsonl"
  game2_train: "playerLogs_game2_playerbasedlines_train.jsonl"
  game2_eval: "playerLogs_game2_playerbasedlines_eval.jsonl"
  
  # Final labeled datasets
  game1_ds1: "game1_DS1_labeled.jsonl"
  game1_ds2: "game1_DS2_labeled.jsonl"
  game2_ds1: "game2_DS1_labeled.jsonl"
  game2_ds2: "game2_DS2_labeled.jsonl"
  
  # Result files
  preparation_results: "preparation_results.json"
  dataset_creation_results: "dataset_creation_results.json"
  pipeline_results: "pipeline_results.json"
  
  # File suffixes
  train_suffix: "_train.jsonl"
  eval_suffix: "_eval.jsonl"
  labeled_suffix: "_labeled.jsonl"

# Logging Configuration
logging:
  level: "INFO"                # DEBUG, INFO, WARNING, ERROR, CRITICAL
  console: true                # Log to console
  file: true                   # Log to file

# Performance Settings
performance:
  batch_size: 1000             # Batch size for processing
  progress_interval: 1000      # Progress logging interval
```

### Customizing Configuration

1. **Edit config.yaml directly** for permanent changes:
   ```yaml
   data_processing:
     observation_days: 7        # Changed from 5 to 7
     churn_period_days: 14      # Changed from 10 to 14
   
   filenames:
     labeled_suffix: "_churn_labeled.jsonl"  # Custom naming
   ```

2. **Override via command line** for temporary changes:
   ```bash
   python src/scripts/run_pipeline.py --observation-days 7 --churn-days 14
   ```

3. **Use custom config file**:
   ```bash
   python src/scripts/run_pipeline.py --config my_custom_config.yaml
   ```

### Benefits of Centralized Configuration

- **Easy Customization**: Change filenames without modifying code
- **Environment Flexibility**: Different configs for dev/test/prod
- **Experiment Management**: Easy to track different parameter combinations
- **Consistency**: All components use the same filename configurations
- **Maintainability**: Single source of truth for all settings

---

## 8. Data Processing Pipeline

The pipeline consists of two main steps:

### Step 1: Data Preparation
- Converts Game 1 CSV to JSONL format
- Splits both games into 80/20 train/eval sets
- Handles different data formats between games

### Step 2: Dataset Creation
- Defines 5-day observation period from each player's first event
- Defines 10-day churn prediction period following observation
- Labels players as churned if no events in churn period
- Creates DS1 (training) and DS2 (evaluation) datasets

### Output Files
- `game1_DS1_labeled.jsonl` - Game 1 training dataset
- `game1_DS2_labeled.jsonl` - Game 1 evaluation dataset
- `game2_DS1_labeled.jsonl` - Game 2 training dataset
- `game2_DS2_labeled.jsonl` - Game 2 evaluation dataset
- Statistics and metadata files

---

## 9. Usage Examples

### As a Library
```python
from src.data_processing import DataPipeline
from src.config import get_settings

# Use default configuration
settings = get_settings()
pipeline = DataPipeline(settings)
results = pipeline.run_full_pipeline()

# Use custom configuration file
settings = get_settings("my_config.yaml")
pipeline = DataPipeline(settings)
```

### Programmatic Access
```python
from src.data_processing import DataPreparation, DatasetCreator
from src.utils import load_jsonl_sample, calculate_dataset_stats
from src.config import get_settings

# Load settings
settings = get_settings()

# Prepare data
prep = DataPreparation(settings)
prep_results = prep.prepare_all_data()

# Create labeled datasets
creator = DatasetCreator(settings)
dataset_results = creator.create_all_datasets()

# Analyze results
stats = calculate_dataset_stats(settings.processed_dir / settings.game1_ds1)
```

### Custom Configuration Examples

```python
# Load with custom observation and churn periods
settings = get_settings()
settings.observation_days = 7
settings.churn_period_days = 14

# Initialize components with modified settings
pipeline = DataPipeline(settings)

# Access configured filenames
print(f"Game 1 training file: {settings.game1_train}")
print(f"Pipeline results will be saved to: {settings.pipeline_results}")
```

---

## 10. Course Information

- **Course**: Mining Media Data II
- **Semester**: Summer Semester 25
- **Assignment**: Assignment-1
- **Instructors**: Prof. Dr. Rafet Sifa, Dr. Lorenz Sparrenberg, Maren Pielka
- **Points**: 50 Pts. + 15 Bonus Pts. for LLM part
- **Deadline**: 03.06.2025 at 11:59 am Germany time
- **Submission**: Single zip file (Python code + PDF or Jupyter Notebook) to `amllab@bit.uni-bonn.de` with the title `MMD SS2025 Assignment 1 [GroupID]`

---

## 11. Key References

- Kim, S., Choi, D., Lee, E., & Rhee, W. (2017). Churn prediction of mobile and online casual games using play log data. _PLoS one_, _12_(7), e0180735.
- Hadiji, F., Sifa, R., Drachen, A., Thurau, C., Kersting, K., & Bauckhage, C. (2014). Predicting player churn in the wild. In _2014 IEEE conference on computational intelligence and games_ (pp. 1-8). IEEE.
- Sifa, R. (2021). Predicting player churn with echo state networks.
