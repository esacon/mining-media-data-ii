# Project Structure

This document outlines the organized structure of the churn prediction project.

## Directory Structure

```
project-1/
├── src/                              # Source code
│   ├── __init__.py                   # Main package init
│   ├── config/                       # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py               # Project settings and environment config
│   ├── utils/                        # Utility modules
│   │   ├── __init__.py
│   │   ├── logging_utils.py          # Logging utilities
│   │   ├── file_utils.py             # File operations
│   │   ├── time_utils.py             # Time/timestamp utilities
│   │   └── data_utils.py             # Data processing utilities
│   ├── data_processing/              # Data processing pipeline
│   │   ├── __init__.py
│   │   ├── data_preparation.py       # CSV to JSONL conversion, splitting
│   │   ├── dataset_creation.py       # Observation/churn period labeling
│   │   └── pipeline.py               # Main pipeline orchestrator
│   ├── scripts/                      # Executable scripts
│   │   ├── __init__.py
│   │   ├── run_pipeline.py           # Main pipeline runner
│   │   └── inspect_datasets.py       # Dataset inspection tool
│   ├── data/                         # Data directory
│   │   ├── dataset_game1/          # Game 1 raw data
│   │   ├── dataset_game2/          # Game 2 raw data
│   │   ├── processed/                # Processed datasets
│   │   │   ├── README.md             # Processed data documentation
│   │   │   ├── *.jsonl               # Player event files
│   │   │   ├── *_labeled.jsonl       # Labeled datasets (DS1, DS2)
│   │   │   ├── *_stats.json          # Dataset statistics
│   │   │   └── pipeline_results.json # Pipeline execution results
│   │   └── _analysis/                # Data analysis files
│   ├── lib/                          # External libraries (if any)
│   ├── models/                       # Model implementations (future)
│   └── tests/                        # Unit tests (future)
├── logs/                             # Log files
├── results/                          # Analysis results and figures
├── references/                       # Reference materials
├── scripts/                          # Additional utility scripts
├── requirements.txt                  # Python dependencies
├── Pipfile                           # Pipenv configuration
├── Pipfile.lock                      # Pipenv lock file
├── config.yaml                       # Main configuration file
├── README.md                         # Main project README
└── PROJECT_STRUCTURE.md              # This file
```

## Module Descriptions

### Core Packages

#### `src/config/`
- **Purpose**: Centralized configuration management
- **Key Files**:
  - `settings.py`: Project settings, environment variables, directory paths, and filename configurations

#### `src/utils/`
- **Purpose**: Reusable utility functions
- **Key Files**:
  - `logging_utils.py`: Consistent logging setup across modules
  - `file_utils.py`: File operations, size calculations, directory management
  - `time_utils.py`: Timestamp format detection and conversion
  - `data_utils.py`: JSONL processing, dataset statistics

#### `src/data_processing/`
- **Purpose**: Data preparation and dataset creation pipeline
- **Key Files**:
  - `data_preparation.py`: CSV to JSONL conversion, train/eval splitting
  - `dataset_creation.py`: Observation/churn period processing, labeling
  - `pipeline.py`: Main orchestrator for the complete pipeline

#### `src/scripts/`
- **Purpose**: Executable command-line scripts
- **Key Files**:
  - `run_pipeline.py`: Main entry point with CLI arguments
  - `inspect_datasets.py`: Dataset inspection and visualization

## Design Principles

### 1. **Separation of Concerns**
- Each module has a single, well-defined responsibility
- Utilities are separated from business logic
- Configuration is centralized

### 2. **Reusability**
- Common operations are abstracted into utility functions
- Modules can be imported and used independently
- Clear interfaces between components

### 3. **Scalability**
- Modular design allows easy addition of new features
- Configuration-driven approach for easy parameter changes
- Logging and error handling throughout

### 4. **Maintainability**
- Clear naming conventions
- Comprehensive documentation
- Type hints for better code clarity
- Consistent error handling

## Usage Patterns

### Running the Pipeline
```bash
# Full pipeline
python src/scripts/run_pipeline.py

# With custom parameters
python src/scripts/run_pipeline.py --observation-days 7 --churn-days 14

# Only data preparation
python src/scripts/run_pipeline.py --prep-only
```

### Inspecting Results
```bash
# Basic inspection
python src/scripts/inspect_datasets.py

# Detailed inspection with event details
python src/scripts/inspect_datasets.py --detailed

# Inspect only Game 1
python src/scripts/inspect_datasets.py --game game1
```

### Using as Library
```python
from src.data_processing import DataPipeline
from src.config import get_settings

# Use with default settings
settings = get_settings()
pipeline = DataPipeline(settings)
results = pipeline.run_full_pipeline()

# Use with custom settings
settings = get_settings("custom_config.yaml")
pipeline = DataPipeline(settings)
```

## Configuration Management

The project uses a YAML-based configuration system with centralized filename management:

- **YAML Configuration**: Main settings in `config.yaml` at project root
- **Settings Class**: Type-safe configuration with validation in `src/config/settings.py`
- **Command Line Overrides**: Temporary parameter changes via CLI arguments
- **Directory Management**: Automatic creation of required directories
- **Filename Configuration**: All filenames centrally managed and configurable

### Configuration Structure

```yaml
data_processing:          # Core processing parameters
  observation_days: 5
  churn_period_days: 10
  train_ratio: 0.8
  random_seed: 42

paths:                    # Directory paths
  data_dir: "src/data"
  processed_dir: "src/data/processed"
  logs_dir: "logs"
  results_dir: "results"

filenames:                # File naming configuration
  # Input files
  game1_csv: "dataset_game1/rawdata_game1.csv"
  game2_jsonl: "dataset_game2/playerLogs_game2_playerbasedlines.jsonl"
  
  # Intermediate files
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

logging:                  # Logging configuration
  level: "INFO"
  console: true
  file: true

performance:              # Performance settings
  batch_size: 1000
  progress_interval: 1000
```

### Usage Patterns

```python
# Load default configuration
settings = get_settings()

# Load custom configuration
settings = get_settings("custom_config.yaml")

# Reload configuration
settings = reload_settings()

# Access filename configurations
print(f"Game 1 training file: {settings.game1_train}")
print(f"Pipeline results file: {settings.pipeline_results}")
```

### Component Initialization

All data processing components now use the settings-based approach:

```python
from src.config import get_settings
from src.data_processing import DataPreparation, DatasetCreator, DataPipeline

# Load settings
settings = get_settings()

# Initialize components with settings
data_prep = DataPreparation(settings)
dataset_creator = DatasetCreator(settings)
pipeline = DataPipeline(settings)
```

## Future Extensions

This structure is designed to easily accommodate:

1. **Feature Engineering**: Add `src/features/` package
2. **Model Training**: Expand `src/models/` package
3. **Evaluation**: Add `src/evaluation/` package
4. **API**: Add `src/api/` for serving models
5. **Visualization**: Add `src/visualization/` for plots and dashboards

## Benefits of This Structure

1. **Clear Organization**: Easy to find and understand code
2. **Modularity**: Components can be developed and tested independently
3. **Reusability**: Utilities can be used across different parts of the project
4. **Scalability**: Easy to add new features without restructuring
5. **Maintainability**: Clear interfaces and separation of concerns
6. **Professional**: Follows Python packaging best practices
7. **Configurable**: Centralized filename and parameter management
8. **Flexible**: Easy to customize for different environments or experiments 