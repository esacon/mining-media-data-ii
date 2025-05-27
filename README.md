## MMD Assignment 1: Telemetry-Based Churn Prediction in Mobile Racing Games

**Version:** 1.0.0
**Last Updated:** May 2025
**Repository:** [mining-media-data-ii](https://github.com/esacon/mining-media-data-ii)

---

## 1. Overview & Goal

This project implements churn prediction for two freemium mobile racing games using telemetry data. We analyze player behavior patterns to predict which players will stop playing (churn) within a 10-day period after observing their activity for 5 days.

The project emphasizes efficient data processing, potentially leveraging libraries optimized for large datasets, and primarily follows the methodology from the Kim et al. (2017) research on mobile game churn prediction.

---

## 2. Assignment Tasks Checklist

### Data Preparation

- [ ] Download telemetry datasets for two mobile racing games.
- [ ] Structure Game 1 dataset into JSONL format (one JSON object per player per line).
- [ ] Verify Game 2 dataset is already in correct JSONL format.
- [ ] Create 80/20 player-based train/evaluation split for both games.

### Dataset Creation

- [ ] Create DS1 (training dataset) from the 80% training split.
- [ ] Define a 5-day Observation Period (OP) for each player based on their log times.
- [ ] Define a 10-day Churn Prediction Period (CP) immediately following the OP.
- [ ] Label players as "churned" (no activity in CP) or "not churned" based on event occurrence.
- [ ] Create DS2 (evaluation dataset) from the 20% evaluation split using the same OP, CP, and labeling process.

### Feature Engineering

- [ ] Extract behavioral features from the 5-day observation periods (OP) in DS1.
- [ ] Implement common features as described in Kim et al. (2017) (e.g., playCount, activeDuration, bestScore, meanScore).
- [ ] Add game-specific features (e.g., purchase data for Game 2, if applicable and identifiable in logs).
- [ ] Apply the same feature extraction steps to DS2.

### Model Training & Evaluation

- [ ] Train a Decision Tree classifier (required).
- [ ] Train at least two additional distinct classifiers (e.g., Random Forest, Logistic Regression, Gradient Boosting).
- [ ] Evaluate all trained models on DS2.
- [ ] Report performance metrics (e.g., AUC, Accuracy, Precision, Recall, F1-score).
- [ ] Analyze and report important features for churn prediction and identify the best-performing classifier.

### Bonus: LLM Integration

- [ ] Choose and set up a Large Language Model (LLM).
- [ ] Create prompts for each user in DS2 for zero-shot churn prediction, including relevant user information.
- [ ] Evaluate the LLM's churn prediction performance on DS2.
- [ ] Compare LLM results with the traditional classifiers.

### Documentation & Submission

- [ ] Document the entire process, methodology, and findings in a Jupyter Notebook or a PDF with an accompanying codebase.
- [ ] Ensure all results and model comparisons are clearly presented.
- [ ] Prepare for a potential presentation/discussion of the results.

---

## 3. Dataset Description

This project utilizes telemetry data from the first two freemium mobile racing games described in Kim et al. (2017), as per the assignment instructions.

### Game 1: "Dodge the Mud"

- **Original Format**: CSV (as per your `data_format_summary.json` for `dataset_1`).
- **Target Format**: JSONL.
- **Key Fields (from source)**: `device.id` (player ID), `time` (end time of the play), `score` (score of the play).
- **Processing**: Raw data needs to be grouped by `device.id` and then each player's sequence of events transformed into a single JSON object per line.

### Game 2: Undisclosed Casual Racing Game

- **Format**: JSONL (as per your `data_format_summary.json` for `dataset_2` and assignment notes).
- **Key Fields (from source)**: `device.id` (unique user ID, referred to as `uid` in your files), `time` (end time of the event), `event` (type of record, e.g., "play" or "softPurchase"), `score` (score of the play, when `event` is "play"), `purchase.price` (when `event` is "softPurchase").
- **Processing**: Utilize the existing JSONL structure.

**Data Source URL**: `https://tinyurl.com/pchurndatasets`

---

## 4. Project Structure (Revised Suggestion)

```
MMD_SS25_Assignment1_YourGroupID/
├── data/
│   ├── dataset_1_game1/              # Game 1 raw data (e.g., rawdata_game1.csv)
│   ├── dataset_2_game2/              # Game 2 JSONL files (e.g., playerLogs_game2_playerbasedlines_part_aa)
│   ├── processed/                    # Output of data preparation steps
│   │   ├── game1_player_events.jsonl
│   │   ├── game1_DS1_features.csv
│   │   ├── game1_DS2_features.csv
│   │   ├── game2_DS1_features.csv
│   │   ├── game2_DS2_features.csv
│   ├── README.md                     # Detailed dataset README.md
│   └── data_format_summary.json      # Data_format_summary.json
├── notebooks/                        # Primary location for analysis and reporting
├── src/                              # Optional: for reusable Python modules
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── utils.py
├── results/                          # For storing output figures, tables, model performance
│   ├── figures/
│   └── performance_summary.csv
├── Pipfile                           # If using Pipenv
├── Pipfile.lock
├── requirements.txt                  # If using pip
└── README.md                         # This project README
```

---

## 5. Setup & Installation

### Prerequisites

- Python 3.10+
- `pipenv` (recommended) or `pip`
- Git

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone git@github.com:esacon/mining-media-data-ii.git
    cd mining-media-data-ii
    ```

2.  **Install dependencies (using `pipenv` is recommended):**

    ```bash
    pipenv install
    pipenv shell  # Activate virtual environment
    ```

3.  **Alternative (using `pip` with `requirements.txt`):**
    Ensure you have a `requirements.txt` file listing packages like `pandas`, `scikit-learn`, etc.
    ```bash
    pip install -r requirements.txt
    ```

---

## 6. Key Technologies & Libraries

### Core Libraries

- **Data Handling**: `polars` (primarily), `json`
- **Machine Learning**: `scikit-learn` (for classifiers, metrics, preprocessing)
- **LLM Interaction (Bonus)**: e.g., `transformers`, `openai` (depending on the chosen LLM)

### Optional: For Performance Optimization on Very Large Datasets

- **Fast I/O & Serialization**:
  - `pyarrow` (For Parquet file format, efficient columnar storage)
  - `orjson` (Faster JSON parsing than the built-in `json` module)
- **Memory Management**:
  - `memory_profiler` (For diagnosing memory bottlenecks)

---

## 7. Usage

### Primary Workflow (Jupyter Notebooks)

The main analysis, from data loading to model evaluation and LLM integration, is expected to be conducted and documented within Jupyter Notebook(s) located in the `notebooks/` directory.

1.  **Activate the virtual environment (if not already):**
    ```bash
    pipenv shell
    ```
    Or ensure your Python interpreter with installed dependencies is active.
2.  **Launch Jupyter Lab/Notebook:**
    ```bash
    pipenv run jupyter lab
    # or
    # jupyter lab
    ```
3.  Navigate to the `notebooks/` directory and open the notebooks. Execute cells sequentially.

---

## 8. Course Information

- **Course**: Mining Media Data II
- **Semester**: Summer Semester 25
- **Assignment**: Assignment-1
- **Instructors**: Prof. Dr. Rafet Sifa, Dr. Lorenz Sparrenberg, Maren Pielka
- **Points**: 50 Pts. + 15 Bonus Pts. for LLM part
- **Deadline**: 03.06.2025 at 11:59 am Germany time
- **Submission**: Single zip file (Python code + PDF or Jupyter Notebook) to `amllab@bit.uni-bonn.de` with the title `MMD SS2025 Assignment 1 [GroupID]`

---

## 9. Key References

- Kim, S., Choi, D., Lee, E., & Rhee, W. (2017). Churn prediction of mobile and online casual games using play log data. _PLoS one_, _12_(7), e0180735.
- Hadiji, F., Sifa, R., Drachen, A., Thurau, C., Kersting, K., & Bauckhage, C. (2014). Predicting player churn in the wild. In _2014 IEEE conference on computational intelligence and games_ (pp. 1-8). IEEE.
- Sifa, R. (2021). Predicting player churn with echo state networks.

---

## 11. Development Guidelines

- All submitted work must be original. Adhere to the university's plagiarism policy.
- Prioritize clarity and reproducibility in your code and documentation.
- If dealing with significantly large datasets, demonstrate consideration for efficient data processing techniques.
- Clearly document all data preprocessing, feature engineering choices, and model training steps.
- (Optional, but good practice) Consider including brief benchmarks or notes on memory usage if you implement specific optimizations for large data.
