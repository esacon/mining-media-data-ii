# Model Training and Evaluation Summary

## Models Used

Three different classification models were implemented:

- **Decision Tree Classifier**
- **Random Forest Classifier** (with 100 estimators)
- **Logistic Regression**

## Metrics Comparison

### Game 1

- **Best Performing:** Logistic Regression
    - Highest accuracy (**92.91%**)
    - Best AUC-ROC score (**0.778**)
    - Excellent recall (**99.19%**)
- **Close Second:** Random Forest
    - Very similar accuracy (**92.82%**)
    - Good AUC-ROC score (**0.758**)
    - High recall (**98.98%**)
- **Worst Performing:** Decision Tree
    - Lower accuracy (**89.62%**)
    - Poor AUC-ROC score (**0.536**)

### Game 2

- **Best Performing:** Logistic Regression
    - Highest accuracy (**78.15%**)
    - Best AUC-ROC score (**0.731**)
    - Highest recall (**95.83%**)
- **Close Second:** Random Forest
    - Good accuracy (**77.83%**)
    - Decent AUC-ROC score (**0.708**)
- **Worst Performing:** Decision Tree
    - Much lower accuracy (**68.53%**)
    - Poor AUC-ROC score (**0.593**)

## Key Observations

- All models performed significantly better on **Game 1** than **Game 2**.
- **Logistic Regression** consistently performed best across both games.
- **Decision Trees** consistently performed worst, suggesting the data may be too complex for simple tree-based decisions.
- The performance gap between models is larger in **Game 2**, indicating a more challenging classification task.

## Model Training Process and Code Insights

The model training process is implemented in Python using scikit-learn and pandas, as detailed in the `model_training.py` script. The workflow for both games is as follows:

1. **Data Loading:**
   - Processed feature datasets for each game and split (train/test) are loaded from CSV files.
   - Features and labels are separated, with `player_id` and `churned` columns handled appropriately.

2. **Feature Preparation:**
   - Features are converted to numpy arrays for compatibility with scikit-learn models.
   - For Logistic Regression, features are standardized using `StandardScaler` to improve model performance.

3. **Model Training:**
   - Three classifiers are initialized: Decision Tree, Random Forest (with 100 estimators), and Logistic Regression.
   - Each model is trained on the training data. Logistic Regression uses the scaled features, while tree-based models use the raw features.

4. **Evaluation:**
   - Each trained model is evaluated on the test set using metrics such as accuracy, precision, recall, F1 score, and AUC-ROC.
   - Predictions and probability scores are generated for metric calculation.

5. **Visualization:**
   - Feature importance is plotted for each model (where available) and saved as PNG files.
   - A comparison plot of all model metrics is also generated for each game.

6. **Results Saving:**
   - All evaluation metrics are saved as JSON files for further analysis and reporting.

**Code Structure Highlights:**
- The code is modular, with functions for loading data, preparing features, training models, evaluating, and plotting.
- The main loop iterates over both games, ensuring a consistent process.
- The use of scikit-learn's API allows for easy extension or modification of models and metrics.

This structured approach ensures reproducibility and clarity in the model evaluation process, making it straightforward to compare model performance across different games and datasets.

## Visualizations

To further support the evaluation, the following visualizations are included:

### Feature Importance Plots

- ![Feature Importance - Decision Tree (Game 1)](../plots/feature_importance_game1_decision_tree.png)
- ![Feature Importance - Random Forest (Game 1)](../plots/feature_importance_game1_random_forest.png)
- ![Feature Importance - Logistic Regression (Game 1)](../plots/feature_importance_game1_logistic_regression.png)
- ![Feature Importance - Decision Tree (Game 2)](../plots/feature_importance_game2_decision_tree.png)
- ![Feature Importance - Random Forest (Game 2)](../plots/feature_importance_game2_random_forest.png)
- ![Feature Importance - Logistic Regression (Game 2)](../plots/feature_importance_game2_logistic_regression.png)

### Model Metrics Comparison

- ![Model Metrics Comparison (Game 1)](../plots/metrics_comparison_game1.png)
- ![Model Metrics Comparison (Game 2)](../plots/metrics_comparison_game2.png)

These images provide a visual summary of feature importance for each model and a direct comparison of model performance metrics for both games.

## Project Structure Overview

Below is a summary of the key modules involved in the pipeline:

```
project-root/
├── src/
│   ├── data_processing/
│   │   ├── data_preparation.py      # Data loading, cleaning, and splitting
│   │   ├── dataset_creation.py      # Observation/churn period labeling
│   │   ├── feature_engineering.py   # Feature extraction from player data
│   │   └── pipeline.py              # Orchestrates the data pipeline
│   ├── model_training.py            # Model training, evaluation, and reporting
│   ├── llm_prediction.py            # LLM-based prediction module
│   └── ...
├── results/
│   ├── features/                    # Extracted feature CSVs
│   ├── models/                      # Trained model files
│   └── ...
├── plots/                           # Visualizations
└── ...
```

- **data_preparation.py**: Handles raw data conversion, cleaning, and splitting into train/eval sets.
- **feature_engineering.py**: Extracts behavioral features from player event logs for ML.
- **model_training.py**: Trains and evaluates machine learning models, generates metrics and plots.
- **llm_prediction.py**: (Optional) Module for LLM-based churn prediction or analysis.

This modular structure ensures each stage of the pipeline is clear, maintainable, and extensible for future work.