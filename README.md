# CS5228-Kaggle-Project

## Requirements

    pip install -r requirements.txt

## How to Run

### 1. Main Model Training and Prediction
To train the model and generate the CSV submission file:

1. Ensure that all dependencies are installed (check `requirements.txt`) in your environment.
2. Run the `main.py` script using the best hyperparameters to train the final model and output the results in a CSV format for Kaggle submission:
   
```bash
python main.py
```

### 2. Hyperparameter Tuning for XGBoost + Model Evaluation

To run Bayesian hyperparameter search for the XGBoost model, execute the following script. The script also generates charts for feature importance and error analysis.

    python xgboost_params_search_and_evaluation.py

### 3. Testing Alternative Models (Benchmarks)
To evaluate other models (Random Forest and Linear Regression) for comparison, use the notebook random_forest_linear_regression_and_evaluation.ipynb. This notebook includes steps to train and evaluate these models as benchmarks for model selection.

    jupyter notebook random_forest_linear_regression_and_evaluation.ipynb

### 4. EDA
To explore and visualize the dataset, use the EDA.ipynb notebook. This will help understand the data distribution, relationships between features, and provide insights into potential feature engineering that we have included in the report.

    jupyter notebook EDA.ipynb
