# Predict Customer Churn

- **Project Name:** Predict Customer Churn  
  This project is part of the ML DevOps Engineer Nanodegree by Udacity. It involves building a machine learning pipeline to predict customer churn.

## Project Description
The goal is to develop a machine learning model to predict customer churn using historical banking data. The pipeline includes data ingestion, feature engineering, model training, evaluation, and deployment, with a focus on logging and testing.

## Files and Data Description
- `churn_library.py`: Core functions for data loading, EDA, encoding, feature engineering, and model training.
- `churn_script_logging_and_tests.py`: Test functions for the churn library with logging.
- `data/`: Contains dataset (`bank_data.csv`).
- `images/`: Stores EDA plots and result visualizations.
- `logs/`: Logs generated from running scripts.
- `models/`: Saved models (`rfc_model.pkl`, `logistic_model.pkl`).

## Running Files
1. Install dependencies:
   ```bash
   pip install -r requirements_py3.10.txt
   ```
2. Run the churn library:
   ```bash
   python churn_library.py
   ```
3. Run tests and log results:
   ```bash
   python churn_script_logging_and_tests.py
   ```



