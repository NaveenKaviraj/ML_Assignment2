# Loan Default Prediction App

This project is a simple Streamlit web application that trains and evaluates several machine‑learning models to predict whether a loan will default.

## Files

- `app.py` – Main Streamlit app
- `loan_data.csv` – Example dataset used by default (you can also upload your own CSV in the app)

## Features

- Load the bundled dataset or upload your own CSV
- Choose which column should be treated as the target (defaults to `loan_status` if present)
- View a quick dataset preview
- Automatic preprocessing:
  - Fill missing numeric values with the median
  - Fill missing categorical and target values with the mode
  - Encode categorical columns with label encoding
  - Split into train, validation, and test sets
  - Standardize features with `StandardScaler`
- Train and evaluate multiple classifiers:
  - Logistic Regression
  - Decision Tree
  - K‑Nearest Neighbors (KNN)
  - Gaussian Naive Bayes
  - Random Forest
  - XGBoost
- For a selected model, display:
  - Accuracy, validation accuracy, AUC, precision, recall, F1, MCC
  - Confusion matrix heatmap
  - Learning curve (train vs validation accuracy)
- Comparison table with metrics for all models

## Installation

Use Python 3.8 or newer.

```bash
pip install streamlit pandas seaborn matplotlib scikit-learn xgboost
```

## How to Run

From the folder containing `app.py` and (optionally) `loan_data.csv`:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Basic Usage

1. Start the app with `streamlit run app.py`.
2. Optionally upload your own CSV, or use the default `loan_data.csv`.
3. Select the target column you want to predict.
4. Open the dataset overview to understand the data.
5. Choose a model to see its metrics, confusion matrix, and learning curve.
6. Use the comparison table to see how all models perform side by side.

## Notes

- The app assumes a classification problem (typically binary, e.g., default vs non‑default).
- XGBoost requires the `xgboost` package to be installed.
- You can modify `app.py` to add more models, tuning options, or visualizations as needed.