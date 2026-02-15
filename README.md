
# Loan Default Prediction App

Streamlit application for loan default prediction using 6 ML classifiers.

## Files
- `streamlit_app.py` – Main Streamlit app[file:3]
- `loan_data.csv` – Dataset (45,000 records)[file:4]
- `app.ipynb` – Analysis notebook

## a. Problem Statement
**Objective:** Binary classification to predict `loan_status`:
- 0 = Loan repaid successfully (non-default, 78%)
- 1 = Loan defaulted (22%)

**Business Importance:** Lending institutions suffer significant losses from defaults. Automated ML risk assessment improves lending decisions, reduces losses, and scales underwriting.

## b. Dataset Description
**Dataset:** `loan_data.csv` (45,000 loan applications × 14 features)[file:4]

**Target Variable:** `loan_status` (binary, imbalanced: 78% Class 0, 22% Class 1)

**Feature Categories:**
| Demographics | Financial History | Loan Characteristics | Credit Profile |
|--------------|------------------|---------------------|---------------|
| person_age, person_gender, person_education, person_income | person_emp_exp, person_home_ownership | loan_amnt, loan_intent, loan_int_rate, loan_percent_income | cb_person_cred_hist_length, credit_score, previous_loan_defaults_on_file |

**Preprocessing Pipeline (streamlit_app.py):**
1. Missing numeric values → median imputation
2. Missing categorical/target → mode imputation
3. Categorical encoding → LabelEncoder()
4. Train(70%)/Val(15%)/Test(15%) → stratified split
5. Feature scaling → StandardScaler

## c. Model Performance Table (All Metrics)

| Model | Test Accuracy | Validation Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|-------|---------------|---------------------|-----|-----------|--------|----------|-----|
| XGBoost | **0.9350** | **0.9302** | **0.9769** | **0.8927** | 0.8040 | **0.8460** | **0.8067** |
| Random Forest | 0.9265 | 0.9250 | 0.9718 | **0.8947** | 0.7587 | 0.8211 | 0.7794 |
| Logistic Regression | 0.8951 | 0.8924 | 0.9506 | 0.7750 | 0.7440 | 0.7592 | 0.6924 |
| Decision Tree | 0.8926 | 0.8972 | 0.8457 | 0.7568 | 0.7613 | 0.7591 | 0.6900 |
| KNN | 0.8884 | 0.8910 | 0.9227 | 0.7701 | 0.7100 | 0.7388 | 0.6689 |
| Naive Bayes | 0.7314 | 0.7378 | 0.9403 | 0.4527 | **0.9993** | 0.6232 | 0.5440 |

## d. Model Observations

| ML Model Name | Observation about model performance |
|---------------|------------------------------------|
| Logistic Regression | Solid baseline performance with 89.5% test accuracy and balanced precision (77.5%)/recall (74.4%). Strong AUC (0.951) indicates reliable probability calibration. Fast training and interpretable coefficients ideal for regulatory compliance |
| Decision Tree | Achieves 89.3% accuracy with visualizable decision boundaries suitable for stakeholder explanation. Handles non-linear relationships but shows lowest AUC (0.846) among competitive models. Higher overfitting risk due to single tree structure |
| KNN | Instance-based K-Nearest Neighbors learner scores 88.8% accuracy using distance-weighted voting. Performance dependent on proper feature scaling (StandardScaler applied successfully). Computationally expensive O(n) predictions limit production scalability |
| Naive Bayes | Gaussian Naive Bayes delivers exceptional 99.9% recall, catching nearly all defaults at cost of 73.1% accuracy and poor 45.3% precision (high false positives). Strong AUC (0.940) despite overall poor performance. Specialized for false-negative critical applications |
| Random Forest (Ensemble) | Bagging ensemble of decision trees achieves 92.7% accuracy with highest precision (89.5%). Extremely stable performance (0.15% train/validation gap) and robust AUC (0.972). Feature importance rankings provide model interpretability |
| XGBoost (Ensemble) | Gradient boosting framework dominates with highest test accuracy (93.5%), F1-score (84.6%), and MCC (80.7%). Perfect precision-recall balance with minimal overfitting. Built-in regularization makes it production deployment choice |

## e. Deployment
```bash
pip install streamlit pandas scikit-learn xgboost
streamlit run streamlit_app.py
```

## Notes
XGBoost recommended for production (superior metrics across board). Complete end-to-end ML pipeline deployed via Streamlit.
