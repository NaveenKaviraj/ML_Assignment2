
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Loan Status Classifier", layout="wide")

# Title and description
st.title("🏦 Loan Default Classification Model")
st.markdown("---")
st.markdown("""
This application demonstrates six machine learning classification models trained on loan data 
to predict whether a loan applicant will default or not.
""")

# Load and prepare data
@st.cache_resource
def load_and_prepare_data():
    """Load and preprocess the loan dataset"""
    try:
        df = pd.read_csv('loan_data.csv')
    except:
        st.error("Dataset not found!")
        return None, None, None, None, None, None

    df_processed = df.copy()

    # Handle categorical columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    le_dict = {}

    for col in categorical_cols:
        if col != 'loan_status':
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le

    # Handle previous_loan_defaults_on_file
    if df_processed['previous_loan_defaults_on_file'].dtype == 'object':
        le_defaults = LabelEncoder()
        df_processed['previous_loan_defaults_on_file'] = le_defaults.fit_transform(df_processed['previous_loan_defaults_on_file'])

    # Separate features and target
    X = df_processed.drop('loan_status', axis=1)
    y = df_processed['loan_status']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                          random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X

# Train models
@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    """Train all 6 classification models"""
    models = {}
    results = []
    predictions = {}

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]
    models['Logistic Regression'] = lr
    predictions['Logistic Regression'] = (y_pred, y_proba)

    results.append({
        'Model': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    })

    # 2. Decision Tree
    dt = DecisionTreeClassifier(max_depth=15, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_proba = dt.predict_proba(X_test)[:, 1]
    models['Decision Tree'] = dt
    predictions['Decision Tree'] = (y_pred, y_proba)

    results.append({
        'Model': 'Decision Tree',
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    })

    # 3. K-Nearest Neighbor
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)[:, 1]
    models['K-Nearest Neighbor'] = knn
    predictions['K-Nearest Neighbor'] = (y_pred, y_proba)

    results.append({
        'Model': 'K-Nearest Neighbor',
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    })

    # 4. Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    y_proba = nb.predict_proba(X_test)[:, 1]
    models['Naive Bayes'] = nb
    predictions['Naive Bayes'] = (y_pred, y_proba)

    results.append({
        'Model': 'Naive Bayes',
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    })

    # 5. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    models['Random Forest'] = rf
    predictions['Random Forest'] = (y_pred, y_proba)

    results.append({
        'Model': 'Random Forest',
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    })

    # 6. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    y_proba = gb.predict_proba(X_test)[:, 1]
    models['Gradient Boosting'] = gb
    predictions['Gradient Boosting'] = (y_pred, y_proba)

    results.append({
        'Model': 'Gradient Boosting',
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    })

    results_df = pd.DataFrame(results)
    return models, predictions, results_df, y_test

# Load data
data_result = load_and_prepare_data()
if data_result[0] is not None:
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X = data_result

    # Train models
    models, predictions, results_df, y_test_final = train_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Sidebar for navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Select Page", ["Overview", "Model Comparison", "Model Details", "About Dataset"])

    if page == "Overview":
        st.header("📋 Project Overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", "45,000")
        with col2:
            st.metric("Test Samples", f"{len(y_test_final):,}")
        with col3:
            st.metric("Number of Features", "13")

        st.subheader("🎯 Project Objective")
        st.write("""
        This machine learning project implements 6 different classification algorithms to predict 
        loan default status. The models are trained on historical loan data and evaluated using 
        multiple performance metrics.
        """)

        st.subheader("📈 Models Implemented")
        model_list = """
        1. **Logistic Regression** - Linear model for binary classification
        2. **Decision Tree** - Tree-based model capturing non-linear relationships
        3. **K-Nearest Neighbor** - Instance-based learning approach
        4. **Naive Bayes** - Probabilistic model based on Bayes' theorem
        5. **Random Forest** - Ensemble of decision trees
        6. **Gradient Boosting** - Sequential boosting ensemble method
        """
        st.markdown(model_list)

    elif page == "Model Comparison":
        st.header("📊 Model Comparison")

        # Display results table
        st.subheader("Performance Metrics Comparison")
        st.dataframe(results_df.style.format({
            'Accuracy': '{:.4f}',
            'AUC': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1 Score': '{:.4f}',
            'MCC': '{:.4f}'
        }), use_container_width=True)

        # Best performers
        col1, col2, col3 = st.columns(3)
        with col1:
            best_acc = results_df.loc[results_df['Accuracy'].idxmax()]
            st.metric("Best Accuracy", f"{best_acc['Accuracy']:.4f}", 
                     f"({best_acc['Model']})")
        with col2:
            best_f1 = results_df.loc[results_df['F1 Score'].idxmax()]
            st.metric("Best F1 Score", f"{best_f1['F1 Score']:.4f}", 
                     f"({best_f1['Model']})")
        with col3:
            best_auc = results_df.loc[results_df['AUC'].idxmax()]
            st.metric("Best AUC", f"{best_auc['AUC']:.4f}", 
                     f"({best_auc['Model']})")

        # Visualization
        st.subheader("📉 Performance Visualization")
        metrics_to_plot = st.multiselect(
            "Select metrics to visualize:",
            ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC'],
            default=['Accuracy', 'F1 Score']
        )

        if metrics_to_plot:
            fig, ax = plt.subplots(figsize=(12, 5))
            results_df.set_index('Model')[metrics_to_plot].plot(kind='bar', ax=ax)
            ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12)
            ax.set_xlabel('Model', fontsize=12)
            ax.legend(loc='lower right')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

    elif page == "Model Details":
        st.header("🔍 Model Details & Confusion Matrices")

        selected_model = st.selectbox(
            "Select a model to view details:",
            list(models.keys())
        )

        if selected_model:
            # Model metrics
            model_metrics = results_df[results_df['Model'] == selected_model].iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
            with col2:
                st.metric("AUC", f"{model_metrics['AUC']:.4f}")
            with col3:
                st.metric("F1 Score", f"{model_metrics['F1 Score']:.4f}")
            with col4:
                st.metric("MCC", f"{model_metrics['MCC']:.4f}")

            # Confusion Matrix
            y_pred, _ = predictions[selected_model]
            cm = confusion_matrix(y_test_final, y_pred)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Default', 'Default'],
                       yticklabels=['No Default', 'Default'])
            ax.set_title(f'Confusion Matrix - {selected_model}', fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)

            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test_final, y_pred, 
                                          target_names=['No Default', 'Default'])
            st.text(report)

    elif page == "About Dataset":
        st.header("📚 Dataset Information")

        st.subheader("Dataset Overview")
        st.write("""
        **Loan Status Classification Dataset**
        - **Total Samples:** 45,000
        - **Features:** 13 (after encoding)
        - **Target Variable:** loan_status (0 = No Default, 1 = Default)
        - **Class Distribution:** Imbalanced (77.8% No Default, 22.2% Default)
        """)

        st.subheader("Features Description")
        features_desc = pd.DataFrame({
            'Feature': ['person_age', 'person_gender', 'person_education', 'person_income', 
                       'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
                       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                       'credit_score', 'previous_loan_defaults_on_file'],
            'Description': [
                'Age of the loan applicant',
                'Gender of the applicant',
                'Education level',
                'Annual income',
                'Employment experience (years)',
                'Home ownership status',
                'Loan amount requested',
                'Purpose of the loan',
                'Interest rate on the loan',
                'Loan as percentage of income',
                'Credit history length (years)',
                'Credit score',
                'Previous loan defaults'
            ]
        })
        st.dataframe(features_desc, use_container_width=True)

        st.subheader("Data Preprocessing Steps")
        preprocessing_steps = """
        1. **Encoding:** Categorical variables (gender, education, home ownership, loan intent, 
           previous defaults) were encoded using LabelEncoder
        2. **Scaling:** All features were standardized using StandardScaler (mean=0, std=1)
        3. **Train-Test Split:** 80% training data, 20% test data with stratification
        4. **Missing Values:** No missing values were found in the dataset
        """
        st.markdown(preprocessing_steps)

else:
    st.error("Unable to load the dataset. Please ensure 'loan_data.csv' is in the correct location.")
