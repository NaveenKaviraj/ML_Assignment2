import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


st.set_page_config(page_title="ML Assignment 2", layout="centered")
st.title("Loan Default Prediction – ML Assignment 2")


uploaded = st.file_uploader("Upload loan_data.csv", type=["csv"])
df = pd.read_csv(uploaded if uploaded else "loan_data.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)


st.subheader("Target Column Selection")

default_index = 0
if "loan_status" in df.columns:
    default_index = list(df.columns).index("loan_status")

target = st.selectbox(
    "Select the target column",
    options=df.columns.tolist(),
    index=default_index,
    help="Choose the column you want to predict"
)

st.success(f"✅ Selected target column: **{target}**")


@st.dialog("Dataset Overview", width="large")
def show_dataset_overview():
    st.markdown("### Basic Information")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])
    with c2:
        st.write("Target Column:", target)
    
    st.markdown("### Column Names")
    st.write(list(df.columns))
    
    st.markdown("### Data Types")
    st.dataframe(
        df.dtypes.reset_index().rename(
            columns={"index": "Column", 0: "Data Type"}
        ),
        use_container_width=True
    )
    
    st.markdown("### Missing Values")
    st.dataframe(
        df.isnull().sum().reset_index().rename(
            columns={"index": "Column", 0: "Missing Count"}
        ),
        use_container_width=True
    )
    
    st.markdown("### Statistical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)
    
    st.markdown("### Target Distribution")
    st.bar_chart(df[target].value_counts())
    
    st.markdown("### Sample Records")
    st.dataframe(df.head(10), use_container_width=True)
    
    if st.button("❌ Close"):
        st.rerun()

if st.button("📊 Show Dataset Overview"):
    show_dataset_overview()


X = df.drop(columns=[target])
y = df[target]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include="object").columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])
y = y.fillna(y.mode()[0])

for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

y = LabelEncoder().fit_transform(y)

assert X.isnull().sum().sum() == 0, "NaNs still present in X!"


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30, 
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}


@st.cache_data(show_spinner=True)
def train_all_models(X_train, X_val, X_test, y_train, y_val, y_test):
    trained = {}
    metrics_all = {}
    learning_curves = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        trained[name] = {
            "y_pred": y_test_pred
        }
        
        metrics_all[name] = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_test_pred),
            "Validation Accuracy": val_acc,
            "AUC": roc_auc_score(y_test, y_test_prob),
            "Precision": precision_score(y_test, y_test_pred),
            "Recall": recall_score(y_test, y_test_pred),
            "F1-score": f1_score(y_test, y_test_pred),
            "MCC": matthews_corrcoef(y_test, y_test_pred)
        }
        
        sizes, train_scores, val_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=3,
            scoring="accuracy",
            train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0]
        )
        
        learning_curves[name] = {
            "sizes": sizes,
            "train": train_scores.mean(axis=1),
            "val": val_scores.mean(axis=1)
        }
    
    return trained, metrics_all, learning_curves

trained, metrics_all, learning_curves = train_all_models(
    X_train, X_val, X_test, y_train, y_val, y_test
)


st.subheader("Model Selection")
model_option = st.selectbox(
    "Choose a model",
    ["-- Select Model --"] + list(models.keys()),
    index=0
)


if model_option != "-- Select Model --":
    y_pred_selected = trained[model_option]["y_pred"]
    metrics = metrics_all[model_option]
    lc = learning_curves[model_option]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{metrics['Accuracy']:.4f}")
    c2.metric("Validation Accuracy", f"{metrics['Validation Accuracy']:.4f}")
    c3.metric("AUC", f"{metrics['AUC']:.4f}")
    
    c1.metric("Precision", f"{metrics['Precision']:.4f}")
    c2.metric("Recall", f"{metrics['Recall']:.4f}")
    c3.metric("F1-score", f"{metrics['F1-score']:.4f}")
    
    col1, col2 = st.columns(2)
    

    with col1:
        fig, ax = plt.subplots(figsize=(2, 2), dpi=130)
        cm = confusion_matrix(y_test, y_pred_selected, labels=[1, 0])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            square=True,
            cbar=False,
            ax=ax,
            annot_kws={"size": 7},
            xticklabels=[1, 0],
            yticklabels=[1, 0]
        )
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Predicted Class", fontsize=7)
        ax.set_ylabel("True Class", fontsize=7)
        ax.tick_params(labelsize=6)
        plt.tight_layout(pad=0.2)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(2, 2), dpi=130)
        ax.plot(lc["sizes"], lc["train"], marker="o", label="Train")
        ax.plot(lc["sizes"], lc["val"], marker="o", label="Validation")
        ax.set_title("Learning Curve", fontsize=8)
        ax.set_xlabel("Samples", fontsize=7)
        ax.set_ylabel("Accuracy", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        plt.tight_layout(pad=0.2)
        st.pyplot(fig)

st.subheader("Model Comparison Table (All Models)")
metrics_df = pd.DataFrame(metrics_all.values()).round(4)
st.dataframe(metrics_df, use_container_width=True)
