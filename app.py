import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.header {
    background: linear-gradient(90deg, #4b6cb7, #182848);
    padding: 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    padding: 1.2rem;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.metric-desc {
    font-size: 13px;
    color: #666;
}

.footer {
    margin-top: 4rem;
    text-align: center;
    color: #888;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<div class="header">
    <h1>💳 Credit Card Default Prediction</h1>
    <p>Interactive Machine Learning Model Comparison Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ------------------ LOAD SCALER ------------------
scaler = joblib.load("model/scaler.pkl")

# ------------------ MODEL PATHS ------------------
MODEL_PATHS = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "k-Nearest Neighbors": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl",
    "XGBoost (Ensemble)": "model/xgboost.pkl"
}

METRIC_EXPLANATIONS = {
    "Accuracy": "Overall correctness of the model across all predictions.",
    "AUC": "Ability of the model to distinguish between defaulters and non-defaulters.",
    "Precision": "How reliable the model is when it predicts a default.",
    "Recall": "How well the model identifies actual defaulters (important for risk).",
    "F1 Score": "Balanced trade-off between precision and recall.",
    "MCC": "Robust metric for imbalanced datasets, considers all confusion matrix terms."
}

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Configuration Panel")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Test Dataset (CSV only)",
    type=["csv"]
)

selected_models = st.sidebar.multiselect(
    "🤖 Select Model(s)",
    list(MODEL_PATHS.keys()),
    default=["Logistic Regression"]
)

st.sidebar.markdown("""
ℹ️ **Instructions**
- Upload **test data only**
- Dataset must include  
  `default.payment.next.month`
- You may select **multiple models**
""")

# ------------------ MAIN LOGIC ------------------
if uploaded_file is not None and selected_models:
    df_test = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Test Data Preview")
    st.dataframe(df_test.head(), use_container_width=True)

    if "default.payment.next.month" not in df_test.columns:
        st.error("❌ Target column `default.payment.next.month` not found.")
    else:
        X_test = df_test.drop("default.payment.next.month", axis=1)
        y_test = df_test["default.payment.next.month"]

        X_test_scaled = scaler.transform(X_test)

        for model_name in selected_models:
            st.markdown("---")
            st.subheader(f"📌 Model: {model_name}")

            model = joblib.load(MODEL_PATHS[model_name])

            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_prob),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "MCC": matthews_corrcoef(y_test, y_pred)
            }

            cols = st.columns(6)
            for col, (name, value) in zip(cols, metrics.items()):
                col.markdown(f"""
                <div class="metric-card">
                    <h4>{name}</h4>
                    <h2>{value:.3f}</h2>
                    <div class="metric-desc">{METRIC_EXPLANATIONS[name]}</div>
                </div>
                """, unsafe_allow_html=True)

            # ------------------ CONFUSION MATRIX ------------------
            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(4,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # ------------------ CLASSIFICATION REPORT ------------------
            st.subheader("📄 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

else:
    st.info("⬅️ Upload a test CSV file and select at least one model.")

# ------------------ FOOTER ------------------
st.markdown("""
<div class="footer">
    Made with ❤️ by <b>Pratik Roy</b>
</div>
""", unsafe_allow_html=True)
