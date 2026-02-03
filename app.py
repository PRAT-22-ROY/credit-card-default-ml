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
    confusion_matrix
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
    padding: 2.2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
}

.metric-card {
    background: #ffffff;
    padding: 1.4rem;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
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

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Configuration Panel")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Test Dataset (CSV only)",
    type=["csv"]
)

model_name = st.sidebar.selectbox(
    "🤖 Select Machine Learning Model",
    list(MODEL_PATHS.keys())
)

st.sidebar.markdown("""
ℹ️ **Instructions**
- Upload **test data only**
- Dataset must include  
  `default.payment.next.month`
- Same feature columns as training
""")

# ------------------ MAIN LOGIC ------------------
if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Test Data Preview")
    st.dataframe(df_test.head(), use_container_width=True)

    if "default.payment.next.month" not in df_test.columns:
        st.error("❌ Target column `default.payment.next.month` not found.")
    else:
        X_test = df_test.drop("default.payment.next.month", axis=1)
        y_test = df_test["default.payment.next.month"]

        # Scale features
        X_test_scaled = scaler.transform(X_test)

        # Load model
        model = joblib.load(MODEL_PATHS[model_name])

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }

        # ------------------ METRICS DISPLAY ------------------
        st.subheader(f"📈 Performance Metrics — {model_name}")

        cols = st.columns(6)
        for col, (name, value) in zip(cols, metrics.items()):
            col.markdown(f"""
            <div class="metric-card">
                <h4>{name}</h4>
                <h2>{value:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)

        # ------------------ CONFUSION MATRIX ------------------
        st.subheader("🧩 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4,4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

else:
    st.info("⬅️ Upload a test CSV file from the sidebar to begin.")

# ------------------ FOOTER ------------------
st.markdown("""
<div class="footer">
    Made with ❤️ by <b>Pratik Roy</b>
</div>
""", unsafe_allow_html=True)
