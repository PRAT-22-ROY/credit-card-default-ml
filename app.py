import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef

# Page Config
st.set_page_config(page_title="ML Classification Dashboard", layout="wide")

st.title("🔬 ML Assignment 2: Interactive Classification Dashboard")
st.markdown("Implemented by **Pratik Roy**")

# --- 1. Sidebar: Multi-Model Selection ---
st.sidebar.header("Configuration")

available_models = [
    "Logistic Regression", "Decision Tree", "KNN", 
    "Naive Bayes", "Random Forest", "XGBoost"
]

# "Select All" Helper
all_models = st.sidebar.checkbox("Select All Models", value=False)
if all_models:
    selected_models = st.sidebar.multiselect("Select Models to Run", available_models, default=available_models)
else:
    selected_models = st.sidebar.multiselect("Select Models to Run", available_models, default=["Logistic Regression"])

if not selected_models:
    st.warning("Please select at least one model from the sidebar to proceed.")
    st.stop()

# Load Scaler (Required for some models)
try:
    scaler = joblib.load('models/scaler.pkl')
except:
    st.error("Scaler not found. Please run 'train_models.py' first.")
    st.stop()

# --- 2. Main Section: Dataset Upload ---
st.header("1. Upload Test Data")
uploaded_file = st.file_uploader("Upload your CSV file (Test Data)", type=["csv"])

# --- Helper Function for Metric Explanations ---
def get_metric_explanation(metric_name):
    explanations = {
        "Accuracy": "Overall correctness (True Positives + True Negatives / Total). Good for balanced datasets.",
        "Precision": "Quality of positive predictions. Out of everyone predicted positive, how many actually were?",
        "Recall": "Coverage of actual positives. Out of all actual positive cases, how many did we catch?",
        "F1 Score": "Harmonic mean of Precision and Recall. Best single metric for uneven class distribution.",
        "AUC Score": "Area Under Curve. Shows how well the model distinguishes between classes (0.5 = Random, 1.0 = Perfect).",
        "MCC": "Matthews Correlation Coefficient. The most robust metric for imbalanced datasets (-1 to +1)."
    }
    return explanations.get(metric_name, "")

# --- Main Logic ---
if uploaded_file is not None:
    # Read Data
    df = pd.read_csv(uploaded_file)
    
    # Basic Preprocessing (Match your training steps)
    if 'id' in df.columns: df = df.drop('id', axis=1)
    if 'Unnamed: 32' in df.columns: df = df.drop('Unnamed: 32', axis=1)
    
    # Target Column Check
    target_col = 'diagnosis'  # UPDATE THIS if your target column name is different
    
    if target_col in df.columns:
        X_test = df.drop(target_col, axis=1)
        y_true = df[target_col]
        # Map text target if necessary
        if y_true.dtype == 'object':
             y_true = y_true.map({'M': 1, 'B': 0}) # Update mapping based on your data
        
        # Scale Features
        X_test_scaled = scaler.transform(X_test)
        
        # --- Store Results for Comparison ---
        model_results = []
        
        # Create Tabs: First tab for Comparison, others for Individual Models
        tabs = st.tabs(["📊 Comparison Board"] + selected_models)
        
        # --- TAB 1: Comparison Board ---
        with tabs[0]:
            st.subheader("Model Performance Comparison")
            
            comparison_data = []
            
            for model_name in selected_models:
                try:
                    # Load Model
                    model_path = f'models/{model_name.replace(" ", "_")}.pkl'
                    model = joblib.load(model_path)
                    
                    # Predict
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
                    
                    # Calculate Metrics
                    metrics = {
                        "Model": model_name,
                        "Accuracy": accuracy_score(y_true, y_pred),
                        "Precision": precision_score(y_true, y_pred),
                        "Recall": recall_score(y_true, y_pred),
                        "F1 Score": f1_score(y_true, y_pred),
                        "AUC Score": roc_auc_score(y_true, y_prob),
                        "MCC": matthews_corrcoef(y_true, y_pred)
                    }
                    comparison_data.append(metrics)
                    
                    # Save results for individual tabs
                    model_results.append({
                        "name": model_name,
                        "y_pred": y_pred,
                        "y_true": y_true,
                        "metrics": metrics
                    })
                    
                except Exception as e:
                    st.error(f"Error loading {model_name}: {e}")

            # Display Comparison Table
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data).set_index("Model")
                st.dataframe(comp_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
                
                st.info("💡 Tip: The cells highlighted in green indicate the best performing model for that specific metric.")

        # --- TABS 2...N: Individual Model Details ---
        for i, result in enumerate(model_results):
            with tabs[i+1]:
                st.header(f"Analysis: {result['name']}")
                
                # 1. Metrics with Explanations
                st.subheader("1. Key Metrics Explained")
                col1, col2, col3 = st.columns(3)
                col4, col5, col6 = st.columns(3)
                
                cols = [col1, col2, col3, col4, col5, col6]
                metric_keys = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC Score", "MCC"]
                
                for col, key in zip(cols, metric_keys):
                    val = result['metrics'][key]
                    col.metric(label=key, value=f"{val:.3f}")
                    col.caption(get_metric_explanation(key))

                st.divider()

                # 2. Confusion Matrix & Report
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    st.subheader("2. Confusion Matrix")
                    cm = confusion_matrix(result['y_true'], result['y_pred'])
                    
                    # Plot using Seaborn
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    st.pyplot(fig)
                    st.caption("Top-Left: True Negatives | Bottom-Right: True Positives")

                with col_viz2:
                    st.subheader("3. Classification Report")
                    report_dict = classification_report(result['y_true'], result['y_pred'], output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['f1-score', 'recall', 'precision']))

    else:
        st.warning(f"Target column '{target_col}' not found in CSV. Please ensure your CSV has the correct target column.")
else:
    st.info("Awaiting CSV upload. Please upload the test dataset.")