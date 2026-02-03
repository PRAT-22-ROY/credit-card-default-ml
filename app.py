import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Page Config
st.set_page_config(page_title="Classification Dashboard", layout="centered")

st.title("🔬 ML Assignment 2")
st.markdown("Implemented by **Pratik Roy**")

# Sidebar
st.sidebar.header("Choose Model")
# The keys here are nice names for the UI, the values are the EXACT filenames in your 'model' folder
model_options = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

model_name = st.sidebar.selectbox(
    "Select Model",
    list(model_options.keys())
)

# Load Models
try:
    # 1. Loading from 'model' folder (singular) instead of 'models'
    # 2. Loading the specific filename mapped above
    model_path = f'model/{model_options[model_name]}'
    model = joblib.load(model_path)
    
    # Load Scaler (from 'model' folder)
    scaler = joblib.load('model/scaler.pkl')
except Exception as e:
    st.error(f"⚠️ Error loading files! Details: {e}")
    st.stop()

# Main App
st.subheader("1. Upload Test Data")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Preprocessing
    if 'id' in df.columns: df = df.drop('id', axis=1)
    if 'Unnamed: 32' in df.columns: df = df.drop('Unnamed: 32', axis=1)
    
    target_col = 'diagnosis' # Update if needed
    
    if target_col in df.columns:
        X_test = df.drop(target_col, axis=1)
        y_true = df[target_col]
        if y_true.dtype == 'object': y_true = y_true.map({'M': 1, 'B': 0})
        
        # Predict
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        # --- Metrics Section ---
        st.divider()
        st.subheader(f"Performance: {model_name}")
        
        # Calculate real metrics
        acc = accuracy_score(y_true, y_pred)
        
        # Simple Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.2%}")
        # Note: You can add Precision/Recall calc here if you want them dynamic
        
        # --- Visuals Section ---
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(3, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            st.pyplot(fig)
            
        with col_viz2:
            st.write("**Detailed Report**")
            report = classification_report(y_true, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

    else:
        st.warning("Target column not found. Showing predictions only.")
        X_test_scaled = scaler.transform(df)
        predictions = model.predict(X_test_scaled)
        st.write(predictions)