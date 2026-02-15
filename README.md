# 💳 Credit Card Default Prediction – Machine Learning Assignment 2

---

## 🔹 Problem Statement

Credit card default prediction is a critical financial risk assessment problem. The objective of this project is to build and compare multiple machine learning classification models to predict whether a credit card customer will default on payment in the next month.

This is a **binary classification problem**, where:

* `0` → No Default
* `1` → Default

The project evaluates six machine learning models and deploys them using an interactive Streamlit web application.

---

## 🔹 Dataset Description

The dataset used is the **Default of Credit Card Clients Dataset** from Kaggle.

* **Total Records:** 30,000
* **Input Features:** 23
* **Target Variable:** `default.payment.next.month`
* **Problem Type:** Binary Classification
* **Domain:** Financial Risk Prediction

The dataset includes demographic details, credit limits, repayment history, bill amounts, and payment information over the previous six months.

---

## 🔹 Machine Learning Models Implemented

The following models were implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. k-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

Each model was evaluated using:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

---

## 🔹 Model Performance Comparison

| ML Model Name            | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC      |
| ------------------------ | -------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression      | 0.808000 | 0.707777 | 0.688172  | 0.241145 | 0.357143 | 0.326143 |
| Decision Tree            | 0.726833 | 0.614733 | 0.389362  | 0.413715 | 0.401169 | 0.224613 |
| k-Nearest Neighbors      | 0.793500 | 0.694213 | 0.553012  | 0.345893 | 0.425591 | 0.320367 |
| Naive Bayes              | 0.752333 | 0.725099 | 0.451317  | 0.555388 | 0.497973 | 0.339102 |
| Random Forest (Ensemble) | 0.814667 | 0.756570 | 0.642762  | 0.364732 | 0.465385 | 0.384818 |
| XGBoost (Ensemble)       | 0.811167 | 0.769796 | 0.628307  | 0.357950 | 0.456073 | 0.372470 |

---

## 🔹 Model Performance Observations

| ML Model Name            | Observation about model performance                                                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Logistic Regression achieved high accuracy due to class imbalance but showed low recall for defaulters, indicating bias toward the majority class and limitations in identifying risky customers. |
| Decision Tree            | Decision Tree improved recall compared to Logistic Regression but showed lower AUC and MCC, suggesting overfitting and weaker generalization performance.                                         |
| k-Nearest Neighbors      | kNN demonstrated balanced performance with moderate improvements in recall and F1-score but was sensitive to local data structure and high-dimensional space.                                     |
| Naive Bayes              | Naive Bayes achieved the highest recall among all models, showing strong sensitivity in detecting defaulters, though precision was lower due to the independence assumption.                      |
| Random Forest (Ensemble) | Random Forest achieved the best overall balance in accuracy, F1-score, and MCC, demonstrating improved generalization and better handling of class imbalance.                                     |
| XGBoost (Ensemble)       | XGBoost achieved the highest AUC score, indicating superior class separation and strong boosting capability for financial risk prediction.                                                        |

---

## 🔹 Streamlit Web Application

An interactive Streamlit web application was developed and deployed with the following features:

* CSV dataset upload (test data only)
* Multiple model selection (single, multiple, or all models)
* Interactive display of evaluation metrics with explanations
* Confusion matrix visualization
* Detailed classification report
* Responsive and modern UI design

The app allows dynamic comparison of multiple models on uploaded test data.

---

## 🔹 Project Structure

```
credit-card-default-ml/
│
│-- app.py
│-- requirements.txt
│-- README.md
│
│-- model/
│   ├── logistic.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
```

---

## 🔹 Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment Steps:

1. Project pushed to GitHub repository
2. Repository connected to Streamlit Cloud
3. `app.py` selected as entry file
4. Application deployed successfully

---

## 🔹 Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Matplotlib & Seaborn
* Streamlit

---

## 🔹 Conclusion

This project demonstrates a complete end-to-end machine learning workflow:

* Data preprocessing
* Model implementation
* Performance evaluation
* Model comparison
* Interactive web deployment

Ensemble models (Random Forest and XGBoost) demonstrated superior performance compared to baseline models, particularly in terms of AUC and MCC, making them more suitable for imbalanced financial risk prediction problems.

---

### ❤️ Made with love by Pratik Roy




