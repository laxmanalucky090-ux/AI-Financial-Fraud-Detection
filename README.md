````markdown
# AI-Powered Financial Fraud Detection & Risk Analytics System

## 1. Project Overview

An end-to-end Machine Learning project for detecting fraudulent credit card transactions using multiple supervised and unsupervised ML algorithms.

The project includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Class imbalance handling using SMOTE
- Feature engineering
- Multiple machine learning models
- Model evaluation using fraud-focused metrics
- Fraud risk scoring
- Single transaction prediction
- Batch transaction prediction
- Interactive Streamlit dashboard

---

## 2. Dataset

Dataset: **Credit Card Fraud Detection**

Source: Kaggle  
Dataset Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset file:

```text
creditcard.csv
````

Dataset characteristics:

* 284,807 transactions
* 31 columns
* 492 fraud transactions
* 284,315 legitimate transactions
* Fraud rate: approximately 0.17%
* Features: `Time`, `V1`–`V28`, `Amount`, `Class`
* `Class = 0` → Legitimate transaction
* `Class = 1` → Fraudulent transaction

The `V1`–`V28` features are anonymized PCA-transformed variables provided by the original dataset.

---

## 3. Problem Statement

Credit card fraud detection is a highly imbalanced classification problem because fraudulent transactions represent only a very small percentage of all transactions.

The objective of this project is to build a machine learning system that can:

1. Identify potentially fraudulent transactions.
2. Reduce false positives while maintaining strong fraud detection.
3. Compare multiple machine learning algorithms.
4. Generate fraud probability and risk levels.
5. Provide an interactive dashboard for analysis and prediction.

---

## 4. Project Architecture

This project uses a **single-file architecture**.

```text
AI-Financial-Fraud-Detection/
│
├── SalapareddiLaxmana_FinancialFraudDetection.py
├── requirements.txt
├── README.md
├── SalapareddiLaxmana_ProjectReport.docx
├── creditcard.csv
├── fraud_models/
└── fraud_assets/
```

The main Python file contains:

* Data loading
* Data cleaning
* Feature engineering
* EDA
* Preprocessing
* SMOTE
* Model training
* Model evaluation
* Fraud prediction
* Batch prediction
* Streamlit dashboard

---

## 5. Machine Learning Models

The project compares five machine learning approaches:

### 1. Logistic Regression

Used as a baseline supervised classification model.

### 2. Random Forest

An ensemble tree-based classifier used for robust fraud classification.

### 3. XGBoost

A gradient boosting algorithm designed for high-performance classification.

### 4. LightGBM

A fast gradient boosting framework suitable for large datasets.

### 5. Isolation Forest

An unsupervised anomaly detection algorithm used to identify unusual transactions.

---

## 6. Data Preprocessing

The preprocessing pipeline includes:

* Duplicate removal
* Missing value checking
* Feature engineering
* Log transformation of transaction amount
* Hour extraction from transaction time
* Stratified train-test split
* Robust scaling
* SMOTE oversampling on the training data only

### Feature Engineering

Two additional features are created:

```text
Amount_log = log1p(Amount)
Hour = (Time // 3600) % 24
```

### Class Imbalance

The dataset contains approximately:

```text
99.83% Legitimate
0.17% Fraud
```

After duplicate removal, the dataset contains approximately:

```text
283,253 legitimate transactions
473 fraudulent transactions
```

This is approximately:

```text
599 legitimate transactions per 1 fraud.
```

---

## 7. Model Evaluation Metrics

Because the dataset is highly imbalanced, accuracy is not used as the primary evaluation metric.

The project evaluates models using:

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion Matrix

### Why these metrics?

**Precision** measures how many predicted fraud transactions are actually fraud.

**Recall** measures how many actual fraud transactions are successfully detected.

**F1 Score** balances precision and recall.

**ROC-AUC** measures the model's overall ranking ability.

**PR-AUC** is especially useful for highly imbalanced fraud detection datasets.

---

## 8. Model Performance

The evaluated model results are:

| Model               | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| ------------------- | --------: | -----: | -------: | ------: | -----: |
| Logistic Regression |    0.0532 | 0.8737 |   0.1004 |  0.9599 | 0.6775 |
| Random Forest       |    0.9268 | 0.8000 |   0.8588 |  0.9606 | 0.8288 |
| XGBoost             |    0.6525 | 0.8105 |   0.7230 |  0.9775 | 0.8128 |
| LightGBM            |    0.6033 | 0.7684 |   0.6759 |  0.9739 | 0.7860 |
| Isolation Forest    |    0.2353 | 0.2947 |   0.2617 |  0.9390 | 0.1616 |



Random Forest achieved an F1 score of approximately **0.8588** in the recorded evaluation.

---

## 9. Random Forest Confusion Matrix

The recorded Random Forest confusion matrix is:

```text
                 Predicted
                 Legit   Fraud

Actual Legit     56645      6
Actual Fraud        19     76
```

This corresponds to:

* True Negatives = 56,645
* False Positives = 6
* False Negatives = 19
* True Positives = 76

---

## 10. Exploratory Data Analysis

The project generates multiple visualizations including:

1. Class distribution
2. Transaction amount distribution
3. Transaction time distribution
4. Fraud transactions by hour
5. Transaction amount boxplot
6. Feature distribution comparison
7. Correlation heatmap

Generated visualization files are stored in:

```text
fraud_assets/
```

---

## 11. Streamlit Dashboard

The project includes an interactive Streamlit application with seven sections:

### 1. Dashboard

Displays:

* Dataset statistics
* Fraud count
* Legitimate transaction count
* Fraud rate
* Model summary

### 2. Dataset Analysis

Displays:

* Dataset information
* Class distribution
* Transaction statistics
* Fraud vs legitimate comparison

### 3. EDA / Visualizations

Displays generated charts and visual analysis.

### 4. Model Performance

Displays:

* Model comparison
* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion matrices
* ROC curve
* Precision-Recall curve

### 5. Fraud Prediction

Allows the user to enter transaction information and receive:

* Fraud probability / anomaly score
* Prediction
* Risk level

### 6. Batch Prediction

Allows users to upload a CSV file and generate predictions for multiple transactions.

### 7. About

Provides project information, methodology, models and limitations.

---

## 12. Risk Levels

The application converts model scores into four risk categories:

```text
Score < 0.30     → LOW
0.30 - 0.59      → MEDIUM
0.60 - 0.79      → HIGH
0.80 - 1.00      → CRITICAL
```

These thresholds are intended for demonstration and analytical purposes.

---

## 13. Project Structure

```text
AI-Financial-Fraud-Detection/
│
├── SalapareddiLaxmana_FinancialFraudDetection.py
│   └── Complete ML pipeline and Streamlit application
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── SalapareddiLaxmana_ProjectReport.docx
│   └── Complete project report
│
├── creditcard.csv
│   └── Kaggle dataset
│
├── fraud_models/
│   └── Saved machine learning models and preprocessing objects
│
└── fraud_assets/
    └── Generated charts and evaluation visualizations
```

---

## 14. Requirements

The project uses the following Python libraries:

```text
streamlit
pandas
numpy
scikit-learn
xgboost
lightgbm
imbalanced-learn
matplotlib
seaborn
joblib
plotly
Pillow
scipy
```

The complete dependency list is available in:

```text
requirements.txt
```

---

## 15. Installation

Clone the repository:

```bash
git clone https://github.com/laxmanalucky090-ux/AI-Financial-Fraud-Detection.git
```

Move into the project directory:

```bash
cd AI-Financial-Fraud-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 16. Run the Machine Learning Pipeline

To train the models and generate project artifacts:

```bash
python SalapareddiLaxmana_FinancialFraudDetection.py
```

The pipeline performs:

1. Dataset loading
2. Duplicate removal
3. Feature engineering
4. Exploratory analysis
5. Train-test split
6. Feature scaling
7. SMOTE balancing
8. Model training
9. Model evaluation
10. Evaluation chart generation
11. Model and artifact saving

---

## 17. Launch the Streamlit Application

Run:

```bash
streamlit run SalapareddiLaxmana_FinancialFraudDetection.py
```

The application opens in the browser and provides the interactive fraud detection dashboard.

---

## 18. Generated Artifacts

After running the project, the following directories are generated:

```text
fraud_models/
```

Contains:

* Trained machine learning models
* Feature scaler
* Feature metadata
* Evaluation results

```text
fraud_assets/
```

Contains:

* EDA charts
* ROC curves
* Precision-Recall curves
* Confusion matrices
* Model comparison charts

---

## 19. Important Considerations

### Class Imbalance

Fraud cases represent only a small percentage of transactions. Therefore, model performance should not be judged using accuracy alone.

### SMOTE

SMOTE is applied only to the training data to avoid contaminating the test set.

### Scaling

RobustScaler is used because transaction amounts and anonymized features can contain outliers.

### PCA Features

The `V1`–`V28` variables are anonymized PCA components. They do not directly represent business attributes such as customer age, gender, merchant name or location.

### Isolation Forest

Isolation Forest produces an anomaly score rather than a calibrated fraud probability.

---

## 20. Limitations

The project has several limitations:

* The dataset contains anonymized PCA features.
* Real-time banking transaction integration is not included.
* The model is trained on a historical dataset.
* Risk thresholds are demonstration thresholds.
* Production deployment would require continuous monitoring and retraining.
* Additional business and customer features could improve practical fraud detection.

---

## 21. Future Enhancements

Possible future improvements include:

* Real-time transaction streaming
* Cloud deployment
* API integration
* Model monitoring
* Automated retraining
* Explainable AI using SHAP
* Advanced hyperparameter tuning
* Cost-sensitive learning
* Real-time alert generation
* Database integration
* Authentication and role-based access

---

## 22. Resume / Portfolio Value

This project demonstrates practical skills in:

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Imbalanced-learn
* Machine Learning
* Fraud Detection
* Feature Engineering
* Exploratory Data Analysis
* Model Evaluation
* Streamlit
* Data Visualization
* End-to-End ML Project Development

### Resume Project Description

**AI-Powered Financial Fraud Detection & Risk Analytics System**

Built an end-to-end machine learning fraud detection system using Logistic Regression, Random Forest, XGBoost, LightGBM and Isolation Forest. Implemented SMOTE for class imbalance, feature engineering, fraud-focused evaluation using Precision, Recall, F1, ROC-AUC and PR-AUC, and developed an interactive Streamlit dashboard for single and batch transaction risk prediction.

---

## 23. Author

**Salapareddi Laxmana**

IBM SkillsBuild Academic Internship 2026
Data Analytics with AI | BharatCares

---

## 24. Repository

```text
GitHub Repository:

https://github.com/laxmanalucky090-ux/AI-Financial-Fraud-Detection
```
