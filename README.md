# AI-Powered Financial Fraud Detection and Risk Analytics System Using Machine Learning

## 1. Project Description

AI-Powered Financial Fraud Detection and Risk Analytics System is an end-to-end machine learning project developed to identify potentially fraudulent credit card transactions.

The project uses the Kaggle Credit Card Fraud Detection dataset and performs data cleaning, exploratory data analysis, feature engineering, class-imbalance handling, machine learning model training, model evaluation, and interactive fraud prediction.

The project is implemented using Python and Streamlit. The complete machine learning pipeline and frontend application are contained in a single Python file:

The system supports:

* Dataset preprocessing
* Exploratory Data Analysis
* Feature engineering
* Imbalanced-data handling using SMOTE
* Multiple machine learning models
* Model performance comparison
* Single transaction fraud prediction
* Batch transaction prediction
* Risk-level classification
* Interactive Streamlit dashboard

---

## 2. What Did We Do?

The project follows a complete machine learning workflow:

1. Loaded the Kaggle Credit Card Fraud Detection dataset.
2. Checked the dataset structure and data types.
3. Checked missing values and duplicate records.
4. Removed 1,081 duplicate transactions.
5. Performed Exploratory Data Analysis (EDA).
6. Created additional features from transaction time and amount.
7. Split the dataset into training and testing sets using stratification.
8. Applied RobustScaler using only the training data.
9. Applied SMOTE only to the training data to handle class imbalance.
10. Trained five machine learning models.
11. Evaluated the models using multiple classification metrics.
12. Generated ROC curves, Precision-Recall curves, confusion matrices, and comparison charts.
13. Saved trained models and preprocessing artifacts.
14. Developed an interactive Streamlit frontend.
15. Implemented single transaction fraud prediction.
16. Implemented batch CSV fraud prediction.
17. Added risk-level classification for prediction results.

---

## 3. Frontend

The frontend of the project is developed using **Streamlit**.

The Streamlit application provides an interactive user interface for exploring the dataset, analyzing model performance, and making fraud predictions.

The frontend is implemented inside:

```text
FinancialFraudDetection.py
```

### Frontend Features

The application contains the following pages:

* Dashboard
* Dataset Analysis
* EDA / Visualizations
* Model Performance
* Fraud Prediction
* Batch Prediction
* About

The frontend allows users to interact with the trained machine learning models without writing Python code.

---

## 4. Backend / Machine Learning Engine

The current version does not use a separate Flask backend.

The backend machine learning functionality is implemented directly inside:

```text
FinancialFraudDetection.py
```

The Python ML engine handles:

* Dataset loading
* Data cleaning
* Duplicate removal
* Feature engineering
* Train-test splitting
* Feature scaling
* SMOTE oversampling
* Model training
* Model evaluation
* Model saving
* Single transaction prediction
* Batch transaction prediction

Therefore, this project currently follows a **single-file Python + Streamlit architecture**.

```text
User
  |
  v
Streamlit Frontend
  |
  v
FinancialFraudDetection.py
  |
  +--> Data Processing
  |
  +--> Feature Engineering
  |
  +--> Machine Learning Models
  |
  +--> Model Evaluation
  |
  +--> Fraud Prediction
```

---

## 5. Dataset

### Kaggle Credit Card Fraud Detection Dataset

Dataset Source:

[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The original dataset contains:

* 284,807 transactions
* 31 columns
* 492 fraudulent transactions
* 284,315 legitimate transactions

After removing 1,081 duplicate records, the project uses:

* 283,726 unique transactions
* 473 fraudulent transactions
* 283,253 legitimate transactions

The dataset is highly imbalanced because fraudulent transactions represent a very small percentage of all transactions.

---

## 6. Dataset Features

The dataset contains the following columns:

```text
Time
V1
V2
V3
V4
V5
V6
V7
V8
V9
V10
V11
V12
V13
V14
V15
V16
V17
V18
V19
V20
V21
V22
V23
V24
V25
V26
V27
V28
Amount
Class
```

### Feature Description

* `Time` - Seconds elapsed between transactions.
* `V1` to `V28` - Anonymized PCA-transformed numerical features.
* `Amount` - Transaction amount.
* `Class` - Target variable.

Target variable:

```text
0 = Legitimate Transaction
1 = Fraudulent Transaction
```

---

## 7. Data Preprocessing

The following preprocessing steps are performed.

### Duplicate Removal

Duplicate transaction records are removed before model training.

```text
Original rows: 284,807
Duplicate rows removed: 1,081
Final rows: 283,726
```

### Train-Test Split

The dataset is divided using an 80/20 stratified split.

```text
Training Set: 80%
Testing Set: 20%
```

Stratification is used to preserve the fraud-to-legitimate transaction ratio.

### Feature Scaling

`RobustScaler` from Scikit-learn is used for feature scaling.

The scaler is fitted only on the training data to reduce the risk of data leakage.

### SMOTE

Because the dataset contains very few fraud transactions compared with legitimate transactions, **SMOTE (Synthetic Minority Oversampling Technique)** is applied only to the training data.

The test dataset remains unchanged for evaluation.

---

## 8. Feature Engineering

Two additional features are generated.

### Amount_log

The transaction amount is transformed using:

```python
Amount_log = log1p(Amount)
```

This helps reduce the effect of highly skewed transaction amounts.

### Hour

Transaction time is converted into an hour-of-day feature:

```python
Hour = (Time // 3600) % 24
```

The final model features include:

```text
V1 - V28
Amount_log
Hour
```

---

## 9. Machine Learning Models

The project trains and compares five machine learning models.

### 1. Logistic Regression

Used as a baseline supervised classification model.

### 2. Random Forest

An ensemble tree-based classification model used to capture nonlinear relationships between transaction features.

### 3. XGBoost

A gradient boosting algorithm used for classification.

### 4. LightGBM

A gradient boosting framework designed for efficient tree-based learning.

### 5. Isolation Forest

An unsupervised anomaly detection algorithm used to identify unusual transaction patterns.

---

## 10. Model Training Pipeline

The complete training pipeline is:

```text
Kaggle Dataset
      |
      v
Load Dataset
      |
      v
Remove Duplicates
      |
      v
Feature Engineering
      |
      v
Exploratory Data Analysis
      |
      v
Train/Test Split
      |
      v
RobustScaler
      |
      v
SMOTE on Training Data
      |
      v
Train Machine Learning Models
      |
      v
Evaluate Models
      |
      v
Save Models
      |
      v
Fraud Prediction
```

---

## 11. Model Performance

The models are evaluated using:

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion Matrix

The obtained experimental results are:

| Model               | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| ------------------- | --------: | -----: | -------: | ------: | -----: |
| Logistic Regression |    0.0532 | 0.8737 |   0.1004 |  0.9599 | 0.6775 |
| Random Forest       |    0.9268 | 0.8000 |   0.8588 |  0.9606 | 0.8288 |
| XGBoost             |    0.6525 | 0.8105 |   0.7230 |  0.9775 | 0.8128 |
| LightGBM            |    0.6033 | 0.7684 |   0.6759 |  0.9739 | 0.7860 |
| Isolation Forest    |    0.2353 | 0.2947 |   0.2617 |  0.9390 | 0.1616 |

The Random Forest experiment produced an F1 Score of **0.8588**.

---

## 12. Evaluation Metrics

### Precision

Precision measures how many transactions predicted as fraud were actually fraudulent.

```text
Precision = True Positives / (True Positives + False Positives)
```

### Recall

Recall measures how many actual fraud transactions were detected.

```text
Recall = True Positives / (True Positives + False Negatives)
```

### F1 Score

F1 Score combines Precision and Recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between legitimate and fraudulent transactions across classification thresholds.

### PR-AUC

PR-AUC is useful for evaluating classification problems with highly imbalanced classes.

---

## 13. Exploratory Data Analysis

The project generates multiple visualizations including:

* Class distribution
* Transaction amount distribution
* Transaction time distribution
* Fraud transactions by hour
* Transaction amount boxplot
* Top feature analysis
* Feature correlation heatmap

These visualizations help understand the transaction dataset and identify patterns in the data.

---

## 14. Fraud Prediction

The application supports single transaction prediction.

The user can provide:

```text
Time
Amount
V1
V2
V3
...
V28
```

The system processes the transaction using the saved preprocessing artifacts and trained machine learning model.

The prediction output includes:

* Fraud prediction
* Fraud score/probability for supervised models
* Risk level

---

## 15. Batch Prediction

The application also supports batch prediction using a CSV file.

The uploaded CSV should contain:

```text
Time
V1
V2
V3
...
V28
Amount
```

The application processes all rows and generates additional output columns:

```text
fraud_probability
prediction
risk_level
```

The processed results can be downloaded as a CSV file.

---

## 16. Risk Classification

The application provides four risk levels:

| Fraud Score    | Risk Level |
| -------------- | ---------- |
| Less than 0.30 | LOW        |
| 0.30 - 0.59    | MEDIUM     |
| 0.60 - 0.79    | HIGH       |
| 0.80 and above | CRITICAL   |

These thresholds are application-level risk categories used by the project interface.

They should not be interpreted as calibrated real-world banking decision thresholds.

---

## 17. Frontend Pages

### Dashboard

The Dashboard provides an overview of the project and dataset.

It displays:

* Total transactions
* Legitimate transactions
* Fraudulent transactions
* Fraud percentage
* Average transaction amount
* Model performance summary
* Class distribution

### Dataset Analysis

The Dataset Analysis page provides:

* Dataset information
* Dataset shape
* Feature information
* Class distribution
* Transaction statistics
* Sample records
* Fraud and legitimate transaction counts

### EDA / Visualizations

This page displays the generated exploratory data analysis charts.

It includes:

* Class distribution
* Amount distribution
* Time distribution
* Fraud by hour
* Amount boxplot
* Feature analysis
* Correlation heatmap

### Model Performance

This page displays:

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion matrices
* ROC curves
* Precision-Recall curves
* Model comparison

### Fraud Prediction

This page allows users to enter transaction information manually.

The application then generates:

```text
Prediction
Fraud Score / Probability
Risk Level
```

### Batch Prediction

This page allows users to upload a CSV file and generate predictions for multiple transactions.

The output contains:

```text
fraud_probability
prediction
risk_level
```

### About

The About page provides:

* Project description
* Dataset information
* Machine learning models
* Technology stack
* Project workflow
* Academic project information

---

## 18. Project Structure

```text
Financial_Fraud_Detection_Project/
│
├── creditcard.csv
│
├── FinancialFraudDetection.py
│
├── requirements.txt
│
├── README.md
│
├── SalapareddiLaxmana_ProjectReport.docx
│
├── fraud_models/
│   ├── scaler.joblib
│   ├── feature_cols.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── lightgbm.joblib
│   ├── isolation_forest.joblib
│   ├── dataset_stats.joblib
│   ├── eval_results.joblib
│   └── eval_summary.json
│
└── fraud_assets/
    ├── EDA charts
    ├── ROC curves
    ├── Precision-Recall curves
    ├── Confusion matrices
    └── Model comparison charts
```

---

## 19. Tech Stack

### Programming Language

* Python 3.11

### Frontend

* Streamlit

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM
* imbalanced-learn

### Data Processing

* Pandas
* NumPy
* SciPy

### Data Visualization

* Matplotlib
* Seaborn
* Plotly

### Model Persistence

* Joblib

### Dataset

* Kaggle Credit Card Fraud Detection Dataset

---

## 20. Requirements

The project dependencies are listed in:

```text
requirements.txt
```

Main dependencies include:

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

---

# 21. Quick Start

## Step 1 - Install Dependencies

First, clone or download the project.

Open the project folder in a terminal.

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 2 - Train the Model

Make sure the dataset file is available in the project directory:

```text
creditcard.csv
```

Run the training pipeline:

```bash
python FinancialFraudDetection.py
```

The training process will:

1. Load the dataset.
2. Remove duplicate records.
3. Perform feature engineering.
4. Generate EDA visualizations.
5. Split the dataset.
6. Scale the features.
7. Apply SMOTE to training data.
8. Train five machine learning models.
9. Evaluate model performance.
10. Save trained models.
11. Save evaluation results.

The generated files will be stored in:

```text
fraud_models/
```

and:

```text
fraud_assets/
```

---

## Step 3 - Launch the Streamlit Frontend

After training is completed, start the Streamlit application:

```bash
streamlit run FinancialFraudDetection.py
```

Streamlit will provide a local URL in the terminal.

Open that URL in a web browser to access the application.

---

# 22. Flask Backend

The current project version does not contain a separate Flask backend.

Therefore, the following command is **not required**:

```bash
python backend/app.py
```

The current project uses:

```text
Python ML Engine
        +
Streamlit Frontend
```

inside the same Python file.

---

# 23. API Endpoints

The current version does not expose REST API endpoints.

There is no separate implementation of:

```text
GET /health
GET /models
GET /stats
POST /predict
POST /predict/batch
```

in the current project.

Instead, prediction functionality is directly available through the Streamlit interface.

The Python prediction functions are:

```python
predict_single()
```

for individual transaction prediction and:

```python
predict_batch_df()
```

for batch transaction prediction.

---

# 24. Prediction Workflow

A single transaction is processed using the following workflow:

```text
Transaction Input
       |
       v
Feature Engineering
       |
       v
Saved Feature Columns
       |
       v
Saved RobustScaler
       |
       v
Trained ML Model
       |
       v
Fraud Score
       |
       v
Prediction
       |
       v
Risk Level
```

Example output:

```text
Prediction: Fraud
Fraud Score: 0.96
Risk Level: CRITICAL
```

The displayed score depends on the selected model and input transaction.

---

# 25. Batch Prediction Example

Example input CSV:

```csv
Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount
406,-1.35,-0.07,2.54,1.38,-0.34,0.46,-0.28,0.11,0.07,-0.28,-0.38,-0.61,-0.99,-0.31,1.47,-0.47,0.21,0.03,0.22,0.01,-0.02,0.28,-0.11,0.06,0.12,-0.01,0.03,-0.02,149.62
```

The application produces additional columns:

```text
fraud_probability
prediction
risk_level
```

Example:

```text
fraud_probability = 0.96
prediction = 1
risk_level = CRITICAL
```

---

# 26. Generated Model Files

After model training, the application saves the trained models using Joblib.

The model directory contains:

```text
fraud_models/
```

Example files:

```text
scaler.joblib
feature_cols.joblib
logistic_regression.joblib
random_forest.joblib
xgboost.joblib
lightgbm.joblib
isolation_forest.joblib
dataset_stats.joblib
eval_results.joblib
eval_summary.json
```

These files allow the Streamlit application to load the trained models and perform predictions without retraining every time.

---

# 27. Generated Visualizations

The project saves visualization assets in:

```text
fraud_assets/
```

The generated visualizations include:

* Class distribution chart
* Amount distribution
* Time distribution
* Fraud by hour
* Amount boxplot
* Feature analysis
* Correlation heatmap
* ROC curve
* Precision-Recall curve
* Confusion matrices
* Model comparison chart

---

# 28. Limitations

The project has the following limitations:

1. The V1-V28 features are anonymized PCA components, so they do not have direct business interpretations.
2. The dataset represents a historical transaction population.
3. Model performance depends on the selected dataset and train-test split.
4. Fraud patterns can change over time.
5. Fraud scores should not automatically be interpreted as calibrated real-world probabilities.
6. The application is an academic project and is not a production banking fraud detection system.
7. Production deployment would require additional security, monitoring, model governance, and validation.

---

# 29. Future Enhancements

Possible future improvements include:

* SHAP-based model explainability
* Hyperparameter optimization
* Probability calibration
* Threshold optimization
* Real-time transaction processing
* Model drift monitoring
* Automated model retraining
* Database integration
* Cloud deployment
* Authentication and authorization
* REST API integration
* Production monitoring
* Advanced fraud investigation dashboard

---

# 30. Academic Project Information

**Project Title:**

AI-Powered Financial Fraud Detection and Risk Analytics System Using Machine Learning

**Program:**

IBM SkillsBuild Academic Internship

**Domain:**

Data Analytics with AI

**Organization:**

BharatCares

**Student:**

Salapareddi Laxmana

---

# 31. Conclusion

This project demonstrates an end-to-end machine learning workflow for financial fraud detection.

It combines:

* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Imbalanced-data handling
* Machine learning
* Model evaluation
* Fraud prediction
* Batch prediction
* Risk classification
* Interactive Streamlit visualization

The project provides a practical demonstration of applying machine learning techniques to highly imbalanced financial transaction data.

---

# 32. Author

**Salapareddi Laxmana**

IBM SkillsBuild Data Analytics with AI Internship Project

