import os
import sys
import json
import time
import warnings
import pickle

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "creditcard.csv"
MODELS_DIR = "fraud_models"
ASSETS_DIR = "fraud_assets"

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_directories():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)


def load_dataset(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Please place creditcard.csv in the project directory."
        )

    df = pd.read_csv(path)
    return df


def clean_dataset(df):
    df = df.copy()

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    return df, duplicate_count


def feature_engineering(df):
    df = df.copy()

    if "Amount" in df.columns:
        df["Amount_log"] = np.log1p(df["Amount"])

    if "Time" in df.columns:
        df["Hour"] = (df["Time"] / 3600) % 24

    return df


def get_feature_columns(df):
    excluded = {"Class"}

    features = [
        col for col in df.columns
        if col not in excluded
    ]

    return features


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df):
    df = feature_engineering(df)

    feature_columns = get_feature_columns(df)

    X = df[feature_columns].copy()
    y = df["Class"].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(
        random_state=RANDOM_STATE,
        sampling_strategy="auto"
    )

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled,
        y_train
    )

    joblib.dump(
        scaler,
        os.path.join(MODELS_DIR, "scaler.joblib")
    )

    joblib.dump(
        feature_columns,
        os.path.join(MODELS_DIR, "feature_columns.joblib")
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        X_train_resampled,
        y_train_resampled,
        feature_columns,
        scaler,
    )


# ============================================================
# EDA
# ============================================================

def generate_eda(df):
    ensure_directories()

    eda_files = {}

    # --------------------------------------------------------
    # Class Distribution
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Class"
    )

    plt.title("Transaction Class Distribution")
    plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Transaction Count")
    plt.tight_layout()

    path = os.path.join(
        ASSETS_DIR,
        "class_distribution.png"
    )

    plt.savefig(path, dpi=160)
    plt.close()

    eda_files["class_distribution"] = path

    # --------------------------------------------------------
    # Amount Distribution
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="Amount",
        bins=50,
        kde=True
    )

    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()

    path = os.path.join(
        ASSETS_DIR,
        "amount_distribution.png"
    )

    plt.savefig(path, dpi=160)
    plt.close()

    eda_files["amount_distribution"] = path

    # --------------------------------------------------------
    # Fraud Amount Distribution
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="Class",
        y="Amount"
    )

    plt.title("Transaction Amount by Class")
    plt.xlabel("Class")
    plt.ylabel("Amount")
    plt.tight_layout()

    path = os.path.join(
        ASSETS_DIR,
        "amount_by_class.png"
    )

    plt.savefig(path, dpi=160)
    plt.close()

    eda_files["amount_by_class"] = path

    # --------------------------------------------------------
    # Transactions Over Time
    # --------------------------------------------------------

    if "Time" in df.columns:

        temp = df.copy()

        temp["Hour"] = (temp["Time"] / 3600) % 24

        plt.figure(figsize=(10, 5))

        sns.histplot(
            data=temp,
            x="Hour",
            hue="Class",
            bins=48,
            multiple="stack"
        )

        plt.title("Transaction Activity by Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Transaction Count")
        plt.tight_layout()

        path = os.path.join(
            ASSETS_DIR,
            "transactions_by_hour.png"
        )

        plt.savefig(path, dpi=160)
        plt.close()

        eda_files["transactions_by_hour"] = path

    # --------------------------------------------------------
    # Correlation Heatmap
    # --------------------------------------------------------

    numeric_df = df.select_dtypes(include=np.number)

    correlation = numeric_df.corr()

    plt.figure(figsize=(13, 10))

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        linewidths=0.1
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    path = os.path.join(
        ASSETS_DIR,
        "correlation_heatmap.png"
    )

    plt.savefig(path, dpi=160)
    plt.close()

    eda_files["correlation_heatmap"] = path

    # --------------------------------------------------------
    # Fraud Percentage
    # --------------------------------------------------------

    fraud_rate = (
        df["Class"].mean() * 100
        if "Class" in df.columns
        else 0
    )

    plt.figure(figsize=(7, 5))

    labels = ["Legitimate", "Fraud"]
    values = [
        100 - fraud_rate,
        fraud_rate
    ]

    plt.bar(
        labels,
        values
    )

    plt.title("Legitimate vs Fraud Percentage")
    plt.ylabel("Percentage")
    plt.tight_layout()

    path = os.path.join(
        ASSETS_DIR,
        "fraud_percentage.png"
    )

    plt.savefig(path, dpi=160)
    plt.close()

    eda_files["fraud_percentage"] = path

    return eda_files


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_models(y_train_resampled):

    models = {}

    models["Logistic Regression"] = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None,
        max_depth=None
    )

    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1
    )

    models["LightGBM"] = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1
    )

    models["Isolation Forest"] = IsolationForest(
        n_estimators=300,
        contamination=0.002,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return models


# ============================================================
# MODEL TRAINING
# ============================================================

def train_models(
    X_train_scaled,
    y_train,
    X_train_resampled,
    y_train_resampled
):

    models = build_models(y_train_resampled)

    trained_models = {}

    for name, model in models.items():

        if name == "Isolation Forest":

            model.fit(X_train_scaled)

        else:

            model.fit(
                X_train_resampled,
                y_train_resampled
            )

        trained_models[name] = model

        filename = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            + ".joblib"
        )

        joblib.dump(
            model,
            os.path.join(MODELS_DIR, filename)
        )

    return trained_models


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def get_prediction_scores(model, X):

    if isinstance(model, IsolationForest):

        raw_predictions = model.decision_function(X)

        scores = -raw_predictions

        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score == 0:
            probabilities = np.zeros_like(scores)

        else:
            probabilities = (
                (scores - min_score)
                / (max_score - min_score)
            )

        predictions = (
            model.predict(X) == -1
        ).astype(int)

        return predictions, probabilities

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        return predictions, probabilities

    predictions = model.predict(X)

    return predictions, predictions.astype(float)


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_models(
    trained_models,
    X_test_scaled,
    y_test
):

    results = {}

    for name, model in trained_models.items():

        predictions, probabilities = get_prediction_scores(
            model,
            X_test_scaled
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        try:
            roc_auc = roc_auc_score(
                y_test,
                probabilities
            )
        except Exception:
            roc_auc = 0.0

        try:
            pr_auc = average_precision_score(
                y_test,
                probabilities
            )
        except Exception:
            pr_auc = 0.0

        cm = confusion_matrix(
            y_test,
            predictions
        )

        results[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": cm.tolist()
        }

    joblib.dump(
        results,
        os.path.join(
            MODELS_DIR,
            "eval_results.joblib"
        )
    )

    with open(
        os.path.join(
            MODELS_DIR,
            "eval_summary.json"
        ),
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    return results


# ============================================================
# EVALUATION VISUALIZATIONS
# ============================================================

def generate_evaluation_charts(
    trained_models,
    X_test_scaled,
    y_test,
    results
):

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    for name, model in trained_models.items():

        _, probabilities = get_prediction_scores(
            model,
            X_test_scaled
        )

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.title("ROC-AUC Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            ASSETS_DIR,
            "roc_curves.png"
        ),
        dpi=160
    )

    plt.close()

    # --------------------------------------------------------
    # PRECISION RECALL CURVE
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    for name, model in trained_models.items():

        _, probabilities = get_prediction_scores(
            model,
            X_test_scaled
        )

        precision, recall, _ = precision_recall_curve(
            y_test,
            probabilities
        )

        ap = average_precision_score(
            y_test,
            probabilities
        )

        plt.plot(
            recall,
            precision,
            label=f"{name} (AP={ap:.3f})"
        )

    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            ASSETS_DIR,
            "precision_recall_curves.png"
        ),
        dpi=160
    )

    plt.close()

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    metric_df = pd.DataFrame(
        results
    ).T

    metrics_to_plot = [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc"
    ]

    metric_df[metrics_to_plot].plot(
        kind="bar",
        figsize=(12, 7)
    )

    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(rotation=35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            ASSETS_DIR,
            "model_metrics.png"
        ),
        dpi=160
    )

    plt.close()

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    for name, model_results in results.items():

        cm = np.array(
            model_results["confusion_matrix"]
        )

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False
        )

        plt.title(
            f"{name} - Confusion Matrix"
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        filename = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(
            os.path.join(
                ASSETS_DIR,
                filename
            ),
            dpi=160
        )

        plt.close()


# ============================================================
# TRAINING PIPELINE
# ============================================================

def run_training_pipeline():

    print("=" * 70)
    print("AI-POWERED FINANCIAL FRAUD DETECTION")
    print("=" * 70)

    ensure_directories()

    print("\n[1/7] Loading dataset...")

    df = load_dataset()

    print(
        f"Original dataset shape: {df.shape}"
    )

    print("\n[2/7] Cleaning dataset...")

    df, duplicate_count = clean_dataset(df)

    print(
        f"Duplicate rows removed: {duplicate_count}"
    )

    print(
        f"Clean dataset shape: {df.shape}"
    )

    stats = {
        "original_rows": int(df.shape[0] + duplicate_count),
        "clean_rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "fraud_count": int(df["Class"].sum()),
        "legitimate_count": int((df["Class"] == 0).sum()),
        "fraud_rate": float(df["Class"].mean() * 100),
        "duplicate_count": int(duplicate_count)
    }

    joblib.dump(
        stats,
        os.path.join(
            MODELS_DIR,
            "dataset_stats.joblib"
        )
    )

    print("\n[3/7] Generating EDA...")

    generate_eda(df)

    print("\n[4/7] Preparing train/test data...")

    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        X_train_resampled,
        y_train_resampled,
        feature_columns,
        scaler
    ) = prepare_data(df)

    print(
        f"Training rows before SMOTE: {len(y_train)}"
    )

    print(
        f"Training rows after SMOTE: {len(y_train_resampled)}"
    )

    print("\n[5/7] Training models...")

    trained_models = train_models(
        X_train_scaled,
        y_train,
        X_train_resampled,
        y_train_resampled
    )

    print("\n[6/7] Evaluating models...")

    results = evaluate_models(
        trained_models,
        X_test_scaled,
        y_test
    )

    print("\nModel Results:")

    for name, result in results.items():

        print(
            f"{name}: "
            f"Precision={result['precision']:.4f}, "
            f"Recall={result['recall']:.4f}, "
            f"F1={result['f1']:.4f}, "
            f"ROC-AUC={result['roc_auc']:.4f}, "
            f"PR-AUC={result['pr_auc']:.4f}"
        )

    print("\n[7/7] Generating evaluation charts...")

    generate_evaluation_charts(
        trained_models,
        X_test_scaled,
        y_test,
        results
    )

    print("\nTraining completed successfully.")

    return results


# ============================================================
# LOAD SAVED ARTIFACTS
# ============================================================

def load_saved_results():

    path = os.path.join(
        MODELS_DIR,
        "eval_results.joblib"
    )

    if not os.path.exists(path):
        return None

    return joblib.load(path)


def load_saved_stats():

    path = os.path.join(
        MODELS_DIR,
        "dataset_stats.joblib"
    )

    if not os.path.exists(path):
        return None

    return joblib.load(path)


def load_saved_model(model_name):

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + ".joblib"
    )

    path = os.path.join(
        MODELS_DIR,
        filename
    )

    if not os.path.exists(path):
        return None

    return joblib.load(path)


def load_saved_scaler():

    path = os.path.join(
        MODELS_DIR,
        "scaler.joblib"
    )

    if not os.path.exists(path):
        return None

    return joblib.load(path)


def load_feature_columns():

    path = os.path.join(
        MODELS_DIR,
        "feature_columns.joblib"
    )

    if not os.path.exists(path):
        return None

    return joblib.load(path)


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def risk_level(probability):

    if probability < 0.30:
        return "LOW RISK"

    elif probability < 0.60:
        return "MEDIUM RISK"

    elif probability < 0.80:
        return "HIGH RISK"

    else:
        return "CRITICAL RISK"


def risk_description(level):

    descriptions = {

        "LOW RISK":
            "The transaction has a relatively low predicted fraud probability.",

        "MEDIUM RISK":
            "The transaction requires additional monitoring.",

        "HIGH RISK":
            "The transaction shows elevated fraud risk and may require review.",

        "CRITICAL RISK":
            "The transaction has a very high predicted fraud probability and should be investigated."
    }

    return descriptions.get(
        level,
        "Risk level unavailable."
    )


# ============================================================
# SINGLE TRANSACTION PREDICTION
# ============================================================

def predict_transaction(
    input_data,
    model_name="Random Forest"
):

    scaler = load_saved_scaler()
    feature_columns = load_feature_columns()
    model = load_saved_model(model_name)

    if scaler is None:
        raise FileNotFoundError(
            "Scaler not found. Train the models first."
        )

    if feature_columns is None:
        raise FileNotFoundError(
            "Feature columns not found. Train the models first."
        )

    if model is None:
        raise FileNotFoundError(
            f"Model not found: {model_name}"
        )

    input_df = pd.DataFrame(
        [input_data]
    )

    input_df = feature_engineering(
        input_df
    )

    for column in feature_columns:

        if column not in input_df.columns:
            input_df[column] = 0

    input_df = input_df[
        feature_columns
    ]

    X_scaled = scaler.transform(
        input_df
    )

    predictions, probabilities = get_prediction_scores(
        model,
        X_scaled
    )

    probability = float(
        probabilities[0]
    )

    prediction = int(
        predictions[0]
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level(probability),
        "description": risk_description(
            risk_level(probability)
        )
    }


# ============================================================
# BATCH PREDICTION
# ============================================================

def batch_predict(
    df,
    model_name="Random Forest"
):

    scaler = load_saved_scaler()
    feature_columns = load_feature_columns()
    model = load_saved_model(model_name)

    if scaler is None:
        raise FileNotFoundError(
            "Scaler not found."
        )

    if feature_columns is None:
        raise FileNotFoundError(
            "Feature columns not found."
        )

    if model is None:
        raise FileNotFoundError(
            f"Model not found: {model_name}"
        )

    work_df = df.copy()

    work_df = feature_engineering(
        work_df
    )

    for column in feature_columns:

        if column not in work_df.columns:
            work_df[column] = 0

    X = work_df[
        feature_columns
    ]

    X_scaled = scaler.transform(
        X
    )

    predictions, probabilities = get_prediction_scores(
        model,
        X_scaled
    )

    output = df.copy()

    output["Fraud_Prediction"] = predictions

    output["Fraud_Probability"] = probabilities

    output["Risk_Level"] = [
        risk_level(p)
        for p in probabilities
    ]

    return output


# ============================================================
# STREAMLIT APPLICATION
# ============================================================

def run_streamlit_app():

    import streamlit as st

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except Exception:
        px = None
        go = None

    st.set_page_config(
        page_title="AI Financial Fraud Detection",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ========================================================
    # PROFESSIONAL DARK UI


# ========================================================
    # HIGH-CONTRAST DARK UI OVERRIDE (COMPLETE FIX)
    # ========================================================

    st.markdown(
        """
        <style>

        /* 1. FORCE ENTIRE BACKGROUND TO DARK SLATE */
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #7DD3FC !important;
            color: #38BDF8 !important;
        }

        /* 2. FORCE ALL TEXT, LABELS, PARAGRAPHS TO VISIBLE WHITE/SLATE */
        p, span, label, li, div, h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF !important;
        }

        /* 3. FIX METRIC CARDS (HIGH CONTRAST & CLEAR VISIBILITY) */
        [data-testid="stMetric"] {
            background-color: #FFFFFF !important; /* Pure White Card for high contrast */
            border: 2px solid #0F172A !important;   /* Solid Dark Navy Border */
            border-radius: 12px !important;
            padding: 16px !important;
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.15) !important;
        }

        [data-testid="stMetricLabel"] * {
            color: #334155 !important;             /* Dark Slate Blue for Subheadings */
            font-weight: 700 !important;
            font-size: 14px !important;
        }

        [data-testid="stMetricValue"] * {
            color: #0F172A !important;             /* Deep Dark Navy for Numbers (Maximum Visibility) */
            font-weight: 900 !important;
            font-size: 32px !important;
        }

        [data-testid="stMetricLabel"] * {
            color: #94A3B8 !important;
            font-weight: 700 !important;
            font-size: 14px !important;
        }
-----------------------------------------------------------------------
# ========================================================
    # HIGH-CONTRAST PROFESSIONAL DARK UI
    # ========================================================

    st.markdown(
        """
        <style>

        /* 1. FORCE APP BACKGROUND & DEFAULT TEXT COLOR */
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #0F172A !important;
            color: #F8FAFC !important;
        }

        /* 2. HEADINGS & GENERAL TEXT VISIBILITY */
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF !important;
            font-weight: 800 !important;
        }

        p, span, label, li, div {
            color: #E2E8F0 !important;
        }

        /* 3. STREAMLIT NATIVE METRIC CARDS */
        [data-testid="stMetric"] {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
            padding: 16px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
        }

        [data-testid="stMetricLabel"] * {
            color: #94A3B8 !important;
            font-weight: 700 !important;
            font-size: 14px !important;
        }

        [data-testid="stMetricValue"] * {
            color: #FFFFFF !important;
            font-weight: 900 !important;
            font-size: 32px !important;
        }

        /* 4. CUSTOM METRIC CARDS & FACT BOXES */
        .metric-card, .info-card, .fact-box {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
            padding: 20px !important;
        }

        .metric-label, .fact-title {
            color: #38BDF8 !important;
            font-weight: 700 !important;
            font-size: 13px !important;
            text-transform: uppercase;
        }

        .metric-value, .fact-value {
            color: #FFFFFF !important;
            font-weight: 900 !important;
            font-size: 28px !important;
        }

        /* 5. SIDEBAR STYLING */
        [data-testid="stSidebar"] {
            background-color: #090D16 !important;
            border-right: 1px solid #334155 !important;
        }

        [data-testid="stSidebar"] * {
            color: #F8FAFC !important;
        }

        /* 6. TABLES AND DATAFRAMES VISIBILITY */
        [data-testid="stDataFrame"] {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
        }

        /* 7. PLOTLY CHART BACKGROUND FIX */
        .js-plotly-plot .plotly .main-svg {
            background-color: #1E293B !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # SIDEBAR
    # ========================================================

    st.sidebar.markdown(
        """
        <div style="
            text-align: center;
            padding: 10px 0px 20px 0px;
            border-bottom: 1px solid #334155;
            margin-bottom: 20px;
        ">
            <div style="font-size: 36px; margin-bottom: 4px;">💳</div>
            <div style="font-size: 18px; font-weight: 800; color: #FFFFFF;">
                Fraud Detection AI
            </div>
            <div style="font-size: 12px; color: #38BDF8; font-weight: 600; margin-top: 2px;">
                Financial Risk Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.sidebar.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Dataset Analysis",
            "EDA / Visualizations",
            "Model Performance",
            "Fraud Prediction",
            "Batch Prediction",
            "About"
        ]
    )

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    st.sidebar.markdown(
        """
        <div style="
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 14px;
        ">
            <div style="color: #38BDF8; font-weight: 800; font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase;">
                PROJECT OVERVIEW
            </div>
            <div style="color: #FFFFFF; font-size: 14px; font-weight: 700; margin-top: 4px;">
                AI Financial Fraud Detection
            </div>
            <div style="color: #94A3B8; font-size: 12px; margin-top: 6px; line-height: 1.4;">
                Machine Learning • Risk Analytics • Streamlit
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    try:
        df = load_dataset()
        clean_df, duplicate_count = clean_dataset(df)
    except Exception as e:

        st.error(
            f"Unable to load dataset: {e}"
        )

        st.stop()

    stats = load_saved_stats()
    results = load_saved_results()

    # ========================================================
    # DASHBOARD
    # ========================================================

    if page == "Dashboard":

        st.markdown(
            """
            <div class="main-title">
                AI-Powered Financial Fraud Detection
            </div>

            <div class="main-subtitle">
                Machine Learning System for Transaction Risk Analytics
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    🔐 Intelligent Financial Transaction Monitoring
                </div>

                <div class="info-card-text">
                    This application uses machine learning to identify
                    potentially fraudulent credit card transactions,
                    compare multiple classification and anomaly detection
                    algorithms, and provide transaction-level risk analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        total_transactions = len(clean_df)

        fraud_count = int(
            clean_df["Class"].sum()
        )

        legitimate_count = int(
            (clean_df["Class"] == 0).sum()
        )

        fraud_rate = (
            fraud_count / total_transactions * 100
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Transactions",
                f"{total_transactions:,}"
            )

        with c2:
            st.metric(
                "Fraud Cases",
                f"{fraud_count:,}"
            )

        with c3:
            st.metric(
                "Legitimate",
                f"{legitimate_count:,}"
            )

        with c4:
            st.metric(
                "Fraud Rate",
                f"{fraud_rate:.3f}%"
            )

        st.markdown("## System Overview")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                """
                <div class="info-card">
                    <div class="info-card-title">
                        🤖 Machine Learning Models
                    </div>

                    <div class="info-card-text">
                        • Logistic Regression<br>
                        • Random Forest<br>
                        • XGBoost<br>
                        • LightGBM<br>
                        • Isolation Forest
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                """
                <div class="info-card">
                    <div class="info-card-title">
                        📊 Analytics Capabilities
                    </div>

                    <div class="info-card-text">
                        • Exploratory Data Analysis<br>
                        • Class Imbalance Analysis<br>
                        • ROC-AUC & PR-AUC<br>
                        • Confusion Matrix Analysis<br>
                        • Single & Batch Prediction
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if results:

            st.markdown("## Model Summary")

            model_df = pd.DataFrame(
                {
                    name: {
                        "Precision": result["precision"],
                        "Recall": result["recall"],
                        "F1 Score": result["f1"],
                        "ROC-AUC": result["roc_auc"],
                        "PR-AUC": result["pr_auc"]
                    }
                    for name, result in results.items()
                }
            ).T

            st.dataframe(
                model_df.style.format(
                    "{:.4f}"
                ),
                use_container_width=True
            )

    # ========================================================
    # DATASET ANALYSIS
    # ========================================================

    elif page == "Dataset Analysis":

        st.markdown(
            """
            <div class="main-title">
                Dataset Analysis
            </div>

            <div class="main-subtitle">
                Credit Card Fraud Detection Dataset
            </div>
            """,
            unsafe_allow_html=True
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Overview",
                "Dataset Facts",
                "Class Imbalance",
                "Data Preview"
            ]
        )

        with tab1:

            st.markdown(
                """
                <div class="info-card">
                    <div class="info-card-title">
                        📌 Dataset Overview
                    </div>

                    <div class="info-card-text">
                        The project uses the Kaggle Credit Card Fraud
                        Detection dataset. The data contains anonymized
                        numerical transaction features and a binary target
                        indicating whether a transaction is fraudulent.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Dataset Structure")

            st.write(
                "The dataset contains transaction timing, transaction "
                "amount and anonymized PCA-transformed features."
            )

            columns = list(
                clean_df.columns
            )

            st.write(
                f"**Number of columns:** {len(columns)}"
            )

            st.write(
                "**Target column:** `Class`"
            )

            st.write(
                "**Fraud label:** `1`"
            )

            st.write(
                "**Legitimate label:** `0`"
            )

        with tab2:

            st.markdown(
                "## Dataset Facts"
            )

            total = len(clean_df)
            fraud = int(
                clean_df["Class"].sum()
            )
            legitimate = int(
                (clean_df["Class"] == 0).sum()
            )

            amount_mean = clean_df[
                "Amount"
            ].mean()

            fraud_amount_mean = clean_df[
                clean_df["Class"] == 1
            ]["Amount"].mean()

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f"""
                    <div class="fact-box">
                        <div class="fact-title">
                            Transactions
                        </div>

                        <div class="fact-value">
                            {total:,}
                        </div>

                        <div class="fact-description">
                            Clean transaction records
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:
                st.markdown(
                    f"""
                    <div class="fact-box">
                        <div class="fact-title">
                            Fraud Transactions
                        </div>

                        <div class="fact-value">
                            {fraud:,}
                        </div>

                        <div class="fact-description">
                            Confirmed fraud labels
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c3:
                st.markdown(
                    f"""
                    <div class="fact-box">
                        <div class="fact-title">
                            Legitimate
                        </div>

                        <div class="fact-value">
                            {legitimate:,}
                        </div>

                        <div class="fact-description">
                            Legitimate transactions
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            c4, c5, c6 = st.columns(3)

            with c4:

                st.markdown(
                    f"""
                    <div class="fact-box">
                        <div class="fact-title">
                            Fraud Rate
                        </div>

                        <div class="fact-value">
                            {fraud / total * 100:.3f}%
                        </div>

                        <div class="fact-description">
                            Share of fraudulent transactions
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c5:

                st.markdown(
                    f"""
                    <div class="fact-box">
                        <div class="fact-title">
                            Average Amount
                        </div>

                        <div class="fact-value">
                            ${amount_mean:.2f}
                        </div>

                        <div class="fact-description">
                            Across all transactions
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c6:

                st.markdown(
                    f"""
                    <div class="fact-box">
                        <div class="fact-title">
                            Fraud Avg. Amount
                        </div>

                        <div class="fact-value">
                            ${fraud_amount_mean:.2f}
                        </div>

                        <div class="fact-description">
                            Average fraudulent transaction
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("### Dataset Columns")

            column_df = pd.DataFrame(
                {
                    "Column": clean_df.columns,
                    "Data Type": [
                        str(clean_df[c].dtype)
                        for c in clean_df.columns
                    ],
                    "Missing Values": [
                        int(clean_df[c].isna().sum())
                        for c in clean_df.columns
                    ]
                }
            )

            st.dataframe(
                column_df,
                use_container_width=True,
                hide_index=True
            )

        with tab3:

            st.markdown(
                """
                <div class="info-card">
                    <div class="info-card-title">
                        ⚠️ Severe Class Imbalance
                    </div>

                    <div class="info-card-text">
                        Fraud represents only a very small percentage of
                        all transactions. This means a model could achieve
                        high accuracy while still failing to detect fraud.
                        Therefore, this project emphasizes Precision,
                        Recall, F1 Score, ROC-AUC and PR-AUC.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            fraud_count = int(
                clean_df["Class"].sum()
            )

            legitimate_count = int(
                (clean_df["Class"] == 0).sum()
            )

            imbalance_df = pd.DataFrame(
                {
                    "Class": [
                        "Legitimate",
                        "Fraud"
                    ],
                    "Count": [
                        legitimate_count,
                        fraud_count
                    ],
                    "Percentage": [
                        legitimate_count / len(clean_df) * 100,
                        fraud_count / len(clean_df) * 100
                    ]
                }
            )

            st.dataframe(
                imbalance_df.style.format(
                    {"Percentage": "{:.4f}%"}
                ),
                use_container_width=True,
                hide_index=True
            )

            if px:

                fig = px.bar(
                    imbalance_df,
                    x="Class",
                    y="Count",
                    text="Count",
                    title="Transaction Class Distribution"
                )

                fig.update_layout(
                    paper_bgcolor="#071A2B",
                    plot_bgcolor="#071A2B",
                    font=dict(
                        color="white"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with tab4:

            st.markdown(
                "## Data Preview"
            )

            st.dataframe(
                clean_df.head(20),
                use_container_width=True,
                hide_index=True
            )

    # ========================================================
    # EDA
    # ========================================================

    elif page == "EDA / Visualizations":

        st.markdown(
            """
            <div class="main-title">
                Exploratory Data Analysis
            </div>

            <div class="main-subtitle">
                Visual exploration of transaction behavior
            </div>
            """,
            unsafe_allow_html=True
        )

        eda_images = [
            (
                "Class Distribution",
                "class_distribution.png"
            ),
            (
                "Amount Distribution",
                "amount_distribution.png"
            ),
            (
                "Amount by Class",
                "amount_by_class.png"
            ),
            (
                "Transactions by Hour",
                "transactions_by_hour.png"
            ),
            (
                "Correlation Heatmap",
                "correlation_heatmap.png"
            ),
            (
                "Fraud Percentage",
                "fraud_percentage.png"
            )
        ]

        for title, filename in eda_images:

            path = os.path.join(
                ASSETS_DIR,
                filename
            )

            if os.path.exists(path):

                st.markdown(
                    f"### {title}"
                )

                st.image(
                    path,
                    use_container_width=True
                )

                st.markdown("---")

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    elif page == "Model Performance":

        st.markdown(
            """
            <div class="main-title">
                Model Performance
            </div>

            <div class="main-subtitle">
                Comparative evaluation of fraud detection algorithms
            </div>
            """,
            unsafe_allow_html=True
        )

        if not results:

            st.warning(
                "Model evaluation results are not available. "
                "Run the training pipeline first."
            )

            st.stop()

        performance_df = pd.DataFrame(
            [
                {
                    "Model": name,
                    "Precision": result["precision"],
                    "Recall": result["recall"],
                    "F1 Score": result["f1"],
                    "ROC-AUC": result["roc_auc"],
                    "PR-AUC": result["pr_auc"]
                }
                for name, result in results.items()
            ]
        )

        st.dataframe(
            performance_df.style.format(
                {
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "ROC-AUC": "{:.4f}",
                    "PR-AUC": "{:.4f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("## Evaluation Charts")

        roc_path = os.path.join(
            ASSETS_DIR,
            "roc_curves.png"
        )

        pr_path = os.path.join(
            ASSETS_DIR,
            "precision_recall_curves.png"
        )

        metrics_path = os.path.join(
            ASSETS_DIR,
            "model_metrics.png"
        )

        if os.path.exists(roc_path):

            st.markdown(
                "### ROC Curves"
            )

            st.image(
                roc_path,
                use_container_width=True
            )

        if os.path.exists(pr_path):

            st.markdown(
                "### Precision-Recall Curves"
            )

            st.image(
                pr_path,
                use_container_width=True
            )

        if os.path.exists(metrics_path):

            st.markdown(
                "### Metrics Comparison"
            )

            st.image(
                metrics_path,
                use_container_width=True
            )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    📌 Why multiple metrics?
                </div>

                <div class="info-card-text">
                    Fraud detection is an imbalanced classification
                    problem. Precision measures how many flagged transactions
                    are actually fraudulent, while Recall measures how many
                    fraudulent transactions were detected. F1 Score balances
                    Precision and Recall. ROC-AUC and PR-AUC evaluate ranking
                    performance across thresholds.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # FRAUD PREDICTION
    # ========================================================

    elif page == "Fraud Prediction":

        st.markdown(
            """
            <div class="main-title">
                Fraud Prediction
            </div>

            <div class="main-subtitle">
                Analyze an individual transaction
            </div>
            """,
            unsafe_allow_html=True
        )

        model_options = [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "LightGBM",
            "Isolation Forest"
        ]

        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=1
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    🧠 Input Transaction Features
                </div>

                <div class="info-card-text">
                    Enter values for the transaction. The anonymized V
                    features correspond to the original dataset's
                    PCA-transformed attributes.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        time_value = st.number_input(
            "Transaction Time",
            min_value=0.0,
            value=10000.0,
            step=100.0
        )

        st.markdown("### PCA Features")

        feature_inputs = {}

        cols = st.columns(4)

        for i in range(1, 29):

            with cols[(i - 1) % 4]:

                feature_inputs[
                    f"V{i}"
                ] = st.number_input(
                    f"V{i}",
                    value=0.0,
                    format="%.6f",
                    key=f"single_v_{i}"
                )

        if st.button(
            "🔍 Analyze Transaction",
            use_container_width=True
        ):

            input_data = {
                "Time": time_value,
                "Amount": amount
            }

            input_data.update(
                feature_inputs
            )

            try:

                result = predict_transaction(
                    input_data,
                    selected_model
                )

                probability = result[
                    "probability"
                ]

                prediction = result[
                    "prediction"
                ]

                level = result[
                    "risk_level"
                ]

                st.markdown("---")

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Fraud Probability",
                        f"{probability * 100:.2f}%"
                    )

                with c2:

                    st.metric(
                        "Prediction",
                        "FRAUD" if prediction else "LEGITIMATE"
                    )

                with c3:

                    st.metric(
                        "Risk Level",
                        level
                    )

                if prediction == 1:

                    st.error(
                        f"⚠️ {level}: Potential fraudulent transaction detected."
                    )

                else:

                    st.success(
                        f"✅ {level}: Transaction classified as legitimate."
                    )

                st.info(
                    result["description"]
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

    # ========================================================
    # BATCH PREDICTION
    # ========================================================

    elif page == "Batch Prediction":

        st.markdown(
            """
            <div class="main-title">
                Batch Prediction
            </div>

            <div class="main-subtitle">
                Analyze multiple transactions at once
            </div>
            """,
            unsafe_allow_html=True
        )

        selected_model = st.selectbox(
            "Select Model",
            [
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "Isolation Forest"
            ],
            index=1,
            key="batch_model"
        )

        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"]
        )

        if uploaded_file:

            try:

                batch_df = pd.read_csv(
                    uploaded_file
                )

                st.markdown(
                    "### Uploaded Data"
                )

                st.dataframe(
                    batch_df.head(20),
                    use_container_width=True
                )

                if st.button(
                    "🚀 Run Batch Prediction",
                    use_container_width=True
                ):

                    result_df = batch_predict(
                        batch_df,
                        selected_model
                    )

                    st.success(
                        "Batch prediction completed successfully."
                    )

                    st.markdown(
                        "### Prediction Results"
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    csv_data = result_df.to_csv(
                        index=False
                    ).encode(
                        "utf-8"
                    )

                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=csv_data,
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            except Exception as e:

                st.error(
                    f"Batch prediction failed: {e}"
                )

        else:

            st.info(
                "Upload a CSV file containing transaction features "
                "to generate batch fraud predictions."
            )

    # ========================================================
    # ABOUT
    # ========================================================

    elif page == "About":

        st.markdown(
            """
            <div class="main-title">
                About the Project
            </div>

            <div class="main-subtitle">
                AI-Powered Financial Fraud Detection & Risk Analytics System
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    🎯 Project Objective
                </div>

                <div class="info-card-text">
                    The objective of this project is to develop a machine
                    learning-based fraud detection system capable of
                    identifying suspicious financial transactions under
                    severe class imbalance.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    🛠️ Technology Stack
                </div>

                <div class="info-card-text">
                    Python • Pandas • NumPy • Scikit-learn •
                    XGBoost • LightGBM • Imbalanced-learn •
                    Matplotlib • Seaborn • Streamlit
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    🔬 Machine Learning Approach
                </div>

                <div class="info-card-text">
                    The project combines supervised classification models
                    with an unsupervised anomaly detection approach.
                    SMOTE is applied only to the training data to address
                    severe class imbalance while preserving the original
                    test distribution.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    📈 Evaluation Strategy
                </div>

                <div class="info-card-text">
                    Model performance is evaluated using Precision,
                    Recall, F1 Score, ROC-AUC, PR-AUC and Confusion
                    Matrices instead of relying only on accuracy.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">
                    ⚠️ Important Dataset Limitation
                </div>

                <div class="info-card-text">
                    The dataset contains anonymized PCA features and does
                    not contain typical customer information such as names,
                    merchant categories, locations or demographic details.
                    Therefore, the application does not claim to perform
                    customer profiling or merchant-specific fraud analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="footer">
                AI-Powered Financial Fraud Detection & Risk Analytics System
                <br>
                Developed as an academic machine learning project
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if "streamlit" in sys.modules:

        run_streamlit_app()

    else:

        print(
            "\nStarting model training pipeline...\n"
        )

        run_training_pipeline()
