"""
SalapareddiLaxmana_FinancialFraudDetection.py
==============================================

AI-Powered Financial Fraud Detection and Risk Analytics System
Using Machine Learning on the Kaggle Credit Card Fraud Detection Dataset

Run modes:
    python SalapareddiLaxmana_FinancialFraudDetection.py
        -> train all models and save artifacts

    streamlit run SalapareddiLaxmana_FinancialFraudDetection.py
        -> launch interactive web application
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

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

import xgboost as xgb
import lightgbm as lgb


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "creditcard.csv"

MODELS_DIR = "fraud_models"
ASSETS_DIR = "fraud_assets"

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


# =============================================================================
# SECTION 1 - DATA LOADING & PREPROCESSING
# =============================================================================

def load_and_clean(path=DATA_PATH):
    """Load CSV and remove exact duplicate transactions."""

    print("[1/7] Loading dataset ...")

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Dataset not found: %s. Please place creditcard.csv "
            "in the same folder as this Python file." % path
        )

    df = pd.read_csv(path)

    original = len(df)

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("      Rows before dedup : %d" % original)
    print("      Duplicates removed : %d" % (original - len(df)))
    print("      Rows after dedup  : %d" % len(df))

    return df


def feature_engineer(df):
    """Create Amount_log and Hour features."""

    print("[2/7] Feature engineering ...")

    df = df.copy()

    df["Amount_log"] = np.log1p(df["Amount"])
    df["Hour"] = (df["Time"] // 3600) % 24

    return df


def split_and_scale(df):
    """
    Stratified 80/20 split.
    RobustScaler is fitted only on training data.
    SMOTE is applied only to the training data.
    """

    print("[3/7] Train/test split (stratified 80/20) ...")

    feature_cols = [
        c for c in df.columns
        if c not in ("Class", "Time", "Amount")
    ]

    X = df[feature_cols]
    y = df["Class"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        "      Train: %d | Fraud: %d"
        % (len(X_tr), y_tr.sum())
    )

    print(
        "      Test : %d | Fraud: %d"
        % (len(X_te), y_te.sum())
    )

    print("[4/7] Scaling (RobustScaler, fit on train only) ...")

    scaler = RobustScaler()

    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    joblib.dump(
        scaler,
        os.path.join(MODELS_DIR, "scaler.joblib")
    )

    joblib.dump(
        feature_cols,
        os.path.join(MODELS_DIR, "feature_cols.joblib")
    )

    # Original unbalanced training data for Isolation Forest
    X_tr_orig = X_tr_s.copy()
    y_tr_orig = y_tr.copy()

    print("[5/7] SMOTE on training set only ...")

    smote = SMOTE(random_state=RANDOM_STATE)

    X_tr_bal, y_tr_bal = smote.fit_resample(
        X_tr_s,
        y_tr
    )

    print(
        "      Fraud after SMOTE: %d | Total: %d"
        % (y_tr_bal.sum(), len(y_tr_bal))
    )

    return (
        X_tr_bal,
        y_tr_bal,
        X_te_s,
        y_te,
        X_tr_orig,
        y_tr_orig,
        feature_cols,
        scaler,
    )


# =============================================================================
# SECTION 2 - EDA VISUALISATIONS
# =============================================================================

def generate_eda(df):
    """Generate and save EDA charts."""

    print("[6/7] Generating EDA visualisations ...")

    sns.set_theme(
        style="whitegrid",
        font_scale=1.0
    )

    # -------------------------------------------------------------------------
    # 1. CLASS DISTRIBUTION
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    fig.suptitle(
        "Class Distribution",
        fontsize=14,
        fontweight="bold"
    )

    counts = df["Class"].value_counts()

    labels = [
        "Legitimate",
        "Fraud"
    ]

    vals = [
        counts.get(0, 0),
        counts.get(1, 0)
    ]

    total = sum(vals)

    colors = [
        "#2563EB",
        "#DC2626"
    ]

    axes[0].bar(
        labels,
        vals,
        color=colors,
        width=0.5
    )

    for i, (v, lbl) in enumerate(
        zip(vals, labels)
    ):
        axes[0].text(
            i,
            v + 500,
            "%d\n(%.2f%%)"
            % (v, 100 * v / total),
            ha="center",
            fontsize=10
        )

    axes[0].set_title(
        "Transaction Counts"
    )

    axes[0].set_ylabel(
        "Count"
    )

    axes[1].pie(
        vals,
        labels=labels,
        colors=colors,
        autopct="%1.3f%%",
        startangle=90,
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 2
        }
    )

    axes[1].set_title(
        "Class Proportion"
    )

    fig.savefig(
        os.path.join(
            ASSETS_DIR,
            "eda_class_distribution.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # 2. AMOUNT DISTRIBUTION
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    fig.suptitle(
        "Transaction Amount by Class",
        fontsize=14,
        fontweight="bold"
    )

    for ax, log in zip(
        axes,
        [False, True]
    ):

        for cls, lbl, col in [
            (0, "Legitimate", "#2563EB"),
            (1, "Fraud", "#DC2626")
        ]:

            values = df.loc[
                df["Class"] == cls,
                "Amount"
            ]

            if log:
                values = np.log1p(values)

            ax.hist(
                values,
                bins=60,
                alpha=0.6,
                label=lbl,
                color=col
            )

        ax.set_xlabel(
            "log1p(Amount)"
            if log
            else "Amount ($)"
        )

        ax.set_ylabel(
            "Count"
        )

        ax.set_title(
            "Log-Transformed"
            if log
            else "Raw Amount"
        )

        ax.legend()

    fig.savefig(
        os.path.join(
            ASSETS_DIR,
            "eda_amount_distribution.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. TIME DISTRIBUTION
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    fig.suptitle(
        "Time Distribution",
        fontsize=14,
        fontweight="bold"
    )

    for ax, cls, lbl, col in [
        (axes[0], 0, "Legitimate", "#2563EB"),
        (axes[1], 1, "Fraud", "#DC2626")
    ]:

        ax.hist(
            df.loc[
                df["Class"] == cls,
                "Time"
            ],
            bins=48,
            color=col,
            alpha=0.85
        )

        ax.set_title(lbl)

        ax.set_xlabel(
            "Time (seconds)"
        )

        ax.set_ylabel(
            "Count"
        )

    fig.savefig(
        os.path.join(
            ASSETS_DIR,
            "eda_time_distribution.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # 4. FRAUD BY HOUR
    # -------------------------------------------------------------------------

    df2 = df.copy()

    df2["Hour"] = (
        df2["Time"] // 3600
    ) % 24

    hourly = (
        df2.groupby(
            ["Hour", "Class"]
        )
        .size()
        .unstack(fill_value=0)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    fig.suptitle(
        "Transactions by Hour of Day",
        fontsize=14,
        fontweight="bold"
    )

    if 0 in hourly.columns:

        axes[0].bar(
            hourly.index,
            hourly[0],
            color="#2563EB",
            alpha=0.85
        )

        axes[0].set_title(
            "Legitimate"
        )

        axes[0].set_xlabel(
            "Hour"
        )

        axes[0].set_ylabel(
            "Count"
        )

    if 1 in hourly.columns:

        axes[1].bar(
            hourly.index,
            hourly[1],
            color="#DC2626",
            alpha=0.85
        )

        axes[1].set_title(
            "Fraudulent"
        )

        axes[1].set_xlabel(
            "Hour"
        )

        axes[1].set_ylabel(
            "Count"
        )

    fig.savefig(
        os.path.join(
            ASSETS_DIR,
            "eda_fraud_by_hour.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # 5. AMOUNT BOXPLOT
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    fig.suptitle(
        "Amount Boxplot by Class",
        fontsize=14,
        fontweight="bold"
    )

    for ax, log in zip(
        axes,
        [False, True]
    ):

        legitimate = df.loc[
            df["Class"] == 0,
            "Amount"
        ]

        fraud = df.loc[
            df["Class"] == 1,
            "Amount"
        ]

        if log:
            legitimate = np.log1p(
                legitimate
            )
            fraud = np.log1p(
                fraud
            )

        ax.boxplot(
            [legitimate, fraud],
            patch_artist=True
        )

        ax.set_xticks(
            [1, 2]
        )

        ax.set_xticklabels(
            [
                "Legitimate",
                "Fraud"
            ]
        )

        ax.set_title(
            "log1p(Amount)"
            if log
            else "Raw Amount ($)"
        )

    fig.savefig(
        os.path.join(
            ASSETS_DIR,
            "eda_amount_boxplot.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # 6. TOP FEATURES
    # -------------------------------------------------------------------------

    sample = df.sample(
        min(10000, len(df)),
        random_state=42
    )

    v_cols = [
        c for c in sample.columns
        if c.startswith("V")
    ]

    corr = (
        sample[
            v_cols + ["Class"]
        ]
        .corr()["Class"]
        .drop("Class")
        .abs()
    )

    top10 = corr.nlargest(
        10
    ).index.tolist()

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(18, 8)
    )

    fig.suptitle(
        "Top 10 Features by Correlation with Class",
        fontsize=14,
        fontweight="bold"
    )

    axes = axes.flatten()

    for ax, feat in zip(
        axes,
        top10
    ):

        ax.boxplot(
            [
                df.loc[
                    df["Class"] == 0,
                    feat
                ],
                df.loc[
                    df["Class"] == 1,
                    feat
                ]
            ],
            patch_artist=True
        )

        ax.set_xticks(
            [1, 2]
        )

        ax.set_xticklabels(
            ["Legit", "Fraud"]
        )

        ax.set_title(
            feat,
            fontsize=11
        )

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            ASSETS_DIR,
            "eda_top_features.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # 7. CORRELATION HEATMAP
    # -------------------------------------------------------------------------

    sample2 = df.sample(
        min(5000, len(df)),
        random_state=42
    ).copy()

    sample2["Amount_log"] = np.log1p(
        sample2["Amount"]
    )

    v2 = [
        c for c in sample2.columns
        if c.startswith("V")
    ]

    corr2 = sample2[
        v2 + ["Amount_log", "Class"]
    ].corr()

    fig, ax = plt.subplots(
        figsize=(18, 14)
    )

    mask = np.triu(
        np.ones_like(
            corr2,
            dtype=bool
        )
    )

    sns.heatmap(
        corr2,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.3,
        annot=False,
        square=True,
        cbar_kws={
            "shrink": 0.6
        }
    )

    ax.set_title(
        "Feature Correlation Heatmap",
        fontsize=14,
        fontweight="bold"
    )

    fig.savefig(
        os.path.join(
            ASSETS_DIR,
            "eda_correlation_heatmap.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "      EDA charts saved to %s/"
        % ASSETS_DIR
    )


# =============================================================================
# SECTION 3 - MODEL TRAINING
# =============================================================================

def train_models(
    X_tr_bal,
    y_tr_bal,
    X_tr_orig,
    y_tr_orig
):
    """Train all five models."""

    print("[7/7] Training models ...")

    models = {}

    # -------------------------------------------------------------------------
    # LOGISTIC REGRESSION
    # -------------------------------------------------------------------------

    t0 = time.time()

    print(
        "      [1/5] Logistic Regression ..."
    )

    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    lr.fit(
        X_tr_bal,
        y_tr_bal
    )

    models[
        "Logistic Regression"
    ] = lr

    print(
        "            done %.1fs"
        % (time.time() - t0)
    )

    # -------------------------------------------------------------------------
    # RANDOM FOREST
    # -------------------------------------------------------------------------

    t0 = time.time()

    print(
        "      [2/5] Random Forest ..."
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf.fit(
        X_tr_bal,
        y_tr_bal
    )

    models[
        "Random Forest"
    ] = rf

    print(
        "            done %.1fs"
        % (time.time() - t0)
    )

    # -------------------------------------------------------------------------
    # XGBOOST
    # -------------------------------------------------------------------------

    t0 = time.time()

    print(
        "      [3/5] XGBoost ..."
    )

    neg = int(
        (y_tr_bal == 0).sum()
    )

    pos = int(
        (y_tr_bal == 1).sum()
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(
            neg / pos
            if pos
            else 1.0
        ),
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )

    xgb_model.fit(
        X_tr_bal,
        y_tr_bal
    )

    models[
        "XGBoost"
    ] = xgb_model

    print(
        "            done %.1fs"
        % (time.time() - t0)
    )

    # -------------------------------------------------------------------------
    # LIGHTGBM
    # -------------------------------------------------------------------------

    t0 = time.time()

    print(
        "      [4/5] LightGBM ..."
    )

    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(
            neg / pos
            if pos
            else 1.0
        ),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )

    lgb_model.fit(
        X_tr_bal,
        y_tr_bal
    )

    models[
        "LightGBM"
    ] = lgb_model

    print(
        "            done %.1fs"
        % (time.time() - t0)
    )

    # -------------------------------------------------------------------------
    # ISOLATION FOREST
    # -------------------------------------------------------------------------

    t0 = time.time()

    print(
        "      [5/5] Isolation Forest ..."
    )

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.002,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    iso.fit(
        X_tr_orig
    )

    models[
        "Isolation Forest"
    ] = iso

    print(
        "            done %.1fs"
        % (time.time() - t0)
    )

    # -------------------------------------------------------------------------
    # SAVE MODELS
    # -------------------------------------------------------------------------

    for name, model in models.items():

        safe_name = (
            name
            .lower()
            .replace(" ", "_")
        )

        joblib.dump(
            model,
            os.path.join(
                MODELS_DIR,
                "%s.joblib" % safe_name
            )
        )

    return models


# =============================================================================
# SECTION 4 - MODEL EVALUATION
# =============================================================================

def evaluate_models(
    models,
    X_te,
    y_te
):
    """Evaluate all trained models."""

    all_results = []

    for name, model in models.items():

        is_iso = (
            name == "Isolation Forest"
        )

        if is_iso:

            raw = model.predict(
                X_te
            )

            y_pred = np.where(
                raw == -1,
                1,
                0
            )

            scores = -model.decision_function(
                X_te
            )

            y_prob = (
                scores - scores.min()
            ) / (
                scores.max()
                - scores.min()
                + 1e-9
            )

        else:

            y_pred = model.predict(
                X_te
            )

            y_prob = model.predict_proba(
                X_te
            )[:, 1]

        cm = confusion_matrix(
            y_te,
            y_pred
        )

        result = {
            "model_name": name,

            "precision": round(
                float(
                    precision_score(
                        y_te,
                        y_pred,
                        zero_division=0
                    )
                ),
                4
            ),

            "recall": round(
                float(
                    recall_score(
                        y_te,
                        y_pred,
                        zero_division=0
                    )
                ),
                4
            ),

            "f1": round(
                float(
                    f1_score(
                        y_te,
                        y_pred,
                        zero_division=0
                    )
                ),
                4
            ),

            "roc_auc": round(
                float(
                    roc_auc_score(
                        y_te,
                        y_prob
                    )
                ),
                4
            ),

            "pr_auc": round(
                float(
                    average_precision_score(
                        y_te,
                        y_prob
                    )
                ),
                4
            ),

            "confusion_matrix":
                cm.tolist(),

            "y_prob":
                y_prob.tolist()
        }

        all_results.append(
            result
        )

        print(
            "  %-22s F1=%.4f ROC-AUC=%.4f PR-AUC=%.4f"
            % (
                name,
                result["f1"],
                result["roc_auc"],
                result["pr_auc"]
            )
        )

    return all_results


def generate_eval_charts(
    all_results,
    y_te
):
    """Generate model evaluation charts."""

    colors = [
        "#2563EB",
        "#16A34A",
        "#F59E0B",
        "#7C3AED",
        "#DC2626"
    ]

    y_arr = np.array(
        y_te
    )

    # -------------------------------------------------------------------------
    # ROC CURVES
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    for result, color in zip(
        all_results,
        colors
    ):

        fpr, tpr, _ = roc_curve(
            y_arr,
            np.array(
                result["y_prob"]
            )
        )

        ax.plot(
            fpr,
            tpr,
            label="%s (%.4f)"
            % (
                result["model_name"],
                result["roc_auc"]
            ),
            color=color,
            lw=2
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        lw=1,
        label="Random"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curves",
        fontsize=14,
        fontweight="bold"
    )

    ax.legend(
        loc="lower right"
    )

    ax.grid(
        True,
        alpha=0.4
    )

    fig.savefig(
        os.path.join(
            ASSETS_DIR,
            "eval_roc_curves.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # PRECISION-RECALL CURVES
    # -------------------------------------------------------------------------

    baseline = y_arr.mean()

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    for result, color in zip(
        all_results,
        colors
    ):

        precision, recall, _ = (
            precision_recall_curve(
                y_arr,
                np.array(
                    result["y_prob"]
                )
            )
        )

        ax.plot(
            recall,
            precision,
            label="%s (%.4f)"
            % (
                result["model_name"],
                result["pr_auc"]
            ),
            color=color,
            lw=2
        )

    ax.axhline(
        baseline,
        color="k",
        linestyle="--",
        lw=1,
        label="Baseline"
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Precision-Recall Curves",
        fontsize=14,
        fontweight="bold"
    )

    ax.legend(
        loc="upper right"
    )

    ax.grid(
        True,
        alpha=0.4
    )

    fig.savefig(
        os.path.join(
            ASSETS_DIR,
            "eval_pr_curves.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # CONFUSION MATRICES
    # -------------------------------------------------------------------------

    n = len(
        all_results
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10)
    )

    axes = axes.flatten()

    for ax, result in zip(
        axes,
        all_results
    ):

        cm = np.array(
            result["confusion_matrix"]
        )

        ax.imshow(
            cm,
            cmap="Blues"
        )

        ax.set_title(
            result["model_name"],
            fontsize=11,
            fontweight="bold"
        )

        ax.set_xticks(
            [0, 1]
        )

        ax.set_yticks(
            [0, 1]
        )

        ax.set_xticklabels(
            ["Legit", "Fraud"]
        )

        ax.set_yticklabels(
            ["Legit", "Fraud"]
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

        for i in range(2):

            for j in range(2):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=(
                        "white"
                        if cm[i, j]
                        > cm.max() / 2
                        else "black"
                    )
                )

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            ASSETS_DIR,
            "eval_confusion_matrices.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # METRICS COMPARISON
    # -------------------------------------------------------------------------

    metrics = [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc"
    ]

    x = np.arange(
        len(metrics)
    )

    width = 0.15

    offsets = (
        np.linspace(
            -(len(all_results) - 1) / 2,
            (len(all_results) - 1) / 2,
            len(all_results)
        )
        * width
    )

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    for result, offset, color in zip(
        all_results,
        offsets,
        colors
    ):

        ax.bar(
            x + offset,
            [
                result[m]
                for m in metrics
            ],
            width,
            label=result["model_name"],
            color=color,
            alpha=0.85
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "PR-AUC"
        ]
    )

    ax.set_ylim(
        0,
        1.1
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Model Performance Comparison",
        fontsize=14,
        fontweight="bold"
    )

    ax.legend(
        loc="lower right"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.4
    )

    fig.savefig(
        os.path.join(
            ASSETS_DIR,
            "eval_metrics_comparison.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# =============================================================================
# SECTION 5 - INFERENCE
# =============================================================================

MODEL_FILES = {
    "Logistic Regression":
        "logistic_regression.joblib",

    "Random Forest":
        "random_forest.joblib",

    "XGBoost":
        "xgboost.joblib",

    "LightGBM":
        "lightgbm.joblib",

    "Isolation Forest":
        "isolation_forest.joblib",
}

SUPERVISED = {
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM"
}


def get_risk_level(probability):

    if probability < 0.30:
        return "LOW RISK", "#16A34A"

    if probability < 0.60:
        return "MEDIUM RISK", "#D97706"

    if probability < 0.80:
        return "HIGH RISK", "#EA580C"

    return "CRITICAL RISK", "#DC2626"


def _load_inference_artifacts(
    model_name
):

    scaler = joblib.load(
        os.path.join(
            MODELS_DIR,
            "scaler.joblib"
        )
    )

    feature_cols = joblib.load(
        os.path.join(
            MODELS_DIR,
            "feature_cols.joblib"
        )
    )

    model = joblib.load(
        os.path.join(
            MODELS_DIR,
            MODEL_FILES[model_name]
        )
    )

    return (
        scaler,
        feature_cols,
        model
    )


def predict_single(
    model_name,
    time_val,
    v_features,
    amount
):
    """Predict one transaction."""

    scaler, feature_cols, model = (
        _load_inference_artifacts(
            model_name
        )
    )

    amount_log = np.log1p(
        amount
    )

    hour = (
        time_val // 3600
    ) % 24

    feature_dict = {
        "V%d" % i:
        v_features[i - 1]
        for i in range(1, 29)
    }

    feature_dict[
        "Amount_log"
    ] = amount_log

    feature_dict[
        "Hour"
    ] = hour

    raw = np.array([
        feature_dict[c]
        for c in feature_cols
    ]).reshape(
        1,
        -1
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        scaled = scaler.transform(
            raw
        )

    if model_name in SUPERVISED:

        probability = float(
            model.predict_proba(
                scaled
            )[0, 1]
        )

        prediction = int(
            model.predict(
                scaled
            )[0]
        )

    else:

        score = float(
            -model.decision_function(
                scaled
            )[0]
        )

        probability = float(
            np.clip(
                score,
                0,
                1
            )
        )

        prediction = (
            1
            if model.predict(
                scaled
            )[0] == -1
            else 0
        )

    risk_level, risk_color = (
        get_risk_level(
            probability
        )
    )

    return {
        "probability": probability,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "prediction": prediction
    }


def predict_batch_df(
    model_name,
    df_in
):
    """Predict fraud risk for a complete dataframe."""

    scaler, feature_cols, model = (
        _load_inference_artifacts(
            model_name
        )
    )

    df = df_in.copy()

    df["Amount_log"] = np.log1p(
        df["Amount"]
    )

    df["Hour"] = (
        df["Time"] // 3600
    ) % 24

    missing = [
        c
        for c in feature_cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: %s"
            % str(missing)
        )

    X = df[
        feature_cols
    ].values

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        X_scaled = scaler.transform(
            X
        )

    if model_name in SUPERVISED:

        probabilities = (
            model.predict_proba(
                X_scaled
            )[:, 1]
        )

        predictions = model.predict(
            X_scaled
        )

    else:

        scores = (
            -model.decision_function(
                X_scaled
            )
        )

        probabilities = np.clip(
            scores,
            0,
            1
        )

        predictions = np.where(
            model.predict(
                X_scaled
            ) == -1,
            1,
            0
        )

    df[
        "fraud_probability"
    ] = np.round(
        probabilities,
        4
    )

    df[
        "prediction"
    ] = predictions

    df[
        "risk_level"
    ] = df[
        "fraud_probability"
    ].apply(
        lambda p:
        get_risk_level(p)[0]
    )

    return df


def list_trained_models():

    return [
        name
        for name, filename
        in MODEL_FILES.items()
        if os.path.exists(
            os.path.join(
                MODELS_DIR,
                filename
            )
        )
    ]


def models_ready():

    required = [
        "scaler.joblib",
        "feature_cols.joblib",
        "random_forest.joblib"
    ]

    return all(
        os.path.exists(
            os.path.join(
                MODELS_DIR,
                filename
            )
        )
        for filename in required
    )


# =============================================================================
# SECTION 6 - TRAINING PIPELINE
# =============================================================================

def run_training_pipeline():

    total_start = time.time()

    print("=" * 65)
    print(
        "  AI FINANCIAL FRAUD DETECTION"
    )
    print(
        "  Machine Learning Training Pipeline"
    )
    print("=" * 65)

    df = load_and_clean()

    df = feature_engineer(
        df
    )

    generate_eda(
        df
    )

    (
        X_tr_bal,
        y_tr_bal,
        X_te,
        y_te,
        X_tr_orig,
        y_tr_orig,
        feature_cols,
        scaler
    ) = split_and_scale(
        df
    )

    models = train_models(
        X_tr_bal,
        y_tr_bal,
        X_tr_orig,
        y_tr_orig
    )

    print(
        "\nEvaluating models on test set ..."
    )

    all_results = evaluate_models(
        models,
        X_te,
        y_te
    )

    generate_eval_charts(
        all_results,
        y_te
    )

    best = max(
        all_results,
        key=lambda result:
        result["f1"]
    )

    # -------------------------------------------------------------------------
    # DATASET STATISTICS
    # -------------------------------------------------------------------------

    stats = {

        "total_rows":
            int(len(df)),

        "fraud_count":
            int(df["Class"].sum()),

        "legit_count":
            int(
                (df["Class"] == 0)
                .sum()
            ),

        "fraud_pct":
            float(
                100
                * df["Class"].sum()
                / len(df)
            ),

        "legit_pct":
            float(
                100
                * (
                    df["Class"] == 0
                ).sum()
                / len(df)
            ),

        "mean_amount":
            float(
                df["Amount"].mean()
            ),

        "mean_amount_fraud":
            float(
                df.loc[
                    df["Class"] == 1,
                    "Amount"
                ].mean()
            ),

        "mean_amount_legit":
            float(
                df.loc[
                    df["Class"] == 0,
                    "Amount"
                ].mean()
            ),

        "max_amount":
            float(
                df["Amount"].max()
            ),

        "best_model":
            best["model_name"]
    }

    joblib.dump(
        stats,
        os.path.join(
            MODELS_DIR,
            "dataset_stats.joblib"
        )
    )

    joblib.dump(
        all_results,
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
    ) as file:

        json.dump(
            [
                {
                    key: value
                    for key, value
                    in result.items()
                    if key != "y_prob"
                }
                for result in all_results
            ],
            file,
            indent=2
        )

    elapsed = (
        time.time()
        - total_start
    ) / 60

    print("\n" + "=" * 65)

    print(
        "  Training complete in %.1f minutes"
        % elapsed
    )

    print(
        "  Best model by F1: %s = %.4f"
        % (
            best["model_name"],
            best["f1"]
        )
    )

    print(
        "  Launch:"
    )

    print(
        "  streamlit run "
        "SalapareddiLaxmana_FinancialFraudDetection.py"
    )

    print("=" * 65)


# =============================================================================
# SECTION 7 - STREAMLIT APPLICATION
# =============================================================================

def run_streamlit_app():

    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from PIL import Image

    # -------------------------------------------------------------------------
    # PAGE CONFIG
    # -------------------------------------------------------------------------

    st.set_page_config(
        page_title="AI Financial Fraud Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # -------------------------------------------------------------------------
    # PROFESSIONAL UI
    # -------------------------------------------------------------------------

    st.markdown(
        """
        <style>

        /* ================================================================
           GLOBAL PAGE
           ================================================================ */

        html,
        body,
        [data-testid="stAppViewContainer"] {
            background: #F1F5F9 !important;
        }

        [data-testid="stHeader"] {
            background: #F1F5F9 !important;
        }

        .main {
            background: #F1F5F9 !important;
        }

        .block-container {
            max-width: 1500px !important;
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
        }

        /* ================================================================
           NORMAL TEXT
           ================================================================ */

        p,
        li,
        label,
        span,
        div {
            font-family:
                Inter,
                "Segoe UI",
                Arial,
                sans-serif;
        }

        p {
            color: #334155;
            font-size: 1rem;
            line-height: 1.65;
        }

        /* ================================================================
           MAIN HEADINGS
           ================================================================ */

        h1 {
            color: #0F172A !important;
            font-size: 2.25rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px !important;
        }

        h2 {
            color: #0F172A !important;
            font-size: 1.75rem !important;
            font-weight: 800 !important;
        }

        h3 {
            color: #0F172A !important;
            font-size: 1.35rem !important;
            font-weight: 750 !important;
        }

        h4 {
            color: #1E293B !important;
            font-size: 1.1rem !important;
            font-weight: 700 !important;
        }

        /* ================================================================
           SIDEBAR
           ================================================================ */

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    #0B1F33 0%,
                    #102A43 100%
                ) !important;

            border-right:
                1px solid #1E3A5F !important;
        }

        section[data-testid="stSidebar"] * {
            color: #E2E8F0 !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #FFFFFF !important;
        }

        .sidebar-brand {
            background: #143653;
            border: 1px solid #285477;
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 18px;
        }

        .sidebar-brand-title {
            color: #FFFFFF !important;
            font-size: 1.15rem;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .sidebar-brand-subtitle {
            color: #B8CCE0 !important;
            font-size: 0.78rem;
            line-height: 1.4;
        }

        section[data-testid="stSidebar"]
        [role="radiogroup"] label {
            border-radius: 10px !important;
            padding: 9px 11px !important;
            margin: 3px 0 !important;
            color: #DCE7F2 !important;
        }

        section[data-testid="stSidebar"]
        [role="radiogroup"] label:hover {
            background: #173B5E !important;
        }

        section[data-testid="stSidebar"]
        [data-testid="stCaptionContainer"] {
            color: #9FB5CA !important;
        }

        /* ================================================================
           HERO HEADER
           ================================================================ */

        .hero {
            background:
                linear-gradient(
                    135deg,
                    #0B1F33 0%,
                    #123A5A 60%,
                    #155E8A 100%
                );

            border-radius: 20px;
            padding: 28px 32px;
            margin-bottom: 24px;

            box-shadow:
                0 12px 30px
                rgba(15, 23, 42, 0.14);
        }

        .hero-title {
            color: #FFFFFF !important;
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
        }

        .hero-subtitle {
            color: #D7E7F5 !important;
            font-size: 1rem;
            margin-top: 7px;
        }

        .hero-badge {
            display: inline-block;
            background: #E0F2FE;
            color: #075985 !important;
            border-radius: 999px;
            padding: 6px 12px;
            margin-top: 14px;
            font-size: 0.78rem;
            font-weight: 800;
        }

        /* ================================================================
           SECTION HEADER
           ================================================================ */

        .section-header {
            background: #FFFFFF;
            border-left: 6px solid #2563EB;
            border-radius: 12px;

            padding: 16px 20px;
            margin-bottom: 22px;

            color: #0F172A !important;

            font-size: 1.45rem;
            font-weight: 800;

            box-shadow:
                0 3px 12px
                rgba(15, 23, 42, 0.07);
        }

        /* ================================================================
           INFORMATION CARDS
           ================================================================ */

        .info-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 18px;
            margin-bottom: 16px;

            box-shadow:
                0 3px 12px
                rgba(15, 23, 42, 0.05);
        }

        .info-card-title {
            color: #0F172A !important;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .info-card-text {
            color: #475569 !important;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        /* ================================================================
           METRICS
           ================================================================ */

        [data-testid="stMetric"] {
            background: #FFFFFF !important;
            border: 1px solid #DDE5EE !important;
            border-radius: 15px !important;
            padding: 18px !important;

            min-height: 125px;

            box-shadow:
                0 4px 14px
                rgba(15, 23, 42, 0.06);
        }

        [data-testid="stMetricLabel"] {
            color: #64748B !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        [data-testid="stMetricLabel"] p {
            color: #64748B !important;
        }

        [data-testid="stMetricValue"] {
            color: #0F172A !important;
            font-size: 2rem !important;
            font-weight: 850 !important;
        }

        [data-testid="stMetricDelta"] {
            font-size: 0.82rem !important;
            font-weight: 700 !important;
        }

        /* ================================================================
           BUTTONS
           ================================================================ */

        .stButton > button {
            min-height: 44px !important;
            border-radius: 10px !important;
            font-weight: 750 !important;
            border: 1px solid #CBD5E1 !important;
            background: #FFFFFF !important;
            color: #0F172A !important;
        }

        .stButton > button:hover {
            border-color: #2563EB !important;
            color: #1D4ED8 !important;
        }

        .stButton > button[kind="primary"] {
            background:
                linear-gradient(
                    135deg,
                    #2563EB,
                    #1D4ED8
                ) !important;

            color: #FFFFFF !important;
            border: none !important;

            box-shadow:
                0 5px 14px
                rgba(37, 99, 235, 0.25);
        }

        .stButton > button[kind="primary"] * {
            color: #FFFFFF !important;
        }

        /* ================================================================
           INPUTS
           ================================================================ */

        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 9px !important;
            font-size: 0.95rem !important;
        }

        [data-testid="stSelectbox"] > div > div {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border-color: #CBD5E1 !important;
        }

        [data-testid="stSelectbox"] * {
            color: #0F172A !important;
        }

        /* ================================================================
           TABS
           ================================================================ */

        [data-testid="stTabs"] button {
            color: #64748B !important;
            font-weight: 700 !important;
        }

        [data-testid="stTabs"]
        button[aria-selected="true"] {
            color: #1D4ED8 !important;
        }

        /* ================================================================
           DATAFRAME
           ================================================================ */

        [data-testid="stDataFrame"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            overflow: hidden;
        }

        /* ================================================================
           ALERTS
           ================================================================ */

        [data-testid="stAlert"] {
            border-radius: 11px !important;
        }

        /* ================================================================
           EXPANDERS
           ================================================================ */

        [data-testid="stExpander"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 11px !important;
        }

        [data-testid="stExpander"] summary {
            color: #0F172A !important;
            font-weight: 700 !important;
        }

        /* ================================================================
           CAPTIONS
           ================================================================ */

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] * {
            color: #64748B !important;
        }

        /* ================================================================
           DIVIDERS
           ================================================================ */

        hr {
            border-color: #DCE4EC !important;
            margin: 24px 0 !important;
        }

        /* ================================================================
           RESULT CARD
           ================================================================ */

        .prediction-card {
            background: #FFFFFF;
            border: 1px solid #DCE5EE;
            border-radius: 18px;
            padding: 24px;
            margin: 15px 0;

            box-shadow:
                0 7px 20px
                rgba(15, 23, 42, 0.08);
        }

        .prediction-title {
            color: #0F172A !important;
            font-size: 1.25rem;
            font-weight: 800;
        }

        .prediction-subtitle {
            color: #64748B !important;
            font-size: 0.9rem;
        }

        /* ================================================================
           RISK BOXES
           ================================================================ */

        .risk-low {
            background: #ECFDF5;
            border: 1px solid #A7F3D0;
            color: #065F46 !important;
            border-radius: 12px;
            padding: 16px;
            font-weight: 800;
        }

        .risk-medium {
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            color: #92400E !important;
            border-radius: 12px;
            padding: 16px;
            font-weight: 800;
        }

        .risk-high {
            background: #FFF7ED;
            border: 1px solid #FED7AA;
            color: #9A3412 !important;
            border-radius: 12px;
            padding: 16px;
            font-weight: 800;
        }

        .risk-critical {
            background: #FEF2F2;
            border: 1px solid #FECACA;
            color: #991B1B !important;
            border-radius: 12px;
            padding: 16px;
            font-weight: 800;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    # =========================================================================
    # CACHED LOADERS
    # =========================================================================

    @st.cache_resource
    def load_stats():

        path = os.path.join(
            MODELS_DIR,
            "dataset_stats.joblib"
        )

        if os.path.exists(path):
            return joblib.load(path)

        return None

    @st.cache_resource
    def load_results():

        path = os.path.join(
            MODELS_DIR,
            "eval_results.joblib"
        )

        if os.path.exists(path):
            return joblib.load(path)

        return None

    @st.cache_data(show_spinner=False)
    def load_sample():

        if not os.path.exists(
            DATA_PATH
        ):
            return None

        sample = pd.read_csv(
            DATA_PATH,
            nrows=5000
        )

        sample.drop_duplicates(
            inplace=True
        )

        return sample.sample(
            min(
                2000,
                len(sample)
            ),
            random_state=42
        )

    def load_image(
        filename
    ):

        path = os.path.join(
            ASSETS_DIR,
            filename
        )

        if os.path.exists(path):
            return Image.open(path)

        return None

    # =========================================================================
    # SIDEBAR
    # =========================================================================

    with st.sidebar:

        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">
                    🛡️ AI Fraud Detection
                </div>
                <div class="sidebar-brand-subtitle">
                    Machine Learning Risk Analytics System
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "### Navigation"
        )

        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Dataset Analysis",
                "EDA / Visualizations",
                "Model Performance",
                "Fraud Prediction",
                "Batch Prediction",
                "About"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        if models_ready():

            st.success(
                "✓ Models Ready"
            )

        else:

            st.warning(
                "Models not trained"
            )

        st.caption(
            "Dataset: Kaggle Credit Card Fraud Detection"
        )

        st.caption(
            "IBM SkillsBuild Academic Internship"
        )

    # =========================================================================
    # LOAD SAVED RESULTS
    # =========================================================================

    stats = load_stats()
    results = load_results()

    # =========================================================================
    # PAGE 1 - DASHBOARD
    # =========================================================================

    if page == "Dashboard":

        st.markdown(
            """
            <div class="hero">
                <div class="hero-title">
                    🛡️ AI-Powered Financial Fraud Detection
                </div>

                <div class="hero-subtitle">
                    Machine Learning-Based Transaction Risk Analytics
                    and Fraud Detection System
                </div>

                <div class="hero-badge">
                    RANDOM FOREST • XGBOOST • LIGHTGBM • ISOLATION FOREST
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not stats:

            st.warning(
                "Training artifacts are not available yet. "
                "Run the training pipeline first."
            )

            st.code(
                "python SalapareddiLaxmana_FinancialFraudDetection.py",
                language="bash"
            )

            st.stop()

        st.markdown(
            '<div class="section-header">System Overview</div>',
            unsafe_allow_html=True
        )

        # ---------------------------------------------------------------------
        # KPI ROW
        # ---------------------------------------------------------------------

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Total Transactions",
            f"{stats['total_rows']:,}"
        )

        c2.metric(
            "Legitimate",
            f"{stats['legit_count']:,}"
        )

        c3.metric(
            "Fraudulent",
            f"{stats['fraud_count']:,}"
        )

        c4.metric(
            "Average Amount",
            f"${stats['mean_amount']:.2f}"
        )

        c5.metric(
            "Fraud Avg. Amount",
            f"${stats['mean_amount_fraud']:.2f}"
        )

        st.markdown("---")

        # ---------------------------------------------------------------------
        # CHARTS
        # ---------------------------------------------------------------------

        col_a, col_b = st.columns(2)

        with col_a:

            st.markdown(
                "### Class Distribution"
            )

            fig = go.Figure(
                go.Pie(
                    labels=[
                        "Legitimate",
                        "Fraud"
                    ],
                    values=[
                        stats["legit_count"],
                        stats["fraud_count"]
                    ],
                    hole=0.58,
                    marker_colors=[
                        "#2563EB",
                        "#DC2626"
                    ],
                    textinfo="label+percent"
                )
            )

            fig.update_layout(
                height=340,
                margin=dict(
                    t=20,
                    b=20,
                    l=20,
                    r=20
                ),
                paper_bgcolor="white",
                plot_bgcolor="white",
                legend=dict(
                    orientation="h",
                    y=-0.05
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col_b:

            st.markdown(
                "### Average Transaction Amount"
            )

            fig2 = go.Figure(
                go.Bar(
                    x=[
                        "Legitimate",
                        "Fraudulent"
                    ],
                    y=[
                        stats[
                            "mean_amount_legit"
                        ],
                        stats[
                            "mean_amount_fraud"
                        ]
                    ],
                    marker_color=[
                        "#2563EB",
                        "#DC2626"
                    ],
                    text=[
                        "$%.2f"
                        % stats[
                            "mean_amount_legit"
                        ],
                        "$%.2f"
                        % stats[
                            "mean_amount_fraud"
                        ]
                    ],
                    textposition="outside"
                )
            )

            fig2.update_layout(
                height=340,
                yaxis_title="Amount ($)",
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(
                    t=30,
                    b=30,
                    l=30,
                    r=30
                )
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        # ---------------------------------------------------------------------
        # MODEL SUMMARY
        # ---------------------------------------------------------------------

        st.markdown("---")

        st.markdown(
            '<div class="section-header">Model Performance Summary</div>',
            unsafe_allow_html=True
        )

        if results:

            rows = []

            for result in results:

                rows.append(
                    {
                        "Model":
                            result["model_name"],

                        "Precision":
                            f"{result['precision']:.4f}",

                        "Recall":
                            f"{result['recall']:.4f}",

                        "F1 Score":
                            f"{result['f1']:.4f}",

                        "ROC-AUC":
                            f"{result['roc_auc']:.4f}",

                        "PR-AUC":
                            f"{result['pr_auc']:.4f}"
                    }
                )

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

            best = max(
                results,
                key=lambda result:
                result["f1"]
            )

            st.success(
                "Best model by F1 Score: "
                f"**{best['model_name']}** "
                f"(F1 = {best['f1']:.4f})"
            )

    # =========================================================================
    # PAGE 2 - DATASET ANALYSIS
    # =========================================================================

    elif page == "Dataset Analysis":

        st.markdown(
            '<div class="section-header">Dataset Analysis</div>',
            unsafe_allow_html=True
        )

        sample = load_sample()

        tab1, tab2, tab3 = st.tabs(
            [
                "Dataset Overview",
                "Feature Statistics",
                "Sample Data"
            ]
        )

        # ---------------------------------------------------------------------
        # OVERVIEW
        # ---------------------------------------------------------------------

        with tab1:

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### Dataset Facts"
                )

                if stats:

                    facts = [
                        (
                            "Source",
                            "Kaggle — ULB Machine Learning Group"
                        ),
                        (
                            "Transactions",
                            f"{stats['total_rows']:,}"
                        ),
                        (
                            "Duplicate Rows Removed",
                            "1,081"
                        ),
                        (
                            "Features",
                            "Time, V1–V28, Amount"
                        ),
                        (
                            "Target",
                            "Class (0 = Legit, 1 = Fraud)"
                        ),
                        (
                            "Missing Values",
                            "None"
                        ),
                        (
                            "Time Window",
                            "~48 hours"
                        )
                    ]

                    for key, value in facts:

                        st.markdown(
                            f"""
                            <div class="info-card">
                                <div class="info-card-title">
                                    {key}
                                </div>

                                <div class="info-card-text">
                                    {value}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            with col2:

                st.markdown(
                    "### Class Imbalance"
                )

                if stats:

                    imbalance_df = pd.DataFrame(
                        {
                            "Class": [
                                "Legitimate",
                                "Fraud"
                            ],
                            "Count": [
                                stats[
                                    "legit_count"
                                ],
                                stats[
                                    "fraud_count"
                                ]
                            ],
                            "Percentage": [
                                f"{stats['legit_pct']:.4f}%",
                                f"{stats['fraud_pct']:.4f}%"
                            ]
                        }
                    )

                    st.dataframe(
                        imbalance_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.warning(
                        "The dataset is highly imbalanced. "
                        "Approximately 599 legitimate transactions "
                        "occur for every 1 fraudulent transaction. "
                        "Therefore accuracy alone is not sufficient."
                    )

        # ---------------------------------------------------------------------
        # STATISTICS
        # ---------------------------------------------------------------------

        with tab2:

            if sample is not None:

                st.markdown(
                    "### Descriptive Statistics"
                )

                st.caption(
                    "Statistics calculated from a 2,000-row sample."
                )

                display_cols = (
                    ["Time", "Amount"]
                    + [
                        f"V{i}"
                        for i in range(1, 11)
                    ]
                )

                st.dataframe(
                    sample[
                        display_cols
                    ].describe().round(4),
                    use_container_width=True
                )

                st.markdown(
                    "### Transaction Amount by Class"
                )

                amount_stats = pd.DataFrame(
                    {
                        "Legitimate":
                            sample.loc[
                                sample["Class"] == 0,
                                "Amount"
                            ].describe(),

                        "Fraud":
                            sample.loc[
                                sample["Class"] == 1,
                                "Amount"
                            ].describe()
                    }
                ).round(4)

                st.dataframe(
                    amount_stats,
                    use_container_width=True
                )

        # ---------------------------------------------------------------------
        # SAMPLE DATA
        # ---------------------------------------------------------------------

        with tab3:

            if sample is not None:

                st.markdown(
                    "### Transaction Sample"
                )

                show_cols = (
                    ["Time", "Amount", "Class"]
                    + [
                        f"V{i}"
                        for i in range(1, 6)
                    ]
                )

                st.dataframe(
                    sample[
                        show_cols
                    ].head(50),
                    use_container_width=True,
                    hide_index=True
                )

    # =========================================================================
    # PAGE 3 - EDA
    # =========================================================================

    elif page == "EDA / Visualizations":

        st.markdown(
            '<div class="section-header">Exploratory Data Analysis</div>',
            unsafe_allow_html=True
        )

        st.info(
            "Visual analysis of transaction distribution, amount patterns, "
            "time patterns, feature relationships, and class imbalance."
        )

        charts = [
            (
                "eda_class_distribution.png",
                "Class Distribution"
            ),
            (
                "eda_amount_distribution.png",
                "Transaction Amount Distribution"
            ),
            (
                "eda_time_distribution.png",
                "Transaction Time Distribution"
            ),
            (
                "eda_fraud_by_hour.png",
                "Transactions by Hour"
            ),
            (
                "eda_amount_boxplot.png",
                "Amount Boxplot by Class"
            ),
            (
                "eda_top_features.png",
                "Top Features vs Class"
            ),
            (
                "eda_correlation_heatmap.png",
                "Feature Correlation Heatmap"
            )
        ]

        for filename, title in charts:

            st.markdown(
                f"### {title}"
            )

            image = load_image(
                filename
            )

            if image:

                st.image(
                    image,
                    use_container_width=True
                )

            else:

                st.warning(
                    "Chart not available. "
                    "Run the training pipeline first."
                )

            st.markdown("---")

    # =========================================================================
    # PAGE 4 - MODEL PERFORMANCE
    # =========================================================================

    elif page == "Model Performance":

        st.markdown(
            '<div class="section-header">Machine Learning Model Performance</div>',
            unsafe_allow_html=True
        )

        if not results:

            st.warning(
                "Model evaluation results are not available."
            )

            st.stop()

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Metrics",
                "ROC & PR Curves",
                "Confusion Matrices",
                "Comparison"
            ]
        )

        # ---------------------------------------------------------------------
        # METRICS
        # ---------------------------------------------------------------------

        with tab1:

            st.markdown(
                "### Evaluation Metrics"
            )

            st.caption(
                "Metrics are calculated on the held-out stratified test set."
            )

            metrics_df = pd.DataFrame(
                [
                    {
                        "Model":
                            result["model_name"],

                        "Precision":
                            result["precision"],

                        "Recall":
                            result["recall"],

                        "F1 Score":
                            result["f1"],

                        "ROC-AUC":
                            result["roc_auc"],

                        "PR-AUC":
                            result["pr_auc"]
                    }

                    for result in results
                ]
            )

            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                "### Metric Visualization"
            )

            metric_choice = st.selectbox(
                "Select Metric",
                [
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "ROC-AUC",
                    "PR-AUC"
                ]
            )

            metric_key = {
                "Precision":
                    "precision",

                "Recall":
                    "recall",

                "F1 Score":
                    "f1",

                "ROC-AUC":
                    "roc_auc",

                "PR-AUC":
                    "pr_auc"
            }[metric_choice]

            fig = go.Figure(
                go.Bar(
                    x=[
                        result["model_name"]
                        for result in results
                    ],
                    y=[
                        result[metric_key]
                        for result in results
                    ],
                    marker_color=[
                        "#2563EB",
                        "#16A34A",
                        "#F59E0B",
                        "#7C3AED",
                        "#DC2626"
                    ],
                    text=[
                        f"{result[metric_key]:.4f}"
                        for result in results
                    ],
                    textposition="outside"
                )
            )

            fig.update_layout(
                title=metric_choice,
                yaxis_range=[
                    0,
                    1.1
                ],
                height=400,
                paper_bgcolor="white",
                plot_bgcolor="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ---------------------------------------------------------------------
        # ROC / PR
        # ---------------------------------------------------------------------

        with tab2:

            col1, col2 = st.columns(2)

            with col1:

                image = load_image(
                    "eval_roc_curves.png"
                )

                if image:

                    st.image(
                        image,
                        caption="ROC Curves",
                        use_container_width=True
                    )

            with col2:

                image = load_image(
                    "eval_pr_curves.png"
                )

                if image:

                    st.image(
                        image,
                        caption="Precision-Recall Curves",
                        use_container_width=True
                    )

        # ---------------------------------------------------------------------
        # CONFUSION MATRICES
        # ---------------------------------------------------------------------

        with tab3:

            image = load_image(
                "eval_confusion_matrices.png"
            )

            if image:

                st.image(
                    image,
                    use_container_width=True
                )

            st.markdown(
                "### Individual Confusion Matrix"
            )

            selected_model = st.selectbox(
                "Select Model",
                [
                    result["model_name"]
                    for result in results
                ]
            )

            selected_result = next(
                result
                for result in results
                if result["model_name"]
                == selected_model
            )

            cm = np.array(
                selected_result[
                    "confusion_matrix"
                ]
            )

            tn, fp, fn, tp = (
                cm[0, 0],
                cm[0, 1],
                cm[1, 0],
                cm[1, 1]
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "True Negatives",
                f"{tn:,}"
            )

            c2.metric(
                "False Positives",
                f"{fp:,}"
            )

            c3.metric(
                "False Negatives",
                f"{fn:,}"
            )

            c4.metric(
                "True Positives",
                f"{tp:,}"
            )

            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels={
                    "x": "Predicted",
                    "y": "Actual"
                },
                x=[
                    "Legitimate",
                    "Fraud"
                ],
                y=[
                    "Legitimate",
                    "Fraud"
                ],
                color_continuous_scale="Blues"
            )

            fig_cm.update_layout(
                height=400,
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig_cm,
                use_container_width=True
            )

        # ---------------------------------------------------------------------
        # COMPARISON
        # ---------------------------------------------------------------------

        with tab4:

            image = load_image(
                "eval_metrics_comparison.png"
            )

            if image:

                st.image(
                    image,
                    use_container_width=True
                )

    # =========================================================================
    # PAGE 5 - FRAUD PREDICTION
    # =========================================================================

    elif page == "Fraud Prediction":

        st.markdown(
            '<div class="section-header">Single Transaction Fraud Prediction</div>',
            unsafe_allow_html=True
        )

        if not models_ready():

            st.error(
                "Models are not available. "
                "Run the training pipeline first."
            )

            st.code(
                "python SalapareddiLaxmana_FinancialFraudDetection.py",
                language="bash"
            )

            st.stop()

        available_models = (
            list_trained_models()
        )

        col1, col2 = st.columns(
            [2, 3]
        )

        with col1:

            selected_model = st.selectbox(
                "Select Machine Learning Model",
                available_models,
                index=(
                    available_models.index(
                        "XGBoost"
                    )
                    if "XGBoost"
                    in available_models
                    else 0
                )
            )

        with col2:

            st.info(
                "Enter transaction attributes. "
                "V1–V28 are anonymized PCA-transformed features."
            )

        st.markdown(
            "### Transaction Information"
        )

        c1, c2 = st.columns(2)

        with c1:

            time_value = st.number_input(
                "Time (seconds)",
                min_value=0.0,
                max_value=200000.0,
                value=50000.0,
                step=100.0
            )

        with c2:

            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                max_value=30000.0,
                value=100.0,
                step=0.01
            )

        st.markdown(
            "### PCA Features V1 – V28"
        )

        st.caption(
            "Default values are 0.0. "
            "For a demonstration fraud example, you can modify selected features."
        )

        v_features = []

        feature_groups = [
            list(
                range(1, 29)
            )[i:i + 4]
            for i in range(
                0,
                28,
                4
            )
        ]

        for group in feature_groups:

            columns = st.columns(4)

            for column, feature_number in zip(
                columns,
                group
            ):

                value = column.number_input(
                    f"V{feature_number}",
                    value=0.0,
                    format="%.4f",
                    key=f"feature_{feature_number}"
                )

                v_features.append(
                    value
                )

        st.markdown("---")

        if st.button(
            "🔍 Predict Fraud Risk",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing transaction..."
            ):

                try:

                    prediction_result = (
                        predict_single(
                            selected_model,
                            time_value,
                            v_features,
                            amount
                        )
                    )

                except Exception as error:

                    st.error(
                        f"Prediction error: {error}"
                    )

                    st.stop()

            probability = (
                prediction_result[
                    "probability"
                ]
            )

            risk_level = (
                prediction_result[
                    "risk_level"
                ]
            )

            prediction = (
                prediction_result[
                    "prediction"
                ]
            )

            risk_color = (
                prediction_result[
                    "risk_color"
                ]
            )

            st.markdown(
                '<div class="prediction-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### Prediction Result"
            )

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Fraud Score",
                f"{probability:.4f}"
            )

            r2.metric(
                "Risk Percentage",
                f"{probability * 100:.2f}%"
            )

            r3.metric(
                "Risk Level",
                risk_level
            )

            if prediction == 1:

                st.error(
                    "⚠️ FRAUD DETECTED — Transaction flagged for review."
                )

            else:

                st.success(
                    "✓ LEGITIMATE — Transaction classified as non-fraudulent."
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            # -----------------------------------------------------------------
            # GAUGE
            # -----------------------------------------------------------------

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={
                        "suffix": "%",
                        "font": {
                            "size": 32
                        }
                    },
                    gauge={
                        "axis": {
                            "range": [
                                0,
                                100
                            ]
                        },

                        "bar": {
                            "color": risk_color
                        },

                        "steps": [
                            {
                                "range": [
                                    0,
                                    30
                                ],
                                "color":
                                    "#DCFCE7"
                            },
                            {
                                "range": [
                                    30,
                                    60
                                ],
                                "color":
                                    "#FEF3C7"
                            },
                            {
                                "range": [
                                    60,
                                    80
                                ],
                                "color":
                                    "#FFEDD5"
                            },
                            {
                                "range": [
                                    80,
                                    100
                                ],
                                "color":
                                    "#FEE2E2"
                            }
                        ],

                        "threshold": {
                            "line": {
                                "color":
                                    "#991B1B",
                                "width": 3
                            },
                            "thickness": 0.75,
                            "value": 50
                        }
                    },

                    title={
                        "text":
                            "Transaction Risk Score"
                    }
                )
            )

            gauge.update_layout(
                height=320,
                paper_bgcolor="white"
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            # -----------------------------------------------------------------
            # RISK INTERPRETATION
            # -----------------------------------------------------------------

            st.markdown(
                "### Risk Interpretation"
            )

            risk_info = [
                (
                    "LOW RISK",
                    "< 30%",
                    "Low fraud score."
                ),
                (
                    "MEDIUM RISK",
                    "30–60%",
                    "Suspicious transaction requiring review."
                ),
                (
                    "HIGH RISK",
                    "60–80%",
                    "Strong fraud indicators detected."
                ),
                (
                    "CRITICAL RISK",
                    "> 80%",
                    "Very high fraud score requiring investigation."
                )
            ]

            for level, range_text, description in risk_info:

                if level == risk_level:

                    st.success(
                        f"**{level}** ({range_text}) — "
                        f"{description}"
                    )

                else:

                    st.markdown(
                        f"**{level}** ({range_text}) — "
                        f"{description}"
                    )

    # =========================================================================
    # PAGE 6 - BATCH PREDICTION
    # =========================================================================

    elif page == "Batch Prediction":

        st.markdown(
            '<div class="section-header">Batch CSV Fraud Prediction</div>',
            unsafe_allow_html=True
        )

        if not models_ready():

            st.error(
                "Models are not available."
            )

            st.stop()

        available_models = (
            list_trained_models()
        )

        col1, col2 = st.columns(
            [2, 3]
        )

        with col1:

            batch_model = st.selectbox(
                "Select Model",
                available_models,
                index=(
                    available_models.index(
                        "XGBoost"
                    )
                    if "XGBoost"
                    in available_models
                    else 0
                ),
                key="batch_model"
            )

        with col2:

            st.info(
                "CSV must contain Time, V1–V28 and Amount columns."
            )

        uploaded_file = st.file_uploader(
            "Upload Transaction CSV",
            type=["csv"]
        )

        if uploaded_file:

            try:

                uploaded_df = pd.read_csv(
                    uploaded_file
                )

                st.success(
                    "Loaded %s rows × %s columns"
                    % (
                        f"{len(uploaded_df):,}",
                        len(uploaded_df.columns)
                    )
                )

                required_columns = (
                    ["Time"]
                    + [
                        f"V{i}"
                        for i in range(1, 29)
                    ]
                    + ["Amount"]
                )

                missing_columns = [
                    column
                    for column in required_columns
                    if column
                    not in uploaded_df.columns
                ]

                if missing_columns:

                    st.error(
                        "Missing columns: "
                        + str(
                            missing_columns
                        )
                    )

                    st.stop()

                st.markdown(
                    "### Input Preview"
                )

                st.dataframe(
                    uploaded_df.head(10),
                    use_container_width=True
                )

                if st.button(
                    "🚀 Run Batch Prediction",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing uploaded transactions..."
                    ):

                        output_df = (
                            predict_batch_df(
                                batch_model,
                                uploaded_df
                            )
                        )

                    fraud_count = int(
                        (
                            output_df[
                                "prediction"
                            ] == 1
                        ).sum()
                    )

                    legitimate_count = (
                        len(output_df)
                        - fraud_count
                    )

                    fraud_rate = (
                        100
                        * fraud_count
                        / len(output_df)
                    )

                    st.markdown(
                        "### Batch Analysis Summary"
                    )

                    b1, b2, b3, b4 = st.columns(4)

                    b1.metric(
                        "Total Transactions",
                        f"{len(output_df):,}"
                    )

                    b2.metric(
                        "Fraud Flagged",
                        f"{fraud_count:,}"
                    )

                    b3.metric(
                        "Legitimate",
                        f"{legitimate_count:,}"
                    )

                    b4.metric(
                        "Fraud Rate",
                        f"{fraud_rate:.2f}%"
                    )

                    risk_counts = (
                        output_df[
                            "risk_level"
                        ].value_counts()
                    )

                    fig_risk = go.Figure(
                        go.Bar(
                            x=risk_counts.index.tolist(),
                            y=risk_counts.values.tolist(),
                            text=risk_counts.values.tolist(),
                            textposition="outside",
                            marker_color=[
                                "#16A34A",
                                "#D97706",
                                "#EA580C",
                                "#DC2626"
                            ][:len(
                                risk_counts
                            )]
                        )
                    )

                    fig_risk.update_layout(
                        title="Risk Level Distribution",
                        yaxis_title="Transaction Count",
                        height=380,
                        paper_bgcolor="white",
                        plot_bgcolor="white"
                    )

                    st.plotly_chart(
                        fig_risk,
                        use_container_width=True
                    )

                    st.markdown(
                        "### Prediction Results"
                    )

                    display_columns = [
                        "Time",
                        "Amount",
                        "fraud_probability",
                        "prediction",
                        "risk_level"
                    ]

                    if "Class" in output_df.columns:

                        display_columns.insert(
                            2,
                            "Class"
                        )

                    st.dataframe(
                        output_df[
                            display_columns
                        ].head(100),
                        use_container_width=True,
                        hide_index=True
                    )

                    csv_data = (
                        output_df
                        .to_csv(
                            index=False
                        )
                        .encode(
                            "utf-8"
                        )
                    )

                    st.download_button(
                        "⬇️ Download Prediction Results",
                        data=csv_data,
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            except Exception as error:

                st.error(
                    f"Batch prediction error: {error}"
                )

    # =========================================================================
    # PAGE 7 - ABOUT
    # =========================================================================

    elif page == "About":

        st.markdown(
            '<div class="section-header">About the Project</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="hero">
                <div class="hero-title">
                    AI-Powered Financial Fraud Detection
                </div>

                <div class="hero-subtitle">
                    A machine learning system for detecting
                    suspicious credit card transactions and
                    presenting transaction risk analytics.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(
            [2, 1]
        )

        with col1:

            st.markdown(
                "### Project Overview"
            )

            st.write(
                "This project uses the Kaggle Credit Card Fraud "
                "Detection dataset to build a machine learning "
                "pipeline for fraud classification and anomaly "
                "detection."
            )

            st.markdown(
                "### Dataset"
            )

            dataset_table = pd.DataFrame(
                {
                    "Property": [
                        "Source",
                        "Transactions",
                        "Legitimate",
                        "Fraudulent",
                        "Features",
                        "Target",
                        "Missing Values"
                    ],

                    "Value": [
                        "Kaggle — ULB Machine Learning Group",
                        "283,726 after deduplication",
                        "283,253",
                        "473",
                        "Time, V1–V28, Amount",
                        "Class",
                        "None"
                    ]
                }
            )

            st.dataframe(
                dataset_table,
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                "### Machine Learning Pipeline"
            )

            pipeline_steps = [
                "1. Load and clean the dataset",
                "2. Remove exact duplicate transactions",
                "3. Engineer Amount_log and Hour features",
                "4. Perform stratified 80/20 train-test split",
                "5. Apply RobustScaler using training data only",
                "6. Apply SMOTE only to the training set",
                "7. Train five machine learning models",
                "8. Evaluate using Precision, Recall, F1, ROC-AUC and PR-AUC",
                "9. Save trained models and evaluation artifacts",
                "10. Provide interactive Streamlit prediction interface"
            ]

            for step in pipeline_steps:

                st.markdown(
                    f"**{step}**"
                )

            st.markdown(
                "### Technology Stack"
            )

            st.write(
                "Python • pandas • NumPy • scikit-learn • "
                "imbalanced-learn • XGBoost • LightGBM • "
                "Streamlit • Plotly • Matplotlib • Seaborn • joblib"
            )

        with col2:

            st.markdown(
                "### Model Results"
            )

            if results:

                for result in results:

                    with st.expander(
                        result["model_name"]
                    ):

                        st.metric(
                            "F1 Score",
                            f"{result['f1']:.4f}"
                        )

                        st.write(
                            "Precision: "
                            f"{result['precision']:.4f}"
                        )

                        st.write(
                            "Recall: "
                            f"{result['recall']:.4f}"
                        )

                        st.write(
                            "ROC-AUC: "
                            f"{result['roc_auc']:.4f}"
                        )

                        st.write(
                            "PR-AUC: "
                            f"{result['pr_auc']:.4f}"
                        )

            st.markdown(
                "### Run Commands"
            )

            st.code(
                "python SalapareddiLaxmana_FinancialFraudDetection.py",
                language="bash"
            )

            st.code(
                "streamlit run SalapareddiLaxmana_FinancialFraudDetection.py",
                language="bash"
            )


# =============================================================================
# SECTION 8 - ENTRY POINT
# =============================================================================

_streamlit_mode = (
    "streamlit"
    in sys.modules
)

if _streamlit_mode:

    run_streamlit_app()

elif __name__ == "__main__":

    if (
        len(sys.argv) > 1
        and sys.argv[1] == "--app"
    ):

        run_streamlit_app()

    else:

        run_training_pipeline()
