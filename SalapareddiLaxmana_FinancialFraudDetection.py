"""
FinancialFraudDetection.py
==========================
AI-Powered Financial Fraud Detection and Risk Analytics System
Using Machine Learning on the Kaggle Credit Card Fraud Detection Dataset

Run modes:
  python FinancialFraudDetection.py        -> train all models, save artifacts
  streamlit run FinancialFraudDetection.py -> launch interactive web application
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import sys
import json
import time
import warnings
import pickle

warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
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
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH    = "creditcard.csv"
MODELS_DIR   = "fraud_models"
ASSETS_DIR   = "fraud_assets"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# =============================================================================
# SECTION 1 — DATA LOADING & PREPROCESSING
# =============================================================================

def load_and_clean(path=DATA_PATH):
    """Load CSV, remove 1,081 exact duplicate rows, return clean DataFrame."""
    print("[1/7] Loading dataset ...")
    df = pd.read_csv(path)
    original = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("      Rows before dedup : %d" % original)
    print("      Duplicates removed : %d" % (original - len(df)))
    print("      Rows after  dedup  : %d" % len(df))
    return df


def feature_engineer(df):
    """Add Amount_log and Hour derived features."""
    print("[2/7] Feature engineering ...")
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Hour"]       = (df["Time"] // 3600) % 24
    return df


def split_and_scale(df):
    """
    Stratified 80/20 split -> fit RobustScaler on train only -> apply SMOTE to train only.
    Returns X_train_bal, y_train_bal, X_test, y_test, feature_cols, scaler.
    """
    print("[3/7] Train/test split (stratified 80/20) ...")
    feature_cols = [c for c in df.columns if c not in ("Class", "Time", "Amount")]
    X = df[feature_cols]
    y = df["Class"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("      Train: %d | Fraud: %d" % (len(X_tr), y_tr.sum()))
    print("      Test : %d | Fraud: %d" % (len(X_te), y_te.sum()))

    print("[4/7] Scaling (RobustScaler, fit on train only) ...")
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Save artifacts
    joblib.dump(scaler,       os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.joblib"))

    # Keep unbalanced scaled train for Isolation Forest
    X_tr_orig = X_tr_s.copy()
    y_tr_orig = y_tr.copy()

    print("[5/7] SMOTE on training set only ...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_s, y_tr)
    print("      Fraud after SMOTE: %d | Total: %d" % (y_tr_bal.sum(), len(y_tr_bal)))

    return (X_tr_bal, y_tr_bal, X_te_s, y_te,
            X_tr_orig, y_tr_orig, feature_cols, scaler)


# =============================================================================
# SECTION 2 — EDA VISUALISATIONS
# =============================================================================

def generate_eda(df):
    """Generate and save 7 EDA charts to fraud_assets/."""
    print("[6/7] Generating EDA visualisations ...")
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # 1. Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution", fontsize=14, fontweight="bold")
    counts = df["Class"].value_counts()
    labels = ["Legitimate", "Fraud"]
    vals   = [counts.get(0, 0), counts.get(1, 0)]
    total  = sum(vals)
    colors = ["#2196F3", "#F44336"]
    axes[0].bar(labels, vals, color=colors, width=0.5, edgecolor="white")
    for i, (v, lbl) in enumerate(zip(vals, labels)):
        axes[0].text(i, v + 500, "%d\n(%.2f%%)" % (v, 100*v/total),
                     ha="center", fontsize=10)
    axes[0].set_title("Transaction Counts")
    axes[0].set_ylabel("Count")
    axes[1].pie(vals, labels=labels, colors=colors, autopct="%1.3f%%",
                startangle=90, wedgeprops={"edgecolor":"white","linewidth":2})
    axes[1].set_title("Class Proportion")
    fig.savefig(os.path.join(ASSETS_DIR, "eda_class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Amount distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transaction Amount by Class", fontsize=14, fontweight="bold")
    for ax, log in zip(axes, [False, True]):
        for cls, lbl, col in [(0,"Legitimate","#2196F3"),(1,"Fraud","#F44336")]:
            v = df.loc[df["Class"]==cls, "Amount"]
            if log: v = np.log1p(v)
            ax.hist(v, bins=60, alpha=0.6, label=lbl, color=col, edgecolor="none")
        ax.set_xlabel("log1p(Amount)" if log else "Amount ($)")
        ax.set_ylabel("Count")
        ax.set_title("Log-Transformed" if log else "Raw Amount")
        ax.legend()
    fig.savefig(os.path.join(ASSETS_DIR, "eda_amount_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Time distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Time Distribution (seconds from dataset start)", fontsize=14, fontweight="bold")
    for ax, cls, lbl, col in [(axes[0],0,"Legitimate","#2196F3"),(axes[1],1,"Fraud","#F44336")]:
        ax.hist(df.loc[df["Class"]==cls,"Time"], bins=48, color=col, alpha=0.85, edgecolor="none")
        ax.set_title(lbl)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count")
    fig.savefig(os.path.join(ASSETS_DIR, "eda_time_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Fraud by hour
    df2 = df.copy()
    df2["Hour"] = (df2["Time"] // 3600) % 24
    hourly = df2.groupby(["Hour","Class"]).size().unstack(fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transactions by Hour of Day", fontsize=14, fontweight="bold")
    if 0 in hourly.columns:
        axes[0].bar(hourly.index, hourly[0], color="#2196F3", alpha=0.85)
        axes[0].set_title("Legitimate")
        axes[0].set_xlabel("Hour")
        axes[0].set_ylabel("Count")
    if 1 in hourly.columns:
        axes[1].bar(hourly.index, hourly[1], color="#F44336", alpha=0.85)
        axes[1].set_title("Fraudulent")
        axes[1].set_xlabel("Hour")
        axes[1].set_ylabel("Count")
    fig.savefig(os.path.join(ASSETS_DIR, "eda_fraud_by_hour.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Amount boxplot by class
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Amount Boxplot by Class", fontsize=14, fontweight="bold")
    for ax, log in zip(axes, [False, True]):
        dl = df.loc[df["Class"]==0,"Amount"]
        df_ = df.loc[df["Class"]==1,"Amount"]
        if log: dl, df_ = np.log1p(dl), np.log1p(df_)
        ax.boxplot([dl, df_], patch_artist=True,
                   boxprops=dict(facecolor="#E3F2FD"),
                   medianprops=dict(color="#F44336", linewidth=2),
                   flierprops=dict(marker=".", markersize=1, alpha=0.3))
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Legitimate","Fraud"])
        ax.set_title("log1p(Amount)" if log else "Raw Amount ($)")
    fig.savefig(os.path.join(ASSETS_DIR, "eda_amount_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6. Top features boxplot
    sample = df.sample(min(10000, len(df)), random_state=42)
    v_cols = [c for c in sample.columns if c.startswith("V")]
    corr   = sample[v_cols+["Class"]].corr()["Class"].drop("Class").abs()
    top10  = corr.nlargest(10).index.tolist()
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle("Top 10 Features (Correlation with Class)", fontsize=14, fontweight="bold")
    axes = axes.flatten()
    for ax, feat in zip(axes, top10):
        ax.boxplot([df.loc[df["Class"]==0,feat], df.loc[df["Class"]==1,feat]],
                   patch_artist=True,
                   boxprops=dict(facecolor="#E3F2FD"),
                   medianprops=dict(color="#F44336",linewidth=2),
                   flierprops=dict(marker=".",markersize=1,alpha=0.3))
        ax.set_xticks([1,2])
        ax.set_xticklabels(["Legit","Fraud"])
        ax.set_title(feat, fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "eda_top_features.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 7. Correlation heatmap
    s2 = df.sample(min(5000, len(df)), random_state=42).copy()
    s2["Amount_log"] = np.log1p(s2["Amount"])
    v2  = [c for c in s2.columns if c.startswith("V")]
    corr2 = s2[v2+["Amount_log","Class"]].corr()
    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr2, dtype=bool))
    sns.heatmap(corr2, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                ax=ax, linewidths=0.3, annot=False, square=True,
                cbar_kws={"shrink":0.6})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(ASSETS_DIR, "eda_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("      EDA charts saved to %s/" % ASSETS_DIR)


# =============================================================================
# SECTION 3 — MODEL TRAINING
# =============================================================================

def train_models(X_tr_bal, y_tr_bal, X_tr_orig, y_tr_orig):
    """Train all 5 models. Return dict of {name: model}."""
    print("[7/7] Training models ...")
    models = {}

    # 1. Logistic Regression
    t0 = time.time()
    print("      [1/5] Logistic Regression ...")
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                             class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(X_tr_bal, y_tr_bal)
    models["Logistic Regression"] = lr
    print("            done %.1fs" % (time.time()-t0))

    # 2. Random Forest
    t0 = time.time()
    print("      [2/5] Random Forest ...")
    rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, min_samples_leaf=2,
                                 class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr_bal, y_tr_bal)
    models["Random Forest"] = rf
    print("            done %.1fs" % (time.time()-t0))

    # 3. XGBoost
    t0 = time.time()
    print("      [3/5] XGBoost ...")
    neg, pos = int((y_tr_bal==0).sum()), int((y_tr_bal==1).sum())
    xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=neg/pos if pos else 1.0,
                               eval_metric="logloss", random_state=RANDOM_STATE,
                               n_jobs=-1, verbosity=0)
    xgb_m.fit(X_tr_bal, y_tr_bal)
    models["XGBoost"] = xgb_m
    print("            done %.1fs" % (time.time()-t0))

    # 4. LightGBM
    t0 = time.time()
    print("      [4/5] LightGBM ...")
    lgb_m = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                scale_pos_weight=neg/pos if pos else 1.0,
                                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgb_m.fit(X_tr_bal, y_tr_bal)
    models["LightGBM"] = lgb_m
    print("            done %.1fs" % (time.time()-t0))

    # 5. Isolation Forest (trained on original unbalanced train)
    t0 = time.time()
    print("      [5/5] Isolation Forest ...")
    iso = IsolationForest(n_estimators=200, contamination=0.002,
                          random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_tr_orig)
    models["Isolation Forest"] = iso
    print("            done %.1fs" % (time.time()-t0))

    # Save all models
    for name, m in models.items():
        safe_name = name.lower().replace(" ","_")
        joblib.dump(m, os.path.join(MODELS_DIR, "%s.joblib" % safe_name))

    return models


# =============================================================================
# SECTION 4 — MODEL EVALUATION
# =============================================================================

def evaluate_models(models, X_te, y_te):
    """Evaluate all models on test set. Return list of result dicts."""
    SUPERVISED = {"Logistic Regression","Random Forest","XGBoost","LightGBM"}
    all_results = []

    for name, model in models.items():
        is_iso = (name == "Isolation Forest")
        if is_iso:
            raw = model.predict(X_te)
            y_pred = np.where(raw == -1, 1, 0)
            scores = -model.decision_function(X_te)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]

        cm = confusion_matrix(y_te, y_pred)
        r = {
            "model_name": name,
            "precision": round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
            "roc_auc":   round(float(roc_auc_score(y_te, y_prob)), 4),
            "pr_auc":    round(float(average_precision_score(y_te, y_prob)), 4),
            "confusion_matrix": cm.tolist(),
            "y_prob": y_prob.tolist(),
        }
        all_results.append(r)
        print("  %-22s F1=%.4f ROC-AUC=%.4f PR-AUC=%.4f" % (name, r["f1"], r["roc_auc"], r["pr_auc"]))

    return all_results


def generate_eval_charts(all_results, y_te):
    """Save ROC, PR, CM, and comparison charts."""
    colors = ["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336"]
    y_arr  = np.array(y_te)

    # ROC curves
    fig, ax = plt.subplots(figsize=(9,7))
    for r, c in zip(all_results, colors):
        fpr, tpr, _ = roc_curve(y_arr, np.array(r["y_prob"]))
        ax.plot(fpr, tpr, label="%s (%.4f)" % (r["model_name"], r["roc_auc"]), color=c, lw=2)
    ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.4)
    fig.savefig(os.path.join(ASSETS_DIR,"eval_roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PR curves
    baseline = y_arr.mean()
    fig, ax = plt.subplots(figsize=(9,7))
    for r, c in zip(all_results, colors):
        prec, rec, _ = precision_recall_curve(y_arr, np.array(r["y_prob"]))
        ax.plot(rec, prec, label="%s (%.4f)" % (r["model_name"], r["pr_auc"]), color=c, lw=2)
    ax.axhline(baseline, color="k", linestyle="--", lw=1, label="Baseline")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.4)
    fig.savefig(os.path.join(ASSETS_DIR,"eval_pr_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrices
    n = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(15,10))
    axes = axes.flatten()
    for ax, r in zip(axes, all_results):
        cm = np.array(r["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(r["model_name"], fontsize=11, fontweight="bold")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Legit","Fraud"]); ax.set_yticklabels(["Legit","Fraud"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=12,
                        color="white" if cm[i,j] > cm.max()/2 else "black")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR,"eval_confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Metrics comparison
    metrics = ["precision","recall","f1","roc_auc","pr_auc"]
    x = np.arange(len(metrics)); w = 0.15
    offsets = np.linspace(-(len(all_results)-1)/2,(len(all_results)-1)/2,len(all_results))*w
    fig, ax = plt.subplots(figsize=(13,6))
    for r, off, c in zip(all_results, offsets, colors):
        ax.bar(x+off, [r[m] for m in metrics], w, label=r["model_name"], color=c, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision","Recall","F1","ROC-AUC","PR-AUC"])
    ax.set_ylim(0,1.1); ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(True, axis="y", alpha=0.4)
    fig.savefig(os.path.join(ASSETS_DIR,"eval_metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# SECTION 5 — INFERENCE HELPERS
# =============================================================================

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest":        "random_forest.joblib",
    "XGBoost":              "xgboost.joblib",
    "LightGBM":             "lightgbm.joblib",
    "Isolation Forest":     "isolation_forest.joblib",
}
SUPERVISED = {"Logistic Regression","Random Forest","XGBoost","LightGBM"}


def get_risk_level(p):
    if p < 0.30: return "LOW RISK",      "#4CAF50"
    if p < 0.60: return "MEDIUM RISK",   "#FF9800"
    if p < 0.80: return "HIGH RISK",     "#FF5722"
    return            "CRITICAL RISK",   "#F44336"


def _load_inference_artifacts(model_name):
    scaler  = joblib.load(os.path.join(MODELS_DIR,"scaler.joblib"))
    feat_cols = joblib.load(os.path.join(MODELS_DIR,"feature_cols.joblib"))
    mfile   = MODEL_FILES[model_name]
    model   = joblib.load(os.path.join(MODELS_DIR, mfile))
    return scaler, feat_cols, model


def predict_single(model_name, time_val, v_features, amount):
    """Return {probability, risk_level, risk_color, prediction}."""
    scaler, feat_cols, model = _load_inference_artifacts(model_name)
    amount_log = np.log1p(amount)
    hour       = (time_val // 3600) % 24
    feat_dict  = {"V%d" % i: v_features[i-1] for i in range(1,29)}
    feat_dict["Amount_log"] = amount_log
    feat_dict["Hour"]       = hour
    raw    = np.array([feat_dict[c] for c in feat_cols]).reshape(1,-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaled = scaler.transform(raw)
    if model_name in SUPERVISED:
        prob = float(model.predict_proba(scaled)[0,1])
        pred = int(model.predict(scaled)[0])
    else:
        score = float(-model.decision_function(scaled)[0])
        prob  = float(np.clip(score,0,1))
        pred  = 1 if model.predict(scaled)[0]==-1 else 0
    lvl, col = get_risk_level(prob)
    return {"probability": prob, "risk_level": lvl, "risk_color": col, "prediction": pred}


def predict_batch_df(model_name, df_in):
    """Add fraud_probability, prediction, risk_level columns to DataFrame."""
    scaler, feat_cols, model = _load_inference_artifacts(model_name)
    df = df_in.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Hour"]       = (df["Time"] // 3600) % 24
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: %s" % str(missing))
    X = df[feat_cols].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_s = scaler.transform(X)
    if model_name in SUPERVISED:
        probs = model.predict_proba(X_s)[:,1]
        preds = model.predict(X_s)
    else:
        scores = -model.decision_function(X_s)
        probs  = np.clip(scores,0,1)
        preds  = np.where(model.predict(X_s)==-1,1,0)
    df["fraud_probability"] = np.round(probs,4)
    df["prediction"]        = preds
    df["risk_level"]        = df["fraud_probability"].apply(lambda p: get_risk_level(p)[0])
    return df


def list_trained_models():
    return [n for n, f in MODEL_FILES.items()
            if os.path.exists(os.path.join(MODELS_DIR, f))]


def models_ready():
    required = ["scaler.joblib","feature_cols.joblib","random_forest.joblib"]
    return all(os.path.exists(os.path.join(MODELS_DIR,f)) for f in required)


# =============================================================================
# SECTION 6 — TRAINING PIPELINE (CLI mode)
# =============================================================================

def run_training_pipeline():
    t_total = time.time()
    print("=" * 60)
    print("  AI Financial Fraud Detection -- Training Pipeline")
    print("=" * 60)

    df = load_and_clean()
    df = feature_engineer(df)
    generate_eda(df)

    X_tr_bal, y_tr_bal, X_te, y_te, X_tr_orig, y_tr_orig, feat_cols, scaler = split_and_scale(df)
    models = train_models(X_tr_bal, y_tr_bal, X_tr_orig, y_tr_orig)

    print("\nEvaluating models on test set ...")
    all_results = evaluate_models(models, X_te, y_te)
    generate_eval_charts(all_results, y_te)

    best = max(all_results, key=lambda r: r["f1"])

    # Save stats & results
    stats = {
        "total_rows":        int(len(df)),
        "fraud_count":       int(df["Class"].sum()),
        "legit_count":       int((df["Class"]==0).sum()),
        "fraud_pct":         float(100*df["Class"].sum()/len(df)),
        "legit_pct":         float(100*(df["Class"]==0).sum()/len(df)),
        "mean_amount":       float(df["Amount"].mean()),
        "mean_amount_fraud": float(df.loc[df["Class"]==1,"Amount"].mean()),
        "mean_amount_legit": float(df.loc[df["Class"]==0,"Amount"].mean()),
        "max_amount":        float(df["Amount"].max()),
        "best_model":        best["model_name"],
    }
    joblib.dump(stats,       os.path.join(MODELS_DIR,"dataset_stats.joblib"))
    joblib.dump(all_results, os.path.join(MODELS_DIR,"eval_results.joblib"))
    with open(os.path.join(MODELS_DIR,"eval_summary.json"),"w") as f:
        json.dump([{k:v for k,v in r.items() if k!="y_prob"} for r in all_results], f, indent=2)

    print("\n" + "=" * 60)
    print("  Training complete in %.1f minutes" % ((time.time()-t_total)/60))
    print("  Best model (F1): %s = %.4f" % (best["model_name"], best["f1"]))
    print("  Launch app : streamlit run FinancialFraudDetection.py")
    print("=" * 60)


# =============================================================================
# SECTION 7 — STREAMLIT APPLICATION
# =============================================================================

def run_streamlit_app():
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from PIL import Image

    # ── Page config ───────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Financial Fraud Detection",
        page_icon="shield",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── CSS ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] { background: #1a1f2e; }
    section[data-testid="stSidebar"] * { color: #e8eaf6 !important; }
    div[data-testid="metric-container"] {
        background:#f8f9fa; border:1px solid #e9ecef;
        border-radius:8px; padding:14px;
    }
    .section-header {
        font-size:1.3rem; font-weight:700; color:#1a1f2e;
        border-left:4px solid #3b82f6; padding-left:10px; margin-bottom:1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Cached loaders ────────────────────────────────────────────────────────
    @st.cache_resource
    def _load_stats():
        p = os.path.join(MODELS_DIR,"dataset_stats.joblib")
        return joblib.load(p) if os.path.exists(p) else None

    @st.cache_resource
    def _load_eval():
        p = os.path.join(MODELS_DIR,"eval_results.joblib")
        return joblib.load(p) if os.path.exists(p) else None

    @st.cache_data(show_spinner=False)
    def _load_sample():
        if not os.path.exists(DATA_PATH): return None
        df = pd.read_csv(DATA_PATH, nrows=5000)
        df.drop_duplicates(inplace=True)
        return df.sample(min(2000,len(df)), random_state=42)

    def _img(name):
        p = os.path.join(ASSETS_DIR, name)
        return Image.open(p) if os.path.exists(p) else None

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Financial Fraud Detection")
        st.markdown("---")
        page = st.radio("Navigation", [
            "Dashboard",
            "Dataset Analysis",
            "EDA / Visualizations",
            "Model Performance",
            "Fraud Prediction",
            "Batch Prediction",
            "About",
        ], label_visibility="collapsed")
        st.markdown("---")
        if models_ready():
            st.success("Models Ready")
        else:
            st.error("Run: python FinancialFraudDetection.py")
        st.caption("Kaggle Credit Card Fraud Detection")

    stats   = _load_stats()
    results = _load_eval()

    # =========================================================================
    # PAGE: DASHBOARD
    # =========================================================================
    if page == "Dashboard":
        st.markdown('<div class="section-header">Dashboard — Overview</div>', unsafe_allow_html=True)

        if not stats:
            st.warning("Run `python FinancialFraudDetection.py` first.")
            st.stop()

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Transactions", f"{stats['total_rows']:,}")
        c2.metric("Legitimate", f"{stats['legit_count']:,}", f"{stats['legit_pct']:.2f}%")
        c3.metric("Fraudulent", f"{stats['fraud_count']:,}", f"{stats['fraud_pct']:.4f}%")
        c4.metric("Mean Amount (All)", f"${stats['mean_amount']:.2f}")
        c5.metric("Mean Amount (Fraud)", f"${stats['mean_amount_fraud']:.2f}")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Class Distribution")
            fig = go.Figure(go.Pie(
                labels=["Legitimate","Fraud"],
                values=[stats["legit_count"], stats["fraud_count"]],
                hole=0.55, marker_colors=["#2196F3","#F44336"],
                textinfo="label+percent",
            ))
            fig.update_layout(margin=dict(t=0,b=0), height=300,
                              legend=dict(orientation="h",y=-0.1))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("#### Mean Amount by Class")
            fig2 = go.Figure(go.Bar(
                x=["Legitimate","Fraudulent"],
                y=[stats["mean_amount_legit"], stats["mean_amount_fraud"]],
                marker_color=["#2196F3","#F44336"],
                text=[f"${stats['mean_amount_legit']:.2f}", f"${stats['mean_amount_fraud']:.2f}"],
                textposition="outside",
            ))
            fig2.update_layout(margin=dict(t=20,b=0), height=300, yaxis_title="Amount ($)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        if results:
            st.markdown("#### Model Performance Summary")
            rows = [{"Model":r["model_name"],"Precision":f"{r['precision']:.4f}",
                     "Recall":f"{r['recall']:.4f}","F1":f"{r['f1']:.4f}",
                     "ROC-AUC":f"{r['roc_auc']:.4f}","PR-AUC":f"{r['pr_auc']:.4f}"}
                    for r in results]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            best = max(results, key=lambda r: r["f1"])
            st.success(f"Best Model by F1: **{best['model_name']}** (F1 = {best['f1']:.4f})")

    # =========================================================================
    # PAGE: DATASET ANALYSIS
    # =========================================================================
    elif page == "Dataset Analysis":
        st.markdown('<div class="section-header">Dataset Analysis</div>', unsafe_allow_html=True)
        sample = _load_sample()

        tab1, tab2, tab3 = st.tabs(["Overview", "Feature Statistics", "Sample Data"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Dataset Facts")
                if stats:
                    for k,v in [
                        ("Source","Kaggle -- ULB Machine Learning Group"),
                        ("Total Rows (after dedup)",f"{stats['total_rows']:,}"),
                        ("Duplicate Rows Removed","1,081"),
                        ("Features","Time, V1-V28 (PCA), Amount"),
                        ("Target","Class (0=Legit, 1=Fraud)"),
                        ("Missing Values","None"),
                        ("Time Window","~48 hours"),
                    ]:
                        st.markdown(f"**{k}:** {v}")
            with col2:
                st.markdown("#### Class Imbalance")
                if stats:
                    st.markdown(f"""
| Class | Count | % |
|---|---|---|
| Legitimate | {stats['legit_count']:,} | {stats['legit_pct']:.4f}% |
| Fraud | {stats['fraud_count']:,} | {stats['fraud_pct']:.4f}% |
                    """)
                    st.warning("578 legitimate transactions per 1 fraud. "
                               "Accuracy is not a valid metric — use F1, ROC-AUC, PR-AUC.")

        with tab2:
            if sample is not None:
                st.markdown("#### Descriptive Statistics (2,000-row sample)")
                display_cols = ["Time","Amount"] + [f"V{i}" for i in range(1,11)]
                st.dataframe(sample[display_cols].describe().round(4), use_container_width=True)

                st.markdown("#### Amount by Class")
                amt_stats = pd.DataFrame({
                    "Legitimate": sample.loc[sample["Class"]==0,"Amount"].describe(),
                    "Fraud":      sample.loc[sample["Class"]==1,"Amount"].describe(),
                }).round(4)
                st.dataframe(amt_stats, use_container_width=True)

        with tab3:
            if sample is not None:
                show = ["Time","Amount","Class"]+[f"V{i}" for i in range(1,6)]
                st.dataframe(sample[show].head(50), use_container_width=True, hide_index=True)

    # =========================================================================
    # PAGE: EDA / VISUALIZATIONS
    # =========================================================================
    elif page == "EDA / Visualizations":
        st.markdown('<div class="section-header">EDA / Visual Analytics</div>', unsafe_allow_html=True)

        charts = [
            ("eda_class_distribution.png",   "Class Distribution"),
            ("eda_amount_distribution.png",  "Amount Distribution by Class"),
            ("eda_time_distribution.png",    "Time Distribution"),
            ("eda_fraud_by_hour.png",        "Fraud Count by Hour of Day"),
            ("eda_amount_boxplot.png",       "Amount Boxplot by Class"),
            ("eda_top_features.png",         "Top 10 Features vs Class"),
            ("eda_correlation_heatmap.png",  "Feature Correlation Heatmap"),
        ]
        for filename, title in charts:
            st.markdown(f"#### {title}")
            img = _img(filename)
            if img:
                st.image(img, use_container_width=True)
            else:
                st.info(f"Chart not found ({filename}). Run training first.")
            st.markdown("---")

    # =========================================================================
    # PAGE: MODEL PERFORMANCE
    # =========================================================================
    elif page == "Model Performance":
        st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

        if not results:
            st.warning("Run training first.")
            st.stop()

        tab1, tab2, tab3, tab4 = st.tabs(["Metrics","ROC & PR","Confusion Matrices","Comparison"])

        with tab1:
            st.markdown("#### Test Set Metrics (56,746 transactions | 95 fraud)")
            df_m = pd.DataFrame([{
                "Model": r["model_name"],
                "Precision": r["precision"], "Recall": r["recall"],
                "F1": r["f1"], "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"],
            } for r in results]).set_index("Model")
            st.dataframe(
                df_m.style.highlight_max(axis=0,
                    props="background-color:#e8f5e9; color:#1b5e20;"),
                use_container_width=True,
            )

            st.markdown("---")
            metric_choice = st.selectbox("Select metric",
                ["Precision","Recall","F1","ROC-AUC","PR-AUC"])
            mkey = {"Precision":"precision","Recall":"recall","F1":"f1",
                    "ROC-AUC":"roc_auc","PR-AUC":"pr_auc"}[metric_choice]
            fig_b = go.Figure(go.Bar(
                x=[r["model_name"] for r in results],
                y=[r[mkey] for r in results],
                marker_color=["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336"],
                text=[f"{r[mkey]:.4f}" for r in results], textposition="outside",
            ))
            fig_b.update_layout(yaxis_range=[0,1.1], height=380,
                                 title=f"{metric_choice} — All Models")
            st.plotly_chart(fig_b, use_container_width=True)

        with tab2:
            c1,c2 = st.columns(2)
            with c1:
                img = _img("eval_roc_curves.png")
                if img: st.image(img, caption="ROC Curves", use_container_width=True)
            with c2:
                img = _img("eval_pr_curves.png")
                if img: st.image(img, caption="PR Curves", use_container_width=True)

        with tab3:
            img = _img("eval_confusion_matrices.png")
            if img: st.image(img, use_container_width=True)

            st.markdown("---")
            sel = st.selectbox("Detail view", [r["model_name"] for r in results])
            for r in results:
                if r["model_name"] == sel:
                    cm = np.array(r["confusion_matrix"])
                    tn,fp,fn,tp = cm[0,0],cm[0,1],cm[1,0],cm[1,1]
                    cc1,cc2,cc3,cc4 = st.columns(4)
                    cc1.metric("True Negatives",  f"{tn:,}")
                    cc2.metric("False Positives", f"{fp:,}")
                    cc3.metric("False Negatives", f"{fn:,}")
                    cc4.metric("True Positives",  f"{tp:,}")
                    fig_cm = px.imshow(cm, text_auto=True,
                        labels=dict(x="Predicted",y="Actual"),
                        x=["Legit","Fraud"], y=["Legit","Fraud"],
                        color_continuous_scale="Blues")
                    fig_cm.update_layout(height=350)
                    st.plotly_chart(fig_cm, use_container_width=True)

        with tab4:
            img = _img("eval_metrics_comparison.png")
            if img: st.image(img, use_container_width=True)

    # =========================================================================
    # PAGE: FRAUD PREDICTION
    # =========================================================================
    elif page == "Fraud Prediction":
        st.markdown('<div class="section-header">Single Transaction Fraud Prediction</div>',
                    unsafe_allow_html=True)

        if not models_ready():
            st.error("Run `python FinancialFraudDetection.py` first.")
            st.stop()

        available = list_trained_models()
        col_m, col_i = st.columns([2,3])
        with col_m:
            model_name = st.selectbox("Select Model", available,
                index=available.index("XGBoost") if "XGBoost" in available else 0)
        with col_i:
            st.info("Enter transaction features. V1-V28 are PCA-transformed components.")

        st.markdown("---")
        c_t, c_a = st.columns(2)
        with c_t:
            time_val = st.number_input("Time (0–172792 seconds)",
                min_value=0.0, max_value=200000.0, value=50000.0, step=100.0)
        with c_a:
            amount = st.number_input("Amount ($)",
                min_value=0.0, max_value=30000.0, value=100.0, step=0.01)

        st.markdown("#### PCA Features V1 – V28")
        st.caption("Default = 0.0 (dataset mean for PCA features). "
                   "For a fraud example: set V14 = -9.47, Amount = 1.00, Time = 406.")

        v_features = []
        groups = [list(range(1,29))[i:i+4] for i in range(0,28,4)]
        for grp in groups:
            cols = st.columns(4)
            for col, vi in zip(cols, grp):
                val = col.number_input(f"V{vi}", value=0.0, format="%.4f", key=f"v{vi}")
                v_features.append(val)

        st.markdown("---")
        if st.button("Predict Fraud Risk", type="primary", use_container_width=True):
            with st.spinner("Running prediction ..."):
                try:
                    result = predict_single(model_name, time_val, v_features, amount)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            prob  = result["probability"]
            level = result["risk_level"]
            pred  = result["prediction"]
            color = result["risk_color"]

            st.markdown("---")
            st.markdown("### Prediction Result")
            r1,r2,r3 = st.columns(3)
            r1.metric("Fraud Probability", f"{prob:.4f}  ({prob*100:.2f}%)")
            with r2:
                if pred == 1:
                    st.error("FRAUD DETECTED")
                else:
                    st.success("LEGITIMATE")
            r3.metric("Risk Level", level)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                number={"suffix":"%","font":{"size":28}},
                gauge={
                    "axis":{"range":[0,100]},
                    "bar":{"color":color},
                    "steps":[
                        {"range":[0,30],"color":"#e8f5e9"},
                        {"range":[30,60],"color":"#fff3e0"},
                        {"range":[60,80],"color":"#fbe9e7"},
                        {"range":[80,100],"color":"#ffebee"},
                    ],
                    "threshold":{"line":{"color":"red","width":3},"thickness":0.75,"value":50},
                },
                title={"text":"Fraud Risk Score","font":{"size":18}},
            ))
            fig_g.update_layout(height=280)
            st.plotly_chart(fig_g, use_container_width=True)

            st.markdown("#### Risk Interpretation")
            risk_info = [
                ("LOW RISK",      "< 30%",   "Very likely legitimate."),
                ("MEDIUM RISK",   "30-60%",  "Suspicious. Review recommended."),
                ("HIGH RISK",     "60-80%",  "Strong fraud indicators. Flag for review."),
                ("CRITICAL RISK", "> 80%",   "Very likely fraudulent. Block and investigate."),
            ]
            for lvl, rng, desc in risk_info:
                marker = ">>>" if lvl==level else "   "
                bold   = "**" if lvl==level else ""
                st.markdown(f"{marker} {bold}{lvl} ({rng}): {desc}{bold}")

    # =========================================================================
    # PAGE: BATCH PREDICTION
    # =========================================================================
    elif page == "Batch Prediction":
        st.markdown('<div class="section-header">Batch CSV Prediction</div>', unsafe_allow_html=True)

        if not models_ready():
            st.error("Run training first.")
            st.stop()

        available = list_trained_models()
        col_m2, col_i2 = st.columns([2,3])
        with col_m2:
            model_b = st.selectbox("Select Model", available,
                index=available.index("XGBoost") if "XGBoost" in available else 0,
                key="batch_model")
        with col_i2:
            st.info("CSV must have columns: **Time, V1-V28, Amount**. "
                    "Results will add: fraud_probability, prediction, risk_level.")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                st.success(f"Loaded {len(df_up):,} rows x {len(df_up.columns)} columns")

                required = ["Time"]+[f"V{i}" for i in range(1,29)]+["Amount"]
                missing  = [c for c in required if c not in df_up.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                    st.stop()

                st.markdown("#### Preview")
                st.dataframe(df_up.head(5), use_container_width=True)

                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner("Predicting ..."):
                        out = predict_batch_df(model_b, df_up)
                    fraud_n = (out["prediction"]==1).sum()
                    b1,b2,b3,b4 = st.columns(4)
                    b1.metric("Total", f"{len(out):,}")
                    b2.metric("Fraud", f"{fraud_n:,}")
                    b3.metric("Legitimate", f"{len(out)-fraud_n:,}")
                    b4.metric("Fraud Rate", f"{100*fraud_n/len(out):.2f}%")

                    rc = out["risk_level"].value_counts()
                    fig_r = go.Figure(go.Bar(
                        x=rc.index.tolist(), y=rc.values.tolist(),
                        text=rc.values.tolist(), textposition="outside",
                        marker_color=["#4CAF50","#FF9800","#FF5722","#F44336"][:len(rc)],
                    ))
                    fig_r.update_layout(yaxis_title="Count", height=350,
                                        title="Risk Level Distribution")
                    st.plotly_chart(fig_r, use_container_width=True)

                    show_cols = ["Time","Amount","fraud_probability","prediction","risk_level"]
                    if "Class" in out.columns: show_cols.insert(2,"Class")
                    st.dataframe(out[show_cols].head(100), use_container_width=True,
                                 hide_index=True)
                    st.download_button("Download Results CSV",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name="fraud_predictions.csv", mime="text/csv",
                        use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    # =========================================================================
    # PAGE: ABOUT
    # =========================================================================
    elif page == "About":
        st.markdown('<div class="section-header">About This System</div>', unsafe_allow_html=True)

        col_ab, col_kpi = st.columns([2,1])
        with col_ab:
            st.markdown("""
### AI-Powered Financial Fraud Detection

Built on the **Kaggle Credit Card Fraud Detection** dataset — 284,807 real European
credit card transactions from September 2013 (48-hour window).

---
#### Dataset
| Property | Value |
|---|---|
| Source | Kaggle — ULB Machine Learning Group |
| Transactions | 283,726 (after dedup) |
| Legitimate | 283,253 (99.83%) |
| Fraudulent | 473 (0.17%) |
| Features | Time, V1–V28 (PCA), Amount |
| Missing Values | None |

---
#### ML Pipeline
1. Remove 1,081 duplicate rows (before split)
2. Feature engineering: `Amount_log`, `Hour`
3. Stratified 80/20 train/test split
4. RobustScaler — fit on training set only
5. SMOTE — applied to training fold only
6. Train 5 models with class-imbalance handling
7. Evaluate: Precision, Recall, F1, ROC-AUC, PR-AUC

---
#### Technology
`Python 3.11` · `scikit-learn` · `XGBoost` · `LightGBM`  
`imbalanced-learn` · `Streamlit` · `Plotly` · `pandas` · `NumPy`
            """)

        with col_kpi:
            st.markdown("#### Actual Results")
            if results:
                for r in results:
                    with st.expander(r["model_name"]):
                        st.write(f"**F1:** {r['f1']:.4f}")
                        st.write(f"**ROC-AUC:** {r['roc_auc']:.4f}")
                        st.write(f"**PR-AUC:** {r['pr_auc']:.4f}")
                        st.write(f"**Precision:** {r['precision']:.4f}")
                        st.write(f"**Recall:** {r['recall']:.4f}")
            st.markdown("---")
            st.code("python FinancialFraudDetection.py", language="bash")
            st.code("streamlit run FinancialFraudDetection.py", language="bash")


# =============================================================================
# SECTION 8 — ENTRY POINT
# =============================================================================

# Detect whether we are running inside Streamlit
_streamlit_mode = "streamlit" in sys.modules

if _streamlit_mode:
    run_streamlit_app()
elif __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--app":
        run_streamlit_app()
    else:
        run_training_pipeline()
