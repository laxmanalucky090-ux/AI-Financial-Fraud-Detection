import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "creditcard.csv"
MODELS_DIR = "fraud_models"
ASSETS_DIR = "fraud_assets"
RANDOM_STATE = 42
TEST_SIZE = 0.20
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


def clean_dataset(df):
    df = df.copy()
    duplicates = int(df.duplicated().sum())
    if duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df, duplicates


def feature_engineering(df):
    df = df.copy()
    if "Amount" in df.columns:
        df["Amount_log"] = np.log1p(df["Amount"])
    if "Time" in df.columns:
        df["Hour"] = (df["Time"] / 3600) % 24
    return df


def load_dataset(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError("creditcard.csv is not included in deployment. Place the Kaggle dataset in the project folder to train locally.")
    return pd.read_csv(path)


def get_feature_columns(df):
    return [c for c in df.columns if c != "Class"]


def prepare_data(df):
    df = feature_engineering(df)
    feature_columns = get_feature_columns(df)
    X = df[feature_columns]
    y = df["Class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.joblib"))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_cols.joblib"))
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train_resampled, y_train_resampled


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1),
        "Isolation Forest": IsolationForest(n_estimators=300, contamination=0.002, random_state=RANDOM_STATE, n_jobs=-1),
    }


def model_filename(name):
    return name.lower().replace(" ", "_").replace("-", "_") + ".joblib"


def train_models(X_train_scaled, X_train_resampled, y_train_resampled):
    trained = {}
    for name, model in build_models().items():
        if name == "Isolation Forest":
            model.fit(X_train_scaled)
        else:
            model.fit(X_train_resampled, y_train_resampled)
        trained[name] = model
        joblib.dump(model, os.path.join(MODELS_DIR, model_filename(name)))
    return trained


def get_prediction_scores(model, X):
    if isinstance(model, IsolationForest):
        raw = model.decision_function(X)
        scores = -raw
        lo, hi = scores.min(), scores.max()
        probabilities = np.zeros_like(scores) if hi == lo else (scores - lo) / (hi - lo)
        predictions = (model.predict(X) == -1).astype(int)
        return predictions, probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int), probabilities
    predictions = model.predict(X).astype(int)
    return predictions, predictions.astype(float)


def evaluate_models(trained, X_test_scaled, y_test):
    results = {}
    for name, model in trained.items():
        pred, prob = get_prediction_scores(model, X_test_scaled)
        results[name] = {
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "pr_auc": float(average_precision_score(y_test, prob)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
    joblib.dump(results, os.path.join(MODELS_DIR, "eval_results.joblib"))
    summary = [{"model_name": name, **value} for name, value in results.items()]
    with open(os.path.join(MODELS_DIR, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return results


def save_dataset_stats(df, duplicate_count):
    total = len(df)
    fraud = int(df["Class"].sum())
    legit = int((df["Class"] == 0).sum())
    stats = {
        "original_rows": int(total + duplicate_count),
        "clean_rows": int(total),
        "total_rows": int(total),
        "columns": int(df.shape[1]),
        "fraud_count": fraud,
        "legitimate_count": legit,
        "legit_count": legit,
        "fraud_rate": float(fraud / total * 100),
        "fraud_pct": float(fraud / total * 100),
        "legit_pct": float(legit / total * 100),
        "duplicate_count": int(duplicate_count),
        "mean_amount": float(df["Amount"].mean()),
        "mean_amount_fraud": float(df.loc[df["Class"] == 1, "Amount"].mean()),
        "mean_amount_legit": float(df.loc[df["Class"] == 0, "Amount"].mean()),
        "max_amount": float(df["Amount"].max()),
        "best_model": "Random Forest",
    }
    joblib.dump(stats, os.path.join(MODELS_DIR, "dataset_stats.joblib"))
    with open(os.path.join(MODELS_DIR, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    df.sample(min(2000, len(df)), random_state=RANDOM_STATE).to_csv(os.path.join(MODELS_DIR, "sample_data.csv"), index=False)
    return stats


def load_saved_stats():
    path = os.path.join(MODELS_DIR, "dataset_stats.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    path = os.path.join(MODELS_DIR, "dataset_stats.joblib")
    if os.path.exists(path):
        try:
            raw = joblib.load(path)
            total = int(raw.get("total_rows", raw.get("clean_rows", 283726)))
            fraud = int(raw.get("fraud_count", 473))
            legit = int(raw.get("legit_count", raw.get("legitimate_count", total - fraud)))
            return {**raw, "total_rows": total, "fraud_count": fraud, "legit_count": legit, "legitimate_count": legit,
                    "fraud_pct": float(raw.get("fraud_pct", raw.get("fraud_rate", fraud / total * 100))),
                    "legit_pct": float(raw.get("legit_pct", legit / total * 100)),
                    "mean_amount": float(raw.get("mean_amount", 88.47268731)),
                    "mean_amount_fraud": float(raw.get("mean_amount_fraud", 123.87186047)),
                    "mean_amount_legit": float(raw.get("mean_amount_legit", 88.41357475)),
                    "duplicate_count": int(raw.get("duplicate_count", 1081))}
        except Exception:
            pass
    return None


def load_saved_results():
    path = os.path.join(MODELS_DIR, "eval_summary.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return {x["model_name"]: {k: v for k, v in x.items() if k != "model_name"} for x in data}
            return data
        except Exception:
            pass
    path = os.path.join(MODELS_DIR, "eval_results.joblib")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass
    return None


def load_saved_model(name):
    path = os.path.join(MODELS_DIR, model_filename(name))
    return joblib.load(path) if os.path.exists(path) else None


def load_saved_scaler():
    path = os.path.join(MODELS_DIR, "scaler.joblib")
    return joblib.load(path) if os.path.exists(path) else None


def load_feature_columns():
    for filename in ("feature_columns.joblib", "feature_cols.joblib"):
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                pass
    return None


def models_ready():
    return load_saved_scaler() is not None and load_feature_columns() is not None and load_saved_model("Random Forest") is not None


def risk_level(p):
    if p < 0.30: return "LOW RISK"
    if p < 0.60: return "MEDIUM RISK"
    if p < 0.80: return "HIGH RISK"
    return "CRITICAL RISK"


def risk_description(level):
    return {
        "LOW RISK": "The transaction has a relatively low predicted fraud probability.",
        "MEDIUM RISK": "The transaction requires additional monitoring.",
        "HIGH RISK": "The transaction shows elevated fraud risk and may require review.",
        "CRITICAL RISK": "The transaction has a very high predicted fraud probability and should be investigated.",
    }.get(level, "Risk level unavailable.")


def predict_transaction(input_data, model_name="Random Forest"):
    scaler, features, model = load_saved_scaler(), load_feature_columns(), load_saved_model(model_name)
    if scaler is None or features is None or model is None:
        raise FileNotFoundError("Required model artifacts are missing.")
    df = feature_engineering(pd.DataFrame([input_data]))
    for col in features:
        if col not in df.columns: df[col] = 0
    X = scaler.transform(df[features])
    pred, prob = get_prediction_scores(model, X)
    p = float(prob[0]); level = risk_level(p)
    return {"prediction": int(pred[0]), "probability": p, "risk_level": level, "description": risk_description(level)}


def batch_predict(df, model_name="Random Forest"):
    scaler, features, model = load_saved_scaler(), load_feature_columns(), load_saved_model(model_name)
    if scaler is None or features is None or model is None: raise FileNotFoundError("Required model artifacts are missing.")
    work = feature_engineering(df.copy())
    for col in features:
        if col not in work.columns: work[col] = 0
    X = scaler.transform(work[features])
    pred, prob = get_prediction_scores(model, X)
    out = df.copy(); out["Fraud_Prediction"] = pred; out["Fraud_Probability"] = prob; out["Risk_Level"] = [risk_level(float(p)) for p in prob]
    return out


def run_training_pipeline():
    print("Loading dataset...")
    df = load_dataset(); df, duplicates = clean_dataset(df)
    save_dataset_stats(df, duplicates)
    Xtr, Xte, ytr, yte, Xres, yres = prepare_data(df)
    trained = train_models(Xtr, Xres, yres)
    results = evaluate_models(trained, Xte, yte)
    for name, r in results.items(): print(f"{name}: Precision={r['precision']:.4f}, Recall={r['recall']:.4f}, F1={r['f1']:.4f}, ROC-AUC={r['roc_auc']:.4f}, PR-AUC={r['pr_auc']:.4f}")
    print("Training completed successfully.")


def run_streamlit_app():
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go

    st.set_page_config(page_title="AI Financial Fraud Detection", page_icon="💳", layout="wide")
    st.markdown("""
    <style>
    html,body,.stApp,[data-testid="stAppViewContainer"]{background:#0F172A!important;color:#F8FAFC!important}
    h1,h2,h3,h4,h5,h6{color:#FFFFFF!important} p,span,label,li{color:#E2E8F0!important}
    [data-testid="stMetric"]{background:#1E293B!important;border:1px solid #334155!important;border-radius:12px!important;padding:16px!important}
    [data-testid="stMetricLabel"] *{color:#94A3B8!important}[data-testid="stMetricValue"] *{color:#FFFFFF!important;font-weight:800!important}
    .info-card{background:#1E293B!important;border:1px solid #334155!important;border-radius:12px!important;padding:20px!important;margin-bottom:16px!important}
    .info-card-title{color:#38BDF8!important;font-weight:800!important}.info-card-text{color:#E2E8F0!important;line-height:1.55}
    .main-title{color:#FFFFFF;font-size:34px;font-weight:900}.main-subtitle{color:#94A3B8;font-size:16px;margin-bottom:24px}
    [data-testid="stSidebar"]{background:#090D16!important}[data-testid="stSidebar"] *{color:#F8FAFC!important}
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<div style='text-align:center;padding:10px 0 20px;border-bottom:1px solid #334155;margin-bottom:20px'><div style='font-size:36px'>💳</div><div style='font-size:18px;font-weight:800;color:#FFFFFF'>Fraud Detection AI</div><div style='font-size:12px;color:#38BDF8'>Financial Risk Analytics</div></div>", unsafe_allow_html=True)
        page = st.radio("NAVIGATION", ["Dashboard","Dataset Analysis","EDA / Visualizations","Model Performance","Fraud Prediction","Batch Prediction","About"])
        if models_ready(): st.success("Models Ready")
        else: st.warning("Model artifacts incomplete")

    stats = load_saved_stats() or {"total_rows":283726,"fraud_count":473,"legit_count":283253,"fraud_pct":0.1667101358,"legit_pct":99.8332898642,"mean_amount":88.47268731,"mean_amount_fraud":123.87186047,"mean_amount_legit":88.41357475,"duplicate_count":1081}
    results = load_saved_results()

    sample = None
    sample_path = os.path.join(MODELS_DIR,"sample_data.csv")
    if os.path.exists(sample_path):
        try: sample = pd.read_csv(sample_path)
        except Exception: pass

    if page == "Dashboard":
        st.markdown('<div class="main-title">AI-Powered Financial Fraud Detection</div><div class="main-subtitle">Machine Learning System for Transaction Risk Analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-card"><div class="info-card-title">🔐 Intelligent Financial Transaction Monitoring</div><div class="info-card-text">This application uses machine learning to identify potentially fraudulent credit card transactions, compare multiple classification and anomaly detection algorithms, and provide transaction-level risk analysis.</div></div>', unsafe_allow_html=True)
        c1,c2,c3,c4=st.columns(4); c1.metric("Transactions",f"{stats['total_rows']:,}"); c2.metric("Fraud Cases",f"{stats['fraud_count']:,}"); c3.metric("Legitimate",f"{stats['legit_count']:,}"); c4.metric("Fraud Rate",f"{stats['fraud_pct']:.3f}%")
        a,b=st.columns(2)
        with a:
            fig=go.Figure(go.Pie(labels=["Legitimate","Fraud"],values=[stats['legit_count'],stats['fraud_count']],hole=.55)); fig.update_layout(template="plotly_dark",height=300); st.plotly_chart(fig,use_container_width=True)
        with b:
            fig=go.Figure(go.Bar(x=["Legitimate","Fraudulent"],y=[stats['mean_amount_legit'],stats['mean_amount_fraud']],text=[f"${stats['mean_amount_legit']:.2f}",f"${stats['mean_amount_fraud']:.2f}"],textposition="outside")); fig.update_layout(template="plotly_dark",height=300,yaxis_title="Amount ($)"); st.plotly_chart(fig,use_container_width=True)
        if results:
            rows=[{"Model":n,"Precision":r["precision"],"Recall":r["recall"],"F1 Score":r["f1"],"ROC-AUC":r["roc_auc"],"PR-AUC":r["pr_auc"]} for n,r in results.items()]
            st.markdown("## Model Summary"); st.dataframe(pd.DataFrame(rows).style.format({c:"{:.4f}" for c in ["Precision","Recall","F1 Score","ROC-AUC","PR-AUC"]}),use_container_width=True,hide_index=True)

    elif page == "Dataset Analysis":
        st.markdown('<div class="main-title">Dataset Analysis</div><div class="main-subtitle">Credit Card Fraud Detection Dataset</div>', unsafe_allow_html=True)
        t1,t2,t3,t4=st.tabs(["Overview","Dataset Facts","Class Imbalance","Data Preview"])
        with t1:
            st.markdown('<div class="info-card"><div class="info-card-title">📌 Dataset Overview</div><div class="info-card-text">The project uses the Kaggle Credit Card Fraud Detection dataset with anonymized numerical transaction features and a binary fraud target.</div></div>',unsafe_allow_html=True)
            st.write("**Rows after deduplication:**",f"{stats['total_rows']:,}"); st.write("**Columns:** 31"); st.write("**Features:** Time, V1–V28, Amount"); st.write("**Target:** Class (0 = Legitimate, 1 = Fraud)"); st.write("**Missing values:** None"); st.write("**Time window:** Approximately 48 hours")
        with t2:
            c1,c2,c3=st.columns(3); c1.metric("Transactions",f"{stats['total_rows']:,}"); c2.metric("Fraud Transactions",f"{stats['fraud_count']:,}"); c3.metric("Legitimate Transactions",f"{stats['legit_count']:,}"); c4,c5,c6=st.columns(3); c4.metric("Fraud Rate",f"{stats['fraud_pct']:.4f}%"); c5.metric("Average Amount",f"${stats['mean_amount']:.2f}"); c6.metric("Fraud Avg. Amount",f"${stats['mean_amount_fraud']:.2f}")
            st.dataframe(pd.DataFrame([["Source","Kaggle – ULB Machine Learning Group"],["Duplicate rows removed",f"{stats.get('duplicate_count',1081):,}"],["Features","Time, V1–V28, Amount"],["Target","Class (0 = Legitimate, 1 = Fraud)"],["Missing values","None"]],columns=["Item","Value"]),use_container_width=True,hide_index=True)
        with t3:
            d=pd.DataFrame({"Class":["Legitimate","Fraud"],"Count":[stats['legit_count'],stats['fraud_count']],"Percentage":[stats['legit_pct'],stats['fraud_pct']]}); st.dataframe(d.style.format({"Percentage":"{:.4f}%"}),use_container_width=True,hide_index=True); fig=px.bar(d,x="Class",y="Count",text="Count",title="Transaction Class Distribution"); fig.update_layout(template="plotly_dark"); st.plotly_chart(fig,use_container_width=True)
        with t4:
            if sample is not None: st.dataframe(sample.head(50),use_container_width=True,hide_index=True); st.caption("Preview uses a small deployment-safe sample. The full Kaggle CSV is not committed to GitHub.")
            else: st.info("Data preview is unavailable in the deployed environment because the full Kaggle CSV is intentionally not committed. Dataset facts remain available.")

    elif page == "EDA / Visualizations":
        st.markdown('<div class="main-title">Exploratory Data Analysis</div><div class="main-subtitle">Visual exploration of transaction behavior</div>', unsafe_allow_html=True)
        images=[("Class Distribution",["eda_class_distribution.png","class_distribution.png"]),("Amount Distribution",["eda_amount_distribution.png","amount_distribution.png"]),("Amount by Class",["eda_amount_boxplot.png","amount_by_class.png"]),("Transactions by Hour",["eda_fraud_by_hour.png","transactions_by_hour.png"]),("Correlation Heatmap",["eda_correlation_heatmap.png","correlation_heatmap.png"]),("Top Features",["eda_top_features.png"])]
        for title,candidates in images:
            path=next((os.path.join(ASSETS_DIR,f) for f in candidates if os.path.exists(os.path.join(ASSETS_DIR,f))),None)
            if path: st.markdown(f"### {title}"); st.image(path,use_container_width=True)

    elif page == "Model Performance":
        st.markdown('<div class="main-title">Model Performance</div><div class="main-subtitle">Comparative evaluation of fraud detection algorithms</div>', unsafe_allow_html=True)
        if not results: st.error("Model evaluation results are unavailable. Ensure fraud_models/eval_summary.json is committed."); st.stop()
        rows=[{"Model":n,"Precision":r["precision"],"Recall":r["recall"],"F1 Score":r["f1"],"ROC-AUC":r["roc_auc"],"PR-AUC":r["pr_auc"]} for n,r in results.items()]
        st.dataframe(pd.DataFrame(rows).style.format({c:"{:.4f}" for c in ["Precision","Recall","F1 Score","ROC-AUC","PR-AUC"]}),use_container_width=True,hide_index=True)
        for title,candidates in [("ROC Curves",["eval_roc_curves.png","roc_curves.png"]),("Precision-Recall Curves",["eval_pr_curves.png","precision_recall_curves.png"]),("Metrics Comparison",["eval_metrics_comparison.png","model_metrics.png"]),("Confusion Matrices",["eval_confusion_matrices.png"])]:
            path=next((os.path.join(ASSETS_DIR,f) for f in candidates if os.path.exists(os.path.join(ASSETS_DIR,f))),None)
            if path: st.markdown(f"### {title}"); st.image(path,use_container_width=True)
        st.markdown('<div class="info-card"><div class="info-card-title">📌 Why multiple metrics?</div><div class="info-card-text">Fraud detection is an imbalanced classification problem. Precision, Recall, F1 Score, ROC-AUC and PR-AUC provide a more informative evaluation than accuracy alone.</div></div>',unsafe_allow_html=True)

    elif page == "Fraud Prediction":
        st.markdown('<div class="main-title">Fraud Prediction</div><div class="main-subtitle">Analyze an individual transaction</div>', unsafe_allow_html=True)
        selected=st.selectbox("Select Model",["Logistic Regression","Random Forest","XGBoost","LightGBM","Isolation Forest"],index=1); amount=st.number_input("Transaction Amount",min_value=0.0,value=100.0,step=1.0); tv=st.number_input("Transaction Time",min_value=0.0,value=10000.0,step=100.0); st.markdown("### PCA Features"); inputs={}; cols=st.columns(4)
        for i in range(1,29):
            with cols[(i-1)%4]: inputs[f"V{i}"]=st.number_input(f"V{i}",value=0.0,format="%.6f",key=f"single_v_{i}")
        if st.button("🔍 Analyze Transaction",use_container_width=True):
            try:
                r=predict_transaction({"Time":tv,"Amount":amount,**inputs},selected); p=r["probability"]; pred=r["prediction"]; level=r["risk_level"]; c1,c2,c3=st.columns(3); c1.metric("Fraud Probability",f"{p*100:.2f}%"); c2.metric("Prediction","FRAUD" if pred else "LEGITIMATE"); c3.metric("Risk Level",level); st.error(f"⚠️ {level}: Potential fraudulent transaction detected.") if pred else st.success(f"✅ {level}: Transaction classified as legitimate."); st.info(r["description"])
            except Exception as e: st.error(f"Prediction failed: {e}")

    elif page == "Batch Prediction":
        st.markdown('<div class="main-title">Batch Prediction</div><div class="main-subtitle">Analyze multiple transactions at once</div>', unsafe_allow_html=True); selected=st.selectbox("Select Model",["Logistic Regression","Random Forest","XGBoost","LightGBM","Isolation Forest"],index=1,key="batch_model"); uploaded=st.file_uploader("Upload CSV File",type=["csv"])
        if uploaded:
            try:
                batch=pd.read_csv(uploaded); st.dataframe(batch.head(20),use_container_width=True)
                if st.button("🚀 Run Batch Prediction",use_container_width=True):
                    out=batch_predict(batch,selected); st.success("Batch prediction completed successfully."); st.dataframe(out,use_container_width=True); st.download_button("⬇️ Download Results CSV",out.to_csv(index=False).encode("utf-8"),"fraud_predictions.csv","text/csv",use_container_width=True)
            except Exception as e: st.error(f"Batch prediction failed: {e}")
        else: st.info("Upload a CSV file containing transaction features to generate batch fraud predictions.")

    elif page == "About":
        st.markdown('<div class="main-title">About the Project</div><div class="main-subtitle">AI-Powered Financial Fraud Detection & Risk Analytics System</div>', unsafe_allow_html=True)
        cards=[("🎯 Project Objective","Develop a machine learning-based fraud detection system capable of identifying suspicious financial transactions under severe class imbalance."),("🛠️ Technology Stack","Python • Pandas • NumPy • Scikit-learn • XGBoost • LightGBM • Imbalanced-learn • Matplotlib • Seaborn • Streamlit"),("🔬 Machine Learning Approach","The project combines supervised classification models with an unsupervised anomaly detection approach. SMOTE is applied only to training data to address severe class imbalance while preserving the original test distribution."),("📈 Evaluation Strategy","Model performance is evaluated using Precision, Recall, F1 Score, ROC-AUC, PR-AUC and Confusion Matrices instead of relying only on accuracy."),("⚠️ Important Dataset Limitation","The dataset contains anonymized PCA features and does not contain customer names, merchant categories, locations or demographic details. The application therefore does not claim customer profiling or merchant-specific fraud analysis.")]
        for title,text in cards: st.markdown(f'<div class="info-card"><div class="info-card-title">{title}</div><div class="info-card-text">{text}</div></div>',unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;color:#64748B;padding:25px">AI-Powered Financial Fraud Detection & Risk Analytics System<br>Developed as an academic machine learning project</div>',unsafe_allow_html=True)


if __name__ == "__main__":
    if "streamlit" in sys.modules:
        run_streamlit_app()
    else:
        run_training_pipeline()
