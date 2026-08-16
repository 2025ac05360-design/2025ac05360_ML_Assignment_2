import io
import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(
    page_title="Sensorless Drive Diagnosis",
    page_icon="⚙️",
    layout="wide",
)

MODEL_DIR = "model"
N_FEATURES = 48
FEATURE_COLS = [f"feature_{i+1}" for i in range(N_FEATURES)]
TARGET_COL = "target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

@st.cache_resource(show_spinner=False)
def load_scaler():
    return joblib.load(f"{MODEL_DIR}/scaler.pkl")


@st.cache_resource(show_spinner=False)
def load_model(filename: str):
    return joblib.load(f"{MODEL_DIR}/{filename}")


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def metric_cards(metrics: dict):
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.4f}")


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.title("⚙️ Controls")
    st.caption("Sensorless Drive Diagnosis — fault classification demo")

    model_name = st.selectbox("Model", list(MODEL_FILES.keys()), index=4)

    uploaded_file = st.file_uploader(
        "Upload test data (CSV)",
        type=["csv"],
        help="Upload the sampled test_data.csv from this repo, or any CSV with "
             "the same 48 feature columns (and optionally a 'target' column).",
    )

    st.divider()
    st.markdown(
        "**About**\n\n"
        "Diagnoses electric motor drive condition (healthy vs. 10 fault types) "
        "from 48 statistical features of the phase current signal. "
        "See `README.md` for the full model comparison and dataset details."
    )

# ----------------------------------------------------------------- main ---
st.title("Sensorless Drive Diagnosis — Fault Classification")
st.caption(
    "Upload test data, pick a model, and see how it performs — "
    "accuracy, AUC, precision, recall, F1, MCC, and the confusion matrix."
)

if uploaded_file is None:
    st.info("⬅️ Upload a CSV from the sidebar to get started (try `test_data.csv` from the repo).")
    st.stop()

try:
    raw_bytes = uploaded_file.getvalue()
    data = pd.read_csv(io.BytesIO(raw_bytes))
except Exception as e:
    st.error(f"Could not read that file as CSV: {e}")
    st.stop()

missing_cols = [c for c in FEATURE_COLS if c not in data.columns]
if missing_cols:
    st.error(
        f"Uploaded file is missing {len(missing_cols)} expected feature column(s), "
        f"e.g. {missing_cols[:5]}. Make sure you're uploading the 48-feature "
        f"`test_data.csv` produced by `model/train_models.ipynb`."
    )
    st.stop()

has_target = TARGET_COL in data.columns

with st.expander("Preview uploaded data", expanded=False):
    st.dataframe(data.head(20), width='stretch')
    st.caption(f"{data.shape[0]} rows × {data.shape[1]} columns")

# ---------------------------------------------------------- load + infer --
try:
    scaler = load_scaler()
    model = load_model(MODEL_FILES[model_name])
except FileNotFoundError as e:
    st.error(
        f"Couldn't find a saved model/scaler file ({e}). "
        f"Run `model/train_models.ipynb` first to generate the .pkl files."
    )
    st.stop()

X = data[FEATURE_COLS]
X_scaled = scaler.transform(X)

y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)

st.subheader(f"Results — {model_name}")

if has_target:
    y_true = data[TARGET_COL]
    metrics = compute_metrics(y_true, y_pred, y_proba)
    metric_cards(metrics)

    col1, col2 = st.columns([1, 1])
    with col1:
        labels = sorted(y_true.unique())
        fig = plot_confusion_matrix(y_true, y_pred, labels)
        st.pyplot(fig, width='stretch')

    with col2:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T.round(4)
        st.markdown("**Classification report**")
        st.dataframe(report_df, width='stretch')
else:
    st.warning(
        "No `target` column found in the uploaded file — showing predictions only "
        "(metrics and the confusion matrix need true labels to compare against)."
    )

results = data.copy()
results["predicted_class"] = y_pred

st.markdown("**Predictions**")
st.dataframe(
    results[[*(["target"] if has_target else []), "predicted_class"]].head(50),
    width='stretch',
)

csv_out = results.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download predictions as CSV",
    data=csv_out,
    file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
    mime="text/csv",
)
