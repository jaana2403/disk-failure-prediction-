"""Streamlit demo: enter a drive's SMART values and get a failure prediction.

Run with:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MODEL_PATH, SMART_FEATURES
from src.features import add_engineered_features

st.set_page_config(page_title="Disk Failure Predictor", page_icon="💽")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# Sensible default inputs and slider ranges per SMART attribute.
INPUT_RANGES = {
    "smart_5_raw": (0, 500, 0),
    "smart_187_raw": (0, 200, 0),
    "smart_188_raw": (0, 200, 0),
    "smart_197_raw": (0, 300, 0),
    "smart_198_raw": (0, 300, 0),
    "smart_9_raw": (0, 80_000, 15_000),
    "smart_194_raw": (20, 70, 30),
    "smart_193_raw": (0, 600_000, 100_000),
}

st.title("💽 Hard Drive Failure Predictor")
st.write(
    "Enter a drive's SMART sensor readings to estimate its probability of "
    "failure. This is a learning demo trained on SMART-style data."
)

model = load_model()
if model is None:
    st.error("No trained model found. Run `python src/train.py` first, then reload.")
    st.stop()

st.subheader("SMART attributes")
values = {}
cols = st.columns(2)
for i, (feature, label) in enumerate(SMART_FEATURES.items()):
    low, high, default = INPUT_RANGES[feature]
    with cols[i % 2]:
        values[feature] = st.number_input(
            f"{label} ({feature})",
            min_value=float(low),
            max_value=float(high),
            value=float(default),
        )

if st.button("Predict", type="primary"):
    row = add_engineered_features(pd.DataFrame([values]))
    proba = float(model.predict_proba(row)[:, 1][0])

    st.subheader("Result")
    st.metric("Failure probability", f"{proba:.1%}")
    if proba >= 0.5:
        st.error("⚠️ High risk — this drive is likely to fail. Consider replacing it.")
    else:
        st.success("✅ Low risk — this drive looks healthy.")
    st.progress(min(proba, 1.0))
