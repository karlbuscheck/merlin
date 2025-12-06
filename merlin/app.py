import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="MERLIN", page_icon="🔮")

st.title("MERLIN 🔮")
st.caption("The what-if engine for YouTube creators")

# -----------------------------------------------------------------------------
# 2. LOAD ARTIFACTS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Paths are relative to this script
    current_dir = Path(__file__).parent
    models_dir = current_dir / "models"
    
    # Define paths
    paths = {
        "views_model": models_dir / "xgb_views_model.pkl",
        "subs_model":  models_dir / "xgb_subs_model.pkl",
        "subs_scaler": models_dir / "xgb_subs_target_scaler.pkl",
        "views_feats": models_dir / "features_views.json",
        "subs_feats":  models_dir / "features_subs.json",
    }
    
    # Load everything
    artifacts = {}
    
    # Load Models and Scalers
    artifacts["views_model"] = joblib.load(paths["views_model"])
    artifacts["subs_model"] = joblib.load(paths["subs_model"])
    artifacts["subs_scaler"] = joblib.load(paths["subs_scaler"])
    
    # Load Feature Metadata
    with open(paths["views_feats"]) as f:
        artifacts["views_schema"] = json.load(f)["feature_cols"]
        
    with open(paths["subs_feats"]) as f:
        artifacts["subs_schema"] = json.load(f)["feature_cols"]
        
    # Load Sentence Transformer (cached)
    artifacts["embedder"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    return artifacts

with st.spinner("Loading models..."):
    artifacts = load_artifacts()

# -----------------------------------------------------------------------------
# 3. USER INPUTS
# -----------------------------------------------------------------------------
with st.form("prediction_form"):
    title = st.text_input("Video Title", value="How the 49ers can win the Super Bowl in Santa Clara")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        duration_minutes = st.number_input("Duration (minutes)", min_value=1, value=12, step=1)
        duration_seconds = duration_minutes * 60
        
    with col2:
        # Mon=0, Sun=6
        days = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        day_of_week = st.selectbox("Day of Week", options=list(days.keys()), format_func=lambda x: days[x], index=2)
        
    with col3:
        # Jan=0, Dec=11
        months = {i: v for i, v in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
        month = st.selectbox("Month", options=list(months.keys()), format_func=lambda x: months[x], index=9)
        
    submit = st.form_submit_button("Predict Performance")

# -----------------------------------------------------------------------------
# 4. PREDICTION LOGIC
# -----------------------------------------------------------------------------
if submit and title:
    # A. Generate Embeddings
    vec = artifacts["embedder"].encode([title], normalize_embeddings=True)[0]
    
    # B. Assemble Feature Dictionary
    feat_dict = {
        "duration_seconds": float(duration_seconds),
        "day_of_week": int(day_of_week),
        "month": int(month),
        # Unpack the 384 embedding dimensions
        **{f"embed_{i}": float(vec[i]) for i in range(384)}
    }
    
    # C. Align with Training Schema
    # Fill missing features with 0.0 (safe default)
    X_views = np.array([[feat_dict.get(c, 0.0) for c in artifacts["views_schema"]]])
    X_subs  = np.array([[feat_dict.get(c, 0.0) for c in artifacts["subs_schema"]]])
    
    # D. Predict
    # Views (Log transformed -> Expm1)
    y_views_log = artifacts["views_model"].predict(X_views)
    pred_views = int(max(0, round(np.expm1(y_views_log)[0])))
    
    # Subs (Scaled -> Inverse Transform)
    y_subs_scaled = artifacts["subs_model"].predict(X_subs)
    pred_subs_float = artifacts["subs_scaler"].inverse_transform(y_subs_scaled.reshape(-1, 1)).ravel()[0]
    pred_subs = round(float(pred_subs_float))
    
    # -----------------------------------------------------------------------------
    # 5. DISPLAY RESULTS
    # -----------------------------------------------------------------------------
    st.divider()
    
    m1, m2 = st.columns(2)
    
    with m1:
        st.metric("Predicted Lifetime Views", f"{pred_views:,}")
        
    with m2:
        delta = f"{'+' if pred_subs > 0 else ''}{pred_subs}"
        st.metric("Predicted Subscriber Change", delta)
        
elif submit and not title:
    st.warning("Please enter a video title.")
