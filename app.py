"""
app.py
Professional Streamlit app for AI-Powered Fabric Comfort Recommender.
"""

import streamlit as st
import pandas as pd
from utils import (
    load_config, load_datasets, train_model,
    construct_feature_vector, rank_fabrics, explain_fabric,
    PROPERTY_DEFINITIONS
)

# -------------------------------
# Load Config + Data
# -------------------------------
cfg = load_config()
st.set_page_config(page_title=cfg["app"]["title"], layout="wide")
st.title("🧵 " + cfg["app"]["title"])

st.markdown("""
This system recommends fabrics based on **comfort prediction** and **sustainability**.  
Each material property is explained, so users understand *why* a fabric was chosen.
""")

# Load datasets (now passes features + target for cleaning)
data = load_datasets(cfg["data"]["paths"], cfg["data"]["features"], cfg["data"]["target"])
features, target = cfg["data"]["features"], cfg["data"]["target"]

# Check dataset validity
if not all(f in data.columns for f in features) or target not in data.columns:
    st.error("❌ Dataset error: required features/target not found!")
    st.stop()

# Train model
model, scaler, metrics = train_model(data[features], data[target], cfg)
st.sidebar.header("📊 Model Performance")
st.sidebar.write(f"MSE: {metrics['mse']:.3f}")
st.sidebar.write(f"R²: {metrics['r2']:.3f}")

# -------------------------------
# User Inputs
# -------------------------------
st.header("Enter User Conditions")
temperature = st.slider("🌡️ Temperature (°C)", 15, 40, 25, help="Ambient environment temperature.")
humidity = st.slider("💧 Humidity (%)", 20, 90, 50, help="Environmental humidity level.")
sweat_num = st.selectbox("Sweat Sensitivity", [1, 2, 3],
                         format_func=lambda x: {1:"Low",2:"Medium",3:"High"}[x])
activity_num = st.selectbox("Activity Intensity", [1, 2, 3],
                            format_func=lambda x: {1:"Light",2:"Moderate",3:"High"}[x])
sustain_w = st.slider("🌱 Sustainability Weight", 0.0, 1.0, 0.3,
                      help="Weight given to eco-friendly fabrics.")

# -------------------------------
# Prediction
# -------------------------------
user_features = construct_feature_vector(temperature, humidity, sweat_num, activity_num)
pred_score = model.predict(scaler.transform(user_features))[0]

st.subheader("Predicted Comfort Score")
st.metric("Comfort Score", f"{pred_score:.2f}")

# -------------------------------
# Ranking
# -------------------------------
ranked = rank_fabrics(data, target, pred_score, sustain_w)
st.subheader("Top Recommended Fabrics")
st.dataframe(ranked.head(5), use_container_width=True)

# -------------------------------
# Explainability
# -------------------------------
st.subheader("Why This Pick?")
top = ranked.head(1).iloc[0]
explanations = explain_fabric(top, data, features)

for f, expl in explanations.items():
    st.write(f"**{f.replace('_',' ').title()}**: {expl} ({PROPERTY_DEFINITIONS.get(f,'')})")

# -------------------------------
# Export
# -------------------------------
st.download_button(
    "📥 Download Top 10 Recommendations",
    ranked.head(10).to_csv(index=False).encode("utf-8"),
    "fabric_recommendations.csv",
    "text/csv"
)
