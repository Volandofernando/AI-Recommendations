"""
app.py
Streamlit UI for AI-Powered Fabric Comfort Recommender
"""

import streamlit as st
from utils import (
    load_config, load_datasets, prepare_xy, train_model,
    construct_feature_vector, rank_fabrics, explain_fabric,
    PROPERTY_DEFINITIONS
)

# -------------------------------
# Setup
# -------------------------------
cfg = load_config()
st.set_page_config(page_title=cfg["app"]["title"], layout="wide")

st.title("🧵 " + cfg["app"]["title"])
st.markdown("""
This system recommends fabrics based on **comfort prediction** and **sustainability**.  
Each material property is explained, so users understand *why* a fabric was chosen.
""")

# -------------------------------
# Load + Train
# -------------------------------
data = load_datasets(cfg["data"]["paths"])
features, target = cfg["data"]["features"], cfg["data"]["target"]
X, y, cleaned_df = prepare_xy(data, features, target)
model, scaler, metrics = train_model(X, y, cfg)

st.sidebar.header("📊 Model Performance")
st.sidebar.write(f"MSE: {metrics['mse']:.3f}")
st.sidebar.write(f"R²: {metrics['r2']:.3f}")
st.sidebar.write(f"Rows used: {len(X)} / {len(data)}")

# -------------------------------
# User Inputs
# -------------------------------
st.header("Set Conditions")
col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("🌡️ Temperature (°C)", 15, 40, 25)
    humidity = st.slider("💧 Humidity (%)", 20, 90, 50)
with col2:
    sweat_num = st.selectbox("Sweat Sensitivity", [1, 2, 3], format_func=lambda x: {1:"Low",2:"Medium",3:"High"}[x])
    activity_num = st.selectbox("Activity Intensity", [1, 2, 3], format_func=lambda x: {1:"Light",2:"Moderate",3:"High"}[x])
sustain_w = st.slider("🌱 Sustainability Weight", 0.0, 1.0, 0.3)

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
ranked = rank_fabrics(cleaned_df, target, pred_score, sustain_w)

st.subheader("Top 5 Recommended Fabrics")
st.dataframe(ranked.head(5), use_container_width=True)

# -------------------------------
# Explainability
# -------------------------------
st.subheader("Why This Pick?")
top = ranked.head(1).iloc[0]
explanations = explain_fabric(top, cleaned_df, features)

for f, expl in explanations.items():
    st.write(f"**{f.replace('_',' ').title()}**: {expl} — {PROPERTY_DEFINITIONS.get(f,'')}")

# -------------------------------
# Export
# -------------------------------
st.download_button(
    "📥 Download Top 10 Recommendations",
    ranked.head(10).to_csv(index=False).encode("utf-8"),
    "fabric_recommendations.csv",
    "text/csv"
)
