import streamlit as st
import pandas as pd
import altair as alt
from utils import load_config, load_datasets, train_model, construct_feature_vector, rank_fabrics, explain_fabric, PROPERTY_DEFS

cfg = load_config()
st.set_page_config(page_title=cfg["app"]["title"], page_icon="👕", layout="wide")
st.title(f"👕 {cfg['app']['title']}")

# -------------------
# Load data
# -------------------
data, report = load_datasets(cfg["data"]["paths"], cfg["data"]["features"], cfg["data"]["target"])
st.sidebar.header("Dataset Report")
st.sidebar.write(report)

features, target = cfg["data"]["features"], cfg["data"]["target"]
model, scaler, metrics = train_model(data[features], data[target], cfg)

st.sidebar.metric("R²", f"{metrics['r2']:.3f}")
st.sidebar.metric("MSE", f"{metrics['mse']:.3f}")

# -------------------
# User Inputs
# -------------------
st.header("Input Conditions")
mode = st.radio("Mode", ["Beginner", "Pro"])
if mode == "Beginner":
    preset = st.selectbox("Scenario", ["Summer Running", "Office Day", "Cold Walk"])
    if preset == "Summer Running":
        temp, hum, sweat, act = 32, 70, "High", "High"
    elif preset == "Office Day":
        temp, hum, sweat, act = 24, 45, "Medium", "Low"
    else:
        temp, hum, sweat, act = 10, 50, "Low", "Low"
else:
    temp = st.slider("Temperature °C", 0, 45, 25)
    hum = st.slider("Humidity %", 20, 90, 50)
    sweat = st.select_slider("Sweat Sensitivity", ["Low", "Medium", "High"])
    act = st.select_slider("Activity", ["Low", "Moderate", "High"])

maps = {"Low": 1, "Medium": 2, "High": 3, "Moderate": 2}
user_vec = construct_feature_vector(temp, hum, maps[sweat], maps[act])
pred_score = float(model.predict(scaler.transform(user_vec))[0])

st.subheader("Predicted Comfort")
st.metric("Comfort Score", f"{pred_score:.2f}")

# -------------------
# Ranking
# -------------------
sustain_w = st.slider("Sustainability Weight", 0.0, 1.0, 0.2, 0.05)
ranked = rank_fabrics(data, target, pred_score, sustain_w).head(5)

st.header("Recommendations")
for i, row in ranked.iterrows():
    st.markdown(f"### {i+1}. {row.get('fabric_type','Unknown')}")
    st.write(f"Comfort: {row[target]:.2f} • Similarity: {row['similarity_score']:.1f}%")
    expl = explain_fabric(row, data, features)
    for f, v in expl.items():
        st.write(f"- {f}: {v} ({PROPERTY_DEFS.get(f,'')})")

st.dataframe(ranked[["fabric_type", target, "similarity_score", "sustainability_score", "gsm", "price"]])
