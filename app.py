import streamlit as st
import pandas as pd
from utils import load_config, load_datasets, preprocess, train_model
from db import save_user_feedback

# -------------------------------
# Config
# -------------------------------
cfg = load_config()
st.set_page_config(page_title=cfg["app"]["title"], layout="wide")

# -------------------------------
# App Header
# -------------------------------
st.title(cfg["app"]["title"])
st.markdown(cfg["app"]["description"])

st.markdown("### 👕 How It Works")
st.info("This system recommends the **best sweat-resistant fabric** by analyzing weather (humidity, temperature), activity level, and user sweat sensitivity. It combines literature-based data with real-world survey inputs.")

# -------------------------------
# Load & Train
# -------------------------------
material_data, survey_data = load_datasets(cfg)
df = preprocess(material_data, survey_data)

try:
    model, scaler = train_model(df)
    st.success("✅ Model trained successfully.")
except Exception as e:
    st.error(f"Model training failed: {e}")

# -------------------------------
# User Inputs
# -------------------------------
st.markdown("### 📝 Input Your Conditions")
humidity = st.slider("Humidity (%)", 10, 100, 60)
temperature = st.slider("Temperature (°C)", 15, 40, 28)
activity = st.selectbox("Activity Level", ["Low", "Medium", "High"])
sensitivity = st.radio("Sweat Sensitivity", ["Low", "Medium", "High"])

# Map inputs into features
activity_map = {"Low": 1, "Medium": 2, "High": 3}
sensitivity_map = {"Low": 1, "Medium": 2, "High": 3}

user_features = pd.DataFrame([{
    "Humidity": humidity,
    "Temperature": temperature,
    "Activity": activity_map[activity],
    "Sensitivity": sensitivity_map[sensitivity]
}])

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Recommend Fabric"):
    try:
        scaled_input = scaler.transform(user_features)
        prediction = model.predict(scaled_input)[0]
        st.success(f"Recommended Fabric: **{prediction}**")
        save_user_feedback(user_features.to_dict(), prediction)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
