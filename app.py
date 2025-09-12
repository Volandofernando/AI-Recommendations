# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ============================
# Dataset URLs (RAW GitHub Links)
# ============================
DATASET_1 = "https://github.com/Volandofernando/Material-Literature-data-/raw/main/Dataset.xlsx"
DATASET_2 = "https://github.com/Volandofernando/REAL-TIME-Dataset/raw/main/IT%20Innovation%20in%20Fabric%20Industry%20%20(Responses).xlsx"

# ============================
# Load Datasets
# ============================
@st.cache_data
def load_datasets():
    try:
        df1 = pd.read_excel(DATASET_1)
        df2 = pd.read_excel(DATASET_2)
        return df1, df2
    except Exception as e:
        st.error(f"❌ Failed to load datasets: {e}")
        return None, None

# ============================
# Train Model (Simple Example)
# ============================
def train_model(df):
    # Example: Assume df has numerical columns for ML
    features = df.select_dtypes(include=np.number).dropna(axis=1)
    if "Comfort_Level" not in df.columns:
        st.warning("⚠️ No 'Comfort_Level' column found in dataset. Using demo labels.")
        df["Comfort_Level"] = np.random.choice(["Low", "Medium", "High"], size=len(df))

    X = features
    y = df["Comfort_Level"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, features.columns.tolist()

# ============================
# App UI
# ============================
st.set_page_config(page_title="👕 Fabric Comfort AI Recommender", layout="wide")

st.title("👕 Fabric Comfort AI Recommender")
st.subheader("Comfort & Performance Insights for Apparel Industry")

st.markdown("""
AI-Powered Fabric Comfort Recommender  
Trusted by **textile R&D, apparel design, and sportswear innovation teams**.  
Powered by **machine learning** trained on fabric properties and real-world comfort data.  

👉 Adjust your conditions and instantly see **top fabric recommendations** optimized for comfort, sweat management, and thermophysiological performance.
""")

# Load datasets
df1, df2 = load_datasets()

if df1 is not None and df2 is not None:
    st.success("✅ Datasets loaded successfully!")

    # Train ML Model
    model, scaler, feature_cols = train_model(df1)

    # User Inputs
    st.sidebar.header("Adjust Your Conditions")
    user_input = {}
    for col in feature_cols:
        min_val = float(df1[col].min())
        max_val = float(df1[col].max())
        mean_val = float(df1[col].mean())
        user_input[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val)

    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    # Predict Comfort
    prediction = model.predict(input_scaled)[0]
    st.subheader("🎯 Recommended Comfort Level")
    st.info(f"Based on your conditions, the fabric is predicted to have **{prediction} Comfort Level**.")

    # Show Recommended Fabrics (Top Similar)
    st.subheader("📊 Similar Fabrics from Dataset")
    df1["Predicted"] = model.predict(scaler.transform(df1[feature_cols]))
    st.dataframe(df1[df1["Predicted"] == prediction].head(10))

else:
    st.error("❌ Could not load datasets. Please check your GitHub raw links.")
