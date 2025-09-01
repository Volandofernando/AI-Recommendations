import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils import load_config, load_datasets, detect_features_and_target, train_model, evaluate_model

# -------------------------------
# Load Config
# -------------------------------
config = load_config()
st.set_page_config(page_title=config["app"]["title"], layout="wide")

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown(f"""
<style>
    .main {{ background-color: #FAFAFA; }}
    h1, h2, h3 {{ color: {config['app']['theme_color']}; font-family: 'Helvetica Neue', sans-serif; }}
    .intro-box {{
        padding: 1rem;
        border-radius: 10px;
        background-color: #FFFFFF;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
    }}
</style>
""", unsafe_allow_html=True)

st.title(f"👕 {config['app']['title']}")
st.subheader(config["app"]["subtitle"])

# -------------------------------
# Introduction Section
# -------------------------------
st.markdown("""
<div class="intro-box">
    <h3>Welcome to the Fabric Comfort Recommender 👕</h3>
    <p>
    This tool helps you choose the <b>most comfortable fabrics</b> based on:
    </p>
    <ul>
        <li>🌡️ <b>Temperature</b>: Outdoor conditions affect fabric breathability and comfort.</li>
        <li>💧 <b>Humidity</b>: Impacts sweat absorption and drying time.</li>
        <li>🧍 <b>Sweat Sensitivity</b>: How much you typically sweat during activity.</li>
        <li>🏃 <b>Activity Intensity</b>: From casual wear to high-performance sports.</li>
    </ul>
    <p>
    Behind the scenes, this app uses <b>Machine Learning</b> trained on academic literature and real-world survey data to recommend fabrics for comfort, sweat management, and performance.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Load Data
# -------------------------------
try:
    df = load_datasets(config)
except Exception as e:
    st.error(f"❌ Failed to load datasets: {e}")
    st.stop()

feature_cols, target_col = detect_features_and_target(df, config)

if target_col is None or len(feature_cols) < 4:
    st.error("❌ Dataset error: required features/target not found!")
    st.stop()

model, scaler, X_test, y_test, df_clean = train_model(df, feature_cols, target_col, config)

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Get Recommendations", "📊 Insights", "🤖 Model Performance", "ℹ️ About"])

# -------------------------------
# TAB 1: Recommendation
# -------------------------------
with tab1:
    st.sidebar.header("Set Your Conditions")

    temperature = st.sidebar.slider("🌡️ Outdoor Temperature (°C)", 10, 45, 28)
    humidity = st.sidebar.slider("💧 Humidity (%)", 10, 100, 60)
    sweat_sensitivity = st.sidebar.radio("🧍 Sweat Sensitivity", ["Low", "Medium", "High"])
    activity_intensity = st.sidebar.radio("🏃 Activity Intensity", ["Low", "Moderate", "High"])

    sweat_map = {"Low": 1, "Medium": 2, "High": 3}
    activity_map = {"Low": 1, "Moderate": 2, "High": 3}
    sweat_num, activity_num = sweat_map[sweat_sensitivity], activity_map[activity_intensity]

    # User Input Vector
    user_input = np.array([[sweat_num * 5,
                            800 + humidity * 5,
                            60 + activity_num * 10,
                            0.04 + (temperature - 25) * 0.001]])
    user_input_scaled = scaler.transform(user_input)

    predicted_score = model.predict(user_input_scaled)[0]
    df_clean["predicted_diff"] = abs(df_clean[target_col] - predicted_score)
    top_matches = df_clean.sort_values(by="predicted_diff").head(3)

    # Show Recommendations
    st.markdown("## 🔹 Recommended Fabrics for You")
    cols = st.columns(3)
    for i, (_, row) in enumerate(top_matches.iterrows()):
        with cols[i]:
            st.markdown(f"### 🧵 {row.get('fabric_type','Unknown')}")
            st.metric("Comfort Score", round(row[target_col], 2))

    # Chart
    chart_data = top_matches[[target_col, "fabric_type"]].rename(columns={target_col: "Comfort Score"})
    chart = alt.Chart(chart_data).mark_bar(color=config["app"]["theme_color"]).encode(
        x=alt.X("fabric_type", sort=None),
        y="Comfort Score"
    )
    st.altair_chart(chart, use_container_width=True)

# -------------------------------
# TAB 2: Dataset Insights
# -------------------------------
with tab2:
    st.markdown("### 📊 Cleaned Dataset Preview")
    clean_cols = [c for c in df_clean.columns if not any(x in c for x in ["email", "timestamp", "what_", "how_", "satisfied"])]
    st.dataframe(df_clean[clean_cols].head(10))

# -------------------------------
# TAB 3: Model Performance
# -------------------------------
with tab3:
    metrics = evaluate_model(model, X_test, y_test)
    st.metric("R² Score", metrics["r2"])
    st.metric("RMSE", metrics["rmse"])

# -------------------------------
# TAB 4: About
# -------------------------------
with tab4:
    st.markdown(f"""
    **{config['app']['title']}**  
    Developed for the **University of West London** as part of a BSc Dissertation.  

    🚀 Key Features:  
    - AI-powered comfort prediction for fabrics  
    - Combines academic + real-world survey data  
    - Designed for sportswear, performance clothing, and fashion industry use  

    👨‍💻 Author: *Volando Fernando*  
    """)
