import os
import time
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from utils import (
    load_config, load_datasets,
    detect_features_and_target, train_model, evaluate_model
)
from db import log_event

# ---------- Config & Theme ----------
config = load_config()
st.set_page_config(page_title=config["app"]["title"], layout="wide")

st.markdown(f"""
<style>
    .main {{ background-color: #F9FAFB; }}
    h1, h2, h3 {{ color: {config['app']['theme_color']}; }}
    .metric-row {{ display:flex; gap:2rem; flex-wrap:wrap; }}
    .metric-card {{ padding:1rem; border-radius:1rem; background:#ffffff; box-shadow:0 1px 3px rgba(0,0,0,0.06); }}
</style>
""", unsafe_allow_html=True)

st.title(f"👕 {config['app']['title']}")
st.caption(config["app"]["subtitle"])

# ---------- Load data ----------
with st.spinner("Loading datasets…"):
    df = load_datasets(config)

st.expander("Show detected columns").write(df.columns.tolist())

# Detect features/target/fabric name
feature_cols, target_col, fabric_col = detect_features_and_target(df, config)

if target_col is None or len(feature_cols) < 4:
    st.error(
        f"❌ Could not detect the required features/target.\n\n"
        f"Detected features: {feature_cols}\n"
        f"Detected target: {target_col}\n\n"
        f"Please verify your dataset columns or update 'config.yaml'."
    )
    st.stop()

# Train model
with st.spinner("Training model…"):
    model, scaler, X_test, y_test, df_clean = train_model(df, feature_cols, target_col, config)

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📌 Recommendation", "📊 Dataset Insights", "🤖 Model Performance", "📝 About Project"]
)

# ---------- Tab 1: Recommendation ----------
with tab1:
    st.sidebar.header("Input Your Conditions")
    temperature = st.sidebar.slider("🌡️ Temperature (°C)", 15, 40, 30)
    humidity = st.sidebar.slider("💧 Humidity (%)", 10, 100, 70)
    sweat_sensitivity = st.sidebar.selectbox("🧍 Sweat Sensitivity", ["Low", "Medium", "High"])
    activity_intensity = st.sidebar.selectbox("🏃 Activity Intensity", ["Low", "Moderate", "High"])

    sweat_map = {"Low": 1, "Medium": 2, "High": 3}
    activity_map = {"Low": 1, "Moderate": 2, "High": 3}
    sweat_num = sweat_map[sweat_sensitivity]
    activity_num = activity_map[activity_intensity]

    # Construct feature vector (domain heuristics)
    user_features = np.array([[
        sweat_num * 5,                    # moisture_regain
        800 + humidity * 5,               # water_absorption
        60 + activity_num * 10,           # drying_time
        0.04 + (temperature - 25) * 0.001 # thermal_conductivity
    ]], dtype=float)

    # Respect the detected column order
    # (Assumes exactly 4 features matched in config order)
    if len(feature_cols) != 4:
        st.warning(f"Detected {len(feature_cols)} features: {feature_cols}. "
                   "The recommender expects 4; results may be off.")
    user_scaled = scaler.transform(user_features)

    predicted_score = float(model.predict(user_scaled)[0])

    # Find nearest items by comfort proximity
    df_view = df_clean.copy()
    df_view["predicted_diff"] = (df_view[target_col] - predicted_score).abs()
    top_matches = df_view.sort_values("predicted_diff").head(3)

    st.subheader("🔹 Top Fabric Recommendations")

    cols = st.columns(3)
    for i, (_, row) in enumerate(top_matches.iterrows()):
        with cols[i]:
            card = st.container(border=True)
            with card:
                name = str(row.get(fabric_col, "Unknown")) if fabric_col else "Unknown"
                st.markdown(f"### 🧵 {name}")
                st.metric("Comfort Score", round(float(row[target_col]), 2))
                st.caption(
                    f"Moisture: {row[feature_cols[0]]} | "
                    f"Absorption: {row[feature_cols[1]]} | "
                    f"Drying: {row[feature_cols[2]]} | "
                    f"Thermal: {row[feature_cols[3]]}"
                )

    # Chart
    cdata = top_matches[[target_col]].copy()
    cdata["fabric_name"] = top_matches[fabric_col] if fabric_col in top_matches.columns else "Unknown"
    cdata.rename(columns={target_col: "Comfort Score"}, inplace=True)
    chart = alt.Chart(cdata).mark_bar().encode(
        x=alt.X("fabric_name:N", title="Fabric"),
        y=alt.Y("Comfort Score:Q"),
        tooltip=["fabric_name", "Comfort Score"]
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

    # Allow CSV download of recommendations
    st.download_button(
        "Download Top Recommendations (CSV)",
        data=top_matches.to_csv(index=False).encode("utf-8"),
        file_name="fabric_recommendations.csv",
        mime="text/csv"
    )

    # Log an analytics event (no-op if Supabase not configured)
    log_event("fabric_events", {
        "ts": int(time.time()),
        "temperature": temperature,
        "humidity": humidity,
        "sweat_sensitivity": sweat_sensitivity,
        "activity_intensity": activity_intensity,
        "predicted_score": predicted_score,
    })

# ---------- Tab 2: Dataset Insights ----------
with tab2:
    st.markdown("### 📊 Explore Dataset")
    st.write(f"Detected features: `{feature_cols}`  •  Target: `{target_col}`  •  Fabric name: `{fabric_col or 'N/A'}`")
    st.dataframe(df_clean.head(20), use_container_width=True)

    # Simple distributions
    st.markdown("#### Feature Distributions")
    for col in feature_cols:
        try:
            hist = alt.Chart(df_clean).mark_bar().encode(
                x=alt.X(f"{col}:Q", bin=True),
                y="count()"
            ).properties(height=200, title=col)
            st.altair_chart(hist, use_container_width=True)
        except Exception:
            pass

# ---------- Tab 3: Model Performance ----------
with tab3:
    st.markdown("### 🤖 Model Performance")
    metrics = evaluate_model(model, X_test, y_test)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("R² Score", round(metrics["r2"], 3))
    with c2:
        st.metric("RMSE", round(metrics["rmse"], 3))

    # Feature importance
    try:
        importances = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
        feat_chart = alt.Chart(feat_df).mark_bar().encode(
            x="Feature:N", y="Importance:Q", tooltip=["Feature", "Importance"]
        ).properties(height=280, title="Feature Importance")
        st.altair_chart(feat_chart, use_container_width=True)
    except Exception:
        st.info("Feature importance not available for this model.")

# ---------- Tab 4: About ----------
with tab4:
    st.markdown(f"""
**{config['app']['title']}**  
Developed as part of a BSc Dissertation at the **University of West London**.

- Combines **literature** + **real-time survey** datasets  
- Uses **Random Forest** to predict comfort score  
- Professional UI + analytics-ready (optional Supabase)  
- Exportable recommendations for stakeholders  

📌 Author: *Volando Fernando*
""")
