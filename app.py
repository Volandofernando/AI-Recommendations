# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils import (
    load_config,
    load_datasets,
    detect_features_and_target,
    train_model,
    evaluate_model,
)

# ===============================
# Page / Config
# ===============================
config = load_config()
APP_TITLE = config["app"]["title"]
THEME = config["app"].get("theme_color", "#2563EB")

st.set_page_config(page_title=APP_TITLE, page_icon="👕", layout="wide")

# ===============================
# Global Styles
# ===============================
st.markdown(
    f"""
<style>
:root {{
    --brand: {THEME};
    --text: #111827;
    --muted: #6B7280;
    --bg: #F8FAFC;
    --card: #FFFFFF;
    --shadow: 0 6px 24px rgba(0,0,0,0.06);
    --radius: 16px;
}}
.main {{ background: var(--bg); }}
.metric-card {{
    background: var(--card); padding: 14px; border-radius: var(--radius);
    box-shadow: var(--shadow); border: 1px solid #EEF2F7;
    margin-bottom: 12px;
}}
.metric-value {{ font-size: 1.2rem; font-weight: 800; color: var(--text); }}
.metric-label {{ font-size: .82rem; color: var(--muted); }}
</style>
""",
    unsafe_allow_html=True,
)

# ===============================
# Header
# ===============================
st.title(f"👕 {APP_TITLE}")
st.caption("AI-Powered Fabric Comfort & Performance Recommender")

# ===============================
# Load Data & Train
# ===============================
@st.cache_data(show_spinner=False)
def _load_df(_config):
    return load_datasets(_config)


try:
    df = _load_df(config)
except Exception as e:
    st.error(f"❌ Failed to load datasets: {e}")
    st.stop()

feature_cols, target_col = detect_features_and_target(df, config)
if target_col is None or len(feature_cols) < 4:
    st.error("❌ Dataset error: required features/target not found!")
    st.stop()


@st.cache_resource(show_spinner=True)
def _train(_df, _feature_cols, _target_col, _config):
    return train_model(_df, _feature_cols, _target_col, _config)


model, scaler, X_test, y_test, df_clean = _train(df, feature_cols, target_col, config)

# ===============================
# Tabs
# ===============================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📌 Recommender", "📊 Insights", "🤖 Model Performance", "🧾 About"]
)

# =========================================================
# TAB 1 – Recommender
# =========================================================
with tab1:
    st.subheader("⚙️ Set Conditions")

    colA, colB = st.columns([1, 1])
    with colA:
        temperature = st.slider("🌡️ Temperature (°C)", 0, 50, 28)
        humidity = st.slider("💧 Humidity (%)", 10, 100, 70)
        sweat_sensitivity = st.selectbox("🧍 Sweat Sensitivity", ["Low", "Medium", "High"])
        activity_intensity = st.selectbox("🏃 Activity Intensity", ["Low", "Moderate", "High"])

    with colB:
        top_k = st.slider("How many recommendations?", 3, 10, 5)

    # Encode input
    sweat_map = {"Low": 1, "Medium": 2, "High": 3}
    activity_map = {"Low": 1, "Moderate": 2, "High": 3}
    sweat_num, activity_num = sweat_map[sweat_sensitivity], activity_map[activity_intensity]

    user_features = np.array(
        [[
            sweat_num * 5,
            800 + humidity * 5,
            60 + activity_num * 10,
            0.04 + (temperature - 25) * 0.001
        ]]
    )
    user_scaled = scaler.transform(user_features)
    predicted_score = float(model.predict(user_scaled)[0])

    # Rank fabrics
    df_clean["predicted_diff"] = (df_clean[target_col] - predicted_score).abs()
    ranked = df_clean.sort_values("predicted_diff").head(top_k).copy()

    # Similarity %
    eps = 1e-9
    inv_prox = 1.0 / (ranked["predicted_diff"] + eps)
    ranked["similarity"] = (
        (inv_prox - inv_prox.min()) / (inv_prox.max() - inv_prox.min() + eps) * 100.0
    ).round(1)

    # UI – Recommendations
    st.markdown("### 🔹 Recommended Fabrics")
    cols = st.columns(min(3, len(ranked)))
    for i, (_, row) in enumerate(ranked.iterrows()):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">🧵 {row.get('fabric_type','Unknown')}</div>
                    <div class="metric-label">Similarity {row['similarity']}%</div>
                    <div class="metric-label">Comfort Score {row[target_col]:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Quick chart
    chart = (
        alt.Chart(ranked.reset_index(drop=True))
        .mark_circle(size=120)
        .encode(
            x=alt.X("similarity:Q", title="Similarity (%)"),
            y=alt.Y(f"{target_col}:Q", title="Comfort Score"),
            tooltip=["fabric_type", "similarity", target_col],
        )
    )
    st.altair_chart(chart, use_container_width=True)

# =========================================================
# TAB 2 – Insights
# =========================================================
with tab2:
    st.subheader("📊 Dataset Insights")
    st.dataframe(df_clean.head(12), use_container_width=True)
    st.markdown("**Summary Statistics**")
    st.dataframe(df_clean.describe(include="all").T, use_container_width=True)

# =========================================================
# TAB 3 – Model Performance
# =========================================================
with tab3:
    st.subheader("🤖 Model Performance")
    metrics = evaluate_model(model, X_test, y_test)
    c1, c2 = st.columns(2)
    c1.metric("R² Score", f"{metrics['r2']:.3f}")
    c2.metric("RMSE", f"{metrics['rmse']:.3f}")

# =========================================================
# TAB 4 – About
# =========================================================
with tab4:
    st.subheader("🧾 About This Project")
    st.markdown(
        f"""
        - **Project**: {APP_TITLE}  
        - **Goal**: AI-assisted fabric comfort & performance recommendation  
        - **Tech**: Streamlit, Pandas, Altair, scikit-learn  
        - **Pipeline**: Climate/activity input → ML model predicts comfort → ranked recommendations  
        - **Author**: Volando Fernando (BSc Dissertation, University of West London)  
        """
    )
