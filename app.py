import streamlit as st
import pandas as pd
import altair as alt
from utils import (
    load_config, load_datasets, train_model,
    construct_feature_vector, rank_fabrics, explain_fabric,
    PROPERTY_DEFINITIONS
)

# -------------------------------
# Config & Setup
# -------------------------------
cfg = load_config()
st.set_page_config(page_title=cfg["app"]["title"], layout="wide")
st.title("🧵 " + cfg["app"]["title"])

# -------------------------------
# Load Data & Train Model
# -------------------------------
data = load_datasets(cfg["data"]["paths"], cfg["data"]["features"], cfg["data"]["target"])
features, target = cfg["data"]["features"], cfg["data"]["target"]

model, scaler, metrics = train_model(data[features], data[target], cfg)

# -------------------------------
# Sidebar: Model Performance
# -------------------------------
st.sidebar.header("📊 Model Performance")
st.sidebar.write(f"MSE: {metrics['mse']:.3f}")
st.sidebar.write(f"R²: {metrics['r2']:.3f}")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Recommender", "Insights", "Model", "About"])

# -------------------------------
# Tab 1: Recommender
# -------------------------------
with tab1:
    st.header("Enter User Conditions")

    temperature = st.slider("🌡️ Temperature (°C)", 15, 40, 25)
    humidity = st.slider("💧 Humidity (%)", 20, 90, 50)
    sweat_num = st.selectbox("Sweat Sensitivity", [1, 2, 3], format_func=lambda x: {1:"Low",2:"Medium",3:"High"}[x])
    activity_num = st.selectbox("Activity Intensity", [1, 2, 3], format_func=lambda x: {1:"Light",2:"Moderate",3:"High"}[x])
    sustain_w = st.slider("🌱 Sustainability Weight", 0.0, 1.0, 0.3)

    user_features = construct_feature_vector(temperature, humidity, sweat_num, activity_num)
    pred_score = model.predict(scaler.transform(user_features))[0]

    st.subheader("Predicted Comfort Score")
    st.metric("Comfort Score", f"{pred_score:.2f}")

    ranked = rank_fabrics(data, target, pred_score, sustain_w)
    st.subheader("Top Recommended Fabrics")
    st.dataframe(ranked.head(5), use_container_width=True)

    st.subheader("Why This Pick?")
    top = ranked.head(1).iloc[0]
    explanations = explain_fabric(top, data, features)
    for f, expl in explanations.items():
        st.write(f"**{f.replace('_',' ').title()}**: {expl} ({PROPERTY_DEFINITIONS.get(f,'')})")

    st.download_button(
        "📥 Download Top 10 Recommendations",
        ranked.head(10).to_csv(index=False).encode("utf-8"),
        "fabric_recommendations.csv",
        "text/csv"
    )

# -------------------------------
# Tab 2: Insights
# -------------------------------
with tab2:
    st.header("📈 Dataset Insights")
    st.write(f"Dataset contains {len(data)} rows and {len(data.columns)} columns.")
    corr = data[features + [target]].corr()
    chart = alt.Chart(corr.reset_index().melt("index")).mark_rect().encode(
        x="index:O", y="variable:O", color="value:Q"
    )
    st.altair_chart(chart, use_container_width=True)

# -------------------------------
# Tab 3: Model
# -------------------------------
with tab3:
    st.header("⚙️ Model Details")
    st.write(cfg["model"])
    st.write("Feature Importance (Random Forest):")
    st.bar_chart(pd.Series(model.feature_importances_, index=features))

# -------------------------------
# Tab 4: About
# -------------------------------
with tab4:
    st.header("📄 About this Project")
    st.markdown("""
    **Fabric Comfort AI Recommender**  
    - Predicts fabric comfort based on activity, climate, and physiology  
    - Ranks fabrics with sustainability weighting  
    - Provides explainability with z-scores  
    - Built with Python, Streamlit, scikit-learn  

    Author: Volando Fernando
    """)
