# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils import (
    load_config,
    load_datasets,
    train_model,
    construct_feature_vector,
    rank_fabrics,
    explain_fabric,
    PROPERTY_DEFINITIONS,
)

st.set_page_config(layout="wide")
cfg = load_config()

APP_TITLE = cfg["app"].get("title", "Fabric Comfort AI Recommender")
THEME = cfg["app"].get("theme_color", "#2563EB")
st.title(f"👕 {APP_TITLE}")
st.caption("Comfort & Performance Recommender — industry-ready for apparel R&D & sourcing")

# ----------------------
# Load data (cached)
# ----------------------
@st.cache_data(show_spinner=True)
def _load(paths, features, target):
    # Use utils.load_datasets which normalizes headers
    return load_datasets(paths, features, target)

try:
    data = _load(cfg["data"]["paths"], cfg["data"]["features"], cfg["data"]["target"])
except Exception as e:
    st.error(f"Failed to load datasets: {e}")
    st.stop()

# Ensure target exists
target = cfg["data"]["target"]
if target not in data.columns:
    st.error(f"Target column '{target}' not found in dataset after normalization. Check COLUMN_MAP in utils.py and config.yaml.")
    st.stop()

# Ensure at least some features exist; create optional cols if missing (robustness)
configured_features = cfg["data"]["features"]
available_features = [f for f in configured_features if f in data.columns]

# If critical features missing, create them as NaN -> fill with median later
for f in configured_features:
    if f not in data.columns:
        data[f] = np.nan

# Fill optional columns with medians (so ranking/explainability won't crash)
for optional in ["sustainability_score", "gsm", "price"]:
    if optional not in data.columns or data[optional].isna().all():
        # set to median of any numeric column or 0.5 if no numeric available
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            med = data[numeric_cols].median().median()
            data[optional] = med
        else:
            data[optional] = 0.5

# Recompute available features (only numeric ones we'll use for explainability)
available_features = [f for f in configured_features if f in data.columns and pd.api.types.is_numeric_dtype(data[f])]

if len(available_features) < 1:
    st.error("No usable numeric features found in dataset. Check your datasets and COLUMN_MAP in utils.py.")
    st.stop()

# Show quick dataset snapshot & counts
with st.expander("Dataset snapshot & info", expanded=False):
    st.write(f"Rows: {len(data)} — Columns: {len(data.columns)}")
    st.dataframe(data.head(6))

# ----------------------
# Train model (cached resource)
# ----------------------
@st.cache_resource(show_spinner=True)
def _train(df, feat_cols, target_col, cfg_local):
    X = df[feat_cols].copy()
    y = df[target_col].copy()
    return train_model(X, y, cfg_local)

# Decide which feature set to train on:
# We'll train on the features present in the dataset (configured features).
train_features = [f for f in configured_features if f in data.columns]
if len(train_features) < 1:
    st.error("No training features found. Check config.yaml and your dataset headers.")
    st.stop()

try:
    model, scaler, metrics = _train(data, train_features, target, cfg)
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

# ----------------------
# Sidebar: model & glossary
# ----------------------
with st.sidebar:
    st.header("Model & Data")
    st.metric("Rows used", len(data))
    st.metric("R²", f"{metrics.get('r2', np.nan):.3f}")
    st.metric("MSE", f"{metrics.get('mse', np.nan):.3f}")
    st.markdown("---")
    st.subheader("Material glossary")
    for k, v in PROPERTY_DEFINITIONS.items():
        st.write(f"**{k.replace('_',' ').title()}** — {v}")
    st.markdown("---")
    st.caption("Config & dataset normalization are handled in utils.py (COLUMN_MAP).")

# ----------------------
# Tabs
# ----------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Recommender", "📊 Insights", "🤖 Model Performance", "🧾 Reports & About"])

# ----------------------
# TAB 1: Recommender
# ----------------------
with tab1:
    st.header("⚙️ Set Conditions & Constraints")

    colL, colR = st.columns([1, 1])
    with colL:
        mode = st.radio("User Mode", ["Beginner (presets)", "Pro (manual)"], index=0, horizontal=True)
        if mode == "Beginner (presets)":
            preset = st.selectbox("Quick Presets", ["Summer Running", "Gym Workout", "Office Day", "Winter Walk"])
            if preset == "Summer Running":
                temperature, humidity, sweat_sensitivity, activity_intensity = 32, 70, "High", "High"
            elif preset == "Gym Workout":
                temperature, humidity, sweat_sensitivity, activity_intensity = 24, 60, "High", "High"
            elif preset == "Office Day":
                temperature, humidity, sweat_sensitivity, activity_intensity = 24, 45, "Medium", "Low"
            else:
                temperature, humidity, sweat_sensitivity, activity_intensity = 10, 50, "Low", "Low"
        else:
            temperature = st.slider("🌡️ Temperature (°C)", 0, 50, 25)
            humidity_level = st.select_slider("💧 Perceived Humidity", options=["Dry", "Comfortable", "Humid", "Very Humid"], value="Comfortable")
            humidity_map = {"Dry": 30, "Comfortable": 50, "Humid": 70, "Very Humid": 90}
            humidity = humidity_map[humidity_level]
            sweat_sensitivity = st.select_slider("🧍 Sweat Sensitivity", ["Low", "Medium", "High"], value="Medium")
            activity_intensity = st.select_slider("🏃 Activity Intensity", ["Low", "Moderate", "High"], value="Moderate")

    with colR:
        st.write("**Business Constraints**")
        fiber_filter = None
        if "fabric_type" in data.columns:
            fiber_filter = st.selectbox("Preferred Fabric Type", ["(Any)"] + sorted(data["fabric_type"].dropna().astype(str).unique().tolist()), index=0)
        price_band = st.select_slider("Target Price Band (if available)", options=["Any", "Budget", "Mid", "Premium"], value="Any")
        max_gsm = None
        if "gsm" in data.columns:
            try:
                gmin = int(data["gsm"].min())
                gmax = int(data["gsm"].max())
            except Exception:
                gmin, gmax = 0, 1000
            max_gsm = st.slider("Max GSM (optional)", gmin, gmax, gmax)
        sustain_w = 0.0
        if "sustainability_score" in data.columns:
            sustain_w = st.slider("Sustainability Priority (0=Off, 1=High)", 0.0, 1.0, 0.2, 0.05)
        top_k = st.slider("How many recommendations?", 3, 12, 5)

    # Encode inputs
    sweat_map = {"Low": 1, "Medium": 2, "High": 3}
    activity_map = {"Low": 1, "Moderate": 2, "High": 3}
    s_num = sweat_map[sweat_sensitivity]
    a_num = activity_map[activity_intensity]

    user_features = construct_feature_vector(temperature, humidity, s_num, a_num)
    user_scaled = scaler.transform(user_features)
    predicted_score = float(model.predict(user_scaled)[0])

    st.markdown("### 🔹 Predicted Comfort")
    st.metric("Predicted Comfort Score", f"{predicted_score:.2f}")

    # Prepare candidate DataFrame (make copies to avoid mutating global)
    df_work = data.copy().reset_index(drop=True)

    # Apply business constraints
    if fiber_filter and fiber_filter != "(Any)" and "fabric_type" in df_work.columns:
        df_work = df_work[df_work["fabric_type"].astype(str) == fiber_filter]

    if price_band != "Any" and "price" in df_work.columns:
        q1, q2 = df_work["price"].quantile([0.33, 0.66])
        if price_band == "Budget":
            df_work = df_work[df_work["price"] <= q1]
        elif price_band == "Mid":
            df_work = df_work[(df_work["price"] > q1) & (df_work["price"] <= q2)]
        else:
            df_work = df_work[df_work["price"] > q2]

    if max_gsm is not None and "gsm" in df_work.columns:
        df_work = df_work[df_work["gsm"] <= max_gsm]

    # Compute ranking (rank_fabrics uses sustainability column if present)
    ranked = rank_fabrics(df_work, target, predicted_score, sustain_w)
    ranked = ranked.head(top_k).reset_index(drop=True)

    st.markdown("### 🔹 Recommendations")
    if ranked.empty:
        st.warning("No candidates after applying constraints. Relax filters or check dataset coverage.")
    else:
        # Responsive columns (1-3)
        n_cols = min(3, len(ranked))
        cols = st.columns(n_cols)
        for i, row in ranked.iterrows():
            with cols[i % n_cols]:
                fabric_name = row.get("fabric_type", f"Fabric {i+1}")
                st.markdown(f"#### 🧵 {fabric_name}")
                st.write(f"**Comfort**: {row.get(target):.2f}  •  **Similarity**: {row.get('similarity_score'):.1f}%")
                badges = []
                if "gsm" in row and not pd.isna(row["gsm"]):
                    badges.append(f"GSM {int(row['gsm'])}")
                if "price" in row and not pd.isna(row["price"]):
                    badges.append(f"Price {row['price']}")
                if "sustainability_score" in row and not pd.isna(row["sustainability_score"]):
                    badges.append(f"🌱 {row['sustainability_score']:.2f}")
                if badges:
                    st.write(" • ".join(badges))
                st.write("**Why this pick:**")
                expl = explain_fabric(row, data, available_features[:6])
                expl_list = [f"{k.replace('_',' ').title()}: {v}" for k, v in expl.items()]
                st.write(", ".join(expl_list))

                # Buttons: shortlist, details
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("📌 Shortlist", key=f"pin-{i}"):
                        st.session_state.setdefault("shortlist", []).append(row.to_dict())
                with c2:
                    if st.button("ℹ️ Details", key=f"det-{i}"):
                        st.session_state["_detail"] = row.to_dict()
        # Chart: similarity vs comfort
        chart = alt.Chart(ranked.reset_index()).mark_circle(size=130).encode(
            x=alt.X("similarity_score:Q", title="Similarity (%)"),
            y=alt.Y(f"{target}:Q", title="Comfort Score"),
            color=alt.Color("similarity_score:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["fabric_type", target, "similarity_score"]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

    # Download
    st.download_button("⬇️ Download Recommendations (CSV)", ranked.to_csv(index=False).encode("utf-8"), "recommendations.csv", "text/csv")

# ----------------------
# TAB 2: Insights
# ----------------------
with tab2:
    st.header("📊 Dataset Insights")
    st.write(f"Dataset rows: {len(data)} — Features used: {', '.join(available_features)}")
    st.markdown("**Summary statistics (features)**")
    st.dataframe(data[available_features].describe().T, use_container_width=True)

    # Correlation heatmap (features + target)
    corr_cols = available_features + [target] if target in data.columns else available_features
    corr = data[corr_cols].corr().reset_index().melt("index")
    heatmap = alt.Chart(corr).mark_rect().encode(
        x=alt.X("index:O", title=None),
        y=alt.Y("variable:O", title=None),
        color=alt.Color("value:Q", title="corr"),
        tooltip=[alt.Tooltip("value:Q", format=".2f")]
    ).properties(height=400)
    st.altair_chart(heatmap, use_container_width=True)

# ----------------------
# TAB 3: Model Performance
# ----------------------
with tab3:
    st.header("🤖 Model Performance & Explainability")
    st.metric("R² score", f"{metrics.get('r2', np.nan):.3f}")
    st.metric("MSE", f"{metrics.get('mse', np.nan):.3f}")

    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=train_features).sort_values(ascending=False).round(4)
        st.markdown("**Feature importances (RandomForest)**")
        st.dataframe(fi.reset_index().rename(columns={"index": "feature", 0: "importance"}).rename(columns={0: "importance"}), use_container_width=True)
        st.bar_chart(fi)
    else:
        st.info("Model does not expose feature importance.")

    # If a detail selection exists, show it (explainability detail)
    if "_detail" in st.session_state:
        st.markdown("### Selected Fabric Details (from shortlist/details)")
        st.json(st.session_state["_detail"])

# ----------------------
# TAB 4: Reports & About
# ----------------------
with tab4:
    st.header("🧾 Reports & About")
    st.markdown(
        f"""
**Project**: {APP_TITLE}  
**Purpose**: AI-assisted fabric comfort recommendation for apparel R&D, sourcing and design.  

**Pipeline**:  
1. Load & normalize literature + survey datasets.  
2. Feature engineering from user inputs (sweat, humidity, activity, temperature) into model features.  
3. RandomForest regression predicts comfort score.  
4. Ranking engine finds nearest fabrics to predicted score and applies sustainability weighting.  
5. Explainability uses z-scores (σ) to show deviations from dataset means.

You can download the shortlists and recommendations as CSV files and use the Dataset Insights tab for dissertation appendices.
"""
    )

    st.markdown("**Quick links / tips**")
    st.write("- Check utils.py COLUMN_MAP to add new header mappings if your Excel files use different column names.")
    st.write("- If training fails due to very small datasets, add more labeled comfort data or reduce model complexity.")
    st.write("- Use the downloads and dataset snapshots when writing the Implementation chapter of your dissertation.")
    st.markdown("---")

    st.markdown("### Mermaid pipeline (copy to mermaid.live)")
    mermaid = """
flowchart TD
  A[User input: Temp, Humidity, Activity, Constraints] --> B[Feature Engineering (construct vector)]
  B --> C[Scaler + Model (RandomForest)]
  C --> D[Predicted Comfort Score]
  D --> E[Ranking Engine (proximity + sustainability weight)]
  E --> F[Recommendations UI (cards, compare, export)]
  subgraph Data
    G[Material Literature Dataset]
    H[Survey / Responses dataset]
  end
  G --> B
  H --> B
"""
    st.code(mermaid, language="mermaid")

# ----------------------
# Shortlist panel (bottom)
# ----------------------
if "shortlist" in st.session_state and st.session_state["shortlist"]:
    st.sidebar.markdown("### 📌 Shortlist (session)")
    short_df = pd.DataFrame(st.session_state["shortlist"])
    st.sidebar.dataframe(short_df.head(10))
    st.sidebar.download_button("Download Shortlist CSV", short_df.to_csv(index=False).encode("utf-8"), "shortlist.csv", "text/csv")
