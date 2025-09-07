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
# Global Styles (mobile-friendly)
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
    .main {{
        background: var(--bg);
    }}
    .hero {{
        padding: 18px 20px;
        border-radius: var(--radius);
        background: linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 100%);
        box-shadow: var(--shadow);
        border: 1px solid #EEF2F7;
    }}
    .badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(37,99,235,0.1);
        color: var(--brand);
        font-weight: 600;
        font-size: 12px;
        margin-right: 6px;
    }}
    .metric-card {{
        background: var(--card);
        padding: 14px;
        border-radius: 14px;
        box-shadow: var(--shadow);
        border: 1px solid #EEF2F7;
        transition: transform .12s ease, box-shadow .12s ease;
        margin-bottom: 12px;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.07);
    }}
    .metric-value {{
        font-size: 1.2rem;
        font-weight: 800;
        color: var(--text);
    }}
    .metric-label {{
        font-size: .82rem;
        color: var(--muted);
    }}
    .rec-title {{
        font-weight: 800;
        font-size: 1.05rem;
        margin-bottom: 6px;
    }}
    .subtle {{
        color: var(--muted);
        font-size: .9rem;
    }}
    .pill {{
        display:inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: #F1F5F9;
        font-size: 12px;
        margin-right: 6px;
        color:#334155;
        border: 1px solid #E5E7EB;
    }}
    .divider {{
        height: 1px; background:#E5E7EB; width:100%; margin: 8px 0 14px 0;
    }}
    .footer-note {{
        color:#64748B; font-size:12px;
    }}
</style>
""",
    unsafe_allow_html=True,
)

# ===============================
# Header
# ===============================
st.title(f"👕 {APP_TITLE}")
st.caption("Comfort & Performance Recommender • Built for apparel R&D, sourcing & product creation teams")

with st.container():
    st.markdown(
        """
<div class="hero">
    <span class="badge">AI-Assisted</span>
    <span class="badge">Industry Ready</span>
    <span class="badge">Mobile Optimized</span>
    <h3 style="margin:10px 0 6px 0;">AI-Powered Fabric Comfort Recommender</h3>
    <p class="subtle">Tune climate, activity and business constraints to instantly get the best fabric options with explainability.</p>
</div>
""",
        unsafe_allow_html=True,
    )

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

# Helpful derived facets
has_price = "price" in df_clean.columns
has_gsm = "gsm" in df_clean.columns
has_fiber = "fabric_type" in df_clean.columns
has_sustain = "sustainability_score" in df_clean.columns  # optional user column

# State for shortlist
if "shortlist" not in st.session_state:
    st.session_state.shortlist = set()

# ===============================
# Tabs
# ===============================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📌 Recommender", "📊 Insights", "🤖 Model Performance", "🧾 Reports & About"]
)

# =========================================================
# TAB 1 – Recommender (Industry workflow)
# =========================================================
with tab1:
    st.subheader("⚙️ Set Conditions & Constraints")

    colA, colB = st.columns([1, 1])
    with colA:
        mode = st.radio("User Mode", ["Beginner (Easy)", "Pro (Detailed)"], horizontal=True)

        if mode == "Beginner (Easy)":
            preset = st.selectbox(
                "Quick Presets",
                ["Summer Running", "Winter Walk", "Gym Workout", "Office Day"],
                index=0,
            )
            if preset == "Summer Running":
                temperature, humidity, sweat_sensitivity, activity_intensity = 32, 70, "High", "High"
            elif preset == "Winter Walk":
                temperature, humidity, sweat_sensitivity, activity_intensity = 10, 50, "Low", "Low"
            elif preset == "Gym Workout":
                temperature, humidity, sweat_sensitivity, activity_intensity = 22, 60, "Medium", "Moderate"
            else:  # Office Day
                temperature, humidity, sweat_sensitivity, activity_intensity = 24, 45, "Medium", "Low"
        else:
            temperature = st.slider("🌡️ Outdoor Temperature (°C)", 0, 50, 28)
            humidity_level = st.select_slider(
                "💧 Perceived Humidity",
                options=["Dry", "Comfortable", "Humid", "Very Humid"],
                value="Humid",
            )
            humidity_map = {"Dry": 30, "Comfortable": 50, "Humid": 70, "Very Humid": 90}
            humidity = humidity_map[humidity_level]
            sweat_sensitivity = st.select_slider("🧍 Sweat Sensitivity", ["Low", "Medium", "High"], value="High")
            activity_intensity = st.select_slider("🏃 Activity Intensity", ["Low", "Moderate", "High"], value="High")

    with colB:
        st.write("**Business Constraints**")
        # Optional dynamic filters based on dataset presence
        fiber_filter = None
        if has_fiber:
            available_fibers = ["(Any)"] + sorted(df_clean["fabric_type"].astype(str).unique().tolist())
            fiber_filter = st.selectbox("Preferred Fabric Type", available_fibers, index=0)

        price_band = None
        if has_price:
            price_band = st.select_slider(
                "Target Price Band (if available)",
                options=["Any", "Budget", "Mid", "Premium"],
                value="Any",
            )

        max_gsm = None
        if has_gsm:
            max_gsm = st.slider("Max GSM (optional)", int(df_clean["gsm"].min()), int(df_clean["gsm"].max()), int(df_clean["gsm"].max()))

        sustain_w = 0.0
        if has_sustain:
            sustain_w = st.slider("Sustainability Priority (0=Off, 1=High)", 0.0, 1.0, 0.2, 0.05)

        top_k = st.slider("How many recommendations?", 3, 10, 5)

    # --------------------------
    # Encode inputs
    # --------------------------
    sweat_map = {"Low": 1, "Medium": 2, "High": 3}
    activity_map = {"Low": 1, "Moderate": 2, "High": 3}
    sweat_num, activity_num = sweat_map[sweat_sensitivity], activity_map[activity_intensity]

    # Heuristic engineered inputs (align to your model features ordering)
    # Adjust these if your utils expect a different feature order.
    user_features = np.array(
        [[
            sweat_num * 5,                 # e.g., perspiration factor
            800 + humidity * 5,           # e.g., moisture mg/m2 baseline + humidity
            60 + activity_num * 10,       # e.g., ventilation proxy
            0.04 + (temperature - 25)*0.001  # e.g., thermal conductivity proxy
        ]]
    )
    user_scaled = scaler.transform(user_features)
    predicted_score = float(model.predict(user_scaled)[0])

    # --------------------------
    # Score & rank candidates
    # --------------------------
    df_work = df_clean.copy()
    df_work["predicted_diff"] = (df_work[target_col] - predicted_score).abs()

    # Apply Business Constraints
    if fiber_filter and fiber_filter != "(Any)" and has_fiber:
        df_work = df_work[df_work["fabric_type"].astype(str) == fiber_filter]

    if has_price and price_band != "Any":
        # You can tune these thresholds to match your pricing schema
        q1, q2, q3 = df_clean["price"].quantile([0.33, 0.66, 0.90])
        if price_band == "Budget":
            df_work = df_work[df_work["price"] <= q1]
        elif price_band == "Mid":
            df_work = df_work[(df_work["price"] > q1) & (df_work["price"] <= q2)]
        else:  # Premium
            df_work = df_work[df_work["price"] > q2]

    if has_gsm and max_gsm is not None:
        df_work = df_work[df_work["gsm"] <= max_gsm]

    # Composite ranking: proximity + (optional) sustainability
    eps = 1e-9
    inv_prox = 1.0 / (df_work["predicted_diff"] + eps)  # higher is better
    if has_sustain and sustain_w > 0:
        # Normalize sustainability 0..1
        s_norm = (df_work["sustainability_score"] - df_work["sustainability_score"].min()) / (
            df_work["sustainability_score"].max() - df_work["sustainability_score"].min() + eps
        )
        df_work["rank_score"] = (1 - sustain_w) * inv_prox + sustain_w * s_norm
    else:
        df_work["rank_score"] = inv_prox

    # Sort by composite score
    ranked = df_work.sort_values("rank_score", ascending=False).head(top_k).copy()

    # Similarity score (0–100): scale inv_prox within current candidate set
    inv_min, inv_max = inv_prox.min(), inv_prox.max()
    ranked["similarity"] = ((1.0 / (ranked["predicted_diff"] + eps)) - inv_min) / (inv_max - inv_min + eps) * 100.0
    ranked["similarity"] = ranked["similarity"].clip(0, 100).round(1)

    # Lightweight explainability: show nearest-feature deltas against user intent proxy
    # We'll compute z-scores across key features to say "good for breathability / moisture / thermal balance".
    expl_cols = feature_cols[: min(6, len(feature_cols))]
    z = (ranked[expl_cols] - df_clean[expl_cols].mean()) / (df_clean[expl_cols].std() + eps)
    ranked["_explain"] = z.apply(
        lambda r: ", ".join(
            [f"{c}: {'+' if r[c]>0 else ''}{r[c]:.1f}σ" for c in expl_cols[:3]]
        ),
        axis=1,
    )

    # --------------------------
    # UI – Recommendations
    # --------------------------
    st.markdown("### 🔹 Recommended Fabrics")

    # Search within results
    search_txt = st.text_input("Search within results (e.g., 'polyester', 'mesh', 'cooling')", "")

    to_show = ranked.copy()
    if search_txt.strip():
        mask = pd.Series(True, index=to_show.index)
        for col in ["fabric_type", "description", "notes"]:
            if col in to_show.columns:
                mask = mask & to_show[col].astype(str).str.contains(search_txt, case=False, na=False)
        to_show = to_show[mask]

    # Responsive columns
    num_cols = 3 if to_show.shape[0] >= 3 else to_show.shape[0] if to_show.shape[0] > 0 else 1
    cols = st.columns(num_cols)

    # Render cards
    for i, (idx, row) in enumerate(to_show.iterrows()):
        with cols[i % num_cols]:
            with st.container():
                st.markdown(
                    f"""
<div class="metric-card">
  <div class="rec-title">🧵 {row.get('fabric_type','Unknown')}</div>
  <div class="pill">Similarity {row['similarity']}%</div>
  <div class="pill">Comfort: <b>{row[target_col]:.2f}</b></div>
  {"<div class='pill'>Sustainability 🌱: <b>"+str(row['sustainability_score'])+"</b></div>" if has_sustain else ""}
  {"<div class='pill'>GSM: <b>"+str(int(row['gsm']))+"</b></div>" if has_gsm else ""}
  {"<div class='pill'>Price: <b>"+str(row['price'])+"</b></div>" if has_price else ""}
  <div class="divider"></div>
  <div class="subtle"><b>Why this pick:</b> {row['_explain']}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    st.button("📌 Shortlist", key=f"pin-{idx}", on_click=lambda k=idx: st.session_state.shortlist.add(k))
                with c2:
                    st.button("ℹ️ Details", key=f"det-{idx}", on_click=lambda r=row: st.session_state.update({"_detail": r.to_dict()}))
                with c3:
                    st.button("🧪 Compare", key=f"cmp-{idx}", on_click=lambda: st.session_state.update({"_compare": True}))

    st.caption("Recommendations are ranked by proximity to your target comfort score, then adjusted by constraints and sustainability weight (if provided).")

    # --------------------------
    # Compare view (top_k table)
    # --------------------------
    st.markdown("#### 🧪 Compare Selected Set")
    cmp_df = ranked[
        ["fabric_type", target_col, "similarity", "predicted_diff"]
        + ([ "sustainability_score"] if has_sustain else [])
        + ([ "gsm"] if has_gsm else [])
        + ([ "price"] if has_price else [])
    ].rename(columns={target_col: "Comfort Score", "predicted_diff": "Δ to Target"})
    st.dataframe(cmp_df, use_container_width=True)

    # Quick chart: comfort vs similarity
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

    # --------------------------
    # Download actions
    # --------------------------
    def _build_downloadables(df_rec: pd.DataFrame) -> tuple[bytes, bytes]:
        csv_bytes = df_rec.to_csv(index=False).encode("utf-8")
        # Basic HTML printable report (works as “PDF via print”)
        html = f"""
<html>
<head><meta charset="utf-8"><title>Fabric Recommendations</title></head>
<body style="font-family:Inter,system-ui,Arial,sans-serif;padding:24px;">
<h2 style="margin:0 0 12px 0;">{APP_TITLE} – Recommendations</h2>
<p>Generated via Streamlit. Top-{len(df_rec)} ranked fabrics.</p>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
<thead><tr>
{"".join([f"<th>{c}</th>" for c in df_rec.columns])}
</tr></thead>
<tbody>
{"".join(["<tr>"+ "".join([f"<td>{row[c]}</td>" for c in df_rec.columns]) + "</tr>" for _,row in df_rec.iterrows()])}
</tbody>
</table>
</body></html>
"""
        return csv_bytes, html.encode("utf-8")

    dl_cols = st.columns([1, 1, 2])
    with dl_cols[0]:
        csv_bytes, html_bytes = _build_downloadables(cmp_df)
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="recommendations.csv", mime="text/csv")
    with dl_cols[1]:
        st.download_button("🖨️ Print/Save PDF (HTML)", data=html_bytes, file_name="recommendations.html", mime="text/html")
    with dl_cols[2]:
        st.caption("Tip: Open the HTML and print to PDF for a quick shareable report.")

    # --------------------------
    # Shortlist (session)
    # --------------------------
    if st.session_state.shortlist:
        st.markdown("#### 📌 Shortlist (this session)")
        short_df = ranked[ranked.index.isin(st.session_state.shortlist)]
        if not short_df.empty:
            st.dataframe(short_df[["fabric_type", target_col, "similarity"]], use_container_width=True)
        else:
            st.write("No shortlisted items from current results yet.")

# =========================================================
# TAB 2 – Insights (Data)
# =========================================================
with tab2:
    st.subheader("📊 Dataset Insights")

    # Snapshot
    st.markdown("**Preview (first 12 rows)**")
    st.dataframe(df_clean.head(12), use_container_width=True)

    st.markdown("**Summary Statistics**")
    st.dataframe(df_clean.describe(include="all").T, use_container_width=True)

    st.markdown("**Correlation Heatmap**")
    corr_cols = list(dict.fromkeys(feature_cols + [target_col]))
    corr = df_clean[corr_cols].corr().reset_index().melt("index")
    heatmap = (
        alt.Chart(corr)
        .mark_rect()
        .encode(
            x=alt.X("index:O", title="Feature"),
            y=alt.Y("variable:O", title="Feature"),
            color=alt.Color("value:Q", title="Correlation"),
            tooltip=["index", "variable", alt.Tooltip("value:Q", format=".2f")],
        )
    )
    st.altair_chart(heatmap, use_container_width=True)

# =========================================================
# TAB 3 – Model Performance
# =========================================================
with tab3:
    st.subheader("🤖 Model Performance")

    metrics = evaluate_model(model, X_test, y_test)
    c1, c2 = st.columns(2)
    c1.metric("R² Score", f"{metrics['r2']:.3f}")
    c2.metric("RMSE", f"{metrics['rmse']:.3f}")

    # Feature importances (if model exposes them)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
        feat_chart = (
            alt.Chart(feat_df.sort_values("Importance", ascending=False))
            .mark_bar()
            .encode(x=alt.X("Feature:N", sort="-y"), y="Importance:Q", tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")])
        )
        st.markdown("**Feature Importances**")
        st.altair_chart(feat_chart, use_container_width=True)
    else:
        st.info("Model does not expose feature importances.")

# =========================================================
# TAB 4 – Reports & About
# =========================================================
with tab4:
    st.subheader("🧾 Reports & About")

    st.markdown(
        f"""
- **Project**: *{APP_TITLE}*  
- **Purpose**: AI-assisted **fabric comfort & performance** recommendation for apparel R&D, sourcing, and design.  
- **How it ranks**: predicted comfort proximity → apply constraints (fiber/price/GSM) → optional sustainability weight → similarity scoring with explainability highlights.  
- **Outputs**: on-screen cards, compare table, CSV/PDF-style HTML report, session shortlist.  

**User Tips**
- Use **Pro Mode** for granular climate control and constraint tuning.  
- Try **Search** to quickly find constructions (e.g., “mesh”, “polyester”).  
- Use **Shortlist** to pin candidates across iterations during workshops or vendor calls.  

<div class="footer-note">© Built by Volando Fernando — Streamlit, Pandas, Altair, and scikit-learn.</div>
""",
        unsafe_allow_html=True,
    )
