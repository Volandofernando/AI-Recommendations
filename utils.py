"""
utils.py
Professional-grade utilities for Fabric Comfort Recommender.
Includes automatic column normalization, dataset cleaning, and model training.
"""

import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Config & Data Loading
# -------------------------------
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Mapping of messy dataset headers -> clean internal names
COLUMN_MAP = {
    "Moisture Absorption (%)": "absorption_rate",
    "Absorption Rate": "absorption_rate",
    "Dry Time (sec)": "drying_time",
    "Drying Time": "drying_time",
    "Thermal Conductivity (W/mK)": "thermal_conductivity",
    "Thermal Conductivity": "thermal_conductivity",
    "Air Flow (mm/s)": "air_permeability",
    "Air Permeability": "air_permeability",
    "Fabric GSM": "gsm",
    "GSM": "gsm",
    "Price ($)": "price",
    "Cost": "price",
    "Eco Index": "sustainability_score",
    "Sustainability": "sustainability_score",
    "Comfort Index": "comfort_score",
    "Comfort Score": "comfort_score",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: COLUMN_MAP.get(c, c) for c in df.columns})

def load_datasets(paths, features=None, target=None):
    dfs = []
    for p in paths:
        try:
            if p.endswith((".xls", ".xlsx")):
                df = pd.read_excel(p)
            else:
                df = pd.read_csv(p)
            dfs.append(normalize_columns(df))
        except Exception as e:
            print(f"⚠️ Could not load {p}: {e}")

    data = pd.concat(dfs, ignore_index=True)

    if features and target:
        cols = features + [target]

        # Convert all to numeric, coerce errors into NaN
        for c in cols:
            if c in data.columns:
                data[c] = pd.to_numeric(data[c], errors="coerce")

        # Replace infinities with NaN
        data = data.replace([np.inf, -np.inf], np.nan)

        # Drop rows missing target
        data = data.dropna(subset=[target])

        # Fill missing feature values with median
        for c in features:
            if c in data.columns:
                data[c] = data[c].fillna(data[c].median())

        print(f"✅ Clean dataset shape: {data.shape}")

    return data

# -------------------------------
# Model Training
# -------------------------------
def train_model(X, y, config):
    # Final safeguard against NaN/inf
    X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = pd.Series(y).replace([np.inf, -np.inf], np.nan).fillna(y.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"]
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestRegressor(
        **config["model"]["params"],
        random_state=config["model"]["random_state"]
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    return model, scaler, {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

# -------------------------------
# Feature Engineering
# -------------------------------
def construct_feature_vector(temperature, humidity, sweat_num, activity_num):
    return np.array([[ 
        sweat_num * 5,                      # Sweat scaling
        800 + humidity * 5,                 # Absorption
        60 + activity_num * 10,             # Ventilation
        0.04 + (temperature - 25) * 0.001   # Conductivity
    ]])

# -------------------------------
# Ranking Fabrics
# -------------------------------
def rank_fabrics(df, target_col, predicted_score, sustain_w=0.3):
    df = df.copy()
    eps = 1e-6
    df["predicted_diff"] = abs(df[target_col] - predicted_score)
    df["inv_prox"] = 1.0 / (df["predicted_diff"] + eps)
    df["sustain_norm"] = df["sustainability_score"].rank(pct=True)
    df["rank_score"] = (1 - sustain_w) * df["inv_prox"] + sustain_w * df["sustain_norm"]
    df["similarity_score"] = 100 * df["rank_score"] / df["rank_score"].max()
    return df.sort_values("similarity_score", ascending=False)

# -------------------------------
# Explainability
# -------------------------------
def explain_fabric(top_row, df_all, features):
    means = df_all[features].mean()
    stds = df_all[features].std(ddof=0).replace(0, np.nan)
    z = (top_row[features] - means) / stds
    explanations = {}
    for f in features:
        val = z[f]
        explanations[f] = "N/A" if pd.isna(val) else f"{val:+.2f}σ"
    return explanations

# -------------------------------
# Definitions for Material Properties
# -------------------------------
PROPERTY_DEFINITIONS = {
    "absorption_rate": "How quickly fabric absorbs sweat (mg/m²). Higher = faster absorption.",
    "drying_time": "Time needed for fabric to dry (seconds). Lower = better for active wear.",
    "thermal_conductivity": "Heat conduction ability (W/mK). Lower = better insulation.",
    "air_permeability": "Air flow through fabric (mm/s). Higher = more breathable.",
    "gsm": "Weight per square meter. Higher = thicker/heavier fabric.",
    "price": "Relative cost of the fabric.",
    "sustainability_score": "Eco-friendliness index (survey/literature derived).",
    "comfort_score": "Target label: perceived comfort index (1–10)."
}
