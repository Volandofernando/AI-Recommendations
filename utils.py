"""
utils.py
Professional-grade utilities for AI-Powered Fabric Comfort Recommender
"""

import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Config
# -------------------------------
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -------------------------------
# Column Normalization
# -------------------------------
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
    "Comfort Score": "comfort_score"
}

def normalize_columns(df):
    return df.rename(columns={c: COLUMN_MAP.get(c, c) for c in df.columns})

# -------------------------------
# Data Loading
# -------------------------------
def load_datasets(paths):
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
    return pd.concat(dfs, ignore_index=True)

# -------------------------------
# Prepare Features + Target
# -------------------------------
def prepare_xy(df, features, target):
    df = df[features + [target]].copy()
    for col in features + [target]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]
    return X, y

# -------------------------------
# Train Model
# -------------------------------
def train_model(X, y, config):
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
        sweat_num * 5,
        800 + humidity * 5,
        60 + activity_num * 10,
        0.04 + (temperature - 25) * 0.001
    ]])

# -------------------------------
# Ranking Fabrics
# -------------------------------
def rank_fabrics(df, target_col, predicted_score, sustain_w=0.3):
    df = df.copy()
    df = df.dropna(subset=[target_col])
    eps = 1e-6
    df["predicted_diff"] = abs(df[target_col] - predicted_score)
    df["inv_prox"] = 1.0 / (df["predicted_diff"] + eps)
    if "sustainability_score" not in df.columns:
        df["sustainability_score"] = 0
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
    return {f: ("N/A" if pd.isna(z[f]) else f"{z[f]:+.2f}σ") for f in features}

# -------------------------------
# Property Definitions
# -------------------------------
PROPERTY_DEFINITIONS = {
    "absorption_rate": "How quickly fabric absorbs sweat. Higher = better absorption.",
    "drying_time": "How long fabric takes to dry. Lower = better.",
    "thermal_conductivity": "Heat transfer ability. Lower = more insulating.",
    "air_permeability": "Breathability of fabric. Higher = more airflow.",
    "gsm": "Fabric weight (g/m²). Higher = thicker fabric.",
    "price": "Relative cost of fabric.",
    "sustainability_score": "Eco-friendliness score from survey/literature.",
    "comfort_score": "Target comfort index (1–10)."
}
