"""
utils.py
Professional-grade utilities for Fabric Comfort Recommender.
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

def load_datasets(paths):
    dfs = []
    for p in paths:
        try:
            if p.endswith((".xls", ".xlsx")):
                dfs.append(pd.read_excel(p))
            else:
                dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"⚠️ Could not load {p}: {e}")
    return pd.concat(dfs, ignore_index=True)

# -------------------------------
# Model Training
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
        if pd.isna(val):
            explanations[f] = "N/A"
        else:
            explanations[f] = f"{val:+.2f}σ"
    return explanations

# -------------------------------
# Definitions for Material Properties
# -------------------------------
PROPERTY_DEFINITIONS = {
    "absorption_rate": "The rate at which a fabric absorbs moisture (mg/m²). Higher values = faster sweat absorption.",
    "drying_time": "The time fabric takes to dry after absorbing sweat. Lower is better for active wear.",
    "thermal_conductivity": "Ability of fabric to conduct heat (W/mK). Lower values = better insulation.",
    "air_permeability": "How easily air passes through fabric (mm/s). Higher = more ventilation & breathability.",
    "gsm": "Grams per square meter (GSM). Indicates fabric weight/thickness.",
    "price": "Relative fabric cost index.",
    "sustainability_score": "Eco-friendliness score, derived from survey/literature data."
}
