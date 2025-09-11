import os
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -------------------
# Config Loader
# -------------------
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -------------------
# Column Normalization Map
# -------------------
COLUMN_MAP = {
    "Moisture Absorption (%)": "absorption_rate",
    "Absorption Rate": "absorption_rate",
    "Dry Time (sec)": "drying_time",
    "Drying Time": "drying_time",
    "Thermal Conductivity (W/mK)": "thermal_conductivity",
    "Air Permeability": "air_permeability",
    "Fabric GSM": "gsm",
    "Price ($)": "price",
    "Eco Index": "sustainability_score",
    "Comfort Score": "comfort_score",
    "Comfort Index": "comfort_score",
    "Perceived Comfort": "comfort_score",
}

PROPERTY_DEFS = {
    "absorption_rate": "Liquid absorption ability (g/m²). Higher = faster sweat uptake.",
    "drying_time": "Time to dry (min). Lower = better.",
    "thermal_conductivity": "Heat conduction (W/mK). Lower = warmer, higher = cooler.",
    "air_permeability": "Airflow through fabric (mm/s). Higher = more breathable.",
    "gsm": "Grams per square meter (weight).",
    "price": "Relative price or cost.",
    "sustainability_score": "Eco index (higher = more sustainable).",
    "comfort_score": "Target comfort rating (1–10)."
}

def normalize_columns(df):
    return df.rename(columns={c: COLUMN_MAP.get(c, c) for c in df.columns})

# -------------------
# Dataset Loader
# -------------------
def load_datasets(paths, features, target):
    dfs, errors = [], []
    for p in paths:
        try:
            if p.endswith((".xls", ".xlsx")):
                df = pd.read_excel(p)
            else:
                df = pd.read_csv(p)
            dfs.append(normalize_columns(df))
        except Exception as e:
            errors.append({p: str(e)})

    if not dfs:
        raise RuntimeError("No datasets could be loaded.")

    data = pd.concat(dfs, ignore_index=True)

    # Ensure numeric
    for col in features + [target]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop missing target
    data = data.dropna(subset=[target])
    # Impute features
    for f in features:
        if f in data.columns:
            data[f] = data[f].fillna(data[f].median())

    return data, {"rows": len(data), "errors": errors}

# -------------------
# Model Training
# -------------------
def train_model(X, y, cfg):
    X = X.fillna(0)
    y = y.fillna(y.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["model"]["test_size"], random_state=cfg["model"]["random_state"]
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestRegressor(random_state=cfg["model"]["random_state"], **cfg["model"]["params"])
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    metrics = {"r2": r2_score(y_test, y_pred), "mse": mean_squared_error(y_test, y_pred)}

    return model, scaler, metrics

# -------------------
# Feature Engineering (User → Model)
# -------------------
def construct_feature_vector(temp, humidity, sweat_num, activity_num):
    return np.array([[ 
        sweat_num * 5,
        800 + humidity * 5,
        60 + activity_num * 10,
        0.04 + (temp - 25) * 0.001
    ]])

# -------------------
# Ranking
# -------------------
def rank_fabrics(df, target, predicted, sustain_w=0.2):
    eps = 1e-9
    df = df.copy()
    df["predicted_diff"] = (df[target] - predicted).abs()
    df["inv_prox"] = 1.0 / (df["predicted_diff"] + eps)

    if "sustainability_score" in df.columns:
        df["sustain_norm"] = (df["sustainability_score"] - df["sustainability_score"].min()) / (
            df["sustainability_score"].max() - df["sustainability_score"].min() + eps
        )
    else:
        df["sustain_norm"] = 0.5

    df["rank_score"] = (1 - sustain_w) * df["inv_prox"] + sustain_w * df["sustain_norm"]
    df["similarity_score"] = 100 * (df["rank_score"] - df["rank_score"].min()) / (
        df["rank_score"].max() - df["rank_score"].min() + eps
    )
    return df.sort_values("similarity_score", ascending=False).reset_index(drop=True)

# -------------------
# Explainability (z-scores)
# -------------------
def explain_fabric(row, df, features):
    means = df[features].mean()
    stds = df[features].std(ddof=0).replace(0, np.nan)
    z = (row[features] - means) / stds
    return {f: f"{z[f]:+.2f}σ" if pd.notna(z[f]) else "N/A" for f in features if f in row}
