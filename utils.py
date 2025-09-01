# ================================
# utils.py – helper functions
# ================================

import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Load Config
# -------------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -------------------------------
# Load Datasets
# -------------------------------
def load_datasets(config):
    lit_url = config["Dataset.xlsx"]["https://github.com/Volandofernando/Material-Literature-data-/blob/main/Dataset.xlsx"]
    survey_url = config["IT Innovation in Fabric Industry (Responses).xlsx"]["https://github.com/Volandofernando/REAL-TIME-Dataset/blob/main/IT%20Innovation%20in%20Fabric%20Industry%20%20(Responses).xlsx"]

    df_lit = pd.read_excel(lit_url)
    df_survey = pd.read_excel(survey_url)

    def clean_columns(df):
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(r"[^\w]", "_", regex=True)
        )
        return df

    df_lit = clean_columns(df_lit)
    df_survey = clean_columns(df_survey)

    return pd.concat([df_lit, df_survey], ignore_index=True, sort=False)

# -------------------------------
# Detect Features & Target
# -------------------------------
def detect_features_and_target(df, config):
    feature_keywords = config["features"]
    target_keywords = config["target_keywords"]

    def find_column(df_cols, keywords):
        for col in df_cols:
            if all(k in col for k in keywords):
                return col
        return None

    feature_cols = [find_column(df.columns, kw) for kw in feature_keywords.values()]
    target_col = find_column(df.columns, target_keywords)

    feature_cols = [c for c in feature_cols if c is not None]
    return feature_cols, target_col

# -------------------------------
# Train Model
# -------------------------------
def train_model(df, feature_cols, target_col, config):
    df_clean = df.dropna(subset=feature_cols + [target_col])
    X, y = df_clean[feature_cols], df_clean[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=config["model"]["test_size"], random_state=config["model"]["random_state"]
    )

    model = RandomForestRegressor(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"]
    )
    model.fit(X_train, y_train)

    return model, scaler, X_test, y_test, df_clean

# -------------------------------
# Evaluate Model
# -------------------------------
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "r2": r2_score(y_test, preds),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds)))
    }
