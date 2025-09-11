import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

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
    try:
        df1 = pd.read_excel(config["data"]["dataset1"])
        df2 = pd.read_excel(config["data"]["dataset2"])
        df = pd.concat([df1, df2], ignore_index=True)
        return df.dropna()
    except Exception as e:
        raise Exception(f"Dataset load failed: {e}")

# -------------------------------
# Detect Features & Target
# -------------------------------
def detect_features_and_target(df, config):
    features = config["ml"]["features"]
    target = config["ml"]["target"]
    if not set(features).issubset(df.columns) or target not in df.columns:
        return [], None
    return features, target

# -------------------------------
# Train Model
# -------------------------------
def train_model(df, features, target, config):
    X = df[features].values
    y = df[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=config["ml"]["n_estimators"],
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, scaler, X_test, y_test, df

# -------------------------------
# Evaluate Model
# -------------------------------
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"rmse": round(rmse, 3), "r2": round(r2, 3)}

# -------------------------------
# Save & Load Model (optional for production)
# -------------------------------
def save_model(model, scaler, path="models/saved_model.pkl"):
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)

def load_model(path="models/saved_model.pkl"):
    if os.path.exists(path):
        return joblib.load(path)
    return None
