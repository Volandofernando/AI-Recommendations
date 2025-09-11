# utils.py
import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# ==============================
# Load Config
# ==============================
def load_config(path="config.yaml"):
    """
    Load YAML config file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"⚠️ Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ==============================
# Load Datasets
# ==============================
def load_datasets(config):
    """
    Load dataset based on config
    """
    dataset_path = config["data"]["path"]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"⚠️ Dataset not found at {dataset_path}")

    df = pd.read_excel(dataset_path) if dataset_path.endswith(".xlsx") else pd.read_csv(dataset_path)
    return df

# ==============================
# Detect Features & Target
# ==============================
def detect_features_and_target(df, config):
    """
    Detect feature columns and target column
    """
    feature_cols = config["model"].get("features", [])
    target_col = config["model"].get("target")

    if not feature_cols:
        # fallback: all numeric except last column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            feature_cols, target_col = numeric_cols[:-1], numeric_cols[-1]
        else:
            return [], None

    return feature_cols, target_col

# ==============================
# Train Model
# ==============================
def train_model(df, feature_cols, target_col, config):
    """
    Train ML model and return fitted model, scaler, and test sets
    """
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"].get("test_size", 0.2), random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = RandomForestRegressor(
        n_estimators=config["model"].get("n_estimators", 100),
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test_scaled, y_test, df_clean

# ==============================
# Evaluate Model
# ==============================
def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model performance
    """
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    return {"r2": round(r2, 3), "rmse": round(rmse, 3)}
