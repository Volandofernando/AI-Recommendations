import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =========================================================
# Load Config
# =========================================================
def load_config(path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# =========================================================
# Load Dataset(s)
# =========================================================
def load_datasets(config: dict) -> pd.DataFrame:
    """
    Load one or more datasets based on config. Supports CSV and Excel.
    If multiple datasets are listed, they will be concatenated.
    """
    paths = config["data"].get("paths")
    if not paths:
        # fallback for backward compatibility
        path = config["data"].get("path")
        if not path:
            raise FileNotFoundError("No dataset path(s) defined in config.")
        paths = [path]

    dfs = []
    for data_path in paths:
        # Handle remote GitHub URLs
        if data_path.startswith("http") and "blob" in data_path:
            data_path = data_path.replace("blob", "raw")

        try:
            if data_path.endswith(".csv"):
                df = pd.read_csv(data_path)
            elif data_path.endswith((".xls", ".xlsx")):
                df = pd.read_excel(data_path)
            else:
                raise ValueError(f"Unsupported dataset format: {data_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {data_path}: {e}")

        # Drop duplicates & reset index
        df = df.drop_duplicates().reset_index(drop=True)

        # Drop rows with missing target
        target = config["data"].get("target")
        if target and target in df.columns:
            df = df.dropna(subset=[target])

        dfs.append(df)

    # Concatenate all datasets
    if not dfs:
        raise ValueError("No valid datasets loaded.")
    return pd.concat(dfs, ignore_index=True)


# =========================================================
# Detect Features & Target
# =========================================================
def detect_features_and_target(df: pd.DataFrame, config: dict):
    """
    Identify features and target column.
    Uses config overrides if provided.
    """
    target = config["data"].get("target")
    features = config["data"].get("features")

    if target and target in df.columns:
        target_col = target
    else:
        # Heuristic: pick numeric column with "comfort" in name
        target_col = None
        for col in df.columns:
            if "comfort" in col.lower():
                target_col = col
                break

    if features:
        feature_cols = [c for c in features if c in df.columns]
    else:
        # Heuristic: all numeric except target
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c != target_col
        ]

    return feature_cols, target_col


# =========================================================
# Train Model
# =========================================================
def train_model(df, feature_cols, target_col, config):
    """
    Train Random Forest regression model on given dataset.
    Returns model, scaler, X_test, y_test, cleaned df.
    """
    X = df[feature_cols].copy()
    y = df[target_col].values

    # Handle NaNs in features
    X = X.fillna(X.median())

    # Split
    test_size = config["model"].get("test_size", 0.2)
    random_state = config["model"].get("random_state", 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    params = config["model"].get("params", {})
    model = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", None),
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # Return
    df_clean = df[feature_cols + [target_col]].copy()
    return model, scaler, X_test_scaled, y_test, df_clean


# =========================================================
# Evaluate Model
# =========================================================
def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate regression model with multiple metrics.
    """
    y_pred = model.predict(X_test)
    return {
        "r2": r2_score(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "mae": mean_absolute_error(y_test, y_pred),
    }
