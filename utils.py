import pandas as pd
import numpy as np
import yaml
from typing import List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_config(path: str = "config.yaml") -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {e}")

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
    )
    return df

def load_datasets(config: dict) -> pd.DataFrame:
    try:
        df_lit = pd.read_excel(config["datasets"]["literature_url"])
        df_survey = pd.read_excel(config["datasets"]["survey_url"])
    except Exception as e:
        raise RuntimeError(f"Error loading datasets: {e}")

    df = pd.concat([clean_columns(df_lit), clean_columns(df_survey)], ignore_index=True, sort=False)
    if df.empty:
        raise ValueError("Combined dataset is empty — check your dataset URLs.")
    return df

def _find_first(df_cols: List[str], keywords_all: List[str]) -> Optional[str]:
    """
    Return first column that contains ALL keywords (lowercase containment match).
    """
    for col in df_cols:
        if all(k in col for k in keywords_all):
            return col
    return None

def detect_features_and_target(df: pd.DataFrame, config: dict) -> Tuple[List[str], Optional[str], Optional[str]]:
    cols = [c.lower() for c in df.columns]
    features_cfg = config.get("features", {})
    target_kw = [k.lower() for k in config.get("target_keywords", [])]
    fabric_kw = [k.lower() for k in config.get("fabric_name_keywords", [])]

    feature_cols: List[str] = []
    for _, kw_list in features_cfg.items():
        col = _find_first(cols, [k.lower() for k in kw_list])
        if col:
            feature_cols.append(col)

    target_col = _find_first(cols, target_kw) if target_kw else None

    # Best-effort fabric name column
    fabric_col = None
    for cand in ("fabric_type", "fabric", "material", "material_type", "type"):
        if cand in cols:
            fabric_col = cand
            break
    if fabric_col is None and fabric_kw:
        fabric_col = _find_first(cols, fabric_kw)

    return feature_cols, target_col, fabric_col

def train_model(df: pd.DataFrame, feature_cols: List[str], target_col: str, config: dict):
    required = feature_cols + [target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}\nAvailable: {df.columns.tolist()}")

    df_clean = df.dropna(subset=required).copy()
    if df_clean.empty:
        raise ValueError("No rows left after dropping NaNs for required columns.")

    X, y = df_clean[feature_cols], df_clean[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"]
    )

    model = RandomForestRegressor(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"]
    )
    model.fit(X_train, y_train)
    return model, scaler, X_test, y_test, df_clean

def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds)))
    }
