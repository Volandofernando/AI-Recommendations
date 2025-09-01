import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_datasets(config):
    path = config["dataset"]["path"]
    if path.endswith(".xlsx"):
        df = pd.read_excel(path, sheet_name=config["dataset"].get("sheet_name", 0))
    else:
        df = pd.read_csv(path)
    return df

def detect_features_and_target(df, config):
    feature_cols = config.get("features", [])
    target_col = config.get("target")

    # Validate presence
    missing = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}. Found: {list(df.columns)}")

    return feature_cols, target_col

def train_model(df, feature_cols, target_col, config):
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, scaler, X_test, y_test, df_clean

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"rmse": round(rmse, 3), "r2": round(r2, 3)}
