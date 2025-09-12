import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# -------------------------------
# Load Config
# -------------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -------------------------------
# Load Datasets (from URLs or local)
# -------------------------------
def load_datasets(config):
    dfs = []
    for key, path in config["data"].items():
        try:
            df = pd.read_excel(path)
            dfs.append(df)
        except Exception as e:
            raise RuntimeError(f"❌ Error loading {key} dataset: {e}")
    return pd.concat(dfs, ignore_index=True)

# -------------------------------
# Detect Features + Target
# -------------------------------
def detect_features_and_target(df, config):
    features = config["ml"]["features"]
    target = config["ml"]["target"]

    if not all(f in df.columns for f in features) or target not in df.columns:
        return None, None
    return features, target

# -------------------------------
# Train Model
# -------------------------------
def train_model(df, features, target, config):
    df_clean = df.dropna(subset=features + [target]).copy()

    X = df_clean[features]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=config["ml"].get("n_estimators", 100),
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test_scaled, y_test, df_clean

# -------------------------------
# Evaluate Model
# -------------------------------
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "r2": round(r2_score(y_test, preds), 3),
        "rmse": round(mean_squared_error(y_test, preds, squared=False), 3),
    }
