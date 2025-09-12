import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_datasets(cfg):
    material_data = pd.read_excel(cfg["datasets"]["material_data"])
    survey_data = pd.read_excel(cfg["datasets"]["survey_data"])
    return material_data, survey_data

def preprocess(material_data, survey_data):
    # Example merge (you may customize how they join)
    df = pd.concat([material_data, survey_data], axis=0, ignore_index=True)
    df = df.dropna()
    return df

def train_model(df, target="Recommended Fabric"):
    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler
