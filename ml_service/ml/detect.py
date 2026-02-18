import pandas as pd
import joblib
import os
from .features import engineer_features

# Paths to the saved models
MODEL_PATHS = {
    "RandomForest": "ml/models/ids_pipeline_randomforest.pkl",
    "XGBoost": "ml/models/ids_pipeline_xgboost.pkl",
    "SVM": "ml/models/ids_pipeline_svm.pkl"
}

# Load all models into a dictionary
pipelines = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        pipelines[name] = joblib.load(path)
        print(f"[+] Loaded {name} pipeline from {path}")
    else:
        print(f"[!] Warning: {name} pipeline not found at {path}. Train it first.")

if not pipelines:
    raise FileNotFoundError("No ML pipelines found. Run train_model.py first.")

def predict(alert: dict) -> float:
    """
    Takes ONE alert dict
    Returns a probabilistic risk score between 0 and 1 using ensemble averaging
    """
    # Minimal feature mapping with defaults
    data = {
        "protocol_type": alert.get("protocol_type", "other"),
        "service": alert.get("service", "other"),
        "flag": alert.get("flag", "other"),
        "src_bytes": int(alert.get("src_bytes", 0)),
        "dst_bytes": int(alert.get("dst_bytes", 0)),
    }

    df = pd.DataFrame([data])
    df = engineer_features(df)

    # Collect predictions from all models
    prob_scores = []
    for name, pipe in pipelines.items():
        prob = pipe.predict_proba(df)[0][1]  # probability of attack
        prob_scores.append(prob)

    # Ensemble: average probabilities
    ensemble_score = sum(prob_scores) / len(prob_scores)
    return round(float(ensemble_score), 3)



