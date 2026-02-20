import pandas as pd
import joblib
import os
import json
from .features import engineer_features


# -------------------------
# Locate Latest Model Version
# -------------------------
BASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

if not os.path.exists(BASE_MODEL_DIR):
    raise FileNotFoundError("Models directory not found. Train the model first.")

# Detect latest version folder (v1, v2, ...)
versions = [
    d for d in os.listdir(BASE_MODEL_DIR)
    if d.startswith("v") and d[1:].isdigit()
]

if not versions:
    raise FileNotFoundError("No versioned models found. Run train_model.py first.")

latest_version = sorted(versions, key=lambda x: int(x[1:]))[-1]
VERSION_DIR = os.path.join(BASE_MODEL_DIR, latest_version)

print(f"[+] Loading models from version: {latest_version}")


# -------------------------
# Load All Models in Version
# -------------------------
pipelines = {}

for file in os.listdir(VERSION_DIR):
    if file.endswith(".pkl") and file != "best.pkl":
        model_name = file.replace(".pkl", "")
        model_path = os.path.join(VERSION_DIR, file)
        pipelines[model_name] = joblib.load(model_path)
        print(f"[+] Loaded {model_name} from {model_path}")

if not pipelines:
    raise FileNotFoundError("No ML pipelines found in latest version folder.")


# -------------------------
# Prediction Function
# -------------------------
def predict(alert: dict) -> float:
    """
    Takes ONE alert dict
    Returns probabilistic risk score (0-1) using ensemble averaging
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


