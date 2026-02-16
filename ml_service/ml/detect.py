import pandas as pd
import joblib
import os
from .features import engineer_features

# Path to the best model
MODEL_PATH = "ml/models/ids_pipeline_best.pkl"

# Load the best pipeline if it exists
if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)
    print(f"[+] Loaded ML pipeline from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Pipeline not found at {MODEL_PATH}. Run train_model.py first.")

def predict(alert: dict) -> float:
    """
    Takes ONE alert dict from Redis
    Returns a probabilistic risk score between 0 and 1
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

    # Predict probability for attack (class=1)
    risk_score = pipeline.predict_proba(df)[0][1]
    return round(float(risk_score), 3)



