import pandas as pd
import joblib
from .features import engineer_features

# Load full pipeline (preprocessing + model)
pipeline = joblib.load("ml/models/ids_pipeline_v1.pkl")

def predict(alert: dict) -> float:
    """
    Takes ONE alert dict from Redis
    Returns a probabilistic risk score between 0 and 1
    """
    # Minimal feature mapping
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



