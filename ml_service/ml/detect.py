import pandas as pd
import joblib
import numpy as np

# Load model and encoders ONCE
model = joblib.load("ml/ids_model.pkl")
encoders = joblib.load("ml/encoders.pkl")

FEATURE_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

def predict(alert: dict) -> float:
    """
    Takes ONE alert dict from Redis
    Returns probabilistic risk score between 0 and 1
    """

    # Minimal feature mapping
    data = {
        "duration": 0,
        "protocol_type": alert.get("protocol_type", "other"),
        "service": alert.get("service", "other"),
        "flag": alert.get("flag", "other"),
        "src_bytes": int(alert.get("src_bytes", 0)),
        "dst_bytes": int(alert.get("dst_bytes", 0)),
        **{col: 0 for col in FEATURE_COLUMNS if col not in [
            "duration","protocol_type","service","flag","src_bytes","dst_bytes"
        ]}
    }

    df = pd.DataFrame([data])

    # Ensure correct column order
    df = df[FEATURE_COLUMNS]

    # Encode categorical columns safely
    for col in ["protocol_type", "service", "flag"]:
        known_classes = set(encoders[col].classes_)
        df[col] = df[col].apply(lambda x: x if x in known_classes else "other")
        df[col] = encoders[col].transform(df[col])

    # 🔥 Probabilistic prediction (attack class = index 1)
    risk_score = model.predict_proba(df)[0][1]

    return round(float(risk_score), 3)


