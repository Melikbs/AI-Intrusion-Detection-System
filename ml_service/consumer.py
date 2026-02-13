import redis
import time
from datetime import datetime
import os
import joblib
import pandas as pd
from ml.features import engineer_features

# -------------------------
# Redis config
# -------------------------
REDIS_HOST = "redis"
REDIS_PORT = 6379

TRAFFIC_STREAM = "traffic_stream"
ALERTS_STREAM = "alerts_stream"

GROUP_NAME = "ml_group"
CONSUMER_NAME = "ml_consumer"

# -------------------------
# ML model config
# -------------------------
MODEL_PATH = "ml_service/ml/models/ids_pipeline_best.pkl"
MODEL_MTIME = None
pipeline = None

# -------------------------
# Connect to Redis
# -------------------------
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True
)

# Create consumer group (only once)
try:
    redis_client.xgroup_create(
        TRAFFIC_STREAM,
        GROUP_NAME,
        id="0",
        mkstream=True
    )
except redis.exceptions.ResponseError:
    pass  # Group already exists

# -------------------------
# Hot-reload ML model
# -------------------------
def reload_model():
    global pipeline, MODEL_MTIME
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Pipeline not found at {MODEL_PATH}. Run train_model.py first.")

    mtime = os.path.getmtime(MODEL_PATH)
    if MODEL_MTIME != mtime:
        pipeline = joblib.load(MODEL_PATH)
        MODEL_MTIME = mtime
        print(f"[ML] Loaded pipeline from {MODEL_PATH}")

# -------------------------
# ML prediction
# -------------------------
def predict(alert: dict, pipeline) -> float:
    """
    Takes ONE alert dict from Redis
    Returns a probabilistic risk score between 0 and 1
    """
    data = {
        "protocol_type": alert.get("protocol_type", "other"),
        "service": alert.get("service", "other"),
        "flag": alert.get("flag", "other"),
        "src_bytes": int(alert.get("src_bytes", 0)),
        "dst_bytes": int(alert.get("dst_bytes", 0)),
    }

    df = pd.DataFrame([data])
    df = engineer_features(df)
    risk_score = pipeline.predict_proba(df)[0][1]
    return round(float(risk_score), 3)

# -------------------------
# Process alerts from Redis
# -------------------------
def process_alert(data):
    reload_model()
    risk = predict(data, pipeline)

    print("DEBUG: input to ML:", data, flush=True)
    print("DEBUG: predicted risk:", risk, flush=True)

    alert = {
        "timestamp": datetime.utcnow().isoformat(),
        "alert_type": "ML_ALERT",
        "severity": data.get("severity", "LOW"),
        "protocol_type": data.get("protocol_type"),
        "service": data.get("service"),
        "src_bytes": data.get("src_bytes"),
        "dst_bytes": data.get("dst_bytes"),
        "risk_score": float(risk)
    }

    redis_client.xadd(ALERTS_STREAM, alert)

    print(
        f"[ML] {alert['protocol_type']} | {alert['service']} | "
        f"severity={alert['severity']} | risk={risk}"
    )

# -------------------------
# Main loop
# -------------------------
print("[ML] ML Service started 🚀")

while True:
    messages = redis_client.xreadgroup(
        groupname=GROUP_NAME,
        consumername=CONSUMER_NAME,
        streams={TRAFFIC_STREAM: ">"},
        count=5,
        block=5000
    )

    if not messages:
        continue

    for stream, events in messages:
        for msg_id, data in events:
            try:
                process_alert(data)
            except Exception as e:
                print(f"[ERROR] Failed to process message {msg_id}: {e}")
            finally:
                redis_client.xack(TRAFFIC_STREAM, GROUP_NAME, msg_id)


