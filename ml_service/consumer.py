import redis
import time
from datetime import datetime
from ml.detect import predict

REDIS_HOST = "redis"
REDIS_PORT = 6379

TRAFFIC_STREAM = "traffic_stream"
ALERTS_STREAM = "alerts_stream"

GROUP_NAME = "ml_group"
CONSUMER_NAME = "ml_consumer"

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

def process_alert(data):
    """
    Run ML prediction and publish result to alerts_stream
    """
    risk = predict(data)
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
            process_alert(data)
            redis_client.xack(TRAFFIC_STREAM, GROUP_NAME, msg_id)

