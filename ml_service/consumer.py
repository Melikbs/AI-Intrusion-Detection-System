import redis
import time
from datetime import datetime
from ml.detect import predict

REDIS_HOST = "redis"
REDIS_PORT = 6379
STREAM_NAME = "traffic_stream"
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
        STREAM_NAME,
        GROUP_NAME,
        id="0",
        mkstream=True
    )
except redis.exceptions.ResponseError:
    pass  # group already exists


def process_alert(data):
    risk = predict(data)  # Make sure this returns a number
    print("[DEBUG] risk =", risk)

    alert = {
        "timestamp": datetime.utcnow().isoformat(),
        "alert_type": "TRAFFIC",
        "severity": data.get("severity", "LOW"),
        "protocol_type": data.get("protocol_type", ""),
        "service": data.get("service", ""),
        "src_bytes": data.get("src_bytes", 0),
        "dst_bytes": data.get("dst_bytes", 0),
        "risk_score": risk
    }

    redis_client.xadd(ALERTS_STREAM, alert)

    print(
        f"[ML] {alert['protocol_type']} | {alert['service']} | "
        f"severity={alert['severity']} | risk={alert['risk_score']}"
    )


print("[ML] ML Service started 🚀")

# First consume **all existing messages**
last_id = "0"

while True:
    messages = redis_client.xreadgroup(
        groupname=GROUP_NAME,
        consumername=CONSUMER_NAME,
        streams={STREAM_NAME: last_id},
        count=5,
        block=5000
    )

    for stream, events in messages:
        for msg_id, data in events:
            process_alert(data)
            redis_client.xack(STREAM_NAME, GROUP_NAME, msg_id)
            last_id = msg_id  # move to next

