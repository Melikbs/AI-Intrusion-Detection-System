import redis
import requests
import time

REDIS_HOST = "redis"
REDIS_PORT = 6379

ALERTS_STREAM = "alerts_stream"

GROUP_NAME = "alert_group"
CONSUMER_NAME = "alert_consumer"

FASTAPI_URL = "http://fastapi:8000/alerts"

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True
)

# Create consumer group (only once)
try:
    redis_client.xgroup_create(
        ALERTS_STREAM,
        GROUP_NAME,
        id="0",
        mkstream=True
    )
except redis.exceptions.ResponseError:
    pass


def send_to_api(alert_data):
    try:
        response = requests.post(FASTAPI_URL, json=alert_data, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"[ALERT_ENGINE] API error: {e}")
        return False


print("[ALERT_ENGINE] Alert Engine started 🚀")

while True:
    messages = redis_client.xreadgroup(
        groupname=GROUP_NAME,
        consumername=CONSUMER_NAME,
        streams={ALERTS_STREAM: ">"},
        count=5,
        block=5000
    )

    if not messages:
        continue

    for stream, events in messages:
        for msg_id, data in events:
            print(f"[ALERT_ENGINE] Processing alert {msg_id}")

            clean_alert = {
                "timestamp": data["timestamp"],
                "alert_type": data["alert_type"],
                "severity": data["severity"],
                "protocol_type": data["protocol_type"],
                "service": data["service"],
                "src_bytes": int(data["src_bytes"]),
                "dst_bytes": int(data["dst_bytes"]),
                "risk_score": float(data["risk_score"]),
            }

            success = send_to_api(clean_alert)


            if success:
                redis_client.xack(ALERTS_STREAM, GROUP_NAME, msg_id)
                print("[ALERT_ENGINE] Saved to Postgres ✅")
            else:
                print("[ALERT_ENGINE] Failed, will retry ⏳")
                time.sleep(3)

