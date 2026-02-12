import redis
import requests
import time

REDIS_HOST = "redis"
REDIS_PORT = 6379
ALERTS_STREAM = "alerts_stream"

GROUP_NAME = "alert_group"
CONSUMER_NAME = "alert_consumer"

API_URL = "http://fastapi:8000/alerts"

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Create consumer group
try:
    redis_client.xgroup_create(ALERTS_STREAM, GROUP_NAME, id="0", mkstream=True)
except redis.exceptions.ResponseError:
    pass  # group already exists

print("[ALERT ENGINE] Started 🚀")

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
            try:
                # Send alert to FastAPI
                resp = requests.post(API_URL, json=data)
                if resp.status_code == 200:
                    # Acknowledge in Redis
                    redis_client.xack(ALERTS_STREAM, GROUP_NAME, msg_id)
                    print(f"[ALERT ENGINE] Alert saved: {data}")
                else:
                    print(f"[ALERT ENGINE] Failed to save alert: {data}, status={resp.status_code}")
            except Exception as e:
                print(f"[ALERT ENGINE] Error: {e}")
