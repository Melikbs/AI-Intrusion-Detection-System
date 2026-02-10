import random
import time
import redis
from datetime import datetime

# Redis config (Docker service name!)
REDIS_HOST = "redis"
REDIS_PORT = 6379
STREAM_NAME = "traffic_stream"

# Connect to Redis
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True
)

PROTOCOL_MAP = {"tcp": 0, "udp": 1, "icmp": 2}
SERVICE_MAP = {"http": 0, "ftp": 1, "smtp": 2, "dns": 3, "other": 4}
FLAG_MAP = {"SF": 0, "S0": 1, "REJ": 2}

protocols = list(PROTOCOL_MAP.keys())
services = list(SERVICE_MAP.keys())
flags = list(FLAG_MAP.keys())

UNUSUAL_SERVICES = ["ftp", "smtp", "dns", "other"]


def determine_severity(src_bytes, dst_bytes, service_name):
    if src_bytes > 4000:
        return "HIGH"
    elif dst_bytes == 0 and service_name in UNUSUAL_SERVICES:
        return "MEDIUM"
    else:
        return "LOW"


def generate_packet():
    proto = random.choice(protocols)
    service = random.choice(services)
    flag = random.choice(flags)

    src_bytes = random.randint(0, 5000)
    dst_bytes = random.randint(0, 5000)

    severity = determine_severity(src_bytes, dst_bytes, service)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "alert_type": "TRAFFIC",
        "severity": severity,
        "protocol_type": proto,
        "service": service,
        "flag": flag,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
    }


def send_to_queue():
    packet = generate_packet()
    redis_client.xadd(STREAM_NAME, packet)
    print(f"[SIMULATOR] Sent to Redis → {packet}")


if __name__ == "__main__":
    print("[SIMULATOR] Traffic simulator started 🚀")
    while True:
        send_to_queue()
        time.sleep(2)

