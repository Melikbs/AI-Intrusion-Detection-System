import random
import time
import requests
from datetime import datetime

API_URL = "http://fastapi:8000/alerts"
HEALTH_URL = "http://fastapi:8000/docs"

# Protocols, services, flags
PROTOCOL_MAP = {"tcp": 0, "udp": 1, "icmp": 2}
SERVICE_MAP = {"http": 0, "ftp": 1, "smtp": 2, "dns": 3, "other": 4}
FLAG_MAP = {"SF": 0, "S0": 1, "REJ": 2}

protocols = list(PROTOCOL_MAP.keys())
services = list(SERVICE_MAP.keys())
flags = list(FLAG_MAP.keys())

# Define unusual services for MEDIUM alerts
UNUSUAL_SERVICES = ["ftp", "smtp", "dns", "other"]


def wait_for_fastapi():
    """Wait until FastAPI is reachable"""
    while True:
        try:
            r = requests.get(HEALTH_URL, timeout=2)
            if r.status_code == 200:
                print("[SIMULATOR] FastAPI is ready ✅")
                return
        except requests.exceptions.RequestException:
            print("[SIMULATOR] Waiting for FastAPI...")
        time.sleep(3)


def determine_severity(src_bytes, dst_bytes, service_name):
    """
    Determine alert severity based on thresholds:
    - src_bytes > 4000 → HIGH
    - dst_bytes == 0 and unusual service → MEDIUM
    - else → LOW
    """
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

    # Determine severity properly
    severity = determine_severity(src_bytes, dst_bytes, service)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "alert_type": "TRAFFIC",
        "severity": severity,
        "protocol_type": proto,  # send real name
        "service": service,      # send real name
        "flag": flag,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
    }


def send_alert():
    packet = generate_packet()
    try:
        r = requests.post(API_URL, json=packet, timeout=5)
        print(f"[SIMULATOR] {r.status_code} → {packet}")
    except requests.exceptions.RequestException as e:
        print(f"[SIMULATOR] Error sending alert: {e}")


if __name__ == "__main__":
    wait_for_fastapi()
    while True:
        send_alert()
        time.sleep(2)

