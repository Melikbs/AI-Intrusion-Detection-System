import random
import time
import requests
from datetime import datetime

API_URL = "http://fastapi:8000/alerts"

protocols = ["tcp", "udp", "icmp"]
services = ["http", "ftp", "smtp", "dns", "other"]
flags = ["SF", "S0", "REJ"]

def generate_alert():
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "severity": random.choice(["LOW", "MEDIUM", "HIGH"]),
        "alert_type": "INTRUSION_DETECTED",
        "protocol_type": random.choice(protocols),
        "service": random.choice(services),
        "flag": random.choice(flags),
        "src_bytes": random.randint(0, 5000),
        "dst_bytes": random.randint(0, 5000),
    }

def main():
    print("[SIMULATOR] Traffic simulator started")

    while True:
        alert = generate_alert()
        try:
            r = requests.post(API_URL, json=alert, timeout=3)
            print(f"[SIMULATOR] Sent alert → {r.status_code}")
        except Exception as e:
            print(f"[SIMULATOR] Error sending alert: {e}")

        time.sleep(1)

if __name__ == "__main__":
    main()

