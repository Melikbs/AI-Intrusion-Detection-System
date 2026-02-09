from datetime import datetime
import random
import requests
import time

API_URL = "http://fastapi:8000/alerts"

def send_packet(packet):
    try:
        response = requests.post(API_URL, json=packet)
        if response.status_code != 200:
            print(f"Failed to send alert: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Error sending packet: {e}")

def generate_traffic():
    protocols = ["tcp", "udp", "icmp"]
    services = ["http", "ftp", "smtp", "dns", "other"]
    flags = ["SF", "S0", "REJ"]

    while True:
        packet = {
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "HIGH",
            "alert_type": "INTRUSION_DETECTED",
            "protocol_type": random.choice(protocols),
            "service": random.choice(services),
            "flag": random.choice(flags),
            "src_bytes": random.randint(0, 5000),
            "dst_bytes": random.randint(0, 5000),
        }
        send_packet(packet)
        time.sleep(1)  # send 1 packet per second

