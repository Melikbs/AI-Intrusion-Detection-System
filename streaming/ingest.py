import time
import json
import requests
import pandas as pd
import joblib
from datetime import datetime
from streaming.traffic_simulator import generate_traffic  # your traffic simulator function

# Load trained model and encoders
model = joblib.load("ml/ids_model.pkl")
encoders = joblib.load("ml/encoders.pkl")

# Columns expected by the ML model
COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

# FastAPI endpoint
FASTAPI_URL = "http://fastapi:8000/alerts"

def prepare_features(packet):
    """
    Convert simulator packet into ML features
    """
    df = pd.DataFrame([packet], columns=COLUMNS)
    
    # Encode categorical columns
    for col in ["protocol_type", "service", "flag"]:
        if col in df:
            df[col] = encoders[col].transform(df[col])
    
    return df

def send_alert(alert):
    """
    Send alert to FastAPI
    """
    try:
        response = requests.post(FASTAPI_URL, json=alert)
        if response.status_code == 200:
            print(f"[+] Alert sent: {alert['alert_type']} at {alert['timestamp']}")
        else:
            print(f"[!] Failed to send alert: {response.status_code}")
    except Exception as e:
        print(f"[!] Exception sending alert: {e}")

def live_ingest():
    """
    Main loop: get traffic, detect, and send alerts
    """
    print("[ML] Live ingestion started...")
    for packet in generate_traffic():  # generator from simulator
        try:
            features = prepare_features(packet)
            prediction = model.predict(features)[0]
            if prediction == 1:
                alert = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": "HIGH",
                    "alert_type": "INTRUSION_DETECTED",
                    **packet
                }
                send_alert(alert)
            time.sleep(0.1)  # adjust speed of ingestion
        except Exception as e:
            print(f"[!] Error processing packet: {e}")

if __name__ == "__main__":
    live_ingest()
