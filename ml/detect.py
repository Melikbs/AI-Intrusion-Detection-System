import pandas as pd
import joblib
from datetime import datetime

# Load model and encoders
model = joblib.load("ml/ids_model.pkl")
encoders = joblib.load("ml/encoders.pkl")

# Column names
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
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

# Load test data
df = pd.read_csv("data/raw/KDDTest+.txt", names=COLUMNS)

# Encode categorical columns
for col in ["protocol_type", "service", "flag"]:
    df[col] = encoders[col].transform(df[col])

# Prepare features
X = df.drop(["label", "difficulty"], axis=1)

# Predict
df["prediction"] = model.predict(X)

# Generate alerts
alerts = df[df["prediction"] == 1].copy()
alerts["timestamp"] = datetime.utcnow().isoformat()
alerts["severity"] = "HIGH"
alerts["alert_type"] = "INTRUSION_DETECTED"

# Save alerts
alerts_log = alerts[["timestamp", "alert_type", "severity", "protocol_type", "s>
alerts_log.to_csv("alerts_log.csv", index=False)

print(f"[+] Alerts generated: {len(alerts_log)}")
print("[+] alerts_log.csv created")


