import pandas as pd
import numpy as np
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -------------------------
# Columns in KDD dataset
# -------------------------
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
    "dst_host_srv_rerror_rate","label","difficulty"
]

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("data/raw/KDDTrain+.txt", names=COLUMNS)

# Binary label
df["attack"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

# -------------------------
# Keep only real-time features
# -------------------------
df = df[[
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "attack"
]]

# -------------------------
# Feature Engineering
# -------------------------
df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
df["total_bytes"] = df["src_bytes"] + df["dst_bytes"]
df["log_src_bytes"] = np.log1p(df["src_bytes"])
df["log_dst_bytes"] = np.log1p(df["dst_bytes"])

# -------------------------
# Features / Target
# -------------------------
X = df.drop("attack", axis=1)
y = df["attack"]

# -------------------------
# Define feature groups
# -------------------------
numeric_features = [
    "src_bytes", "dst_bytes", "byte_ratio",
    "total_bytes", "log_src_bytes", "log_dst_bytes"
]
categorical_features = ["protocol_type", "service", "flag"]

# -------------------------
# Preprocessing
# -------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# -------------------------
# Full Pipeline
# -------------------------
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# -------------------------
# Train / Split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
print(classification_report(y_val, pipeline.predict(X_val)))

# -------------------------
# Save pipeline
# -------------------------
os.makedirs("ml/models", exist_ok=True)
joblib.dump(pipeline, "ml_service/ml/models/ids_pipeline_v1.pkl")

print("[+] Pipeline saved as ids_pipeline_v1.pkl")

