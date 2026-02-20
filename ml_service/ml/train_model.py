import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
df = pd.read_csv("../data/raw/KDDTrain+.txt", names=COLUMNS)

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
# Base classifiers
# -------------------------
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
svm_clf = SVC(kernel='rbf', probability=True, random_state=42)

# Ensemble classifier (soft voting)
ensemble_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('xgb', xgb_clf), ('svm', svm_clf)],
    voting='soft'
)

# -------------------------
# Pipelines
# -------------------------
pipelines = {
    "randomforest": Pipeline([("preprocessor", preprocessor), ("classifier", rf_clf)]),
    "xgboost": Pipeline([("preprocessor", preprocessor), ("classifier", xgb_clf)]),
    "svm": Pipeline([("preprocessor", preprocessor), ("classifier", svm_clf)]),
    "ensemble": Pipeline([("preprocessor", preprocessor), ("classifier", ensemble_clf)])
}

# -------------------------
# Train / Validation Split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Create Versioned Model Directory
# -------------------------
BASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

existing_versions = [
    int(d[1:]) for d in os.listdir(BASE_MODEL_DIR)
    if d.startswith("v") and d[1:].isdigit()
]

next_version = max(existing_versions) + 1 if existing_versions else 1
version_name = f"v{next_version}"
VERSION_DIR = os.path.join(BASE_MODEL_DIR, version_name)
os.makedirs(VERSION_DIR, exist_ok=True)

print(f"\n[+] Saving models to version folder: {VERSION_DIR}\n")

# -------------------------
# Train and Evaluate Models
# -------------------------
results = {}

for name, pipe in pipelines.items():
    print(f"[+] Training {name}...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)

    results[name] = {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1-score": f1_score(y_val, y_pred)
    }

    model_path = os.path.join(VERSION_DIR, f"{name}.pkl")
    joblib.dump(pipe, model_path)
    print(f"[+] Saved {name} model at {model_path}")

# -------------------------
# Metrics Report
# -------------------------
df_results = pd.DataFrame(results).T
print("\n[+] Model Comparison Metrics:\n")
print(df_results)

# -------------------------
# Save Best Model
# -------------------------
best_model_name = df_results["F1-score"].idxmax()
best_model = pipelines[best_model_name]
best_model_path = os.path.join(VERSION_DIR, "best.pkl")
joblib.dump(best_model, best_model_path)

print(
    f"\n[+] Best model: {best_model_name} "
    f"(F1={df_results.loc[best_model_name,'F1-score']:.3f})"
)

# -------------------------
# Save Metadata
# -------------------------
metadata = {
    "version": version_name,
    "trained_at": datetime.utcnow().isoformat(),
    "best_model": best_model_name,
    "metrics": df_results.to_dict(),
    "features": list(X.columns)
}

metadata_path = os.path.join(VERSION_DIR, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"[+] Metadata saved at {metadata_path}")
print(f"[+] Training complete for version {version_name}\n")


