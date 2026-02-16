import pandas as pd
import numpy as np
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
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
# Define pipelines for each model
# -------------------------
pipelines = {
    "RandomForest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
    "XGBoost": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]),
    "SVM": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", SVC(kernel='rbf', probability=True, random_state=42))
    ])
}

# -------------------------
# Train / Split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Train and evaluate all models
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

# -------------------------
# Metrics report
# -------------------------
df_results = pd.DataFrame(results).T
print("\n[+] Model Comparison Metrics:\n")
print(df_results)

# -------------------------
# Save the best-performing model
# -------------------------
best_model_name = df_results["F1-score"].idxmax()
best_model = pipelines[best_model_name]

os.makedirs("ml_service/ml/models", exist_ok=True)
best_model_path = "ml_service/ml/models/ids_pipeline_best.pkl"
joblib.dump(best_model, best_model_path)

print(f"[+] Best model ({best_model_name}) saved as ids_pipeline_best.pkl with F1-score={df_results.loc[best_model_name,'F1-score']:.3f}")


