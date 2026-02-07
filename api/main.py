from fastapi import FastAPI
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Alerts API")

# Allow requests from dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later replace with dashboard URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Absolute or relative path to alerts_log.csv
ALERTS_CSV_PATH = os.path.join(os.path.dirname(__file__), "../alerts_log.csv")

@app.get("/alerts")
def get_alerts():
    try:
        df = pd.read_csv(ALERTS_CSV_PATH)
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}
