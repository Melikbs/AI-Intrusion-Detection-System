from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from db import Alert, SessionLocal, init_db
import pandas as pd
import os

app = FastAPI(title="Alerts API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to dashboard URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CSV fallback path
ALERTS_CSV_PATH = os.path.join(os.path.dirname(__file__), "alerts_log.csv")

# DB dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize DB
init_db()

# Populate DB from CSV if empty
@app.on_event("startup")
def load_alerts():
    db = SessionLocal()
    if db.query(Alert).first() is None and os.path.exists(ALERTS_CSV_PATH):
        df = pd.read_csv(ALERTS_CSV_PATH)
        for _, row in df.iterrows():
            alert = Alert(
                timestamp=str(row['timestamp']),
                alert_type=str(row['alert_type']),
                severity=str(row['severity']),
                protocol_type=int(row['protocol_type']),
                service=int(row['service']),
                src_bytes=int(row['src_bytes']),
                dst_bytes=int(row['dst_bytes'])
            )
            db.add(alert)
        db.commit()
    db.close()

# Pydantic model for POST requests
class AlertSchema(BaseModel):
    timestamp: str
    severity: str
    alert_type: str
    protocol_type: int
    service: int
    src_bytes: int
    dst_bytes: int

# POST /alerts endpoint to ingest new alerts
@app.post("/alerts")
def create_alert(alert: AlertSchema, db: Session = Depends(get_db)):
    new_alert = Alert(
        timestamp=alert.timestamp,
        alert_type=alert.alert_type,
        severity=alert.severity,
        protocol_type=alert.protocol_type,
        service=alert.service,
        src_bytes=alert.src_bytes,
        dst_bytes=alert.dst_bytes
    )
    db.add(new_alert)
    db.commit()
    db.refresh(new_alert)
    return {"status": "ok", "id": new_alert.id}

# GET /alerts endpoint to read all alerts
@app.get("/alerts")
def read_alerts(db: Session = Depends(get_db)):
    alerts = db.query(Alert).all()
    return [
        {
            "id": alert.id,
            "timestamp": alert.timestamp,
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "protocol_type": alert.protocol_type,
            "service": alert.service,
            "src_bytes": alert.src_bytes,
            "dst_bytes": alert.dst_bytes
        }
        for alert in alerts
    ]

def get_alerts(db: Session = Depends(get_db)):
    return db.query(Alert).all()

