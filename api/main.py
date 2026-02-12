from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from db import Alert, SessionLocal, init_db


app = FastAPI(title="Alerts API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to dashboard URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# DB dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize DB
init_db()



# Pydantic model for POST requests
class AlertSchema(BaseModel):
    timestamp: str
    severity: str
    alert_type: str
    protocol_type: str  # <-- string now
    service: str        # <-- string now
    src_bytes: int
    dst_bytes: int
    risk_score: float


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
        dst_bytes=alert.dst_bytes,
        risk_score=alert.risk_score

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

