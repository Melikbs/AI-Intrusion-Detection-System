from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
import asyncio

from db import Alert, SessionLocal, init_db

app = FastAPI(title="Alerts API")

# -----------------------------
# CORS setup
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to dashboard URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# WebSocket Connection Manager
# -----------------------------
active_connections: List[WebSocket] = []


@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    active_connections.append(websocket)

    print(
        f"[WS] Client connected. "
        f"Active clients: {len(active_connections)}"
    )

    try:
        while True:
            await asyncio.sleep(3600)

    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)

        print(
            f"[WS] Client disconnected. "
            f"Active clients: {len(active_connections)}"
        )

    except Exception as e:
        if websocket in active_connections:
            active_connections.remove(websocket)

        print(f"[WS] Connection error: {e}")


# Broadcast new alerts to all connected clients
async def broadcast_alert(alert_data: dict):
    if not active_connections:
        print("[WS] No connected clients")
        return

    print(
        f"[WS] Broadcasting alert to "
        f"{len(active_connections)} client(s)"
    )

    disconnected = []

    # Work on a snapshot so the original list can safely change
    # while we're sending.
    for connection in list(active_connections):
        try:
            await connection.send_json(alert_data)

        except Exception as e:
            print(f"[WS] Send failed: {e}")
            disconnected.append(connection)

    # Remove dead connections
    for connection in disconnected:
        if connection in active_connections:
            active_connections.remove(connection)

    print(
        f"[WS] Broadcast complete. "
        f"Active clients: {len(active_connections)}"
    )
# -----------------------------
# Database
# -----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

init_db()

# -----------------------------
# Schemas
# -----------------------------
class AlertSchema(BaseModel):
    timestamp: str
    severity: str
    alert_type: str
    protocol_type: str
    service: str
    src_bytes: int
    dst_bytes: int
    risk_score: float

# -----------------------------
# REST Endpoints
# -----------------------------
@app.post("/alerts")
async def create_alert(alert: AlertSchema, db: Session = Depends(get_db)):
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

    alert_payload = {
        "id": new_alert.id,
        "timestamp": new_alert.timestamp,
        "alert_type": new_alert.alert_type,
        "severity": new_alert.severity,
        "protocol_type": new_alert.protocol_type,
        "service": new_alert.service,
        "src_bytes": new_alert.src_bytes,
        "dst_bytes": new_alert.dst_bytes,
        "risk_score": new_alert.risk_score
    }

    # Send to all connected WebSocket clients
    asyncio.create_task(broadcast_alert(alert_payload))

    return {"status": "ok", "id": new_alert.id}

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
            "dst_bytes": alert.dst_bytes,
            "risk_score": alert.risk_score
        }
        for alert in alerts
    ]
