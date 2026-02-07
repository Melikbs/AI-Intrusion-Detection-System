import pandas as pd
from datetime import datetime

ALERTS_FILE = "alerts_log.csv"
INCIDENTS_FILE = "incidents_log.csv"

def respond_to_alert(alert):
    """
    Simulated SOAR response logic
    """
    response_actions = []

    if alert["severity"] == "HIGH":
        response_actions.append("BLOCK_SOURCE_IP")
        response_actions.append("ISOLATE_AFFECTED_SERVICE")
        response_actions.append("NOTIFY_SOC_TEAM")

    return response_actions


def run_soar():
    alerts = pd.read_csv(ALERTS_FILE)
    incidents = []

    for _, alert in alerts.iterrows():
        actions = respond_to_alert(alert)

        incident = {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_type": alert["alert_type"],
            "severity": alert["severity"],
            "actions_taken": ",".join(actions),
            "status": "CONTAINED"
        }

        incidents.append(incident)

    incidents_df = pd.DataFrame(incidents)
    incidents_df.to_csv(INCIDENTS_FILE, index=False)

    print(f"[+] SOAR processed {len(incidents_df)} alerts")
    print("[+] incidents_log.csv created")


if __name__ == "__main__":
    run_soar()
