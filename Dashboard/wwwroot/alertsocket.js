window.alertsocket = {
    socket: null,
    dotnetRef: null,

    connect(dotnetRef) {
        this.dotnetRef = dotnetRef;

        const hostname = window.location.hostname === 'localhost' ? 'localhost' : 'fastapi';
        const wsUrl = `ws://${hostname}:8000/ws/alerts`;

        console.log("Connecting WebSocket to:", wsUrl);
        this._connect(wsUrl);
    },

    _connect(wsUrl) {
        this.socket = new WebSocket(wsUrl);

        this.socket.onmessage = (event) => {
            try {
                const alert = JSON.parse(event.data);
                this.dotnetRef.invokeMethodAsync('ReceiveAlert', alert);
            } catch (err) {
                console.error("Error parsing alert:", err);
            }
        };

        this.socket.onopen = () => console.log("WebSocket connected:", wsUrl);

        this.socket.onclose = (event) => {
            console.warn("WebSocket closed, retrying in 2s", event.reason);
            setTimeout(() => this._connect(wsUrl), 2000); // reconnect after 2s
        };

        this.socket.onerror = (err) => {
            console.error("WebSocket error:", err);
            this.socket.close(); // trigger reconnect
        };
    }
};
