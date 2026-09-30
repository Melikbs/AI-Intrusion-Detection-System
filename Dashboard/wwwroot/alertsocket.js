window.alertsocket = {
    socket: null,

    connect: function (dotnetHelper) {
        console.log("[AlertSocket] Connecting to FastAPI WebSocket...");

        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";

        // IMPORTANT:
        // localhost:8000 is the FastAPI port exposed by Docker.
        const wsUrl = protocol + "//" + window.location.hostname + ":8000/ws/alerts";

        console.log("[AlertSocket] URL:", wsUrl);

        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = function () {
            console.log("[AlertSocket] CONNECTED");
        };

        this.socket.onmessage = async function (event) {
            console.log("[AlertSocket] MESSAGE RECEIVED:", event.data);

            try {
                const alert = JSON.parse(event.data);

                console.log("[AlertSocket] Parsed alert:", alert);

                await dotnetHelper.invokeMethodAsync(
                    "ReceiveAlert",
                    alert
                );

                console.log("[AlertSocket] Alert sent to Blazor");
            }
            catch (error) {
                console.error(
                    "[AlertSocket] Error processing message:",
                    error
                );
            }
        };

        this.socket.onerror = function (error) {
            console.error("[AlertSocket] WebSocket ERROR:", error);
        };

        this.socket.onclose = function (event) {
            console.warn(
                "[AlertSocket] WebSocket CLOSED:",
                event.code,
                event.reason
            );
        };
    },

    disconnect: function () {
        if (this.socket) {
            console.log("[AlertSocket] Disconnecting...");
            this.socket.close();
            this.socket = null;
        }
    }
};
