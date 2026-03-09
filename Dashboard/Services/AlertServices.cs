using Dashboard.Models;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;

namespace Dashboard.Services
{
    public class AlertService
    {
        private readonly HttpClient _http;
        private readonly string _apiUrl;

        public event Action<Alert>? OnAlertReceived;

        public AlertService(HttpClient http)
        {
            _http = http;
            _apiUrl = Environment.GetEnvironmentVariable("API_URL") 
                      ?? "http://fastapi:8000/alerts";
        }

        // Fetch existing alerts via HTTP
        public async Task<List<Alert>> GetAlertsAsync()
        {
            try
            {
                var alerts = await _http.GetFromJsonAsync<List<Alert>>(_apiUrl);
                return alerts ?? new List<Alert>();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"FastAPI HTTP error: {ex.Message}");
                return new List<Alert>();
            }
        }

        // Start listening to WebSocket alerts (call this explicitly)
        public async Task StartWebSocketAsync()
        {
            while (true)
            {
                try
                {
                    using var ws = new ClientWebSocket();
                    var wsUrl = _apiUrl.Replace("http://", "ws://")
                                       .Replace("/alerts", "/ws/alerts");

                    await ws.ConnectAsync(new Uri(wsUrl), CancellationToken.None);
                    Console.WriteLine($"Connected to WebSocket: {wsUrl}");

                    var buffer = new byte[4096];

                    while (ws.State == WebSocketState.Open)
                    {
                        var result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);

                        if (result.MessageType == WebSocketMessageType.Text)
                        {
                            var json = Encoding.UTF8.GetString(buffer, 0, result.Count);
                            var alert = JsonSerializer.Deserialize<Alert>(json);
                            if (alert != null)
                            {
                                OnAlertReceived?.Invoke(alert);
                            }
                        }
                        else if (result.MessageType == WebSocketMessageType.Close)
                        {
                            await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None);
                            break;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"WebSocket error: {ex.Message}");
                }

                // Retry after 2s if disconnected
                await Task.Delay(2000);
            }
        }
    }
}

