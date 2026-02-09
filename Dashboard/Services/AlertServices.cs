using Dashboard.Models;
using System.Net.Http.Json;

namespace Dashboard.Services
{
    public class AlertService
    {
        private readonly HttpClient _http;
        private readonly string _apiUrl;

        public AlertService(HttpClient http)
        {
            _http = http;
            _apiUrl = Environment.GetEnvironmentVariable("API_URL")
                      ?? "http://fastapi:8000/alerts"; // fallback
        }

        public async Task<List<Alert>> GetAlertsAsync()
        {
            try
            {
                var alerts = await _http.GetFromJsonAsync<List<Alert>>(_apiUrl);
                return alerts ?? new List<Alert>();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"FastAPI error: {ex.Message}");
                return new List<Alert>();
            }
        }
    }
}

