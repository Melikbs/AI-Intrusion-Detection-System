using Dashboard.Models;
using System.Net.Http.Json;

namespace Dashboard.Services
{
    public class AlertService
    {
        private readonly HttpClient _http;

        public AlertService(HttpClient http)
        {
            _http = http;
        }

        public async Task<List<Alert>> GetAlertsAsync()
{
    try
    {
        // Make sure to include the full FastAPI URL
        var alerts = await _http.GetFromJsonAsync<List<Alert>>("http://127.0.0.1:8000/alerts");
        return alerts ?? new List<Alert>();
    }
    catch
    {
        // If FastAPI is down, just return an empty list
        return new List<Alert>();
    }
}

}
}  
