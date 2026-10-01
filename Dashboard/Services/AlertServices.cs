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
                      ?? "http://fastapi:8000/alerts";
        }

        // Fetch existing alerts from FastAPI
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
        // Fetch ML model performance metrics from FastAPI
	public async Task<MetricsResponse?> GetMetricsAsync()
	{
    	    try
    	    {
        	var metricsUrl = _apiUrl.Replace("/alerts", "/metrics");

        	var metrics =
            	    await _http.GetFromJsonAsync<MetricsResponse>(metricsUrl);

        	return metrics;
    	    }
    	    catch (Exception ex)
    	    {
        	Console.WriteLine($"FastAPI Metrics error: {ex.Message}");

        	return null;
    	    }
        }
    }
}

