using System.Text.Json.Serialization;
namespace Dashboard.Models
{
    public class ModelMetrics
    {
        public double Accuracy { get; set; }
        public double Precision { get; set; }
        public double Recall { get; set; }

        [JsonPropertyName("F1-score")]
        public double F1Score { get; set; }
    }

    public class MetricsResponse
    {
        public string Version { get; set; } = "";
        public string GeneratedAt { get; set; } = "";

        public Dictionary<string, ModelMetrics> Models { get; set; }
            = new();
    }
}
