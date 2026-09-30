using System.Text.Json.Serialization;

namespace Dashboard.Models
{
    public class Alert
    {
        [JsonPropertyName("id")]
        public int Id { get; set; }

        [JsonPropertyName("timestamp")]
        public string Timestamp { get; set; } = "";

        [JsonPropertyName("alert_type")]
        public string Alert_Type { get; set; } = "";

        [JsonPropertyName("severity")]
        public string Severity { get; set; } = "";

        [JsonPropertyName("protocol_type")]
        public string ProtocolType { get; set; } = "";

        [JsonPropertyName("service")]
        public string Service { get; set; } = "";

        [JsonPropertyName("src_bytes")]
        public int Src_Bytes { get; set; }

        [JsonPropertyName("dst_bytes")]
        public int Dst_Bytes { get; set; }

        [JsonPropertyName("risk_score")]
        public double RiskScore { get; set; }
    }
}
