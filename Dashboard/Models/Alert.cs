using System.Text.Json.Serialization;

namespace Dashboard.Models
{
    public class Alert
    {
        [JsonPropertyName("timestamp")]
        public string Timestamp { get; set; } = string.Empty;

        [JsonPropertyName("alert_type")]
        public string Alert_Type { get; set; } = string.Empty;

        [JsonPropertyName("severity")]
        public string Severity { get; set; } = string.Empty;

        [JsonPropertyName("protocol_type")]
        public string ProtocolType { get; set; } = string.Empty;

        [JsonPropertyName("service")]
        public string Service { get; set; } = string.Empty;

        [JsonPropertyName("src_bytes")]
        public int Src_Bytes { get; set; }

        [JsonPropertyName("dst_bytes")]
        public int Dst_Bytes { get; set; }
    }
}

