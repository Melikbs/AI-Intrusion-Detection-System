namespace Dashboard.Models
{
    public class Alert
    {
        // Adding '= string.Empty;' removes the CS8618 warnings
        public string Timestamp { get; set; } = string.Empty;
        public string Alert_Type { get; set; } = string.Empty;
        public string Severity { get; set; } = string.Empty;
        public int Protocol_Type { get; set; }
        public int Service { get; set; }
        public int Src_Bytes { get; set; }
        public int Dst_Bytes { get; set; }
    }
}
