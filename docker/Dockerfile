FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY ml/ ml/
COPY soar/ soar/
COPY data/ data/
COPY alerts_log.csv .
COPY incidents_log.csv .

# Default command
CMD ["python", "ml/detect.py"]
