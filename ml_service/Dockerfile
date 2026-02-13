FROM python:3.10-slim
WORKDIR /app

# Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Ensure models folder exists
RUN ls -lh ml/models/

# Default command: start the ML consumer
CMD ["python", "consumer.py"]


