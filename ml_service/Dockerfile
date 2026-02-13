FROM python:3.10-slim
WORKDIR /app

# Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Make sure the pipeline exists in the container
RUN ls -lh ml/models/

CMD ["python", "consumer.py"]

