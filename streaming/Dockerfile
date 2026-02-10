FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install redis
COPY traffic_simulator.py .

CMD ["python", "traffic_simulator.py"]

