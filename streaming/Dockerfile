# Use Python 3.10 base image
FROM python:3.10-slim

WORKDIR /app

# Copy simulator code
COPY traffic_simulator.py ./

# Install requests library
RUN pip install --no-cache-dir requests

# Run the simulator
CMD ["python", "traffic_simulator.py"]
