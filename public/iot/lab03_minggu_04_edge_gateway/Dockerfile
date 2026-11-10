FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin script dan artefak kompilasi skema protobuf
COPY gateway_service.py .
COPY telemetry.proto .
COPY telemetry_pb2.py .
COPY benchmark_serialization.py .

# Buat direktori volume untuk data buffer
RUN mkdir -p /app/data

ENV PYTHONUNBUFFERED=1
ENV DB_PATH=/app/data/telemetry_cache.db

CMD ["python", "-u", "gateway_service.py"]
