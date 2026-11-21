FROM python:3.11-slim

# Pasang iproute2 agar utilitas emulasi jaringan Linux `tc netem` tersedia di semua OS (Windows/macOS/Linux)
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 iputils-ping \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY coap_server.py coap_client_demo.py http_sink_server.py benchmark_protocols.py ./
