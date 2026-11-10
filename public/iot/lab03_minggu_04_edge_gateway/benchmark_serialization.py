#!/usr/bin/env python3
"""
UNIB S1 Informatika - Teknologi IoT 2026
Praktikum Minggu 4 (Lab-3): Serial Ingestion & Serialization Benchmark
Membandingkan efisiensi JSON (Text), CBOR (RFC 8949 Binary), dan Protocol Buffers (v3).
"""

import sys
import json
import time

try:
    import cbor2
except ImportError:
    print("[WARN] Modul 'cbor2' belum terpasang. Jalankan: pip install cbor2")
    cbor2 = None

try:
    import telemetry_pb2
except ImportError:
    print("[WARN] File 'telemetry_pb2.py' tidak ditemukan. Jalankan: protoc --python_out=. telemetry.proto")
    telemetry_pb2 = None

# Dataset telemetri standar stasiun cuaca
SAMPLE_DATA = {
    "node_id": "ESP32_UNIB_01",
    "timestamp": 1794212500,
    "seq": 1024,
    "temp": 29.85,
    "hum": 78.4,
    "lux": 650,
    "battery": 3.95,
    "status": "ONLINE"
}

ITERATIONS = 10000

def run_benchmark():
    print("=" * 72)
    print("  LAB-3: BENCHMARK SERIALISASI IOT (JSON vs CBOR vs PROTOCOL BUFFERS)")
    print("=" * 72)
    print(f"Data Sampel: {json.dumps(SAMPLE_DATA, indent=2)}\n")

    # 1. JSON
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        json_str = json.dumps(SAMPLE_DATA)
        json_bytes = json_str.encode('utf-8')
    t_encode_json = (time.perf_counter() - t0) / ITERATIONS * 1e6

    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        _ = json.loads(json_bytes.decode('utf-8'))
    t_decode_json = (time.perf_counter() - t0) / ITERATIONS * 1e6
    len_json = len(json_bytes)

    # 2. CBOR
    if cbor2:
        t0 = time.perf_counter()
        for _ in range(ITERATIONS):
            cbor_bytes = cbor2.dumps(SAMPLE_DATA)
        t_encode_cbor = (time.perf_counter() - t0) / ITERATIONS * 1e6

        t0 = time.perf_counter()
        for _ in range(ITERATIONS):
            _ = cbor2.loads(cbor_bytes)
        t_decode_cbor = (time.perf_counter() - t0) / ITERATIONS * 1e6
        len_cbor = len(cbor_bytes)
    else:
        len_cbor, t_encode_cbor, t_decode_cbor = 0, 0, 0

    # 3. Protobuf
    if telemetry_pb2:
        proto_msg = telemetry_pb2.TelemetryPacket(
            node_id=SAMPLE_DATA["node_id"],
            timestamp=SAMPLE_DATA["timestamp"],
            seq=SAMPLE_DATA["seq"],
            temp=SAMPLE_DATA["temp"],
            hum=SAMPLE_DATA["hum"],
            lux=SAMPLE_DATA["lux"],
            battery=SAMPLE_DATA["battery"],
            status=SAMPLE_DATA["status"]
        )
        t0 = time.perf_counter()
        for _ in range(ITERATIONS):
            proto_bytes = proto_msg.SerializeToString()
        t_encode_proto = (time.perf_counter() - t0) / ITERATIONS * 1e6

        t0 = time.perf_counter()
        for _ in range(ITERATIONS):
            parsed_proto = telemetry_pb2.TelemetryPacket()
            parsed_proto.ParseFromString(proto_bytes)
        t_decode_proto = (time.perf_counter() - t0) / ITERATIONS * 1e6
        len_proto = len(proto_bytes)
    else:
        len_proto, t_encode_proto, t_decode_proto = 0, 0, 0

    # Ringkasan Kuantitatif
    print(f"{'Format':<15} | {'Payload (Bytes)':<16} | {'Encode (us)':<12} | {'Decode (us)':<12} | {'Reduksi Ukuran':<14}")
    print("-" * 78)
    print(f"{'JSON (Text)':<15} | {len_json:<16} | {t_encode_json:<12.2f} | {t_decode_json:<12.2f} | {'Baseline (0%)':<14}")
    if cbor2:
        red_cbor = (len_json - len_cbor) / len_json * 100
        print(f"{'CBOR (RFC 8949)':<15} | {len_cbor:<16} | {t_encode_cbor:<12.2f} | {t_decode_cbor:<12.2f} | {f'-{red_cbor:.1f}%':<14}")
    if telemetry_pb2:
        red_proto = (len_json - len_proto) / len_json * 100
        print(f"{'Protobuf (v3)':<15} | {len_proto:<16} | {t_encode_proto:<12.2f} | {t_decode_proto:<12.2f} | {f'-{red_proto:.1f}%':<14}")
    print("-" * 78)

    # Analisis Kuota Data Seluler
    print("\n[ANALISIS KUOTA OPERASIONAL]")
    print("Asumsi transmisi: 1 sampel setiap 10 detik selama 30 hari (259.200 sampel/bulan)")
    monthly_json = (len_json * 259200) / (1024 * 1024)
    monthly_cbor = (len_cbor * 259200) / (1024 * 1024) if cbor2 else 0
    monthly_proto = (len_proto * 259200) / (1024 * 1024) if telemetry_pb2 else 0
    print(f"- Total Kuota Bulanan (JSON)     : {monthly_json:.2f} MB")
    if cbor2:
        print(f"- Total Kuota Bulanan (CBOR)     : {monthly_cbor:.2f} MB (Hemat {monthly_json - monthly_cbor:.2f} MB)")
    if telemetry_pb2:
        print(f"- Total Kuota Bulanan (Protobuf) : {monthly_proto:.2f} MB (Hemat {monthly_json - monthly_proto:.2f} MB)")
    print("=" * 72)

if __name__ == "__main__":
    run_benchmark()
