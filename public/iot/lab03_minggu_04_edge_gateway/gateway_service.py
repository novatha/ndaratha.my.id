#!/usr/bin/env python3
"""
UNIB S1 Informatika - Teknologi IoT 2026
Layanan Edge Gateway Terpadu (gateway_service.py)
Fungsi:
1. Membaca stream serial UART (atau virtual pty) dari node sensor.
2. Memvalidasi skema JSON dan mendeteksi data korup / terfragmentasi.
3. Mengonversi data ke format biner CBOR & Protocol Buffers.
4. Menyimpan data ke basis data lokal SQLite (Pola Offline Store-and-Forward).
5. Dapat dieksekusi secara mandiri (systemd) maupun di dalam kontainer Docker.
"""

import os
import sys
import time
import json
import sqlite3
import argparse

try:
    import serial
except ImportError:
    serial = None

try:
    import cbor2
except ImportError:
    cbor2 = None

try:
    import telemetry_pb2
except ImportError:
    telemetry_pb2 = None

DEFAULT_PORT = os.getenv("SERIAL_PORT", "/dev/ttyUSB0")
DEFAULT_BAUD = int(os.getenv("BAUD_RATE", "115200"))
DB_PATH = os.getenv("DB_PATH", "telemetry_cache.db")

def init_database(db_file):
    """Membuat tabel cache lokal jika belum ada."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS local_buffer (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            temp REAL,
            hum REAL,
            lux INTEGER,
            battery REAL,
            status TEXT,
            cbor_payload BLOB,
            proto_payload BLOB,
            synced INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DATABASE] Database lokal siap di: {db_file}")

def store_to_buffer(db_file, data, cbor_bytes, proto_bytes):
    """Menyimpan paket telemetri ke buffer SQLite lokal."""
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO local_buffer 
            (node_id, timestamp, temp, hum, lux, battery, status, cbor_payload, proto_payload, synced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            data.get("node_id", "UNKNOWN"),
            data.get("timestamp", int(time.time())),
            data.get("temp", 0.0),
            data.get("hum", 0.0),
            data.get("lux", 0),
            data.get("battery", 0.0),
            data.get("status", "OK"),
            sqlite3.Binary(cbor_bytes) if cbor_bytes else None,
            sqlite3.Binary(proto_bytes) if proto_bytes else None
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB-ERR] Gagal menyimpan ke cache lokal: {e}")

def run_gateway(port, baudrate, db_file):
    init_database(db_file)
    print(f"[GATEWAY] Menghubungkan ke port serial: {port} @ {baudrate} bps...")

    while True:
        try:
            if not os.path.exists(port) and not port.startswith("/dev/"):
                print(f"[WAIT] Port {port} belum aktif. Mencoba lagi dalam 3 detik...")
                time.sleep(3)
                continue

            with serial.Serial(port, baudrate, timeout=2.0) as ser:
                ser.reset_input_buffer()
                print(f"[ONLINE] Berhasil terhubung ke {port}! Menunggu aliran telemetri...\n")

                while True:
                    if ser.in_waiting > 0:
                        raw_bytes = ser.readline()
                        line = raw_bytes.decode('utf-8', errors='ignore').strip()
                        if not line:
                            continue

                        # 1. Parsing & Validasi
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError as err:
                            print(f"[REJECT] Frame serial korup: '{line}' ({err})")
                            continue

                        # 2. Verifikasi Field Inti
                        req_fields = ["node_id", "temp", "hum"]
                        if not all(k in data for k in req_fields):
                            print(f"[WARN] Field tidak lengkap: {data}")
                            continue

                        # 3. Konversi Biner CBOR
                        cbor_bytes = cbor2.dumps(data) if cbor2 else b""

                        # 4. Konversi Biner Protobuf
                        proto_bytes = b""
                        if telemetry_pb2:
                            proto_obj = telemetry_pb2.TelemetryPacket(
                                node_id=str(data.get("node_id", "")),
                                timestamp=int(data.get("timestamp", time.time())),
                                seq=int(data.get("seq", 0)),
                                temp=float(data.get("temp", 0.0)),
                                hum=float(data.get("hum", 0.0)),
                                lux=int(data.get("lux", 0)),
                                battery=float(data.get("battery", 0.0)),
                                status=str(data.get("status", "OK"))
                            )
                            proto_bytes = proto_obj.SerializeToString()

                        # 5. Simpan ke SQLite Buffer
                        store_to_buffer(db_file, data, cbor_bytes, proto_bytes)

                        # Output status ke log
                        sz_json = len(line.encode('utf-8'))
                        sz_cbor = len(cbor_bytes)
                        sz_proto = len(proto_bytes)
                        print(f"[RECV] Node: {data.get('node_id')} | T: {data.get('temp')} C | RH: {data.get('hum')}% | "
                              f"Bytes: JSON={sz_json}B, CBOR={sz_cbor}B, Protobuf={sz_proto}B -> Cached [OK]")

                    time.sleep(0.01)

        except (serial.SerialException, FileNotFoundError) as err:
            print(f"[WARN] Koneksi serial terputus/gagal ({err}). Inisialisasi ulang dalam 3 detik...")
            time.sleep(3)
        except KeyboardInterrupt:
            print("\n[INFO] Menghentikan layanan Edge Gateway secara normal.")
            sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNIB IoT Edge Gateway Daemon")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Port Serial UART / pty virtual")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baud rate (default 115200)")
    parser.add_argument("--db", default=DB_PATH, help="Path file SQLite cache")
    args = parser.parse_args()

    run_gateway(args.port, args.baud, args.db)
