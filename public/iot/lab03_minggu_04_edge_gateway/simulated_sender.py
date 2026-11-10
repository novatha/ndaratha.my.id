#!/usr/bin/env python3
"""
UNIB S1 Informatika - Teknologi IoT 2026
Simulator Node Sensor Serial Virtual (simulated_sender.py)
Membuat virtual serial port (pty) pada sistem operasi UNIX/Linux/macOS
dan mengalirkan data telemetri tiruan ESP32 dengan interval 2 detik.
"""

import os
import sys
import pty
import time
import json
import random

def main():
    if not hasattr(os, "openpty"):
        print("[ERROR] Sistem operasi ini tidak mendukung os.openpty() secara native.")
        sys.exit(1)

    master, slave = pty.openpty()
    slave_name = os.ttyname(slave)

    print("=" * 64)
    print("  UNIB SIMULATOR NODE SENSOR ESP32 (VIRTUAL SERIAL PORT)")
    print("=" * 64)
    print(f"\n[INFO] Port Serial Virtual Berhasil Dibuat Pada:")
    print(f"       >>>  {slave_name}  <<<\n")
    print(f"Jalankan gateway pada terminal lain dengan perintah:")
    print(f"python3 gateway_service.py --port {slave_name}\n")
    print("=" * 64)

    seq = 0
    try:
        while True:
            seq += 1
            sample = {
                "node_id": "ESP32_UNIB_01",
                "timestamp": int(time.time()),
                "seq": seq,
                "temp": round(random.uniform(26.5, 31.8), 2),
                "hum": round(random.uniform(68.0, 84.0), 1),
                "lux": random.randint(200, 900),
                "battery": round(random.uniform(3.70, 4.15), 2),
                "status": "ONLINE"
            }
            line = json.dumps(sample) + "\n"
            os.write(master, line.encode('utf-8'))
            print(f"[SIMULATOR TX #{seq:04d}] {line.strip()}")
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n[INFO] Simulator dihentikan.")
        os.close(master)
        os.close(slave)

if __name__ == "__main__":
    main()
