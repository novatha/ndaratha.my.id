# Lab-3: Edge Gateway, Serial Ingestion, Docker & Serialization Benchmark

Selamat datang di repositori praktikum **Minggu 4: Teknologi Internet of Things (IoT)** - Program Studi S1 Informatika, Fakultas Teknik, Universitas Bengkulu.

## Tujuan Praktikum
1. Membangun jembatan komunikasi serial UART yang andal antara ESP32 dan komputer *Edge Gateway* (Python `pyserial`).
2. Mengimplementasikan parser telemetri JSON dengan validasi skema dan penanganan galat pembingkaian (*delimiter-based framing*).
3. Melakukan *benchmarking* komparasi kuantitatif ukuran muatan dan efisiensi serialisasi: **JSON (Teks)** vs **CBOR (Biner RFC 8949)** vs **Protocol Buffers (Biner Berskema)**.
4. Mengemas layanan gateway ke dalam kontainer **Docker** dan menjalankan arsitektur multi-layanan (Gateway Bridge + Local Broker) menggunakan **Docker Compose**.

---

## Struktur Direktori
```
lab03_minggu_04_edge_gateway/
├── Dockerfile                      # Spesifikasi image kontainer gateway
├── docker-compose.yml              # Orkestrasi multi-service edge stack
├── requirements.txt                # Dependensi Python (pyserial, cbor2, protobuf)
├── telemetry.proto                 # Definisi kontrak skema Protocol Buffers
├── telemetry_pb2.py                # Stub Python hasil kompilasi protoc
├── benchmark_serialization.py      # Skrip uji performa dan perbandingan kuota
├── gateway_service.py              # Daemon gateway pembaca serial & SQLite cache
├── simulated_sender.py             # Simulator serial virtual ESP32 (pty loopback)
├── mosquitto.conf                  # Konfigurasi broker MQTT lokal
├── systemd/
│   └── iot-gateway.service         # Unit file systemd untuk Linux/Raspberry Pi OS
└── esp32_sender_wokwi/             # Simulasi firmware ESP32 di Wokwi
    ├── diagram.json
    ├── platformio.ini
    ├── libraries.txt
    └── src/main.cpp
```

---

## Panduan Langkah Kerja

### 1. Persiapan Lingkungan Python
Buat dan aktifkan *virtual environment* lokal:
```bash
python3 -m venv venv
source venv/bin/activate  # Di Linux / macOS
# atau: venv\Scripts\activate di Windows PowerShell

pip install -r requirements.txt
```

### 2. Kompilasi Skema Protocol Buffers (Jika Mengubah `.proto`)
```bash
protoc --python_out=. telemetry.proto
```

### 3. Menjalankan Eksperimen Benchmark Serialisasi
```bash
python3 benchmark_serialization.py
```
*Amati perbandingan ukuran payload byte, waktu encode/decode, dan estimasi kuota data bulanan.*

### 4. Menjalankan Pengujian Aliran Serial Gateway
**Opsi A: Menggunakan Simulator Serial Virtual (Tanpa Hardware Fisik)**
1. Di Terminal 1, jalankan simulator:
   ```bash
   python3 simulated_sender.py
   ```
   *Catat nama port serial virtual yang muncul, misalnya `/dev/ttys003`.*
2. Di Terminal 2, jalankan gateway:
   ```bash
   python3 gateway_service.py --port /dev/ttys003
   ```

**Opsi B: Menggunakan Modul ESP32 Fisik**
Sambungkan ESP32 via kabel USB, lalu jalankan:
```bash
python3 gateway_service.py --port /dev/ttyUSB0 --baud 115200
```

### 5. Menjalankan Gateway Menggunakan Docker Compose
```bash
docker compose up -d --build
docker compose ps
docker compose logs -f gateway-bridge
```

---
*Laboratorium Jaringan Komputer & Sistem Tertanam, Program Studi S1 Informatika FT UNIB.*
