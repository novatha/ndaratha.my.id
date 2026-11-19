# Lab-3 Minggu 5 — Komunikasi HTTP/REST & WebSocket Full-Duplex

**Mata Kuliah:** Teknologi Internet of Things (IoT) — S1 Informatika FT UNIB  
**Sub-CPMK 3.1:** Implementasi komunikasi *request–response* (HTTP/REST) dan *full-duplex* (WebSocket) menggunakan FastAPI, ESP32/Simulator, dan Wireshark.

## Struktur Folder
- `backend/main.py` — Server FastAPI (REST POST/GET + WebSocket `/ws/device/{node_id}` + `/ws/dashboard` + UI monitor HTML).
- `backend/Dockerfile` & `docker-compose.yml` — Kontainerisasi backend.
- `esp32_wokwi/` — Proyek ESP32 untuk PlatformIO & Wokwi (`HTTPClient` + `WebSocketsClient`).
- `simulated_esp32_client.py` — Simulator Python lokal untuk uji coba langsung & tangkap paket Wireshark loopback tanpa memerlukan perangkat keras maupun tunnel eksternal.

## Cara Menjalankan Cepat (Mode Lokal + Wireshark)
1. Jalankan server FastAPI:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Atau via Docker:
   ```bash
   docker compose up --build
   ```
2. Buka browser ke `http://127.0.0.1:8000` (Dashboard Monitor) dan `http://127.0.0.1:8000/docs` (Swagger UI OpenAPI).
3. Jalankan simulator klien di terminal kedua:
   ```bash
   python3 simulated_esp32_client.py
   ```
4. Buka **Wireshark** pada antarmuka `Loopback: lo0` (macOS/Linux) atau `Adapter for loopback traffic capture` (Windows) dengan filter:
   ```text
   tcp.port == 8000 && (http || websocket)
   ```
   Klik tombol **Nyalakan Pompa / Matikan** pada dashboard browser dan amati frame WebSocket (`Opcode 0x1 Text Frame`) muncul secara instan tanpa *HTTP handshake* ulang!
