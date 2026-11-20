# Lab-4 Minggu 6 — Broker Mosquitto, MQTT Publish/Subscribe, QoS, Retained & LWT

**Mata Kuliah:** Teknologi Internet of Things (IoT) — S1 Informatika FT UNIB  
**Sub-CPMK 3.2:** Perancangan hierarki topik dan implementasi komunikasi MQTT 3.1.1/5.0 dengan konfigurasi QoS, Retained Message, Persistent Session, dan Last Will & Testament (LWT).

## Struktur Folder
- `docker-compose.yml` & `mosquitto/config/mosquitto.conf` — Broker **Eclipse Mosquitto 2.0** (TCP 1883 & WS 9001).
- `python_tools/mqtt_monitor_sub.py` — Subscriber monitor wildcard (`+`, `#`) & penguji *Persistent Session*.
- `python_tools/mqtt_device_sim.py` — Simulator perangkat IoT untuk uji QoS 0/1/2, Retained Message, dan simulasi *crash* (`--crash-after`) pemicu LWT.
- `esp32_wokwi/` — Proyek ESP32 PlatformIO / Wokwi dengan `PubSubClient` (LWT + Retained + Subscribe `cmd/pump`).

## Langkah Cepat Menjalankan Lab-4 (Lokal)
1. Jalankan broker Mosquitto via Docker:
   ```bash
   docker compose up
   ```
   Biarkan terminal ini terbuka agar Anda dapat mengamati log paket (`PUBLISH`, `PUBACK`, `PUBREC`, `PUBREL`, `PUBCOMP`, `PINGREQ`, dan eksekusi LWT).
2. Pasang pustaka Python di terminal kedua:
   ```bash
   pip install -r python_tools/requirements.txt
   python3 python_tools/mqtt_monitor_sub.py --topic "unib/rejang-lebong/kebun-kopi/#" --qos 1
   ```
3. Uji simulasi perangkat & pemutusan mendadak (LWT) di terminal ketiga:
   ```bash
   python3 python_tools/mqtt_device_sim.py --qos 2 --keepalive 10 --crash-after 15
   ```
   Perhatikan bahwa 15 detik setelah mulai, perangkat mati tanpa mengirim paket `DISCONNECT`. Sekitar 15 detik kemudian ($1.5 \times \text{keepalive}$), broker Mosquitto secara otomatis mempublikasikan pesan LWT (`"status": "offline"`) ke seluruh subscriber!
4. Untuk uji **ESP32 di Wokwi Web**, gunakan `esp32_wokwi/src/main.cpp` (default terhubung ke `broker.hivemq.com`) lalu pantau dan kirim perintah `"ON"` / `"OFF"` via **MQTT Explorer** pada topik `unib/tim01/rejang-lebong/kebun-kopi/node-01/cmd/pump`.
