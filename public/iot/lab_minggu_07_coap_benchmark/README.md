# Lab-5 Minggu 7 — Protokol CoAP (RFC 7252) & Benchmarking Eksperimental (`tc netem`)

**Mata Kuliah:** Teknologi Internet of Things (IoT) — S1 Informatika FT UNIB  
**Sub-CPMK 3.3:** Implementasi server/klien CoAP (`aiocoap`) beserta *Observe* (RFC 7641) dan *Resource Discovery* (`/.well-known/core`), serta evaluasi kuantitatif HTTP vs MQTT vs CoAP di bawah emulasi jaringan `tc netem`.

## Struktur Folder
- `coap_server.py` — Server CoAP UDP port `5683` (`/.well-known/core`, `/sensor/water` Observable, `/actuator/siren`).
- `coap_client_demo.py` — Klien `aiocoap` (mode `discover`, `get`, `put`, `observe`).
- `http_sink_server.py` — Server pembanding FastAPI HTTP POST port `8000`.
- `benchmark_protocols.py` — Pengukur otomatis HTTP (Close & Keep-Alive) vs MQTT (QoS 0 & 1) vs CoAP (CON & NON) + ekspor CSV.
- `docker-compose.yml` & `Dockerfile` — Lingkungan terisolasi yang menyediakan `tc netem` asli (berjalan lintas OS: Windows, macOS, Linux).

## Langkah Cepat Menjalankan Lab-5 (Docker + `tc netem`)
1. Bangun dan jalankan seluruh layanan:
   ```bash
   docker compose up -d --build
   ```
2. Uji Resource Discovery & Observe CoAP dari dalam `bench-client`:
   ```bash
   docker exec -it lab5_bench_client python3 coap_client_demo.py --host coap-server --mode discover
   docker exec -it lab5_bench_client python3 coap_client_demo.py --host coap-server --mode observe
   ```
3. Jalankan **Skenario 1 (Baseline — Loss 0%, Delay 0 ms)**:
   ```bash
   docker exec -it lab5_bench_client python3 benchmark_protocols.py \
     --http-host http-server --mqtt-host mosquitto --coap-host coap-server \
     --samples 30 --label "baseline_0loss_0ms"
   ```
4. Aktifkan emulasi jaringan seluler buruk (**Skenario 2: Delay 100 ms ± 20 ms, Packet Loss 10%**) menggunakan `tc netem`:
   ```bash
   docker exec -it lab5_bench_client tc qdisc add dev eth0 root netem delay 100ms 20ms loss 10%
   docker exec -it lab5_bench_client python3 benchmark_protocols.py \
     --http-host http-server --mqtt-host mosquitto --coap-host coap-server \
     --samples 30 --label "cellular_10loss_100ms"
   ```
5. Kembalikan jaringan ke normal setelah selesai:
   ```bash
   docker exec -it lab5_bench_client tc qdisc del dev eth0 root
   ```
