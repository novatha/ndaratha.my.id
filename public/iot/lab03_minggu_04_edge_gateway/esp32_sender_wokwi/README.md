# ESP32 Serial Telemetry Streamer (Simulasi Wokwi Lab-3)

Proyek ini mensimulasikan simpul sensor ESP32 yang mengirimkan stream JSON melalui UART serial (`115200 bps`) ke perangkat *Edge Gateway* (Raspberry Pi atau skrip gateway Python di komputer Anda).

## Komponen Rangkaian (diagram.json)
1. **ESP32 DevKit v4**
2. **Potensiometer 10k**: Terhubung ke pin **GPIO 34** (ADC1) untuk simulasi variasi suhu, kelembapan, dan voltase baterai.
3. **LED Indikator**: Terhubung ke **GPIO 2** untuk indikasi visual transmisi paket serial.

## Cara Menjalankan di Wokwi
1. Buka Wokwi Web / VS Code Wokwi extension.
2. Muat direktori ini.
3. Klik tombol **Play/Start Simulation**.
4. Amati Serial Monitor: data telemetri JSON dialirkan setiap 2 detik dengan delimiter newline `\n`.
