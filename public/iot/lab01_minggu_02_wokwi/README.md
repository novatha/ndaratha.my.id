# Proyek Simulasi Wokwi — Praktikum Minggu 2 (Lab-1)
**Mata Kuliah:** Teknologi Internet of Things (IoT) — Prodi S1 Informatika FT UNIB  
**Topik:** GPIO Digital/Analog (ADC1), Modulasi PWM (`LEDC`), Sensor DHT22, *Capacitive Soil Moisture*, *Moving Average Filter*, & Kendali Histeresis Relai

---

## Cara 1: Menjalankan di VS Code + PlatformIO + Ekstensi Wokwi (Direkomendasikan)
1. Buka folder `lab01_minggu_02_wokwi` ini di **Visual Studio Code** (`File -> Open Folder...`).
2. Pastikan ekstensi **PlatformIO IDE** dan **Wokwi Simulator** telah aktif.
3. Kompilasi *firmware* terlebih dahulu dengan menekan ikon centang (**PlatformIO: Build**) di bilah bawah VS Code hingga menghasilkan `.pio/build/esp32doit-devkit-v1/firmware.bin`.
4. Klik berkas [`diagram.json`](./diagram.json), lalu tekan tombol hijau **Start Simulation** (atau `F1 -> Wokwi: Start Simulator`).

## Cara 2: Menjalankan Langsung di Peramban Web (`https://wokwi.com`) Tanpa Instalasi Lokal
1. Buka [https://wokwi.com/projects/new/esp32](https://wokwi.com/projects/new/esp32).
2. Pada tab **`sketch.ino`**, tempelkan seluruh isi berkas [`src/main.cpp`](./src/main.cpp).
3. Pada tab **`diagram.json`**, ganti seluruh isinya dengan isi berkas [`diagram.json`](./diagram.json).
4. Klik tab **Library Manager (`+`)** atau buat tab berkas **`libraries.txt`** dan tempelkan isi [`libraries.txt`](./libraries.txt) (`DHT sensor library`, `Adafruit Unified Sensor`, `ArduinoJson`).
5. Klik tombol hijau **Start the simulation**.

---

## Panduan Interaksi Saat Simulasi Berjalan
- **Uji Kelembapan Tanah & Histeresis Pompa:** Putar knop potensiometer **`potSoil`** (`GPIO 34`). Saat `soil_pct` turun di bawah `35.0%`, relai & LED hijau **Pompa** menyala (`pump_on: true`) dan tetap menyala hingga `soil_pct` naik mencapai `60.0%`.
- **Uji Gejala *Chattering*:** Ubah `#define GUNAKAN_HISTERESIS 0` pada `src/main.cpp`, kompilasi ulang, lalu goyangkan sedikit potensiometer di sekitar `45%` dan amati lonjakan nilai `relay_switches` di Serial Monitor.
- **Uji Kecepatan Kipas PWM:** Klik komponen **`dht1`**, geser *slider* suhu antara `28°C` hingga `38°C`, dan amati perubahan kecerahan LED biru **Kipas PWM** serta nilai `fan_pwm_pct`.
- **Uji *Manual Override*:** Klik tombol kuning **Override** (`GPIO 15`) untuk memaksa pompa menyala (`ctrl_mode: "MANUAL"`).
