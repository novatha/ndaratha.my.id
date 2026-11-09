# Proyek Simulasi Wokwi — Praktikum Minggu 3 (Lab-2)
**Mata Kuliah:** Teknologi Internet of Things (IoT) — Prodi S1 Informatika FT UNIB  
**Topik:** Bus Komunikasi I2C Multi-Peranti, Sensor BMP280/BME280, Layar Grafis OLED SSD1306, FreeRTOS Multi-Tasking Dual-Core, Manajemen Daya *Deep Sleep*, & Retensi Memori RTC (`RTC_DATA_ATTR`)

---

## Cara 1: Menjalankan di VS Code + PlatformIO + Ekstensi Wokwi (Direkomendasikan)
1. Buka folder `lab02_minggu_03_wokwi` ini di **Visual Studio Code** (`File -> Open Folder...`).
2. Pastikan ekstensi **PlatformIO IDE** dan **Wokwi Simulator** telah aktif.
3. Kompilasi *firmware* terlebih dahulu dengan menekan ikon centang (**PlatformIO: Build**) di bilah status VS Code hingga menghasilkan `.pio/build/esp32doit-devkit-v1/firmware.bin`.
4. Klik berkas [`diagram.json`](./diagram.json), lalu tekan tombol hijau **Start Simulation** (atau `F1 -> Wokwi: Start Simulator`).

## Cara 2: Menjalankan Langsung di Peramban Web (`https://wokwi.com`) Tanpa Instalasi Lokal
1. Buka [https://wokwi.com/projects/new/esp32](https://wokwi.com/projects/new/esp32).
2. Pada tab **`sketch.ino`**, tempelkan seluruh isi berkas [`src/main.cpp`](./src/main.cpp).
3. Pada tab **`diagram.json`**, ganti seluruh isinya dengan isi berkas [`diagram.json`](./diagram.json).
4. Klik tab **Library Manager (`+`)** atau buat tab berkas **`libraries.txt`** dan tempelkan pustaka berikut:
   ```text
   Adafruit BMP280 Library
   Adafruit SSD1306
   Adafruit GFX Library
   Adafruit Unified Sensor
   ArduinoJson
   ```
5. Klik tombol hijau **Start the simulation**.

---

## Panduan Interaksi & Pengamatan Saat Simulasi Berjalan
- **Fase Aktif vs Fase Tidur (Duty Cycling):** Saat simulasi dimulai, LED biru **Active Mode** (`GPIO 2`) menyala terang selama ~500 ms menandakan ESP32 sedang membaca sensor I2C, memperbarui tampilan OLED, dan mengirimkan data JSON ke Serial Monitor. Setelah itu, LED padam dan ESP32 memasuki *Deep Sleep*.
- **Pewaktu Siklis (Timer Wakeup):** Biarkan simulasi berjalan tanpa menekan tombol. Setiap 10 detik, ESP32 akan bangun secara otomatis (*Timer Wakeup*), mencetak log diagnostik boot, dan memperbarui riwayat suhu/tekanan di memori RTC.
- **Pemicu Bangun Eksternal Seketika (EXT0 Wakeup):** Saat ESP32 sedang tidur (LED biru padam), klik tombol kuning **EXT0 Wakeup** (`GPIO 15`). ESP32 akan langsung bangun seketika tanpa menunggu timer 10 detik selesai, dengan pemicu diagnostik `[RTC WAKEUP] Pemicu: Tombol Darurat Luar (EXT0 pada GPIO 15)`.
- **Verifikasi Retensi Memori RTC:** Perhatikan Serial Monitor. Variabel `nomorSiklusBoot`, `riwayatSuhuTerakhir`, dan `riwayatTekananTerakhir` tetap tersimpan dan bertambah di setiap siklus bangun, membuktikan bahwa variabel berpenanda `RTC_DATA_ATTR` tidak hilang meskipun RAM normal terhapus saat *Deep Sleep*.
- **Simulasi Perubahan Parameter Cuaca:** Klik komponen **`bmp1`**, ubah *slider* suhu atau tekanan barometrik, lalu tunggu siklus bangun berikutnya untuk melihat pembaruan grafis pada layar OLED SSD1306.
