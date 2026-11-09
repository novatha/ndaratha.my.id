#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP280.h>
#include <Adafruit_SSD1306.h>
#include <ArduinoJson.h>

#define LEBAR_LAYAR         128
#define TINGGI_LAYAR        64
#define ALAMAT_OLED         0x3C
#define PIN_LED_STATUS      2
#define PIN_WAKEUP_TOMBOL   GPIO_NUM_15
#define DURASI_TIDUR_DETIK  10

Adafruit_SSD1306 display(LEBAR_LAYAR, TINGGI_LAYAR, &Wire, -1);
Adafruit_BMP280 bmp; // Berkomunikasi melalui Wire pada I2C default

// --- VARIABEL PERSISTEN DI RTC SLOW MEMORY ---
// Variabel ini TETAP TERJAGA nilainya selama siklus Deep Sleep!
RTC_DATA_ATTR int nomorSiklusBoot = 0;
RTC_DATA_ATTR float riwayatSuhuTerakhir = 0.0f;
RTC_DATA_ATTR float riwayatTekananTerakhir = 0.0f;

// Struktur data telemetri bersama antar-tugas FreeRTOS
struct DataSensor {
  float suhu;
  float tekanan;
  float elevasi;
  bool valid;
} dataCuaca;

SemaphoreHandle_t mutexData;

void cetakDiagnostikWakeup() {
  esp_sleep_wakeup_cause_t alasan = esp_sleep_get_wakeup_cause();
  Serial.printf("\n========================================\n");
  Serial.printf("[RTC] Siklus Bangun Ke-%d\n", nomorSiklusBoot);
  switch (alasan) {
    case ESP_SLEEP_WAKEUP_TIMER:
      Serial.println("[RTC WAKEUP] Pemicu: Pewaktu Internal (RTC Timer)");
      break;
    case ESP_SLEEP_WAKEUP_EXT0:
      Serial.println("[RTC WAKEUP] Pemicu: Tombol Darurat Luar (EXT0 pada GPIO 15)");
      break;
    default:
      Serial.println("[RTC WAKEUP] Pemicu: Power-on Reset / Boot Pertama");
      break;
  }
  Serial.printf("[RTC MEM] Riwayat Siklus Lalu -> Suhu: %.2f C | Tekanan: %.2f hPa\n",
                riwayatSuhuTerakhir, riwayatTekananTerakhir);
}

void perbaruiTampilanOled(float t, float p, float alt, int siklus, const char* pemicu) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  // Baris Judul
  display.setCursor(0, 0);
  display.printf("UNIB IoT - STASIUN CUACA");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  // Parameter Lingkungan
  display.setCursor(0, 14);
  display.printf("Suhu    : %.1f C", t);
  display.setCursor(0, 26);
  display.printf("Tekanan : %.1f hPa", p);
  display.setCursor(0, 38);
  display.printf("Elevasi : %.1f m", alt);

  // Baris Status Manajemen Daya
  display.drawLine(0, 50, 127, 50, SSD1306_WHITE);
  display.setCursor(0, 54);
  display.printf("#%d | %s", siklus, pemicu);
  display.display();
}

// Task Khusus FreeRTOS pada Core 0: Menangani UI dan Serialisasi
void TaskTampilanDanSerial(void *pvParameters) {
  xSemaphoreTake(mutexData, portMAX_DELAY);
  DataSensor lokal = dataCuaca;
  xSemaphoreGive(mutexData);

  if (lokal.valid) {
    esp_sleep_wakeup_cause_t alasan = esp_sleep_get_wakeup_cause();
    const char* labelPemicu = (alasan == ESP_SLEEP_WAKEUP_EXT0) ? "EXT0 BTN" : "TIMER";

    // 1. Tampilkan ke Layar OLED SSD1306
    perbaruiTampilanOled(lokal.suhu, lokal.tekanan, lokal.elevasi, nomorSiklusBoot, labelPemicu);

    // 2. Serialisasi JSON ke Serial Monitor
    JsonDocument doc;
    doc["device_id"]      = "esp32-weather-st01";
    doc["boot_count"]     = nomorSiklusBoot;
    doc["wakeup_cause"]   = labelPemicu;
    doc["temp_c"]         = serialized(String(lokal.suhu, 2));
    doc["pressure_hpa"]   = serialized(String(lokal.tekanan, 2));
    doc["altitude_m"]     = serialized(String(lokal.elevasi, 1));
    doc["prev_temp_c"]    = serialized(String(riwayatSuhuTerakhir, 2));
    serializeJson(doc, Serial);
    Serial.println();

    // Simpan ke memori RTC untuk siklus berikutnya
    riwayatSuhuTerakhir = lokal.suhu;
    riwayatTekananTerakhir = lokal.tekanan;
  }
  vTaskDelete(NULL); // Tugas selesai dan menghapus dirinya sendiri
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_LED_STATUS, OUTPUT);
  pinMode(PIN_WAKEUP_TOMBOL, INPUT_PULLUP);
  digitalWrite(PIN_LED_STATUS, HIGH); // LED MENYALA: FASE AKTIF

  nomorSiklusBoot++;
  cetakDiagnostikWakeup();

  mutexData = xSemaphoreCreateMutex();
  Wire.begin(21, 22); // SDA = 21, SCL = 22

  // Inisialisasi Layar OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, ALAMAT_OLED)) {
    Serial.println("Gagal menemukan layar OLED SSD1306!");
  }

  // Inisialisasi Sensor BMP280 / BME280 pada alamat 0x76
  if (!bmp.begin(0x76)) {
    Serial.println("Gagal mendeteksi sensor BMP/BME280 pada alamat 0x76!");
    dataCuaca.valid = false;
  } else {
    // Parameter konfigurasi oversampling presisi
    bmp.setSampling(Adafruit_BMP280::MODE_FORCED,
                    Adafruit_BMP280::SAMPLING_X2,
                    Adafruit_BMP280::SAMPLING_X16,
                    Adafruit_BMP280::FILTER_X16,
                    Adafruit_BMP280::STANDBY_MS_500);
    bmp.takeForcedMeasurement(); // Paksa pengukuran cepat

    xSemaphoreTake(mutexData, portMAX_DELAY);
    dataCuaca.suhu    = bmp.readTemperature();
    dataCuaca.tekanan = bmp.readPressure() / 100.0F; // Pa -> hPa
    dataCuaca.elevasi = bmp.readAltitude(1013.25);   // Standar sea-level
    dataCuaca.valid   = true;
    xSemaphoreGive(mutexData);
  }

  // Buat Task UI pada Core 0 (Protocol Core) secara konkuren
  xTaskCreatePinnedToCore(TaskTampilanDanSerial, "TaskUI", 4096, NULL, 1, NULL, 0);

  // Berikan waktu sejenak agar Task UI menyelesaikan tugasnya (500 ms)
  delay(500);

  // --- PERSIAPAN MASUK DEEP SLEEP ---
  Serial.printf("[POWER] Menyiapkan Deep Sleep selama %d detik...\n", DURASI_TIDUR_DETIK);
  esp_sleep_enable_timer_wakeup((uint64_t)DURASI_TIDUR_DETIK * 1000000ULL);
  esp_sleep_enable_ext0_wakeup(PIN_WAKEUP_TOMBOL, 0); // Bangun jika GPIO 15 ditekan (LOW)

  digitalWrite(PIN_LED_STATUS, LOW); // LED MATI: MEMASUKI DEEP SLEEP
  Serial.println("[POWER] Zzz... Memasuki Deep Sleep. Arus turun ke ~10-150 uA.");
  Serial.flush();
  esp_deep_sleep_start();
}

void loop() {
  // Loop kosong karena sistem bekerja pada arsitektur Deep Sleep penuh
}
