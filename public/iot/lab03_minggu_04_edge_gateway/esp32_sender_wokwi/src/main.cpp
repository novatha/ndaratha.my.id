/**
 * Lab-3: ESP32 Serial Telemetry Streamer to Edge Gateway
 * Prodi S1 Informatika - Universitas Bengkulu
 */

#include <Arduino.h>
#include <ArduinoJson.h>

const char* NODE_ID = "ESP32_UNIB_01";
const int POT_PIN = 34;   // Input ADC analog
const int LED_PIN = 2;    // Indikator TX transmisi

uint32_t packet_seq = 0;

void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }

    pinMode(POT_PIN, INPUT);
    pinMode(LED_PIN, OUTPUT);
    analogReadResolution(12);

    digitalWrite(LED_PIN, HIGH);
    delay(500);
    digitalWrite(LED_PIN, LOW);
}

void loop() {
    packet_seq++;

    // Baca sensor analog (potensiometer simulasi tegangan)
    int raw_adc = analogRead(POT_PIN);
    float voltage = (raw_adc / 4095.0f) * 3.3f;
    float temp = 25.0f + (voltage * 4.0f);                      // 25.0 - 38.2 C
    float hum = 60.0f + ((4095 - raw_adc) / 4095.0f) * 30.0f;  // 60 - 90%
    float battery = 3.6f + (voltage / 3.3f) * 0.6f;             // 3.6 - 4.2 V

    // Susun dokumen JSON
    StaticJsonDocument<256> doc;
    doc["node_id"] = NODE_ID;
    doc["timestamp"] = (uint32_t)(millis() / 1000);
    doc["seq"] = packet_seq;
    doc["temp"] = round(temp * 100.0) / 100.0;
    doc["hum"] = round(hum * 10.0) / 10.0;
    doc["lux"] = map(raw_adc, 0, 4095, 100, 1200);
    doc["battery"] = round(battery * 100.0) / 100.0;
    doc["status"] = (battery > 3.75f) ? "NORMAL" : "LOW_BAT";

    // Kedipkan LED indikator pengiriman
    digitalWrite(LED_PIN, HIGH);

    // Kirim payload dengan delimiter newline (\r\n)
    serializeJson(doc, Serial);
    Serial.println();

    digitalWrite(LED_PIN, LOW);

    // Interval pengiriman 2 detik
    delay(2000);
}
