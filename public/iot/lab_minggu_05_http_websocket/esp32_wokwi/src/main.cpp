#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

// Konfigurasi Wi-Fi (Default Simulator Wokwi)
const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASS = "";

// Ganti HOST dengan URL tunnel publik (mis. ngrok / cloudflared) saat demo dari Wokwi web,
// atau IP host lokal saat memakai Wokwi Private Gateway / ESP32 fisik.
const char* SERVER_HOST = "example-tunnel.trycloudflare.com";
const uint16_t SERVER_PORT = 80;
const char* NODE_ID = "esp32-bengkulu-01";

#define PIN_DHT        4
#define PIN_POT_WATER  34   // ADC1_CH6 (aman saat Wi-Fi aktif!)
#define PIN_PUMP_LED   26   // Simulasi Relai Pompa / Pintu Air
#define PIN_TOMBOL     15   // Pushbutton INPUT_PULLUP override lokal

DHT dht(PIN_DHT, DHT22);
WebSocketsClient webSocket;

bool pumpOn = false;
uint32_t seqCounter = 0;
uint32_t lastTelemetryMs = 0;
const uint32_t TELEMETRY_INTERVAL_MS = 4000;

void applyPumpState(bool on) {
  pumpOn = on;
  digitalWrite(PIN_PUMP_LED, pumpOn ? HIGH : LOW);
  Serial.printf("[AKTUATOR] Status Pompa diubah menjadi: %s\n", pumpOn ? "ON" : "OFF");
}

void onWebSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      Serial.println("[WS] Handshake 101 Switching Protocols berhasil!");
      break;

    case WStype_DISCONNECTED:
      Serial.println("[WS] Koneksi terputus, mencoba hubung ulang...");
      break;

    case WStype_TEXT: {
      JsonDocument doc;
      DeserializationError err = deserializeJson(doc, payload, length);
      if (!err && strcmp(doc["type"] | "", "ACTUATOR_CMD") == 0) {
        const char* targetState = doc["pump_state"] | "OFF";
        applyPumpState(strcmp(targetState, "ON") == 0);

        // Kirim ACK balik melalui kanal full-duplex yang sama
        JsonDocument ack;
        ack["type"] = "ACK";
        ack["node_id"] = NODE_ID;
        ack["pump_state"] = pumpOn ? "ON" : "OFF";
        String out;
        serializeJson(ack, out);
        webSocket.sendTXT(out);
      }
      break;
    }
    default:
      break;
  }
}

void sendTelemetryHttp(float temp, float hum, float waterCm) {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  String url = String("http://") + SERVER_HOST + ":" + SERVER_PORT + "/api/v1/telemetry";
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(3000);

  JsonDocument doc;
  doc["node_id"] = NODE_ID;
  doc["seq"] = seqCounter;
  doc["temp_c"] = roundf(temp * 100.0f) / 100.0f;
  doc["hum_pct"] = roundf(hum * 10.0f) / 10.0f;
  doc["water_level_cm"] = roundf(waterCm * 10.0f) / 10.0f;
  doc["pump_state"] = pumpOn ? "ON" : "OFF";
  doc["client_ts_ms"] = millis();

  String body;
  serializeJson(doc, body);

  uint32_t t0 = millis();
  int httpCode = http.POST(body);
  uint32_t rtt = millis() - t0;

  if (httpCode > 0) {
    Serial.printf("[HTTP POST #%lu] Kode=%d RTT=%lu ms Payload=%u B\n",
                  seqCounter, httpCode, rtt, body.length());
  } else {
    Serial.printf("[HTTP POST #%lu] Gagal: %s\n",
                  seqCounter, http.errorToString(httpCode).c_str());
  }
  http.end();
}

void sendTelemetryWebSocket(float temp, float hum, float waterCm) {
  if (!webSocket.isConnected()) return;

  JsonDocument doc;
  doc["type"] = "TELEMETRY";
  doc["node_id"] = NODE_ID;
  doc["seq"] = seqCounter;
  doc["temp_c"] = roundf(temp * 100.0f) / 100.0f;
  doc["hum_pct"] = roundf(hum * 10.0f) / 10.0f;
  doc["water_level_cm"] = roundf(waterCm * 10.0f) / 10.0f;
  doc["pump_state"] = pumpOn ? "ON" : "OFF";
  doc["client_ts_ms"] = millis();

  String body;
  serializeJson(doc, body);
  webSocket.sendTXT(body);
  Serial.printf("[WS SEND   #%lu] Frame terkirim penuh Payload=%u B\n",
                seqCounter, body.length());
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_PUMP_LED, OUTPUT);
  pinMode(PIN_TOMBOL, INPUT_PULLUP);
  analogReadResolution(12);
  dht.begin();

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WIFI] Menghubungkan");
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
  }
  Serial.printf("\n[WIFI] Terhubung! IP=%s\n", WiFi.localIP().toString().c_str());

  String wsPath = String("/ws/device/") + NODE_ID;
  webSocket.begin(SERVER_HOST, SERVER_PORT, wsPath);
  webSocket.onEvent(onWebSocketEvent);
  webSocket.setReconnectInterval(3000);
  webSocket.enableHeartbeat(15000, 3000, 2);
}

void loop() {
  webSocket.loop(); // Non-blocking event loop RFC 6455

  // Override manual tombol fisik (active-low)
  static uint32_t lastBtnMs = 0;
  if (digitalRead(PIN_TOMBOL) == LOW && (millis() - lastBtnMs > 400)) {
    lastBtnMs = millis();
    applyPumpState(!pumpOn);
  }

  // Kirim telemetri bergantian HTTP POST (ganjil) vs WebSocket (genap)
  if (millis() - lastTelemetryMs >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryMs = millis();
    seqCounter++;

    float temp = dht.readTemperature();
    float hum = dht.readHumidity();
    if (isnan(temp)) temp = 29.5f;
    if (isnan(hum)) hum = 76.0f;

    int rawAdc = analogRead(PIN_POT_WATER);
    float waterCm = (rawAdc / 4095.0f) * 250.0f; // 0 - 250 cm

    if (seqCounter % 2 == 1) {
      sendTelemetryHttp(temp, hum, waterCm);
    } else {
      sendTelemetryWebSocket(temp, hum, waterCm);
    }
  }
}
