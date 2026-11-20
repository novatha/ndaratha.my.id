#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

// Konfigurasi Wi-Fi (Default Simulator Wokwi)
const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASS = "";

// Broker MQTT: gunakan "broker.hivemq.com" atau "test.mosquitto.org" untuk uji langsung dari Wokwi Web,
// atau ganti dengan IP komputer lokal jika memakai Mosquitto Docker + ESP32 fisik.
const char* MQTT_BROKER = "broker.hivemq.com";
const uint16_t MQTT_PORT = 1883;

// Ganti "tim01" dengan nomor kelompok PjBL Anda agar topik tidak bentrok di broker publik!
const char* GROUP_ID = "tim01";
const char* NODE_ID  = "node-01";

#define PIN_DHT        4
#define PIN_SOIL_ADC   34   // ADC1_CH6 (aman saat Wi-Fi aktif)
#define PIN_PUMP_LED   26   // Simulasi Relai Irigasi Tetes

DHT dht(PIN_DHT, DHT22);
WiFiClient espClient;
PubSubClient mqtt(espClient);

String topicStatus;
String topicTelemetry;
String topicCmdPump;
String topicAckPump;

bool pumpOn = false;
uint32_t seqCounter = 0;
uint32_t lastPubMs = 0;
uint32_t lastReconnectMs = 0;

void applyPump(bool state) {
  pumpOn = state;
  digitalWrite(PIN_PUMP_LED, pumpOn ? HIGH : LOW);
  Serial.printf("[AKTUATOR] Pompa Irigasi -> %s\n", pumpOn ? "ON" : "OFF");
}

void onMqttMessage(char* topic, byte* payload, unsigned int length) {
  String msg;
  msg.reserve(length);
  for (unsigned int i = 0; i < length; i++) {
    msg += (char)payload[i];
  }
  msg.trim();
  Serial.printf("[MQTT-SUB] %s => %s\n", topic, msg.c_str());

  if (String(topic) == topicCmdPump) {
    if (msg.equalsIgnoreCase("ON") || msg.indexOf("\"ON\"") >= 0) {
      applyPump(true);
    } else if (msg.equalsIgnoreCase("OFF") || msg.indexOf("\"OFF\"") >= 0) {
      applyPump(false);
    }
    JsonDocument ack;
    ack["node_id"] = NODE_ID;
    ack["pump_state"] = pumpOn ? "ON" : "OFF";
    ack["uptime_s"] = millis() / 1000;
    String out;
    serializeJson(ack, out);
    mqtt.publish(topicAckPump.c_str(), out.c_str(), false);
  }
}

bool reconnectMqttNonBlocking() {
  String clientId = String("unib-") + GROUP_ID + "-" + NODE_ID + "-" + String((uint32_t)ESP.getEfuseMac(), HEX);
  String lwtPayload = String("{\"node_id\":\"") + NODE_ID + "\",\"status\":\"offline\",\"reason\":\"lwt\"}";

  Serial.printf("[MQTT] Menghubungkan sebagai %s dengan LWT...\n", clientId.c_str());
  // connect(clientId, willTopic, willQos, willRetain, willMessage)
  if (mqtt.connect(clientId.c_str(), topicStatus.c_str(), 1, true, lwtPayload.c_str())) {
    Serial.println("[MQTT] Terhubung ke broker!");

    // Publikasikan status ONLINE sebagai Retained Message
    String onlineMsg = String("{\"node_id\":\"") + NODE_ID + "\",\"status\":\"online\"}";
    mqtt.publish(topicStatus.c_str(), onlineMsg.c_str(), true);

    // Subscribe topik kendali aktuator dengan QoS 1
    mqtt.subscribe(topicCmdPump.c_str(), 1);
    Serial.printf("[MQTT] Subscribe: %s (QoS 1)\n", topicCmdPump.c_str());
    return true;
  }
  Serial.printf("[MQTT] Gagal konek, rc=%d\n", mqtt.state());
  return false;
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_PUMP_LED, OUTPUT);
  analogReadResolution(12);
  dht.begin();

  // Hierarki topik terstruktur (Kasus B: Kebun Kopi Rejang Lebong)
  String base = String("unib/") + GROUP_ID + "/rejang-lebong/kebun-kopi/" + NODE_ID;
  topicStatus    = base + "/status";
  topicTelemetry = base + "/telemetry";
  topicCmdPump   = base + "/cmd/pump";
  topicAckPump   = base + "/ack/pump";

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WIFI] Menghubungkan");
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
  }
  Serial.printf("\n[WIFI] Terhubung! IP=%s\n", WiFi.localIP().toString().c_str());

  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setKeepAlive(15); // Keep-alive 15 detik (LWT terpicu ~22.5 detik jika putus mendadak)
  mqtt.setBufferSize(512);
  mqtt.setCallback(onMqttMessage);
}

void loop() {
  if (!mqtt.connected()) {
    if (millis() - lastReconnectMs >= 4000) {
      lastReconnectMs = millis();
      reconnectMqttNonBlocking();
    }
  } else {
    mqtt.loop();
  }

  if (mqtt.connected() && (millis() - lastPubMs >= 5000)) {
    lastPubMs = millis();
    seqCounter++;

    float temp = dht.readTemperature();
    float hum = dht.readHumidity();
    if (isnan(temp)) temp = 24.5f;
    if (isnan(hum)) hum = 78.0f;

    int rawAdc = analogRead(PIN_SOIL_ADC);
    float soilPct = (rawAdc / 4095.0f) * 100.0f;

    JsonDocument doc;
    doc["node_id"] = NODE_ID;
    doc["seq"] = seqCounter;
    doc["temp_c"] = roundf(temp * 100.0f) / 100.0f;
    doc["hum_pct"] = roundf(hum * 10.0f) / 10.0f;
    doc["soil_moisture_pct"] = roundf(soilPct * 10.0f) / 10.0f;
    doc["pump_state"] = pumpOn ? "ON" : "OFF";

    String payload;
    serializeJson(doc, payload);
    bool ok = mqtt.publish(topicTelemetry.c_str(), payload.c_str(), false);
    Serial.printf("[MQTT-PUB #%lu] ok=%d | %s -> %s\n",
                  seqCounter, ok, topicTelemetry.c_str(), payload.c_str());
  }
}
