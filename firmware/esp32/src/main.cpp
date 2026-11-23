// Vadodara Flood EWS - ESP32 node: JSN-SR04T water level + tipping-bucket rain gauge -> MQTT JSON every 60 s.
// Not compiled in the build sandbox (no PlatformIO there). Please compile & test on your board.
#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <time.h>
#include <esp_task_wdt.h>
#include "config.h"

WiFiClient wifi;
PubSubClient mqtt(wifi);

volatile uint32_t tipTimes[64];      // ring buffer of tip timestamps (ms)
volatile uint8_t tipHead = 0;
volatile uint32_t lastTipMs = 0;
uint32_t lastPublish = 0, lastWifiTry = 0, lastMqttTry = 0;

void IRAM_ATTR onTip() {             // reed switch bounces: ignore tips closer than 200 ms
  uint32_t now = millis();
  if (now - lastTipMs < 200) return;
  lastTipMs = now;
  tipTimes[tipHead] = now;
  tipHead = (tipHead + 1) & 63;
}

float readDistanceCm() {
  digitalWrite(PIN_TRIG, LOW); delayMicroseconds(4);
  digitalWrite(PIN_TRIG, HIGH); delayMicroseconds(12);
  digitalWrite(PIN_TRIG, LOW);
  unsigned long us = pulseIn(PIN_ECHO, HIGH, 30000UL);       // 30 ms timeout (~5 m)
  if (us == 0) return NAN;
  return us * 0.0343f / 2.0f;                                  // speed of sound ~343 m/s (20 C)
}

float medianLevelCm() {                                        // median of N valid readings = spike filter
  float v[N_SAMPLES]; int n = 0;
  for (int i = 0; i < N_SAMPLES; i++) {
    float d = readDistanceCm();
    if (!isnan(d) && d >= SENSOR_MIN_CM && d <= SENSOR_MAX_CM) v[n++] = d;
    delay(60);
  }
  if (n < 3) return NAN;                                       // too few good echoes -> report nothing
  for (int i = 1; i < n; i++) { float k = v[i]; int j = i - 1; while (j >= 0 && v[j] > k) { v[j + 1] = v[j]; j--; } v[j + 1] = k; }
  float dist = (n % 2) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
  float level = SENSOR_HEIGHT_CM - dist;
  return level < 0 ? 0 : level;
}

float rainRateMmH() {                                          // tips in the last RAIN_WINDOW_SEC -> mm/h
  uint32_t now = millis(); int tips = 0;
  noInterrupts();
  for (int i = 0; i < 64; i++) if (tipTimes[i] && now - tipTimes[i] <= RAIN_WINDOW_SEC * 1000UL) tips++;
  interrupts();
  return tips * MM_PER_TIP * 3600.0f / RAIN_WINDOW_SEC;
}

float batteryV() {                                             // 100k/100k divider -> x2
  uint32_t mv = 0; for (int i = 0; i < 16; i++) mv += analogReadMilliVolts(PIN_BATT);
  return (mv / 16.0f) * 2.0f / 1000.0f;
}

void ensureWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  if (millis() - lastWifiTry < 10000 && lastWifiTry) return;   // retry every 10 s, non-blocking
  lastWifiTry = millis();
  WiFi.disconnect(); WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

void ensureMqtt() {
  if (WiFi.status() != WL_CONNECTED || mqtt.connected()) return;
  if (millis() - lastMqttTry < 5000 && lastMqttTry) return;
  lastMqttTry = millis();
  String id = String("flood-") + STATION_ID;
  bool ok = strlen(MQTT_USER) ? mqtt.connect(id.c_str(), MQTT_USER, MQTT_PASSWORD) : mqtt.connect(id.c_str());
  if (ok) Serial.println("MQTT connected");
}

String isoTime() {
  struct tm t;
  if (!getLocalTime(&t, 200)) return "";                       // NTP not ready -> caller skips publish
  char b[40]; strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S+05:30", &t);
  return String(b);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_TRIG, OUTPUT); pinMode(PIN_ECHO, INPUT); pinMode(PIN_RAIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_RAIN), onTip, FALLING);
  analogSetPinAttenuation(PIN_BATT, ADC_11db);
  WiFi.mode(WIFI_STA); WiFi.setAutoReconnect(true);
  mqtt.setServer(MQTT_HOST, MQTT_PORT); mqtt.setBufferSize(256);
  configTime(TZ_OFFSET_SEC, 0, NTP_SERVER);
  esp_task_wdt_init(60, true); esp_task_wdt_add(NULL);          // reboot if the loop hangs for 60 s
}

void loop() {
  esp_task_wdt_reset();
  ensureWifi(); ensureMqtt(); mqtt.loop();
  if (millis() - lastPublish < PUBLISH_EVERY_MS) { delay(20); return; }
  lastPublish = millis();
  float level = medianLevelCm();
  String ts = isoTime();
  if (isnan(level) || ts.length() == 0) { Serial.println("skip: no valid level or time"); return; }
  JsonDocument d;
  d["station"] = STATION_ID; d["level_cm"] = roundf(level * 10) / 10.0f;
  d["rain_mmh"] = roundf(rainRateMmH() * 10) / 10.0f; d["battery_v"] = roundf(batteryV() * 100) / 100.0f; d["ts"] = ts;
  char out[200]; size_t n = serializeJson(d, out, sizeof out);
  if (mqtt.connected()) { mqtt.publish((String("flood/") + STATION_ID + "/telemetry").c_str(), (uint8_t*)out, n, false); Serial.println(out); }
  else Serial.println("offline: reading dropped (add SD/flash buffer if you need store-and-forward)");
}
