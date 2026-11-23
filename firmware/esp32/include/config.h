// Copy this file's values for your site. DO NOT commit real Wi-Fi/MQTT passwords.
#pragma once
#define STATION_ID        "akota"          // must match config/stations.yaml
#define WIFI_SSID         "YOUR_WIFI"
#define WIFI_PASSWORD     "YOUR_PASSWORD"
#define MQTT_HOST         ""               // optional broker IP / hostname; leave empty to use HTTPS HTTP ingest
#define MQTT_PORT         1883
#define MQTT_USER         ""               // leave empty if the broker allows anonymous (demo only)
#define MQTT_PASSWORD     ""
#define API_URL           "https://YOUR-BACKEND-URL" // Render/Railway HTTPS URL, or http://LAN-IP:8000
#define API_KEY           "YOUR_API_KEY"
#define NTP_SERVER        "pool.ntp.org"
#define TZ_OFFSET_SEC     19800            // IST = UTC+5:30

// Wiring (see docs/HARDWARE.md)
#define PIN_TRIG          25               // JSN-SR04T trigger
#define PIN_ECHO          26               // JSN-SR04T echo (through a 5V->3.3V divider!)
#define PIN_RAIN          27               // tipping bucket reed switch (to GND, INPUT_PULLUP)
#define PIN_BATT          35               // battery divider 100k/100k -> ADC

// Calibration
#define SENSOR_HEIGHT_CM  200.0            // distance from sensor face to the zero-level (riverbed / road surface)
#define SENSOR_MIN_CM     25.0             // JSN-SR04T blind zone
#define SENSOR_MAX_CM     450.0
#define N_SAMPLES         9                // median of N readings
#define MM_PER_TIP        0.2794           // typical 0.01 inch bucket; check your gauge
#define RAIN_WINDOW_SEC   600              // rain rate averaged over the last 10 min
#define PUBLISH_EVERY_MS  60000
