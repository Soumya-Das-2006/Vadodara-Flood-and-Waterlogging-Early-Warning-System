# Hardware: wiring and bill of materials

Not built or tested in the sandbox. Prices are rough Indian market estimates (INR, 2026) and vary a lot by seller: check before ordering.

## Wiring (ESP32 DevKit v1)
| Part | Pin / connection |
|---|---|
| JSN-SR04T transducer board | VCC 5 V (use the 5 V / VIN pin), GND, TRIG -> GPIO25, ECHO -> GPIO26 **through a divider** (1 k + 2 k to GND) because ECHO is 5 V and ESP32 pins are 3.3 V |
| Tipping-bucket rain gauge (reed switch) | one wire GPIO27 (INPUT_PULLUP), other wire GND; 100 nF across the switch helps debounce |
| Battery monitor | 18650 +  -> 100 k -> GPIO35 -> 100 k -> GND (reads half the voltage; firmware multiplies by 2) |
| Power | 6 V 2 W solar panel -> TP4056 (with protection) -> 18650 -> 3.3/5 V boost-buck -> ESP32 VIN and sensor 5 V |

Mounting: sensor faces straight down over the water from a fixed bracket (bridge rail / underpass wall), at least 25 cm above the highest water, away from walls (beam is a cone). Measure the distance from the sensor face to the zero level and set `SENSOR_HEIGHT_CM` in `firmware/esp32/include/config.h`. Level = height - measured distance. Rain gauge in open sky, level, away from trees.

## ESP32 live-data setup

1. Copy the firmware folder into a PlatformIO project and open `firmware/esp32/include/config.h`.
2. Set `STATION_ID` to the exact station ID from `config/stations.yaml`, for example `akota`.
3. Set `WIFI_SSID`, `WIFI_PASSWORD`, and `SENSOR_HEIGHT_CM`.
4. For HTTP mode, leave `MQTT_HOST` empty and set `API_URL` and `API_KEY`:
   - Local/LAN backend: `API_URL=http://BACKEND-LAN-IP:8000`
   - Hosted backend: `API_URL=https://your-backend-host`
   - `API_KEY` must exactly match the backend `API_KEY` environment variable.
5. Upload the firmware with PlatformIO. Open the Serial Monitor at `115200`; a successful reading reports `reading accepted by HTTP ingest`.
6. Verify the live reading:

```bash
curl "http://BACKEND-LAN-IP:8000/stations" -H "X-API-Key: YOUR_API_KEY"
curl "http://BACKEND-LAN-IP:8000/readings?station=akota&limit=10" -H "X-API-Key: YOUR_API_KEY"
```

For MQTT mode, set `MQTT_HOST`, `MQTT_PORT`, and credentials if enabled. The ESP32 will publish `flood/<station>/telemetry`; the backend must have the same broker host configured in `MQTT_HOST` and credentials in `MQTT_USER` / `MQTT_PASSWORD`.

## Bill of materials (per station)
| Item | Qty | Approx. price (INR) |
|---|---|---|
| ESP32 DevKit v1 | 1 | 450 - 650 |
| JSN-SR04T waterproof ultrasonic sensor | 1 | 450 - 800 |
| Tipping-bucket rain gauge (0.2794 mm/tip type) | 1 | 1,500 - 4,000 |
| 18650 Li-ion cell + holder + TP4056 charger | 1 set | 350 - 600 |
| 6 V 2 W solar panel | 1 | 250 - 450 |
| Boost/buck regulator module | 1 | 100 - 200 |
| Resistors, capacitor, terminals, wire | 1 set | 100 - 200 |
| IP65 enclosure + cable glands | 1 | 300 - 600 |
| Mounting bracket / pole clamps | 1 | 300 - 800 |
| Optional: 4G modem (if no Wi-Fi) | 1 | 1,500 - 2,800 |
| **Total (no modem / with modem)** | | **about 3,800 - 8,000 / 5,300 - 10,800** |

Network: Wi-Fi or a 4G router with a fixed MQTT broker reachable from the node. For production add MQTT username/password and TLS.
