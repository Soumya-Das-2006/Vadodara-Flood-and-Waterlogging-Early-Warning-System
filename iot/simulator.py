"""Sensor SIMULATOR: publishes the same JSON as the ESP32, so the whole system runs without hardware.

  python -m iot.simulator --mqtt-host localhost --scenario heavy_rain
  python -m iot.simulator --http-url http://localhost:8000 --api-key dev-key     # no broker needed
  python -m iot.simulator --dry-run                                               # print JSON lines

Rain: SYNTHETIC storm scenario by default. If data/raw/openmeteo_archive_hourly.csv exists (make data-real) and
--replay-start 2024-08-24 is given, REAL hourly rainfall is replayed instead (levels are still SYNTHETIC-TARGET).
Message: {"station":"akota","level_cm":85.4,"rain_mmh":12.0,"battery_v":3.9,"ts":"2026-11-22T10:00:00+05:30"}
"""
import argparse
import json
import math
import sys
import time
import numpy as np
import pandas as pd
from src.config import DATA_RAW, TZ, SEED, load_config
from src.level_sim import StationSim


def drizzle(i, rng):
    """Small showers for the dry/backfill period (SYNTHETIC)."""
    return float(max(0.0, 1.5 * math.sin(i / 37.0) ** 6 + rng.normal(0, 0.05)))


def storm(hours_since_start, peak=30.0, dur=9.0):
    """Bell-shaped storm (mm/h) starting at 0."""
    if 0 <= hours_since_start <= dur:
        return peak * math.sin(math.pi * hours_since_start / dur) ** 1.3
    return 0.0


def rain_profile(scenario, step_h, n_back, rng, replay=None, storm_delay_h=5.0, cycle_h=40.0, peak=30.0):
    """Returns f(i) -> rain mm/h at step i (i < n_back = backfill)."""
    def f(i):
        if replay is not None:
            return float(replay[i % len(replay)])
        if i < n_back or scenario == "normal":
            return drizzle(i, rng)
        t = (i - n_back) * step_h
        tc = t % cycle_h if cycle_h else t
        return drizzle(i, rng) + storm(tc - storm_delay_h, peak)
    return f


def iter_messages(cfg=None, scenario="heavy_rain", step_min=10, backfill_hours=110, start=None, live_hours=40,
                  seed=SEED, replay_start=None, peak=30.0, cycle_hours=0, forever=False):
    """Yield dicts {kind, topic, payload, live}. Pure generator (used by the CLI and by tests)."""
    cfg = cfg or load_config()
    rng = np.random.default_rng(seed)
    step_h = step_min / 60.0
    n_back = int(backfill_hours / step_h)
    n_live = int(live_hours / step_h)
    replay = None
    if replay_start:
        p = DATA_RAW / "openmeteo_archive_hourly.csv"
        d = pd.read_csv(p); d["timestamp"] = pd.to_datetime(d["timestamp"])
        d = d[d["timestamp"] >= pd.Timestamp(replay_start).tz_localize(None)]
        replay = np.repeat(d["precipitation"].fillna(0).values, int(60 / step_min))
        if len(replay) == 0:
            raise SystemExit("replay-start is beyond the downloaded data")
    prof = rain_profile(scenario, step_h, n_back, rng, replay, cycle_h=cycle_hours, peak=peak)
    t0 = pd.Timestamp(start) if start else pd.Timestamp.now(tz=TZ).floor(f"{step_min}min") - pd.Timedelta(hours=backfill_hours)
    t0 = t0.tz_localize(TZ) if t0.tzinfo is None else t0.tz_convert(TZ)
    sims = {sid: StationSim.from_cfg(sc["sim"]) for sid, sc in cfg["stations"].items()}
    i = 0
    while forever or i < n_back + n_live:
        r = prof(i)
        ts = t0 + pd.Timedelta(minutes=step_min * (i + 1))
        for sid, sim in sims.items():
            lvl = sim.step(r, step_h)
            m = max(0.0, lvl + rng.normal(0, 0.8))
            batt = 4.1 - 0.0004 * i / 6.0
            yield {"kind": "telemetry", "topic": f"flood/{sid}/telemetry", "live": i >= n_back,
                   "payload": {"station": sid, "level_cm": round(m, 1), "rain_mmh": round(max(0.0, r * rng.uniform(0.97, 1.03)), 1),
                               "battery_v": round(max(3.3, batt), 2), "ts": ts.isoformat()}}
        fut = []
        for k in range(1, 7):                     # simulated forecast = future rain x noise (SYNTHETIC)
            ri = prof(i + int(k / step_h)) if replay is None else replay[(i + int(k / step_h)) % len(replay)]
            fut.append(round(float(max(0.0, ri * rng.lognormal(0, 0.3))), 1))
        yield {"kind": "forecast", "topic": "flood/forecast", "live": i >= n_back,
               "payload": {"ts": ts.isoformat(), "rain_next_6h": fut}}
        i += 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mqtt-host"); ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--mqtt-user"); ap.add_argument("--mqtt-password")
    ap.add_argument("--http-url"); ap.add_argument("--api-key", default="dev-key")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scenario", choices=["normal", "heavy_rain"], default="heavy_rain")
    ap.add_argument("--speed", type=float, default=1.0, help="real seconds between live steps (each = --step-min sim minutes)")
    ap.add_argument("--step-min", type=int, default=10)
    ap.add_argument("--backfill-hours", type=int, default=110)
    ap.add_argument("--live-hours", type=int, default=40)
    ap.add_argument("--peak", type=float, default=30.0, help="storm peak mm/h")
    ap.add_argument("--loop", action="store_true", help="repeat a storm every --cycle-hours forever")
    ap.add_argument("--cycle-hours", type=float, default=40.0)
    ap.add_argument("--replay-start", help="REAL rain replay from this date (needs data/raw/openmeteo_archive_hourly.csv)")
    a = ap.parse_args(argv)
    cfg = load_config()

    start = None
    if a.http_url:
        import requests
        try:
            r = requests.get(f"{a.http_url}/readings", params={"station": next(iter(cfg["stations"])), "limit": 1},
                             headers={"X-API-Key": a.api_key}, timeout=5).json()
            if r:
                start = pd.Timestamp(r[-1]["ts"]) 
                print("resuming after last stored reading", start)
        except Exception:
            pass
    pub = None
    if a.mqtt_host:
        import paho.mqtt.client as mqtt
        pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="flood-sim")
        if a.mqtt_user:
            pub.username_pw_set(a.mqtt_user, a.mqtt_password)
        for attempt in range(30):
            try:
                pub.connect(a.mqtt_host, a.mqtt_port, 30); break
            except OSError:
                print("waiting for broker..."); time.sleep(2)
        pub.loop_start()
    if a.http_url:
        import requests

    n = 0
    last_live_ts = None
    for m in iter_messages(cfg, a.scenario, a.step_min, a.backfill_hours, start, a.live_hours, SEED,
                           a.replay_start, a.peak, a.cycle_hours if a.loop else 0, forever=a.loop):
        body = json.dumps(m["payload"])
        if pub:
            pub.publish(m["topic"], body, qos=1)
        elif a.http_url:
            path = "/ingest" if m["kind"] == "telemetry" else "/forecast-input"
            try:
                requests.post(a.http_url + path, data=body, headers={"X-API-Key": a.api_key, "Content-Type": "application/json"}, timeout=10)
            except Exception as e:
                print("post failed:", e)
        else:
            print(body)
        n += 1
        if m["live"] and m["kind"] == "forecast":
            time.sleep(a.speed)
        elif not m["live"]:
            time.sleep(0.002)
        if m["live"] and m["kind"] == "telemetry" and m["payload"]["station"] == next(iter(cfg["stations"])) and n % 60 == 0:
            print(f"live {m['payload']['ts']} level={m['payload']['level_cm']} rain={m['payload']['rain_mmh']}", flush=True)
    print("scenario finished,", n, "messages")


if __name__ == "__main__":
    main()
