import json
from iot.simulator import iter_messages

def test_message_format_matches_firmware(cfg):
    msgs = list(iter_messages(cfg, "heavy_rain", backfill_hours=2, live_hours=1, start="2026-07-01T00:00:00+05:30"))
    tele = [m for m in msgs if m["kind"] == "telemetry"][0]
    assert tele["topic"] == "flood/akota/telemetry"
    assert set(tele["payload"]) == {"station", "level_cm", "rain_mmh", "battery_v", "ts"}
    assert tele["payload"]["ts"].endswith("+05:30")
    json.dumps(tele["payload"])

def test_storm_raises_level(cfg):
    lv = [m["payload"]["level_cm"] for m in iter_messages(cfg, "heavy_rain", backfill_hours=110, live_hours=26,
          start="2026-07-01T00:00:00+05:30") if m["kind"] == "telemetry" and m["payload"]["station"] == "akota"]
    assert max(lv) > cfg["stations"]["akota"]["danger_cm"] and lv[0] < 20

def test_normal_scenario_stays_low(cfg):
    lv = [m["payload"]["level_cm"] for m in iter_messages(cfg, "normal", backfill_hours=20, live_hours=20,
          start="2026-07-01T00:00:00+05:30") if m["kind"] == "telemetry" and m["payload"]["station"] == "akota"]
    assert max(lv) < 40
