import pandas as pd
import pytest
from backend.app.db import DB
from backend.app.service import FloodService
from backend.app.telegram import TelegramNotifier
from iot.simulator import iter_messages
from src import alerts as A


class FakePred:
    """Predictor stub: status follows the last level, so the state machine is tested in isolation."""
    def __init__(self, seq): self.seq, self.i = seq, 0
    def predict(self, station, h, fc=None):
        s = self.seq[min(self.i, len(self.seq) - 1)]; self.i += 1
        return {"station": station, "as_of": str(h["timestamp"].iloc[-1]), "current_cm": 70.0, "danger_cm": 80.0,
                "forecast_cm": {"1h": 72, "2h": 79, "3h": 85}, "interval_cm": {f"{k}h": {"p10": 60, "p50": 75, "p90": 90} for k in (1, 2, 3)},
                "flood_probability": 0.8 if s != "safe" else 0.01, "eta_minutes": 180 if s != "safe" else None,
                "status": s, "threshold": 0.4}


def _svc(pred, cfg, notifier=None):
    db = DB()
    svc = FloodService(db, pred, cfg, notifier=notifier)
    for i in range(130):      # 130 hourly readings so history >= 96 h
        svc.ingest({"station": "akota", "ts": pd.Timestamp("2025-07-01", tz="Asia/Kolkata") + pd.Timedelta(hours=i),
                    "level_cm": 10.0, "rain_mmh": 0.0})
    return svc


def test_escalation_alert_and_hysteresis(cfg):
    svc = _svc(FakePred(["safe", "warning", "danger", "warning", "safe", "safe", "safe", "safe"]), cfg)
    out = [svc.evaluate("akota")["status"] for _ in range(8)]
    assert out[:3] == ["safe", "warning", "danger"]
    assert out[3] == "danger" and out[-1] in ("warning", "safe")      # held by hysteresis, then steps down
    seq = [(a["from_status"], a["to_status"]) for a in svc.db.alerts()[::-1]]
    assert seq[0] == ("safe", "warning") and seq[1] == ("warning", "danger")
    assert all(a["texts"].keys() == {"en", "hi", "gu"} for a in svc.db.alerts())

def test_no_alert_while_safe(cfg):
    svc = _svc(FakePred(["safe"] * 5), cfg)
    for _ in range(5): svc.evaluate("akota")
    assert svc.db.alerts() == []

def test_insufficient_history_returns_none(cfg):
    svc = FloodService(DB(), FakePred(["safe"]), cfg)
    svc.ingest({"station": "akota", "ts": pd.Timestamp.now(tz="Asia/Kolkata"), "level_cm": 5.0, "rain_mmh": 0.0})
    assert svc.evaluate("akota") is None

def test_telegram_dedupe_through_service(cfg):
    sent = []
    n = TelegramNotifier(token="x", chat_id="y", sender=sent.append, min_interval=0)
    svc = _svc(FakePred(["warning", "safe", "safe", "safe", "warning"]), cfg, notifier=n)
    for _ in range(5): svc.evaluate("akota")
    assert len(sent) >= 2 and svc.db.alerts()[-1]["notify"] == "sent"


def test_heavy_rain_scenario_triggers_trilingual_alert(predictor, cfg):
    """End to end (no MQTT): simulator -> service -> real model -> alert in en/hi/gu that passes the validator."""
    db = DB(); svc = FloodService(db, predictor, cfg)
    for m in iter_messages(cfg, "heavy_rain", step_min=10, backfill_hours=110, live_hours=26, start="2026-07-01T00:00:00+05:30"):
        p = m["payload"]
        if m["kind"] == "forecast":
            svc.set_forecast(p["ts"], p["rain_next_6h"])
        else:
            svc.ingest(p)
            if p["station"] == "akota" and m["live"] and p["ts"][14:16] == "00":      # evaluate hourly
                svc.evaluate("akota")
    al = db.alerts(station="akota")
    assert al, "heavy rain must raise at least one alert"
    levels = {a["to_status"] for a in al}
    assert levels & {"warning", "danger"}
    first = al[-1]
    assert set(first["texts"]) == {"en", "hi", "gu"}
    assert A.validate_alert(first["texts"], first["facts"]) == []
