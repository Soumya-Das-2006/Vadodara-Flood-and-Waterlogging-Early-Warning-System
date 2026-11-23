"""Core logic (no web framework): ingest readings, run the model, track status, create alerts."""
import logging
import threading
import time
import numpy as np
import pandas as pd
from src import alerts as A
from src.clean import spike_filter
from src.config import TZ
from .telegram import SEV

log = logging.getLogger("flood")


class FloodService:
    def __init__(self, db, predictor, cfg, llm=None, notifier=None, forecast_provider=None, clock=time.time):
        self.db, self.p, self.cfg = db, predictor, cfg
        self.llm, self.notifier, self.clock = llm, notifier, clock
        self.forecast_provider = forecast_provider
        self.stations = cfg["stations"]
        self.s = cfg["settings"]
        self._fc = None                       # latest pushed forecast {"ts": Timestamp, "values": [..6]}
        self._last_eval = {}
        self.lock = threading.RLock()

    # ---- inputs
    def ingest(self, r):
        self.db.insert_reading(r["station"], r["ts"], r["level_cm"], r.get("rain_mmh", 0.0), r.get("battery_v"))

    def set_forecast(self, ts, values):
        self._fc = {"ts": pd.Timestamp(ts).tz_convert(TZ) if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).tz_localize(TZ),
                    "values": [float(v) for v in values][:6]}

    def _forecast_for(self, as_of):
        if self._fc is not None and abs(as_of - self._fc["ts"]) <= pd.Timedelta(minutes=90):
            return self._fc["values"]
        if self.forecast_provider:
            try:
                return self.forecast_provider(as_of)
            except Exception as e:
                log.warning("forecast provider failed: %s", e)
        return None

    # ---- model + state machine
    def evaluate(self, station):
        with self.lock:
            h = self.db.hourly(station, hours=240)
            need = self.s["min_history_hours"]
            if len(h) < need or h["level_cm"].notna().sum() < need * 0.8:
                return None
            h = h.copy()
            h["level_cm"] = spike_filter(h["level_cm"].values)[0]
            h = h.ffill(limit=3).dropna(subset=["level_cm", "rain_mmh"]).reset_index(drop=True)
            as_of = h["timestamp"].iloc[-1]
            pred = self.p.predict(station, h, self._forecast_for(as_of))
            self.db.save_forecast(station, pred["as_of"], pred, self.clock())
            st = self.db.get_state(station)
            cur, streak, raw = st["status"], st["streak"], pred["status"]
            new = cur
            if SEV[raw] > SEV[cur]:
                new, streak = raw, 0
            elif SEV[raw] < SEV[cur]:
                streak += 1
                if streak >= self.s["deescalate_after"]:
                    new, streak = raw, 0
            else:
                streak = 0
            self.db.set_state(station, new, streak, pred["as_of"])
            pred["status_raw"], pred["status"] = raw, new
            if new != cur:
                self._alert(station, cur, new, pred)
            return pred

    def _alert(self, station, old, new, pred):
        facts = A.make_facts(self.stations[station], pred)
        res = A.generate_alert(facts, llm=self.llm)
        notify = ""
        aid = self.db.add_alert(ts=pred["as_of"], created=self.clock(), station=station, from_status=old, to_status=new,
                                facts=facts, pred=pred, texts=res["texts"], source=res["source"], errors=res["errors"])
        if self.notifier:
            notify = self.notifier.send(station, new, res["texts"])
            self.db.update_alert_notify(aid, notify)
        log.info("ALERT %s %s->%s via %s (%s)", station, old, new, res["source"], notify)
        return aid

    def evaluate_all(self):
        out = {}
        for sid in self.stations:
            last = self.db.latest_ts(sid)
            if last and last != self._last_eval.get(sid):
                self._last_eval[sid] = last
                try:
                    out[sid] = self.evaluate(sid)
                except Exception:
                    log.exception("evaluate failed for %s", sid)
        return out

    def snapshot(self):
        rows = []
        for sid, sc in self.stations.items():
            fc = self.db.latest_forecast(sid)
            st = self.db.get_state(sid)
            rows.append({"id": sid, "name": sc["name"], "name_hi": sc.get("name_hi"), "name_gu": sc.get("name_gu"),
                         "lat": sc["lat"], "lon": sc["lon"], "danger_cm": sc["danger_cm"],
                         "alt_route": sc.get("alt_route", {}), "status": st["status"] if fc else "unknown",
                         "current_cm": fc["current_cm"] if fc else None,
                         "flood_probability": fc["flood_probability"] if fc else None,
                         "eta_minutes": fc["eta_minutes"] if fc else None, "as_of": fc["as_of"] if fc else None})
        return rows


class OpenMeteoForecast:
    """Real rain forecast (Open-Meteo, no key). Only meaningful when as_of is close to real time."""
    URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, lat, lon, ttl=1800):
        self.lat, self.lon, self.ttl, self.cache = lat, lon, ttl, (0, None)

    def __call__(self, as_of):
        import requests
        now = pd.Timestamp.now(tz=TZ)
        if abs(as_of - now) > pd.Timedelta(hours=2):
            return None
        if time.time() - self.cache[0] > self.ttl:
            js = requests.get(self.URL, params={"latitude": self.lat, "longitude": self.lon, "hourly": "precipitation",
                                                "forecast_days": 2, "timezone": TZ}, timeout=10).json()
            s = pd.Series(js["hourly"]["precipitation"], index=pd.to_datetime(js["hourly"]["time"]).tz_localize(TZ))
            self.cache = (time.time(), s)
        s = self.cache[1]
        nxt = s[s.index > as_of].iloc[:6]
        return [float(v) for v in nxt.values] if len(nxt) == 6 else None
