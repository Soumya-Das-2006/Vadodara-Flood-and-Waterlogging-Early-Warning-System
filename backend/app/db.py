"""SQLite storage (lite mode). Swap for TimescaleDB later: only this file touches SQL."""
import json
import sqlite3
import threading
import pandas as pd
from src.config import TZ

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings(station TEXT, ts TEXT, level_cm REAL, rain_mmh REAL, battery_v REAL,
  PRIMARY KEY(station, ts) ON CONFLICT REPLACE);
CREATE TABLE IF NOT EXISTS forecasts(station TEXT, as_of TEXT, created REAL, body TEXT);
CREATE INDEX IF NOT EXISTS fc_idx ON forecasts(station, created);
CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, created REAL, station TEXT,
  from_status TEXT, to_status TEXT, facts TEXT, pred TEXT, texts TEXT, source TEXT, errors TEXT, notify TEXT);
CREATE TABLE IF NOT EXISTS state(station TEXT PRIMARY KEY, status TEXT, streak INTEGER, last_ts TEXT);
"""


def iso(ts):
    t = pd.Timestamp(ts)
    t = t.tz_localize(TZ) if t.tzinfo is None else t.tz_convert(TZ)
    return t.isoformat()


class DB:
    def __init__(self, path=":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            if path != ":memory:":
                self.conn.execute("PRAGMA journal_mode=WAL")

    def insert_reading(self, station, ts, level, rain, battery=None):
        with self.lock:
            self.conn.execute("INSERT INTO readings VALUES(?,?,?,?,?)", (station, iso(ts), level, rain, battery))
            self.conn.commit()

    def readings(self, station, limit=500, since=None):
        q, a = "SELECT * FROM readings WHERE station=?", [station]
        if since:
            q += " AND ts>=?"; a.append(iso(since))
        q += " ORDER BY ts DESC LIMIT ?"; a.append(limit)
        with self.lock:
            rows = [dict(r) for r in self.conn.execute(q, a)]
        return rows[::-1]

    def latest_ts(self, station=None):
        with self.lock:
            r = self.conn.execute("SELECT MAX(ts) m FROM readings" + (" WHERE station=?" if station else ""),
                                  (station,) if station else ()).fetchone()
        return r["m"]

    def hourly(self, station, hours=240):
        """Hourly frame (timestamp IST, level_cm, rain_mmh) bucketed (t-1h, t]; last `hours` buckets."""
        last = self.latest_ts(station)
        if not last:
            return pd.DataFrame(columns=["timestamp", "level_cm", "rain_mmh"])
        since = pd.Timestamp(last) - pd.Timedelta(hours=hours)
        rows = self.readings(station, limit=hours * 70, since=since)
        d = pd.DataFrame(rows)
        d["ts"] = pd.to_datetime(d["ts"], utc=True).dt.tz_convert(TZ)
        g = d.set_index("ts")
        h = pd.DataFrame({"level_cm": g["level_cm"].resample("1h", label="right", closed="right").median(),
                          "rain_mmh": g["rain_mmh"].resample("1h", label="right", closed="right").mean()})
        h = h.ffill(limit=3)
        return h.reset_index().rename(columns={"ts": "timestamp"})

    def get_state(self, station):
        with self.lock:
            r = self.conn.execute("SELECT * FROM state WHERE station=?", (station,)).fetchone()
        return dict(r) if r else {"station": station, "status": "safe", "streak": 0, "last_ts": None}

    def set_state(self, station, status, streak, last_ts):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO state VALUES(?,?,?,?)", (station, status, streak, last_ts))
            self.conn.commit()

    def save_forecast(self, station, as_of, body, created):
        with self.lock:
            self.conn.execute("INSERT INTO forecasts VALUES(?,?,?,?)", (station, as_of, created, json.dumps(body)))
            self.conn.execute("DELETE FROM forecasts WHERE station=? AND created < "
                              "(SELECT MIN(created) FROM (SELECT created FROM forecasts WHERE station=? ORDER BY created DESC LIMIT 500))",
                              (station, station))
            self.conn.commit()

    def latest_forecast(self, station):
        with self.lock:
            r = self.conn.execute("SELECT body FROM forecasts WHERE station=? ORDER BY created DESC LIMIT 1", (station,)).fetchone()
        return json.loads(r["body"]) if r else None

    def forecast_history(self, station, limit=200):
        with self.lock:
            rows = self.conn.execute("SELECT body FROM forecasts WHERE station=? ORDER BY created DESC LIMIT ?", (station, limit)).fetchall()
        return [json.loads(r["body"]) for r in rows][::-1]

    def add_alert(self, **k):
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO alerts(ts,created,station,from_status,to_status,facts,pred,texts,source,errors,notify) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (k["ts"], k["created"], k["station"], k["from_status"], k["to_status"], json.dumps(k["facts"], ensure_ascii=False),
                 json.dumps(k["pred"], ensure_ascii=False), json.dumps(k["texts"], ensure_ascii=False), k["source"],
                 json.dumps(k["errors"], ensure_ascii=False), k.get("notify", "")))
            self.conn.commit()
            return cur.lastrowid

    def update_alert_notify(self, alert_id, notify):
        with self.lock:
            self.conn.execute("UPDATE alerts SET notify=? WHERE id=?", (notify, alert_id)); self.conn.commit()

    def alerts(self, limit=50, station=None):
        q, a = "SELECT * FROM alerts", []
        if station:
            q += " WHERE station=?"; a.append(station)
        q += " ORDER BY id DESC LIMIT ?"; a.append(limit)
        with self.lock:
            rows = [dict(r) for r in self.conn.execute(q, a)]
        for r in rows:
            for c in ("facts", "pred", "texts", "errors"):
                r[c] = json.loads(r[c])
        return rows
