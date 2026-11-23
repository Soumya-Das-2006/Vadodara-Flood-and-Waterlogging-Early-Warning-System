"""FastAPI app: MQTT ingest + REST API. Run: uvicorn backend.app.main:app"""
import hmac
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from src import rag
from src.config import MODELS, TZ, load_config
from .db import DB
from .service import FloodService, OpenMeteoForecast
from .telegram import TelegramNotifier

log = logging.getLogger("flood")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Reading(BaseModel):
    station: str
    level_cm: float = Field(ge=0, le=400)
    rain_mmh: float = Field(default=0.0, ge=0, le=300)
    battery_v: Optional[float] = Field(default=None, ge=2.5, le=4.6)
    ts: datetime

    @field_validator("ts")
    @classmethod
    def _tz(cls, v):
        return pd.Timestamp(v).tz_localize(TZ) if v.tzinfo is None else pd.Timestamp(v).tz_convert(TZ)


class ChatIn(BaseModel):
    question: str = Field(min_length=2, max_length=500)


def build_service(db_path=None, model_path=None, predictor=None, llm=None, notifier="auto"):
    cfg = load_config()
    db = DB(db_path or os.environ.get("DB_PATH", "data/flood.db"))
    if predictor is None:
        from src.predict import FloodPredictor
        predictor = FloodPredictor(model_path or os.environ.get("MODEL_PATH") or MODELS / "flood_model.joblib", cfg)
    if notifier == "auto":
        notifier = TelegramNotifier(min_interval=cfg["settings"]["telegram_min_interval_seconds"])
    fp = None
    if os.environ.get("FORECAST_SOURCE", "mqtt") == "openmeteo":
        s0 = next(iter(cfg["stations"].values()))
        fp = OpenMeteoForecast(s0["lat"], s0["lon"])
    return FloodService(db, predictor, cfg, llm=llm, notifier=notifier, forecast_provider=fp)


def create_app(service=None, start_background=True, api_key=None, allow_sim_time=None):
    key = api_key or os.environ.get("API_KEY", "dev-key")
    sim_ok = allow_sim_time if allow_sim_time is not None else os.environ.get("ALLOW_SIM_TIME", "0") == "1"
    state = {"mqtt": False}

    @asynccontextmanager
    async def lifespan(app):
        if key == "dev-key":
            log.warning("API_KEY is the insecure default 'dev-key'. Set API_KEY in .env for real use.")
        if start_background:
            threading.Thread(target=_eval_loop, daemon=True).start()
            _start_mqtt()
        yield

    app = FastAPI(title="Vadodara Flood Early Warning API", version="1.0", lifespan=lifespan)
    app.state.service = service or build_service()
    svc: FloodService = app.state.service

    def auth(x_api_key: str = Header(default="")):
        if not hmac.compare_digest(x_api_key.encode(), key.encode()):
            raise HTTPException(401, "invalid or missing X-API-Key")

    def _eval_loop():
        every = float(os.environ.get("EVAL_INTERVAL_S", svc.s["eval_interval_seconds"]))
        while True:
            try:
                svc.evaluate_all()
            except Exception:
                log.exception("eval loop")
            time.sleep(every)

    def _handle(topic, payload):
        try:
            d = json.loads(payload)
            if topic.endswith("/forecast"):
                svc.set_forecast(d["ts"], d["rain_next_6h"])
                return
            r = Reading(**d)
            _accept(r)
        except Exception as e:
            log.warning("bad MQTT message on %s: %s", topic, e)

    def _start_mqtt():
        host = os.environ.get("MQTT_HOST")
        if not host:
            return
        import paho.mqtt.client as mqtt
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="flood-api")
        user = os.environ.get("MQTT_USER")
        if user:
            c.username_pw_set(user, os.environ.get("MQTT_PASSWORD", ""))

        def on_connect(cl, ud, flags, rc, props=None):
            state["mqtt"] = True
            cl.subscribe([("flood/+/telemetry", 1), ("flood/forecast", 1)])
            log.info("MQTT connected")

        def on_disconnect(cl, ud, flags, rc, props=None):
            state["mqtt"] = False

        c.on_connect, c.on_disconnect = on_connect, on_disconnect
        c.on_message = lambda cl, ud, m: _handle(m.topic, m.payload)
        c.reconnect_delay_set(1, 30)
        c.connect_async(host, int(os.environ.get("MQTT_PORT", 1883)), 30)
        c.loop_start()

    def _accept(r: Reading):
        if r.station not in svc.stations:
            raise ValueError(f"unknown station {r.station}")
        now = pd.Timestamp.now(tz=TZ)
        if not sim_ok and r.ts > now + pd.Timedelta(minutes=10):
            raise ValueError("timestamp is in the future")
        svc.ingest({"station": r.station, "level_cm": r.level_cm, "rain_mmh": r.rain_mmh,
                    "battery_v": r.battery_v, "ts": r.ts})

    @app.get("/health")
    def health():
        ok_db = True
        try:
            svc.db.latest_ts()
        except Exception:
            ok_db = False
        return {"status": "ok" if ok_db else "degraded", "db": ok_db, "model_loaded": svc.p is not None,
                "mqtt_connected": state["mqtt"], "stations": len(svc.stations), "last_reading": svc.db.latest_ts()}

    @app.get("/stations", dependencies=[Depends(auth)])
    def stations():
        return svc.snapshot()

    @app.get("/readings", dependencies=[Depends(auth)])
    def readings(station: str, limit: int = Query(200, ge=1, le=5000), since: Optional[datetime] = None):
        if station not in svc.stations:
            raise HTTPException(404, "unknown station")
        return svc.db.readings(station, limit, since)

    @app.get("/forecast/{station}", dependencies=[Depends(auth)])
    def forecast(station: str, history: int = Query(0, ge=0, le=500)):
        if station not in svc.stations:
            raise HTTPException(404, "unknown station")
        fc = svc.db.latest_forecast(station)
        if fc is None:
            raise HTTPException(404, "no forecast yet (not enough history)")
        fc["status"] = svc.db.get_state(station)["status"]
        if history:
            fc["history"] = svc.db.forecast_history(station, history)
        return fc

    @app.get("/alerts", dependencies=[Depends(auth)])
    def alerts(limit: int = Query(50, ge=1, le=500), station: Optional[str] = None):
        return svc.db.alerts(limit, station)

    @app.post("/ingest", dependencies=[Depends(auth)])
    def ingest(r: Reading):
        try:
            _accept(r)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"ok": True}

    @app.post("/forecast-input", dependencies=[Depends(auth)])
    def forecast_input(body: dict):
        svc.set_forecast(body["ts"], body["rain_next_6h"])
        return {"ok": True}

    @app.post("/chat", dependencies=[Depends(auth)])
    def chat(body: ChatIn):
        r = rag.answer(body.question, svc.db, svc.cfg)
        return {"answer": r["answer"], "source": r["source"], "language": r["language"]}

    @app.get("/model/metrics", dependencies=[Depends(auth)])
    def model_metrics():
        p = MODELS / "metrics.json"
        if not p.exists():
            raise HTTPException(404, "no metrics.json - run `make train`")
        return json.loads(p.read_text(), parse_constant=lambda _: None)

    return app


_app = None


def __getattr__(name):          # lazy: `uvicorn backend.app.main:app` builds the app on first access
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
