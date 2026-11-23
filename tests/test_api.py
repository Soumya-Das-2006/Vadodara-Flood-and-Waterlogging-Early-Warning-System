import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.app.db import DB
from backend.app.main import create_app
from backend.app.service import FloodService
from tests.test_service import FakePred

H = {"X-API-Key": "secret"}

@pytest.fixture
def client(cfg):
    svc = FloodService(DB(), FakePred(["warning"] * 20), cfg)
    app = create_app(svc, start_background=False, api_key="secret", allow_sim_time=False)
    return TestClient(app), svc

def _reading(ts, **k):
    return {"station": "akota", "level_cm": 12.5, "rain_mmh": 1.0, "battery_v": 3.9, "ts": ts, **k}

def test_health_is_open_others_need_key(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"
    for p in ("/stations", "/alerts", "/readings?station=akota", "/forecast/akota"):
        assert c.get(p).status_code == 401
        assert c.get(p, headers={"X-API-Key": "wrong"}).status_code == 401

def test_model_metrics_normalizes_non_finite_values(client, tmp_path, monkeypatch):
    from backend.app import main

    (tmp_path / "metrics.json").write_text('{"metric": NaN}', encoding="utf-8")
    monkeypatch.setattr(main, "MODELS", tmp_path)
    c, _ = client
    r = c.get("/model/metrics", headers=H)
    assert r.status_code == 200
    assert r.json() == {"metric": None}

def test_ingest_validation(client):
    c, _ = client
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    assert c.post("/ingest", json=_reading(now.isoformat()), headers=H).status_code == 200
    assert c.post("/ingest", json=_reading(now.isoformat(), level_cm=-5), headers=H).status_code == 422
    assert c.post("/ingest", json=_reading(now.isoformat(), level_cm=9999), headers=H).status_code == 422
    assert c.post("/ingest", json=_reading(now.isoformat(), battery_v=9), headers=H).status_code == 422
    assert c.post("/ingest", json=_reading("not a date"), headers=H).status_code == 422
    assert c.post("/ingest", json=_reading(now.isoformat(), station="nowhere"), headers=H).status_code == 422
    assert c.post("/ingest", json=_reading((now + pd.Timedelta(days=2)).isoformat()), headers=H).status_code == 422
    r = c.get("/readings", params={"station": "akota"}, headers=H).json()
    assert len(r) == 1 and r[0]["level_cm"] == 12.5

def test_endpoints_after_data(client):
    c, svc = client
    t0 = pd.Timestamp.now(tz="Asia/Kolkata").floor("h") - pd.Timedelta(hours=130)
    for i in range(130):
        c.post("/ingest", json=_reading((t0 + pd.Timedelta(hours=i)).isoformat()), headers=H)
    assert c.get("/forecast/akota", headers=H).status_code == 404            # nothing evaluated yet
    svc.evaluate_all()
    f = c.get("/forecast/akota", headers=H).json()
    assert f["status"] == "warning" and "interval_cm" in f
    assert c.get("/forecast/nowhere", headers=H).status_code == 404
    st = {s["id"]: s for s in c.get("/stations", headers=H).json()}
    assert st["akota"]["status"] == "warning" and st["alkapuri"]["status"] == "unknown"
    a = c.get("/alerts", headers=H).json()
    assert len(a) == 1 and set(a[0]["texts"]) == {"en", "hi", "gu"}
    ans = c.post("/chat", json={"question": "Is Akota Bridge safe now?"}, headers=H).json()
    assert "Akota" in ans["answer"] and ans["language"] == "en"
    assert c.post("/chat", json={"question": "x"}, headers=H).status_code == 422
    assert c.get("/readings", params={"station": "akota", "limit": 0}, headers=H).status_code == 422
