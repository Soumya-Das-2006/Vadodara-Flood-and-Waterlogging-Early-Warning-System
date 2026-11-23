import numpy as np
import pandas as pd

def _hist(n=120, rain=0.0, level=6.0):
    ts = pd.date_range("2025-07-10", periods=n, freq="h", tz="Asia/Kolkata")
    return pd.DataFrame({"timestamp": ts, "rain_mmh": rain, "level_cm": level})

def test_predict_returns_valid_structure(predictor):
    out = predictor.predict("akota", _hist())
    assert out["status"] == "safe"
    assert 0 <= out["flood_probability"] <= 1
    for h in ("1h", "2h", "3h"):
        i = out["interval_cm"][h]
        assert i["p10"] <= i["p50"] <= i["p90"] and out["forecast_cm"][h] >= 0
    assert out["danger_cm"] == 80

def test_heavy_rain_and_high_level_raise_risk(predictor):
    calm = predictor.predict("akota", _hist())
    h = _hist(); h.loc[h.index[-6:], "rain_mmh"] = [10, 20, 30, 35, 35, 30]
    h.loc[h.index[-6:], "level_cm"] = [10, 25, 45, 62, 72, 78]
    wet = predictor.predict("akota", h, forecast_rain=[30, 30, 25, 20, 10, 5])
    assert wet["flood_probability"] > calm["flood_probability"] + 0.2
    assert wet["status"] in ("warning", "danger")

def test_level_at_danger_is_danger(predictor):
    h = _hist(level=6.0); h.loc[h.index[-1], "level_cm"] = 85
    assert predictor.predict("akota", h)["status"] == "danger"

def test_works_without_forecast_and_short_history(predictor):
    out = predictor.predict("alkapuri", _hist(n=30))        # many NaN features, must not crash
    assert "flood_probability" in out
