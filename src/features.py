"""Feature engineering + targets for ONE station on an hourly grid.

Row timestamp t = hour label; every feature uses data up to and including t only,
except fc_* columns which are an (external) rain FORECAST for t+1..t+6 h.
"""
import numpy as np
import pandas as pd
from .config import HORIZONS, load_config, station_ids
from .level_sim import soil_proxy, discharge_proxy

RAIN_WINDOWS = (1, 3, 6, 12, 24, 72)
FC_COLS = [f"fc_rain_{k}h" for k in range(1, 7)]

FEATURES = (["level_cm", "rain_mmh", "level_ratio", "danger_cm"]
            + [f"rain_sum_{w}h" for w in RAIN_WINDOWS] + ["rain_max_3h"]
            + ["soil_moisture", "discharge_upstream"]
            + ["level_change_1h", "level_change_3h", "level_change_6h", "level_slope_6h"]
            + ["hours_since_rain_start", "hour", "month", "monsoon"]
            + ["fc_rain_next_1h", "fc_rain_next_3h", "fc_rain_next_6h", "fc_rain_max_6h"]
            + ["station_code"])
TARGETS = [f"level_in_{h}h" for h in HORIZONS] + ["flood_in_3h"]


def _slope(y):
    x = np.arange(len(y)) - (len(y) - 1) / 2.0
    return float((x * (y - y.mean())).sum() / (x ** 2).sum())


def build_features(g, danger_cm, station_code=0, with_targets=True):
    """g: DataFrame sorted hourly with timestamp(IST), level_cm, rain_mmh and optional
    soil_moisture, discharge_upstream, fc_rain_1h..6h. Returns features (+targets), NaN rows kept."""
    g = g.sort_values("timestamp").reset_index(drop=True).copy()
    rain, lvl = g["rain_mmh"], g["level_cm"]
    if "soil_moisture" not in g:                       # live fallback: derive from rain history
        g["soil_moisture"] = soil_proxy(rain.values)
    if "discharge_upstream" not in g:
        g["discharge_upstream"] = discharge_proxy(rain.values)
    g["danger_cm"] = float(danger_cm)
    g["level_ratio"] = lvl / danger_cm
    for w in RAIN_WINDOWS:
        g[f"rain_sum_{w}h"] = rain.rolling(w, min_periods=w).sum()
    g["rain_max_3h"] = rain.rolling(3, min_periods=3).max()
    for w in (1, 3, 6):
        g[f"level_change_{w}h"] = lvl - lvl.shift(w)
    g["level_slope_6h"] = lvl.rolling(6, min_periods=6).apply(_slope, raw=True)
    wet = rain > 0.5
    grp = (~wet).cumsum()
    g["hours_since_rain_start"] = wet.groupby(grp).cumsum().astype(float)
    g["hour"] = g["timestamp"].dt.hour
    g["month"] = g["timestamp"].dt.month
    g["monsoon"] = g["month"].isin([6, 7, 8, 9]).astype(int)
    for c in FC_COLS:
        if c not in g:
            g[c] = np.nan
    fc = g[FC_COLS]
    g["fc_rain_next_1h"] = fc["fc_rain_1h"]
    g["fc_rain_next_3h"] = fc[FC_COLS[:3]].sum(axis=1, min_count=3)
    g["fc_rain_next_6h"] = fc.sum(axis=1, min_count=6)
    g["fc_rain_max_6h"] = fc.max(axis=1, skipna=False)
    g["station_code"] = int(station_code)
    if with_targets:
        for h in HORIZONS:
            g[f"level_in_{h}h"] = lvl.shift(-h)
        fut = pd.concat([lvl.shift(-h) for h in HORIZONS], axis=1).max(axis=1, skipna=False)
        g["flood_in_3h"] = (fut >= danger_cm).astype(float).where(fut.notna())
    return g


def build_all(stations_frames, cfg=None, with_targets=True):
    """stations_frames: {station_id: cleaned hourly df}. Returns stacked feature frame."""
    cfg = cfg or load_config()
    ids = station_ids(cfg)
    out = []
    for sid, g in stations_frames.items():
        f = build_features(g, cfg["stations"][sid]["danger_cm"], ids.index(sid), with_targets)
        f.insert(1, "station", sid)
        out.append(f)
    return pd.concat(out, ignore_index=True).sort_values(["timestamp", "station"]).reset_index(drop=True)
