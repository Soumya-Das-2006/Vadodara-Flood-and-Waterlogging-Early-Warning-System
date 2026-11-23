"""Causal cleaning of raw sensor tables. No future data is used to fix the past
(spike rule and gap filling look backwards only), so it behaves the same live and offline."""
import numpy as np
import pandas as pd
from .config import TZ

MAX_RATE_CM_H = 160.0      # physical limit on level change per hour; bigger jumps = sensor spike
LEVEL_RANGE = (0.0, 400.0)
RAIN_MAX = 200.0           # mm/h
FFILL_LIMIT = 3            # hours of gap that may be bridged (backward-looking only)


def to_ist(ts):
    """Parse timestamps and return tz-aware Asia/Kolkata. Naive input is assumed to be IST."""
    t = pd.to_datetime(ts, utc=False, format="ISO8601") if not hasattr(ts, "dt") else ts
    t = pd.to_datetime(t)
    if t.dt.tz is None:
        return t.dt.tz_localize(TZ)
    return t.dt.tz_convert(TZ)


def spike_filter(values, max_rate=MAX_RATE_CM_H):
    """Mark readings that jump more than max_rate*dt from the last VALID reading as NaN.
    values: 1-D array on an hourly grid. Returns (clean_values, n_spikes)."""
    out = values.astype(float).copy()
    last, gap, n = np.nan, 0, 0
    for i, v in enumerate(out):
        if v != v:
            gap += 1
            continue
        if v < LEVEL_RANGE[0] or v > LEVEL_RANGE[1]:
            out[i] = np.nan; n += 1; gap += 1; continue
        if last == last and abs(v - last) > max_rate * (gap + 1):
            out[i] = np.nan; n += 1; gap += 1; continue
        last, gap = v, 0
    return out, n


def clean_station(df):
    """df: raw rows of ONE station (timestamp, rain_mmh, level_cm, any sub-hourly rate).
    Returns (hourly cleaned df on a full grid, report dict)."""
    rep = {"rows_in": int(len(df))}
    df = df.copy()
    df["timestamp"] = to_ist(df["timestamp"])
    df = df.sort_values("timestamp")
    before = len(df)
    df = df.drop_duplicates("timestamp", keep="last")
    rep["duplicates_removed"] = int(before - len(df))
    # hourly bucket (t-1h, t] labelled t; median level, mean rain rate
    g = df.set_index("timestamp")
    hourly = pd.DataFrame({
        "level_cm": g["level_cm"].resample("1h", label="right", closed="right").median(),
        "rain_mmh": g["rain_mmh"].resample("1h", label="right", closed="right").mean(),
    })
    rep["missing_hours_before"] = int(hourly["level_cm"].isna().sum())
    lv, n_sp = spike_filter(hourly["level_cm"].values)
    hourly["level_cm"] = lv
    rep["level_spikes_removed"] = int(n_sp)
    bad_rain = (hourly["rain_mmh"] < 0) | (hourly["rain_mmh"] > RAIN_MAX)
    rep["rain_outliers_removed"] = int(bad_rain.sum())
    hourly.loc[bad_rain, "rain_mmh"] = np.nan
    hourly = hourly.ffill(limit=FFILL_LIMIT)          # backward-looking only
    rep["missing_hours_after"] = int(hourly["level_cm"].isna().sum())
    hourly = hourly.reset_index().rename(columns={"index": "timestamp"})
    hourly.rename(columns={hourly.columns[0]: "timestamp"}, inplace=True)
    return hourly, rep
