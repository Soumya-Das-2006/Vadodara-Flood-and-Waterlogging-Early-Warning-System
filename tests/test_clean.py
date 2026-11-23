import numpy as np
import pandas as pd
from src.clean import clean_station, spike_filter, to_ist

def _raw(n=60):
    ts = pd.date_range("2025-07-01", periods=n, freq="h")            # naive -> IST
    return pd.DataFrame({"timestamp": ts, "station": "akota", "rain_mmh": 1.0, "level_cm": 20.0 + np.arange(n) * 0.1})

def test_timezone_is_ist():
    t = to_ist(pd.Series(["2025-07-01T00:00:00+00:00"]))
    assert str(t.dt.tz) == "Asia/Kolkata" and t.dt.hour.iloc[0] == 5

def test_duplicates_and_spikes_removed():
    df = _raw()
    df = pd.concat([df, df.iloc[[5]]])
    df.loc[df.index[20], "level_cm"] = 350.0                          # sensor spike
    out, rep = clean_station(df)
    assert rep["duplicates_removed"] == 1 and rep["level_spikes_removed"] >= 1
    assert out["level_cm"].max() < 100                                  # spike gone
    assert out["level_cm"].notna().all()                               # short gap bridged by backward fill

def test_spike_filter_is_causal():
    v = np.array([10, 11, 12, 300, 13, 14.0])
    c, n = spike_filter(v)
    assert n == 1 and np.isnan(c[3]) and c[4] == 13
    v2 = v.copy(); v2[5] = 999
    c2, _ = spike_filter(v2)
    assert np.array_equal(c[:5], c2[:5], equal_nan=True)               # later data cannot change the past

def test_long_gap_not_filled():
    df = _raw(); df = df.drop(df.index[20:30])
    out, rep = clean_station(df)
    assert out["level_cm"].isna().sum() == 10 - 3                      # only 3 h bridged
