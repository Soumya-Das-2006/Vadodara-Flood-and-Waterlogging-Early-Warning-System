import numpy as np
import pandas as pd
from src.features import build_features, FEATURES, TARGETS

def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-07-01", periods=n, freq="h", tz="Asia/Kolkata")
    return pd.DataFrame({"timestamp": ts, "rain_mmh": rng.gamma(0.5, 4, n).round(2), "level_cm": 5 + rng.random(n) * 30})

def test_all_features_present_and_rolling_sums():
    g = _frame()
    f = build_features(g, danger_cm=80)
    assert all(c in f.columns for c in FEATURES + TARGETS)
    assert np.isclose(f["rain_sum_3h"].iloc[50], g["rain_mmh"].iloc[48:51].sum())
    assert np.isclose(f["level_change_6h"].iloc[50], g["level_cm"].iloc[50] - g["level_cm"].iloc[44])
    assert f["rain_sum_72h"].iloc[:71].isna().all()          # strict windows: no partial sums

def test_targets_align_with_future():
    g = _frame()
    f = build_features(g, danger_cm=30)
    assert np.isclose(f["level_in_2h"].iloc[10], g["level_cm"].iloc[12])
    fut = g["level_cm"].iloc[11:14].max()
    assert f["flood_in_3h"].iloc[10] == float(fut >= 30)
    assert f["flood_in_3h"].iloc[-3:].isna().all()           # unknown future -> NaN, not 0

def test_no_future_leakage():
    """Changing rain/levels AFTER t must not change any feature at t."""
    g = _frame(); f1 = build_features(g, 80, with_targets=False)
    g2 = g.copy(); g2.loc[120:, ["rain_mmh", "level_cm"]] = 999.0
    f2 = build_features(g2, 80, with_targets=False)
    pd.testing.assert_frame_equal(f1.loc[:119, FEATURES], f2.loc[:119, FEATURES])

def test_hours_since_rain_start_and_monsoon():
    g = _frame(); g["rain_mmh"] = 0.0; g.loc[10:14, "rain_mmh"] = 5.0
    f = build_features(g, 80, with_targets=False)
    assert list(f["hours_since_rain_start"].iloc[9:16]) == [0, 1, 2, 3, 4, 5, 0]
    assert (f["monsoon"] == 1).all()                         # July

def test_missing_forecast_gives_nan_not_error():
    f = build_features(_frame(), 80, with_targets=False)
    assert f["fc_rain_next_3h"].isna().all()
