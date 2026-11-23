"""SYNTHETIC data generators. Everything made here is labelled SYNTHETIC in the dataset card."""
import numpy as np
import pandas as pd
from .config import SEED, TZ
from .level_sim import StationSim, soil_proxy, discharge_proxy

START_P = {5: 0.0016, 6: 0.005, 7: 0.008, 8: 0.008, 9: 0.0056, 10: 0.0016}
MONSOON = (6, 7, 8, 9)


def synth_rain(start="2010-01-01", end="2025-12-31 23:00", seed=SEED):
    """Monsoon-like hourly rainfall (mm/h), IST. SYNTHETIC. Not a copy of any real record."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h", tz=TZ)
    n = len(idx)
    months = idx.month.values
    rain = np.zeros(n)
    active = False
    for t in range(n):
        m = months[t]
        if m in MONSOON:
            if rng.random() < (0.010 if active else 0.008):
                active = not active
        else:
            active = False
        p = START_P.get(m, 0.0008)
        if m in MONSOON:
            p *= 2.2 if active else 0.35
        if rng.random() < p:
            dur = int(min(30, 1 + rng.geometric(1 / 5)))
            peak = min(70.0, rng.lognormal(np.log(6.0), 0.75))
            if m in (7, 8, 9) and rng.random() < 0.05:      # rare extreme storm
                dur = int(rng.integers(12, 36))
                peak = rng.uniform(16, 32)
            shape = np.sin(np.linspace(0.15, np.pi - 0.15, max(dur, 2))) ** 1.2
            shape = shape[:dur]
            jitter = rng.uniform(0.6, 1.3, dur)
            end_i = min(n, t + dur)
            rain[t:end_i] += (peak * shape * jitter)[: end_i - t]
    rain = np.clip(rain, 0, 120).round(2)
    return pd.Series(rain, index=idx, name="rain_mmh")


def make_context(rain, seed=SEED):
    """Context columns for the synthetic mode: soil moisture + discharge (DERIVED from synthetic rain)."""
    ctx = pd.DataFrame({"timestamp": rain.index,
                        "soil_moisture": np.round(soil_proxy(rain.values), 4),
                        "discharge_upstream": np.round(discharge_proxy(rain.values), 4)})
    return ctx


def make_raw_levels(rain, stations_cfg, seed=SEED, inject_faults=True):
    """Simulate raw sensor tables (what an ESP32 fleet would log) from rain. SYNTHETIC-TARGET.

    Faults injected on purpose (so the cleaning step is meaningful): spikes, gaps, duplicate rows.
    """
    rng = np.random.default_rng(seed + 1)
    frames = []
    for name, sc in stations_cfg.items():
        sim = StationSim.from_cfg(sc["sim"])
        lvl = np.empty(len(rain))
        for i, r in enumerate(rain.values):
            lvl[i] = sim.step(r, 1.0)
        meas = np.clip(lvl + rng.normal(0, 0.8, len(lvl)), 0, None)
        df = pd.DataFrame({"timestamp": rain.index, "station": name,
                           "rain_mmh": rain.values, "level_cm": meas.round(2)})
        if inject_faults:
            n = len(df)
            spikes = rng.random(n) < 0.0015
            df.loc[spikes, "level_cm"] += rng.choice([-1, 1], spikes.sum()) * rng.uniform(200, 320, spikes.sum())
            df.loc[df.level_cm < 0, "level_cm"] = 0
            drop = np.zeros(n, bool)
            for _ in range(int(len(df) / 8760 * 4)):            # ~4 gaps per year
                s = int(rng.integers(0, n - 40))
                drop[s:s + int(rng.integers(2, 30))] = True
            df = df[~drop]
            dups = df.sample(frac=0.0005, random_state=seed)
            df = pd.concat([df, dups]).sort_values("timestamp")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return out


def add_forecast_columns(df, rain_col="rain_mmh", seed=SEED, mask_frac=0.25):
    """Simulated rain forecast for t+1..t+6 h (DERIVED/SYNTHETIC: noisy copy of future rain).
    mask_frac of rows get no forecast (NaN) so the model also works when no forecast is available.
    Input df: ONE station, hourly, sorted."""
    rng = np.random.default_rng(seed + 7)
    df = df.copy()
    r = df[rain_col].values
    n = len(df)
    for k in range(1, 7):
        fut = pd.Series(r).shift(-k).values
        noise = rng.lognormal(0, 0.30 + 0.08 * k, n)
        df[f"fc_rain_{k}h"] = np.clip(fut * noise, 0, None).round(2)
    masked = rng.random(n) < mask_frac
    df.loc[masked, [f"fc_rain_{k}h" for k in range(1, 7)]] = np.nan
    return df
