"""Step 2: raw -> clean -> features/targets -> data/processed (Parquet + CSV) + dataset_card.json.

Mode is chosen automatically:
  REAL rain mode      : data/raw/openmeteo_archive_hourly.csv exists  (run `make data-real` first)
  SYNTHETIC rain mode : otherwise (loudly labelled SYNTHETIC)
In both modes the station water level is SYNTHETIC-TARGET (no open street/gauge data exists).
"""
import argparse, json, sys
import numpy as np
import pandas as pd
from .config import DATA_RAW, DATA_SYN, DATA_PROC, SEED, TZ, load_config
from .clean import clean_station, to_ist
from .features import build_all
from .synth import synth_rain, make_context, make_raw_levels, add_forecast_columns
from .level_sim import soil_proxy


def load_real_rain():
    p = DATA_RAW / "openmeteo_archive_hourly.csv"
    if not p.exists():
        return None, None
    d = pd.read_csv(p)
    d["timestamp"] = to_ist(d["timestamp"])
    d = d.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    rain = d["precipitation"].astype(float).rename("rain_mmh")
    sm = d.get("soil_moisture_0_to_7cm", pd.Series(np.nan, index=d.index)).astype(float)
    ctx = pd.DataFrame({"timestamp": d.index, "soil_moisture": sm.values})
    q = DATA_RAW / "openmeteo_flood_discharge_daily.csv"
    if q.exists():
        dq = pd.read_csv(q)
        dq["date"] = pd.to_datetime(dq["date"]).dt.tz_localize(TZ)
        qh = dq.set_index("date")["river_discharge"].reindex(d.index, method="ffill").shift(24)  # 1-day lag
        ctx["discharge_upstream"] = qh.values
    else:
        ctx["discharge_upstream"] = np.nan
    return rain, ctx


def make_dataset(rain, ctx, cfg, verbose=True):
    """rain: hourly Series (IST). ctx: DataFrame(timestamp, soil_moisture, discharge_upstream).
    Returns (features+targets DataFrame, per-station cleaning reports, raw sensor table)."""
    raw = make_raw_levels(rain, cfg["stations"], SEED)
    ctx_i = ctx.copy(); ctx_i["timestamp"] = to_ist(ctx_i["timestamp"]); ctx_i = ctx_i.set_index("timestamp")
    frames, reports = {}, {}
    for sid, g in raw.groupby("station"):
        clean, rep = clean_station(g)
        clean = clean.set_index("timestamp").join(ctx_i, how="left").reset_index()
        clean = add_forecast_columns(clean, seed=SEED)
        frames[sid], reports[sid] = clean, rep
        if verbose:
            print(sid, rep)
    data = build_all(frames, cfg)
    need = [c for c in data.columns if c.startswith(("level_in_", "flood_in")) or c in
            ("level_cm", "rain_mmh", "rain_sum_72h", "level_slope_6h")]
    n0 = len(data)
    data = data.dropna(subset=need).reset_index(drop=True)
    if verbose:
        print(f"Rows: {n0:,} -> {len(data):,} after dropping rows with missing required values")
    return data, reports, raw


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-real", action="store_true", help="fail if no real rain file")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31 23:00")
    a = ap.parse_args(argv)
    cfg = load_config()
    DATA_PROC.mkdir(parents=True, exist_ok=True); DATA_SYN.mkdir(parents=True, exist_ok=True)

    rain, ctx = load_real_rain()
    if rain is not None:
        mode = "REAL_RAIN__SYNTHETIC_TARGET"
        print("Using REAL rainfall (Open-Meteo/ERA5). Levels are SYNTHETIC-TARGET.")
    elif a.require_real:
        sys.exit("No real rain file found. Run `make data-real` on a machine with internet.")
    else:
        mode = "SYNTHETIC_RAIN__SYNTHETIC_TARGET"
        print("!" * 70, "\nNo real data found: building FULLY SYNTHETIC dataset (labelled SYNTHETIC).\n" + "!" * 70)
        rain = synth_rain(a.start, a.end, SEED)
        ctx = make_context(rain)
    data, reports, raw = make_dataset(rain, ctx, cfg)
    raw.to_csv(DATA_SYN / "station_levels_raw.csv", index=False)       # raw sensor-like table
    frames = reports
    for c in data.select_dtypes("float64").columns:
        data[c] = data[c].astype("float32")
    data["timestamp"] = data["timestamp"].dt.tz_convert(TZ)
    data.to_parquet(DATA_PROC / "flood_dataset.parquet", index=False)
    data.to_csv(DATA_PROC / "flood_dataset.csv", index=False)
    real = mode.startswith("REAL")
    card = {
        "mode": mode, "seed": SEED, "rows": int(len(data)),
        "time_range": [str(data.timestamp.min()), str(data.timestamp.max())],
        "stations": list(reports), "cleaning_report": reports,
        "labels": {
            "rain_mmh": "REAL (ERA5 reanalysis)" if real else "SYNTHETIC",
            "soil_moisture": "REAL (ERA5-Land)" if real else "DERIVED from SYNTHETIC rain",
            "discharge_upstream": "REAL (GloFAS, 1-day lag)" if real else "DERIVED from SYNTHETIC rain (index units)",
            "level_cm and level_in_*h": "SYNTHETIC-TARGET (physics sim driven by rain + injected sensor faults)",
            "flood_in_3h": "DERIVED from the SYNTHETIC-TARGET level and config danger_cm",
            "fc_rain_*": "DERIVED/SYNTHETIC (noisy copy of future rain; 25% rows masked)",
        },
        "flood_rate": {s: float(data.loc[data.station == s, "flood_in_3h"].mean()) for s in reports},
    }
    (DATA_PROC / "dataset_card.json").write_text(json.dumps(card, indent=2))
    sample = data.groupby("station").head(500)
    sample.to_csv(DATA_PROC / "flood_dataset_sample.csv", index=False)
    print(json.dumps(card["flood_rate"], indent=1))


if __name__ == "__main__":
    main()
