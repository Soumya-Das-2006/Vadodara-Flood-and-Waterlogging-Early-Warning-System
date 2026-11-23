"""Step 1: download REAL data for Vadodara. Run on a machine with internet.
    python -m src.fetch_data            # 2000 -> today
Writes data/raw/*.csv and data/raw/SOURCES.md (URL, date, licence, coverage).
"""
import datetime as dt, io, sys, time, pathlib
import pandas as pd, requests

LAT, LON = 22.31, 73.18
RAW = pathlib.Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
TODAY = dt.date.today()
START = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
log = []

def get(url, params, tries=4):
    for i in range(tries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(30 * (i + 1)); continue
        raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
    raise RuntimeError("rate limited")

def hourly_df(js):
    return pd.DataFrame(js["hourly"]).rename(columns={"time": "timestamp"})

def archive():
    url = "https://archive-api.open-meteo.com/v1/archive"
    vars_ = ("precipitation,rain,temperature_2m,relative_humidity_2m,surface_pressure,"
             "wind_speed_10m,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm")
    parts = []
    for y in range(START, TODAY.year + 1):
        end = min(dt.date(y, 12, 31), TODAY - dt.timedelta(days=6))
        if end < dt.date(y, 1, 1): break
        js = get(url, dict(latitude=LAT, longitude=LON, start_date=f"{y}-01-01",
                           end_date=str(end), hourly=vars_, timezone="Asia/Kolkata"))
        parts.append(hourly_df(js)); print("archive", y); time.sleep(2)
    df = pd.concat(parts).drop_duplicates("timestamp")
    df.to_csv(RAW / "openmeteo_archive_hourly.csv", index=False)
    log.append(("Open-Meteo Historical (ERA5/ERA5-Land)", url, "CC BY 4.0",
                f"{df.timestamp.min()} to {df.timestamp.max()}, hourly, {len(df)} rows", "REAL (reanalysis)"))

def flood():
    url = "https://flood-api.open-meteo.com/v1/flood"
    js = get(url, dict(latitude=LAT, longitude=LON, daily="river_discharge",
                       start_date="1984-01-01", end_date=str(TODAY + dt.timedelta(days=7))))
    df = pd.DataFrame(js["daily"]).rename(columns={"time": "date"})
    df.to_csv(RAW / "openmeteo_flood_discharge_daily.csv", index=False)
    log.append(("Open-Meteo Flood (GloFAS discharge)", url, "CC BY 4.0",
                f"{df.date.min()} to {df.date.max()}, daily", "REAL (model-based)"))

def forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    js = get(url, dict(latitude=LAT, longitude=LON, hourly="precipitation,rain",
                       forecast_days=3, timezone="Asia/Kolkata"))
    df = hourly_df(js)
    df.to_csv(RAW / "openmeteo_forecast_72h.csv", index=False)
    log.append(("Open-Meteo Forecast 72 h", url, "CC BY 4.0", f"downloaded {TODAY}", "REAL (forecast)"))

for fn in (archive, flood, forecast):
    try: fn()
    except Exception as e:
        print("FAILED", fn.__name__, e); log.append((fn.__name__, "-", "-", f"FAILED: {e}", "-"))

with open(RAW / "SOURCES.md", "w") as f:
    f.write(f"# Sources (downloaded {TODAY})\n\n| Dataset | URL | Licence | Coverage | Label |\n|---|---|---|---|---|\n")
    for r in log: f.write("| " + " | ".join(r) + " |\n")
print("Done. See data/raw/SOURCES.md")
