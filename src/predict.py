"""Live prediction from recent hourly data of ONE station."""
import numpy as np
import pandas as pd
import joblib
from .config import MODELS, HORIZONS, load_config, station_ids
from .features import build_features, FEATURES, FC_COLS


class FloodPredictor:
    def __init__(self, model_path=None, cfg=None):
        self.cfg = cfg or load_config()
        self.m = joblib.load(model_path or MODELS / "flood_model.joblib")
        self.ids = station_ids(self.cfg)
        self.threshold = float(self.cfg["settings"].get("warning_prob_threshold") or self.m["threshold"])
        self.meta = self.m["meta"]

    def predict(self, station, hourly, forecast_rain=None):
        """hourly: DataFrame timestamp(IST), level_cm, rain_mmh (>= 80 rows advisable), hourly, sorted.
        forecast_rain: optional list of up to 6 floats = rain mm/h expected for next 1..6 h.
        Returns dict (see keys below)."""
        danger = float(self.cfg["stations"][station]["danger_cm"])
        df = hourly.copy()
        if forecast_rain is not None:
            fc = list(forecast_rain)[:6] + [np.nan] * (6 - len(forecast_rain))
            for k, c in enumerate(FC_COLS):
                df[c] = np.nan
                df.loc[df.index[-1], c] = fc[k]
        f = build_features(df, danger, self.ids.index(station), with_targets=False)
        x = f[self.m["features"]].iloc[[-1]]
        cur = float(x["level_cm"].iloc[0])
        out = {"station": station, "as_of": str(f["timestamp"].iloc[-1]), "current_cm": cur, "danger_cm": danger,
               "forecast_cm": {}, "interval_cm": {}}
        for h in HORIZONS:
            out["forecast_cm"][f"{h}h"] = max(0.0, float(self.m["regressors"][h].predict(x)[0]))
            q = {a: max(0.0, float(self.m["quantile"][h][a].predict(x)[0])) for a in (0.1, 0.5, 0.9)}
            lo, hi = min(q[0.1], q[0.5]), max(q[0.9], q[0.5])
            out["interval_cm"][f"{h}h"] = {"p10": lo, "p50": q[0.5], "p90": hi}
        raw = float(self.m["classifier"].predict_proba(x)[0, 1])
        prob = float(self.m["calibrator"].predict([raw])[0])
        out["flood_probability"] = round(prob, 4)
        # ETA: first horizon where the median forecast reaches the danger level (minutes)
        out["eta_minutes"] = next((60 * h for h in HORIZONS if out["forecast_cm"][f"{h}h"] >= danger), None)
        if cur >= danger:
            status = "danger"
        elif prob >= self.threshold or cur >= 0.85 * danger:   # rule: very high water is never "safe"
            status = "warning"
        else:
            status = "safe"
        out["status"] = status
        out["threshold"] = self.threshold
        out["features_missing"] = [c for c in FEATURES if pd.isna(x[c].iloc[0])]
        return out
