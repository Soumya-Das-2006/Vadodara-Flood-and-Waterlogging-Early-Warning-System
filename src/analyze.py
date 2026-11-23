"""Error analysis on the TEST set -> docs/error_analysis.md (where does the model fail?)."""
import numpy as np
import pandas as pd
import joblib
from .config import DATA_PROC, MODELS, DOCS, TZ, load_config
from .features import FEATURES


def main():
    cfg = load_config()
    m = joblib.load(MODELS / "flood_model.joblib")
    df = pd.read_parquet(DATA_PROC / "flood_dataset.parquet")
    te = df[df.timestamp >= pd.Timestamp(cfg["settings"]["splits"]["test_start"], tz=TZ)].reset_index(drop=True)
    X = te[FEATURES]
    te["pred3"] = m["regressors"][3].predict(X)
    te["err3"] = (te["pred3"] - te["level_in_3h"]).abs()
    te["prob"] = m["calibrator"].predict(m["classifier"].predict_proba(X)[:, 1])
    te["flag"] = te["prob"] >= m["threshold"]
    out = ["# Error analysis (TEST set, SYNTHETIC-TARGET)\n"]
    te["rain6"] = pd.cut(te["rain_sum_6h"], [-1, 1, 10, 30, 1e9], labels=["dry (<1 mm/6h)", "light (1-10)", "moderate (10-30)", "heavy (>30)"])
    out += ["## +3 h level MAE by rain in the last 6 h (cm)\n", te.groupby("rain6", observed=True)["err3"].agg(["count", "mean", "max"]).round(2).to_markdown(), "\n"]
    te["lr"] = pd.cut(te["level_ratio"], [-1, .25, .5, 1, 99], labels=["<25% of danger", "25-50%", "50-100%", ">=100%"])
    out += ["## +3 h level MAE by current level (cm)\n", te.groupby("lr", observed=True)["err3"].agg(["count", "mean", "max"]).round(2).to_markdown(), "\n"]
    te["fc"] = np.where(te["fc_rain_next_6h"].isna(), "no forecast", "forecast available")
    out += ["## Effect of the rain forecast input\n", te.groupby("fc")["err3"].agg(["count", "mean"]).round(2).to_markdown(), "\n"]
    rows = []
    for st, g in te.groupby("station"):
        g = g.set_index("timestamp").sort_index()
        full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        above = (full.level_cm >= full.danger_cm.ffill()).fillna(False).values
        prev = pd.Series(above).shift(1, fill_value=False).rolling(6, min_periods=1).max().astype(bool).values
        fl = full["flag"].fillna(False).astype(bool).values
        for i in np.where(above & ~prev)[0]:
            lead = next((60 * j for j in (3, 2, 1) if i - j >= 0 and fl[i - j]), 0)
            w = full.iloc[max(0, i - 3):i]
            rows.append({"station": st, "event_hour": str(full.index[i])[:16],
                         "peak_rain_mmh_6h_before": round(float(full["rain_mmh"].iloc[max(0, i - 6):i].max()), 1),
                         "level_3h_before_cm": round(float(full["level_cm"].iloc[max(0, i - 3)]), 1),
                         "max_prob_3h_before": round(float(w["prob"].max()), 3) if len(w) else None,
                         "lead_min": lead, "detected": lead > 0})
    ev = pd.DataFrame(rows)
    out += ["## Flood events in TEST: detection and lead time\n", ev.to_markdown(index=False), "\n"]
    out.append(f"Missed events: {int((~ev.detected).sum())} of {len(ev)}.\n")
    out.append("Caveat: a single test year holds only a handful of events, so every number has wide uncertainty. "
               "Hourly resolution also limits how early a very fast rise can be seen.\n")
    (DOCS / "error_analysis.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
