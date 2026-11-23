"""Fill the RESULTS blocks of README.md and docs/REPORT.md from models/metrics.json (exact measured values)."""
import json, re
import pandas as pd
from .config import ROOT, MODELS, DATA_PROC


def f(x, n=3):
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.{n}f}"


def results_md():
    m = json.loads((MODELS / "metrics.json").read_text())
    meta, card = m["meta"], json.loads((DATA_PROC / "dataset_card.json").read_text())
    L = [f"_Data mode: **{card['mode']}**. Water levels are **SYNTHETIC-TARGET**. Test = {meta['splits']['test'][0][:10]} to "
         f"{meta['splits']['test'][1][:10]} ({meta['rows']['test']:,} rows, never used for tuning). Measured {meta['trained_at']}._\n"]
    L.append("**Water level forecast, TEST (cm; lower is better)**\n")
    L.append("| Model | MAE +1h | MAE +2h | MAE +3h | RMSE +3h | R2 +3h |\n|---|---|---|---|---|---|")
    for k, v in m["regression_test"].items():
        L.append(f"| {k} | {f(v['1h']['mae'],2)} | {f(v['2h']['mae'],2)} | {f(v['3h']['mae'],2)} | {f(v['3h']['rmse'],2)} | {f(v['3h']['r2'])} |")
    L.append("\n**Flood within 3 h, TEST** (hourly metrics on rows not yet above danger; events = flood episodes, lead time over detected events)\n")
    L.append("| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier | False-alarm ratio | Events detected | Mean lead (min) | False-alarm episodes |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in m["flood_test"].items():
        L.append(f"| {k} | {f(v['precision'])} | {f(v['recall'])} | {f(v['f1'])} | {f(v['pr_auc'])} | {f(v['roc_auc'])} | {f(v['brier'],5) if v['brier'] is not None else '-'} | "
                 f"{f(v['false_alarm_ratio'])} | {v['detected_events']}/{v['events']} | {v['lead_mean_min']:.0f} | {v['false_alarm_episodes']}/{v['alarm_episodes']} |")
    L.append("\n**Prediction interval (quantile LightGBM, nominal 80% = p10-p90), TEST**\n")
    L.append("| Horizon | Coverage | Mean width (cm) | Median MAE (cm) |\n|---|---|---|---|")
    for h, v in m["quantile_test"].items():
        L.append(f"| +{h} | {f(v['coverage_10_90'])} | {f(v['mean_width_cm'],2)} | {f(v['median_mae'],2)} |")
    if "walk_forward_cv_3h" in m:
        L.append("\n**Walk-forward CV (train period only, +3 h MAE, cm)**\n")
        L.append("| Fold | Train rows | LightGBM | Persistence |\n|---|---|---|---|")
        for c in m["walk_forward_cv_3h"]:
            L.append(f"| {c['fold']} | {c['train_rows']:,} | {f(c['lgbm_mae_3h'],2)} | {f(c['persistence_mae_3h'],2)} |")
    L.append(f"\nDeployed regressor: **{meta['deployed_regressor']}** (best mean validation MAE). Deployed classifier: **{meta['deployed_classifier']}**, "
             f"isotonic-calibrated, threshold **{meta['threshold']:.3f}** (max F2 on validation PR curve). GRU: {meta['gru']}.")
    return "\n".join(L)


def fill(path, block):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    s = re.sub(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)", lambda m: m.group(1) + "\n" + block + "\n" + m.group(2), s, flags=re.S)
    p.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    b = results_md()
    for p in ("README.md", "docs/REPORT.md"):
        if (ROOT / p).exists():
            fill(p, b)
    print(b)
