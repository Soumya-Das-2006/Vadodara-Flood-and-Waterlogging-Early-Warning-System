"""Metrics: level regression, flood classification, event lead time, false alarms."""
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, precision_score,
                             recall_score, f1_score, average_precision_score, roc_auc_score,
                             brier_score_loss, confusion_matrix)


def reg_metrics(y, p):
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p))}


def event_stats(df, flag):
    """df: timestamp, station, level_cm, danger_cm (aligned with flag). Hourly grid per station.
    Event = first hour of a flood episode (level >= danger after >=6 h below).
    Lead time (minutes) = earliest warning hour among the 3 h before the event."""
    d = df[["timestamp", "station", "level_cm", "danger_cm"]].copy()
    d["flag"] = np.asarray(flag).astype(bool)
    leads, n_events, n_alarm, n_false = [], 0, 0, 0
    for _, g in d.groupby("station"):
        g = g.set_index("timestamp").sort_index()
        full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        above = (full["level_cm"] >= full["danger_cm"].ffill()).fillna(False).values
        fl = full["flag"].fillna(False).astype(bool).values
        prev = pd.Series(above).shift(1, fill_value=False).rolling(6, min_periods=1).max().astype(bool).values
        ev = np.where(above & ~prev)[0]
        for i in ev:
            n_events += 1
            lead = 0
            for j in (3, 2, 1):
                if i - j >= 0 and fl[i - j]:
                    lead = 60 * j
                    break
            leads.append(lead)
        # alarm episodes (runs of flag==True)
        starts = np.where(fl & ~np.r_[False, fl[:-1]])[0]
        for s in starts:
            e = s
            while e + 1 < len(fl) and fl[e + 1]:
                e += 1
            n_alarm += 1
            if not above[s:min(len(above), e + 4)].any():
                n_false += 1
    leads = np.array(leads) if leads else np.array([0])
    det = leads[leads > 0]
    return {"events": int(n_events), "detected_events": int(len(det)) if n_events else 0,
            "event_recall": float(len(det) / n_events) if n_events else float("nan"),
            "lead_mean_min": float(det.mean()) if len(det) else 0.0,
            "lead_median_min": float(np.median(det)) if len(det) else 0.0,
            "alarm_episodes": int(n_alarm), "false_alarm_episodes": int(n_false),
            "false_alarm_episode_rate": float(n_false / n_alarm) if n_alarm else float("nan")}


def flood_metrics(df, y, flag, score=None, prob=None, mask=None):
    """y: true 0/1. flag: predicted 0/1. score: any ranking score (PR/ROC AUC). prob: calibrated prob (Brier).
    mask: rows used for the hourly metrics (we use rows NOT yet above danger: a warning is only useful before
    the flood). Event/lead-time stats always use all rows (they need the continuous hourly series)."""
    y = np.asarray(y).astype(int); flag_all = np.asarray(flag).astype(int)
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask).astype(bool)
    score_all = score
    y, flag = y[m], flag_all[m]
    score = None if score is None else np.asarray(score)[m]
    prob = None if prob is None else np.asarray(prob)[m]
    tn, fp, fn, tp = confusion_matrix(y, flag, labels=[0, 1]).ravel()
    out = {"precision": float(precision_score(y, flag, zero_division=0)),
           "recall": float(recall_score(y, flag, zero_division=0)),
           "f1": float(f1_score(y, flag, zero_division=0)),
           "false_alarm_ratio": float(fp / (fp + tp)) if (fp + tp) else float("nan"),   # FP/(FP+TP)
           "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
           "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
           "positives": int(y.sum())}
    s = score if score is not None else flag
    out["pr_auc"] = float(average_precision_score(y, s)) if y.sum() else float("nan")
    out["roc_auc"] = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
    out["brier"] = float(brier_score_loss(y, prob)) if prob is not None else None
    out["rows_evaluated"] = int(m.sum())
    out.update(event_stats(df, flag_all))
    return out


def pinball(y, q, alpha):
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))
