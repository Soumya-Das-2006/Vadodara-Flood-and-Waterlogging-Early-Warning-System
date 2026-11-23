"""Step 3: train + compare models on a TIME split, save best deployable model to models/.

Split (never shuffled): train < val_start <= validation < test_start <= test (final, most recent).
Hyper-parameters are tuned with Optuna on VALIDATION only. Test set is touched once for reporting.
Usage: python -m src.train [--quick] [--skip-gru] [--trials 10]
"""
import argparse, json, time, warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from .config import (DATA_PROC, MODELS, FIGS, DOCS, HORIZONS, SEED, TZ, load_config)
from .features import FEATURES
from .evalx import reg_metrics, flood_metrics, pinball
from .models_util import Col

warnings.filterwarnings("ignore")
TGT = [f"level_in_{h}h" for h in HORIZONS]
QUANTILES = (0.1, 0.5, 0.9)
LGB_BASE = dict(n_estimators=1000, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=SEED, n_jobs=-1, verbose=-1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def split_data(df, cfg):
    vs = pd.Timestamp(cfg["settings"]["splits"]["val_start"], tz=TZ)
    ts = pd.Timestamp(cfg["settings"]["splits"]["test_start"], tz=TZ)
    tr, va, te = df[df.timestamp < vs], df[(df.timestamp >= vs) & (df.timestamp < ts)], df[df.timestamp >= ts]
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def baselines(d):
    pers = {h: d["level_cm"].values for h in HORIZONS}
    trend = {h: np.clip(d["level_cm"].values + h * d["level_change_3h"].values / 3.0, 0, None) for h in HORIZONS}
    return {"Persistence": pers, "LinearTrend": trend}


def flag_from_levels(d, preds):
    mx = np.max(np.stack([preds[h] for h in HORIZONS]), axis=0)
    score = mx / d["danger_cm"].values
    return (score >= 1.0), score


def tune_reg(Xtr, ytr, Xva, yva, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def obj(tr):
        p = dict(LGB_BASE, num_leaves=tr.suggest_int("num_leaves", 15, 127),
                 learning_rate=tr.suggest_float("learning_rate", 0.03, 0.15, log=True),
                 min_child_samples=tr.suggest_int("min_child_samples", 20, 200),
                 colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0),
                 subsample=tr.suggest_float("subsample", 0.5, 1.0),
                 reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10, log=True), n_estimators=600)
        m = lgb.LGBMRegressor(**p).fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l1",
                                       callbacks=[lgb.early_stopping(40, verbose=False)])
        tr.set_user_attr("best_iter", int(m.best_iteration_ or p["n_estimators"]))
        return float(np.mean(np.abs(m.predict(Xva) - yva)))
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=n_trials)
    p = dict(LGB_BASE, **st.best_params)
    p["n_estimators"] = int(st.best_trial.user_attrs["best_iter"])
    return p, float(st.best_value)


def tune_clf(Xtr, ytr, Xva, yva, n_trials):
    import optuna
    from sklearn.metrics import average_precision_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def obj(tr):
        p = dict(LGB_BASE, num_leaves=tr.suggest_int("num_leaves", 7, 63),
                 learning_rate=tr.suggest_float("learning_rate", 0.03, 0.15, log=True),
                 min_child_samples=tr.suggest_int("min_child_samples", 20, 200),
                 colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0),
                 scale_pos_weight=tr.suggest_float("scale_pos_weight", 1, 60, log=True),
                 reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10, log=True), n_estimators=500)
        m = lgb.LGBMClassifier(**p).fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
                                        callbacks=[lgb.early_stopping(40, verbose=False)])
        tr.set_user_attr("best_iter", int(m.best_iteration_ or p["n_estimators"]))
        return float(average_precision_score(yva, m.predict_proba(Xva)[:, 1]))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=n_trials)
    p = dict(LGB_BASE, **st.best_params)
    p["n_estimators"] = int(st.best_trial.user_attrs["best_iter"])
    return p, float(st.best_value)


def choose_threshold(y, prob, beta=2.0):
    """Max F-beta (recall-favoured) on the precision-recall curve of VALIDATION data."""
    pr, rc, th = precision_recall_curve(y, prob)
    pr, rc = pr[:-1], rc[:-1]
    f = (1 + beta ** 2) * pr * rc / np.maximum(beta ** 2 * pr + rc, 1e-9)
    i = int(np.nanargmax(f))
    return float(th[i]), {"precision": float(pr[i]), "recall": float(rc[i]), f"f{beta:g}": float(f[i])}


def run(data_path=None, out_dir=None, quick=False, skip_gru=False, n_trials=10, figs=True, df=None):
    cfg = load_config()
    out_dir = out_dir or MODELS
    out_dir.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.read_parquet(data_path or DATA_PROC / "flood_dataset.parquet")
    df = df.sort_values(["timestamp", "station"]).reset_index(drop=True)
    tr, va, te = split_data(df, cfg)
    log(f"rows train {len(tr):,} | val {len(va):,} | test {len(te):,}; flood rate "
        f"{tr.flood_in_3h.mean():.4f}/{va.flood_in_3h.mean():.4f}/{te.flood_in_3h.mean():.4f}")
    if quick:
        tr = tr.iloc[::6].reset_index(drop=True)
    Xtr, Xva, Xte = tr[FEATURES], va[FEATURES], te[FEATURES]
    R = {"regression_test": {}, "regression_val": {}, "flood_test": {}, "flood_val": {}}
    score_store = SCORES = {}     # model -> (flag, score) on test for curves

    def add_reg(name, dtest, dval=None):
        R["regression_test"][name] = {f"{h}h": reg_metrics(te[f"level_in_{h}h"], dtest[h]) for h in HORIZONS}
        if dval is not None:
            R["regression_val"][name] = {f"{h}h": reg_metrics(va[f"level_in_{h}h"], dval[h]) for h in HORIZONS}

    def add_flood(name, d, flag, score, prob=None, split="test"):
        ref = te if split == "test" else va
        pre = (ref["level_cm"] < ref["danger_cm"]).values
        R[f"flood_{split}"][name] = flood_metrics(d, ref["flood_in_3h"].values, flag, score, prob, mask=pre)
        if split == "test":
            score_store[name] = (flag, score)

    # ---- 1. baselines
    for name in ("Persistence", "LinearTrend"):
        bt, bv = baselines(te)[name], baselines(va)[name]
        add_reg(name, bt, bv)
        for sp, d, b in (("test", te, bt), ("val", va, bv)):
            fl, sc = flag_from_levels(d, b)
            add_flood(name + " (levels->flag)", d, fl, sc, split=sp)
    log("baselines done")

    # ---- 2. LightGBM regression (tuned on validation)
    sub = slice(None, None, 2) if not quick else slice(None)
    if quick:
        p_reg = dict(LGB_BASE, n_estimators=150, num_leaves=31); tune_val = None
    else:
        p_reg, tune_val = tune_reg(Xtr.iloc[sub], tr[TGT[2]].iloc[sub], Xva, va[TGT[2]], n_trials)
    log("LGBM reg params", {k: p_reg[k] for k in ("num_leaves", "learning_rate", "n_estimators", "min_child_samples")}, "val MAE h3", tune_val)
    lgb_reg = {h: lgb.LGBMRegressor(**p_reg).fit(Xtr, tr[f"level_in_{h}h"]) for h in HORIZONS}
    pl_te = {h: np.clip(lgb_reg[h].predict(Xte), 0, None) for h in HORIZONS}
    pl_va = {h: np.clip(lgb_reg[h].predict(Xva), 0, None) for h in HORIZONS}
    add_reg("LightGBM", pl_te, pl_va)
    for sp, d, b in (("test", te, pl_te), ("val", va, pl_va)):
        fl, sc = flag_from_levels(d, b)
        add_flood("LightGBM (levels->flag)", d, fl, sc, split=sp)
    log("LightGBM done")

    # ---- 3. Random Forest regression (multi-output) on a subsample
    rng = np.random.default_rng(SEED)
    n_rf = 30000 if quick else 120000
    idx = rng.choice(len(tr), min(n_rf, len(tr)), replace=False)
    rf = RandomForestRegressor(n_estimators=15 if quick else 40, max_depth=16, min_samples_leaf=5,
                               max_features=0.4, n_jobs=-1, random_state=SEED)
    rf.fit(Xtr.iloc[idx], tr[TGT].iloc[idx])
    prf_te = rf.predict(Xte); prf_va = rf.predict(Xva)
    rt = {h: prf_te[:, i] for i, h in enumerate(HORIZONS)}; rv = {h: prf_va[:, i] for i, h in enumerate(HORIZONS)}
    add_reg("RandomForest", rt, rv)
    for sp, d, b in (("test", te, rt), ("val", va, rv)):
        fl, sc = flag_from_levels(d, b)
        add_flood("RandomForest (levels->flag)", d, fl, sc, split=sp)
    log("RandomForest done")

    # ---- 4. GRU (TensorFlow) - comparison only
    gru_note = "skipped"
    if not skip_gru and not quick:
        try:
            from .gru_model import fit_predict
            gva, gte, _ = fit_predict(tr, va, te, epochs=6)
            mt = te.merge(gte, on=["timestamp", "station"], how="inner")
            mv = va.merge(gva, on=["timestamp", "station"], how="inner")
            R["regression_test"]["GRU (24h window)"] = {f"{h}h": reg_metrics(mt[f"level_in_{h}h"], mt[f"gru_{h}h"]) for h in HORIZONS}
            R["regression_val"]["GRU (24h window)"] = {f"{h}h": reg_metrics(mv[f"level_in_{h}h"], mv[f"gru_{h}h"]) for h in HORIZONS}
            gp = {h: mt[f"gru_{h}h"].values for h in HORIZONS}
            fl, sc = flag_from_levels(mt, gp)
            R["flood_test"]["GRU (levels->flag)"] = flood_metrics(mt, mt["flood_in_3h"].values, fl, sc,
                                                                  mask=(mt["level_cm"] < mt["danger_cm"]).values)
            gru_note = f"trained; evaluated on {len(mt):,}/{len(te):,} test rows (rows with a full 24 h window)"
        except Exception as e:      # keep the pipeline alive without TF
            gru_note = f"skipped: {e}"
    log("GRU:", gru_note)

    # ---- 5. Quantile LightGBM (10/50/90)
    qm, q_pred = {h: {} for h in HORIZONS}, {h: {} for h in HORIZONS}
    pq = dict(p_reg, n_estimators=min(p_reg["n_estimators"], 300))
    for h in HORIZONS:
        for a in QUANTILES:
            m = lgb.LGBMRegressor(**dict(pq, objective="quantile", alpha=a)).fit(Xtr, tr[f"level_in_{h}h"])
            qm[h][a] = m
            q_pred[h][a] = np.clip(m.predict(Xte), 0, None)
        q_pred[h][0.1] = np.minimum(q_pred[h][0.1], q_pred[h][0.5]); q_pred[h][0.9] = np.maximum(q_pred[h][0.9], q_pred[h][0.5])
    R["quantile_test"] = {}
    for h in HORIZONS:
        y = te[f"level_in_{h}h"].values
        lo, md, hi = q_pred[h][0.1], q_pred[h][0.5], q_pred[h][0.9]
        R["quantile_test"][f"{h}h"] = {"coverage_10_90": float(np.mean((y >= lo) & (y <= hi))),
                                       "nominal": 0.8, "mean_width_cm": float(np.mean(hi - lo)),
                                       "median_mae": float(np.mean(np.abs(y - md))),
                                       "pinball_q10": pinball(y, lo, .1), "pinball_q50": pinball(y, md, .5), "pinball_q90": pinball(y, hi, .9)}
    log("quantile done")

    # ---- 6. Classifiers (class imbalance: weights; calibrated; threshold from validation PR curve)
    ytr_all, yva_all = tr["flood_in_3h"].values, va["flood_in_3h"].values
    pre_tr, pre_va = (tr.level_cm < tr.danger_cm).values, (va.level_cm < va.danger_cm).values
    Xtr_c, ytr = Xtr[pre_tr], ytr_all[pre_tr]          # classifier = "will it flood?" for not-yet-flooded states
    Xva_c, yva = Xva[pre_va], yva_all[pre_va]
    if quick:
        p_clf = dict(LGB_BASE, n_estimators=150, num_leaves=15, scale_pos_weight=20.0); clf_val = None
    else:
        p_clf, clf_val = tune_clf(Xtr_c.iloc[sub], ytr[sub], Xva_c, yva, n_trials)
    log("LGBM clf params", {k: p_clf[k] for k in ("num_leaves", "learning_rate", "n_estimators", "scale_pos_weight")}, "val PR-AUC", clf_val)
    clf = lgb.LGBMClassifier(**p_clf).fit(Xtr_c, ytr)
    rf_clf_idx = np.r_[np.where(ytr == 1)[0], rng.choice(np.where(ytr == 0)[0], min(int((ytr == 0).sum()), 20000 if quick else 100000), replace=False)]
    rfc = RandomForestClassifier(n_estimators=20 if quick else 80, max_depth=18, min_samples_leaf=3,
                                 class_weight="balanced_subsample", max_features=0.4, n_jobs=-1, random_state=SEED)
    rfc.fit(Xtr_c.iloc[rf_clf_idx], ytr[rf_clf_idx])
    clf_info = {}
    for name, model in (("LightGBM classifier", clf), ("RandomForest classifier", rfc)):
        raw_va = model.predict_proba(Xva)[:, 1]; raw_te = model.predict_proba(Xte)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_va[pre_va], yva)
        p_va, p_te = iso.predict(raw_va), iso.predict(raw_te)
        thr, thr_info = choose_threshold(yva, p_va[pre_va])
        add_flood(name + " (calibrated)", va, p_va >= thr, p_va, p_va, split="val")
        add_flood(name + " (calibrated)", te, p_te >= thr, p_te, p_te, split="test")
        pre_te = (te.level_cm < te.danger_cm).values
        R["flood_test"][name + " (calibrated)"]["brier_uncalibrated"] = float(np.mean((raw_te[pre_te] - te["flood_in_3h"].values[pre_te]) ** 2))
        R["flood_test"][name + " (calibrated)"]["threshold"] = thr
        clf_info[name] = {"model": model, "iso": iso, "thr": thr, "thr_info": thr_info, "p_te": p_te, "raw_te": raw_te}
        log(name, "threshold", round(thr, 4), thr_info)

    # ---- 7. choose deployed model (validation only)
    val_mae = {n: np.mean([v[f"{h}h"]["mae"] for h in HORIZONS]) for n, v in R["regression_val"].items()
               if n in ("LightGBM", "RandomForest")}
    best_reg = min(val_mae, key=val_mae.get)
    reg_models = ({h: lgb_reg[h] for h in HORIZONS} if best_reg == "LightGBM"
                  else {h: Col(rf, i) for i, h in enumerate(HORIZONS)})
    val_pr = {n: R["flood_val"][n + " (calibrated)"]["pr_auc"] for n in ("LightGBM classifier", "RandomForest classifier")}
    best_clf = max(val_pr, key=val_pr.get)
    ci = clf_info[best_clf]
    meta = {
        "label": "Levels are SYNTHETIC-TARGET. See dataset_card.json for per-column labels.",
        "features": FEATURES, "horizons_h": HORIZONS, "quantiles": list(QUANTILES),
        "threshold": ci["thr"], "threshold_selection": "max F2 on validation PR curve (calibrated probs)",
        "deployed_regressor": best_reg, "deployed_classifier": best_clf,
        "val_mae_mean": val_mae, "val_pr_auc": val_pr,
        "splits": {"train": [str(tr.timestamp.min()), str(tr.timestamp.max())],
                   "val": [str(va.timestamp.min()), str(va.timestamp.max())],
                   "test": [str(te.timestamp.min()), str(te.timestamp.max())]},
        "rows": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "lgbm_reg_params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in p_reg.items()},
        "lgbm_clf_params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in p_clf.items()},
        "optuna": {"trials": 0 if quick else n_trials, "val_mae_h3": tune_val, "val_pr_auc": clf_val},
        "gru": gru_note, "quick_mode": bool(quick), "seed": SEED,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # ---- 8. walk-forward CV (train period only)
    if not quick:
        cv = []
        tss = TimeSeriesSplit(n_splits=4, gap=3 * 72)
        for k, (a, b) in enumerate(tss.split(Xtr)):
            m = lgb.LGBMRegressor(**dict(p_reg, n_estimators=min(300, p_reg["n_estimators"]))).fit(Xtr.iloc[a], tr[TGT[2]].iloc[a])
            y = tr[TGT[2]].iloc[b].values
            cv.append({"fold": k, "train_rows": int(len(a)), "val_rows": int(len(b)),
                       "lgbm_mae_3h": float(np.mean(np.abs(m.predict(Xtr.iloc[b]) - y))),
                       "persistence_mae_3h": float(np.mean(np.abs(tr["level_cm"].iloc[b].values - y)))})
        R["walk_forward_cv_3h"] = cv
        log("CV", cv)
    R["meta"] = meta
    joblib.dump({"regressors": reg_models, "quantile": qm, "classifier": ci["model"], "calibrator": ci["iso"],
                 "threshold": ci["thr"], "features": FEATURES, "horizons": HORIZONS, "meta": meta},
                out_dir / "flood_model.joblib", compress=3)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    (out_dir / "metrics.json").write_text(json.dumps(R, indent=2, default=str))
    log("saved model + metrics to", out_dir)

    if figs and not quick:
        make_figures(te, va, lgb_reg, qm, q_pred, clf, clf_info, best_clf, R, score_store)
    return R


def make_figures(te, va, lgb_reg, qm, q_pred, clf, clf_info, best_clf, R, SCORES):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    from sklearn.calibration import calibration_curve
    FIGS.mkdir(parents=True, exist_ok=True)
    pre = (te.level_cm < te.danger_cm).values
    y = te["flood_in_3h"].values[pre]
    # PR curves (rows not yet above danger)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    curves = {n: sc for n, (fl, sc) in SCORES.items()}
    for n, ci_ in clf_info.items():
        curves[n + " (calibrated)"] = ci_["p_te"]
    for name, sc in curves.items():
        p, r, _ = precision_recall_curve(y, np.asarray(sc)[pre]); ax.plot(r, p, label=name)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Flood within 3 h: precision-recall (TEST, SYNTHETIC-TARGET)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGS / "pr_curves.png", dpi=130); plt.close(fig)
    # reliability
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ci = clf_info[best_clf]
    fp_, mp_ = calibration_curve(y, ci["p_te"][pre], n_bins=10, strategy="quantile")
    fu, mu = calibration_curve(y, ci["raw_te"][pre], n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.plot(mu, fu, "o-", label="raw"); ax.plot(mp_, fp_, "s-", label="isotonic")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency"); ax.legend(); ax.set_title("Reliability (TEST)")
    fig.tight_layout(); fig.savefig(FIGS / "reliability.png", dpi=130); plt.close(fig)
    # SHAP
    try:
        import shap
        tp = te[te.level_cm < te.danger_cm]; samp = tp.sample(min(3000, len(tp)), random_state=SEED)
        ex = shap.TreeExplainer(clf)
        sv = ex.shap_values(samp[FEATURES])
        sv = sv[1] if isinstance(sv, list) else sv
        plt.figure(); shap.summary_plot(sv, samp[FEATURES], show=False, max_display=15)
        plt.tight_layout(); plt.savefig(FIGS / "shap_summary.png", dpi=130, bbox_inches="tight"); plt.close()
        imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
        imp.to_csv(DOCS / "shap_importance.csv")
    except Exception as e:
        log("SHAP failed:", e)
        imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
        imp.to_csv(DOCS / "shap_importance.csv")
    # example test event with interval
    d = te.copy()
    for a in (0.1, 0.5, 0.9):
        d[f"q{int(a*100)}"] = q_pred[3][a]
    ev = d[d.level_cm == d.level_cm.max()].iloc[0]
    g = d[(d.station == ev.station) & (abs(d.timestamp - ev.timestamp) <= pd.Timedelta(hours=40))].sort_values("timestamp")
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    t3 = g.timestamp + pd.Timedelta(hours=3)
    ax[0].plot(g.timestamp, g.level_cm, "k", label="observed level")
    ax[0].plot(t3, g.q50, "C0", label="forecast made 3 h earlier (p50)")
    ax[0].fill_between(t3, g.q10, g.q90, color="C0", alpha=.2, label="p10-p90")
    ax[0].axhline(g.danger_cm.iloc[0], color="r", ls="--", label="danger level")
    ax[0].set_ylabel("level (cm)"); ax[0].legend(fontsize=7); ax[0].set_title(f"{ev.station}: test-set event (SYNTHETIC-TARGET)")
    ax[1].bar(g.timestamp, g.rain_mmh, width=0.03, color="C2"); ax[1].set_ylabel("rain mm/h")
    fig.tight_layout(); fig.savefig(FIGS / "test_event_forecast.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--skip-gru", action="store_true")
    ap.add_argument("--trials", type=int, default=10)
    a = ap.parse_args()
    run(quick=a.quick, skip_gru=a.skip_gru, n_trials=a.trials)


if __name__ == "__main__":
    main()
