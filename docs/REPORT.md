# Vadodara Flood and Waterlogging Early Warning System: IoT, Machine Learning and Generative-AI Alerts

## Abstract
Urban flooding in Vadodara develops within hours when intense rain meets a high Vishwamitri river, as in August 2019 and August 2024. This project builds an end-to-end early warning prototype: low-cost ESP32 sensors (ultrasonic water level and tipping-bucket rain gauge) publish readings over MQTT; a backend turns them into hourly features and forecasts water level 1, 2 and 3 hours ahead with prediction intervals and a calibrated probability that the danger level will be crossed within 3 hours; when the status changes, a large language model writes a short alert in English, Hindi and Gujarati from validated facts only; alerts reach a dashboard and a Telegram bot. **Important:** the sandbox used to build the prototype could not download real rainfall or gauge data, and no open street-level gauge data exists for the chosen sites. All reported scores therefore use a SYNTHETIC dataset (synthetic rainfall and a physics-based synthetic water level). They demonstrate a working, leakage-free pipeline, not real-world accuracy.

## 1. Introduction
Warnings are useful only if they are early, location-specific, and understandable by residents. The aim was a system that (i) runs on cheap hardware, (ii) gives 1-3 h of lead time, (iii) explains itself in the local languages, and (iv) can be reproduced by a student on a laptop.

## 2. Data
| Data | Label | Note |
|---|---|---|
| Hourly rainfall 2010-2025 | SYNTHETIC | storm generator with monsoon seasonality and rare extreme storms; ~1100 mm/yr |
| Soil moisture, upstream discharge | DERIVED | linear-reservoir equations driven by the synthetic rain |
| Water level at Akota Bridge, Alkapuri Underpass, Kamnath Bridge | SYNTHETIC-TARGET | soil-dependent runoff -> fast drain response + slow river component; injected sensor spikes, gaps, duplicates |
| Rain forecast for next 1-6 h | DERIVED | future rain with log-normal noise; 25% of rows have no forecast |
| Historical flood events (2019, 2024) | REAL (news) | `data/raw/flood_events.csv`, with URLs |

Real sources (Open-Meteo ERA5, GloFAS, Open-Meteo forecast) are implemented in `src/fetch_data.py` and switch on automatically when the files exist, but could not be downloaded in the build environment (connection blocked). IMD, GPM IMERG, CHIRPS, India-WRIS, CWC, Kaggle and Zenodo were not retrievable (see `data/raw/SOURCES.md`).

Cleaning is causal: timestamps converted to Asia/Kolkata, duplicates removed, spikes rejected when a reading differs from the last valid one by more than 160 cm/h, and gaps filled forward for at most 3 hours. Features at hour t use only data up to t (a unit test changes the future and checks that features do not move): rain sums over 1, 3, 6, 12, 24, 72 h, maximum intensity, soil moisture, upstream discharge, level changes over 1, 3, 6 h and 6 h slope, hours since rain started, hour, month, monsoon flag, forecast rain totals and maximum for the next 1-6 h, station code. Targets: level at +1, +2, +3 h and a binary label "level reaches the danger level within 3 h".

## 3. Methodology
**Split.** Chronological: training to 2022, validation 2023-2024, final test 2025 (including its monsoon). No shuffling. Walk-forward cross-validation (4 folds, 216 h gap) on the training period.
**Baselines.** Persistence and linear trend extrapolation.
**Models.** LightGBM, Random Forest, and a GRU (24 h window, TensorFlow, CPU). Quantile LightGBM (p10, p50, p90) gives intervals. Classifiers (LightGBM and Random Forest) predict the flood label; class imbalance (~1% positives) is handled with class weights; probabilities are calibrated with isotonic regression on validation data; the decision threshold maximises F2 on the validation precision-recall curve. Optuna (10 trials) tunes LightGBM on validation only. The test set is evaluated once.
**Metrics.** MAE, RMSE, R2 for levels; precision, recall, F1, PR-AUC, ROC-AUC, Brier score, false-alarm ratio and event-level lead time (minutes between the first warning and the crossing of the danger level) for floods. Hourly metrics are computed on rows that are not yet above danger, because a warning is only useful before the flood.
**Explainability.** SHAP summary of the LightGBM classifier (`docs/figures/shap_summary.png`); error analysis by rain intensity and level (`docs/error_analysis.md`).
**Alerts.** The LLM (Groq API) receives a JSON of facts (station names in three languages, level, danger level, ETA, probability, alternative route) and must reply with JSON in three languages. A validator checks presence of all languages, script, length, station names, route text, and that every number appears in the facts; otherwise fixed templates are used. A small retrieval-augmented chat answers questions from the live forecast and alert log.
**System.** Mosquitto (MQTT), FastAPI with API-key authentication and input validation, SQLite storage, Streamlit dashboard (English, Hindi, Gujarati, Noto Sans), Telegram bot with de-duplication and rate limiting, Docker Compose. A simulator publishes the same JSON as the ESP32 firmware. A status machine (safe, warning, danger) uses hysteresis so alerts do not flap.

## 4. Results
All values are measured on the SYNTHETIC-TARGET test year.
<!-- RESULTS:START -->
_Data mode: **SYNTHETIC_RAIN__SYNTHETIC_TARGET**. Water levels are **SYNTHETIC-TARGET**. Test = 2025-01-01 to 2025-12-31 (25,154 rows, never used for tuning). Measured 2026-11-22 20:33:22._

**Water level forecast, TEST (cm; lower is better)**

| Model | MAE +1h | MAE +2h | MAE +3h | RMSE +3h | R2 +3h |
|---|---|---|---|---|---|
| Persistence | 1.03 | 1.17 | 1.30 | 3.68 | 0.942 |
| LinearTrend | 1.18 | 1.52 | 1.89 | 3.78 | 0.939 |
| LightGBM | 0.76 | 0.81 | 0.82 | 1.96 | 0.984 |
| RandomForest | 0.74 | 0.79 | 0.83 | 2.12 | 0.981 |
| GRU (24h window) | 0.80 | 0.91 | 1.00 | 2.85 | 0.966 |

**Flood within 3 h, TEST** (hourly metrics on rows not yet above danger; events = flood episodes, lead time over detected events)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier | False-alarm ratio | Events detected | Mean lead (min) | False-alarm episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| Persistence (levels->flag) | 0.000 | 0.000 | 0.000 | 0.091 | 0.988 | - | - | 0/10 | 0 | 0/10 |
| LinearTrend (levels->flag) | 0.667 | 0.467 | 0.549 | 0.482 | 0.992 | - | 0.333 | 8/10 | 105 | 6/16 |
| LightGBM (levels->flag) | 0.778 | 0.700 | 0.737 | 0.798 | 0.999 | - | 0.222 | 9/10 | 147 | 5/16 |
| RandomForest (levels->flag) | 0.944 | 0.567 | 0.708 | 0.747 | 0.998 | - | 0.056 | 9/10 | 127 | 1/13 |
| GRU (levels->flag) | 0.818 | 0.300 | 0.439 | 0.522 | 0.995 | - | 0.182 | 5/10 | 108 | 1/11 |
| LightGBM classifier (calibrated) | 0.718 | 0.933 | 0.812 | 0.886 | 1.000 | 0.00031 | 0.282 | 10/10 | 168 | 6/19 |
| RandomForest classifier (calibrated) | 0.426 | 0.967 | 0.592 | 0.864 | 1.000 | 0.00037 | 0.574 | 10/10 | 180 | 17/37 |

**Prediction interval (quantile LightGBM, nominal 80% = p10-p90), TEST**

| Horizon | Coverage | Mean width (cm) | Median MAE (cm) |
|---|---|---|---|
| +1h | 0.793 | 2.90 | 0.76 |
| +2h | 0.791 | 2.93 | 0.81 |
| +3h | 0.793 | 3.02 | 0.82 |

**Walk-forward CV (train period only, +3 h MAE, cm)**

| Fold | Train rows | LightGBM | Persistence |
|---|---|---|---|
| 0 | 65,639 | 0.85 | 1.19 |
| 1 | 131,492 | 0.88 | 1.32 |
| 2 | 197,345 | 0.76 | 1.12 |
| 3 | 263,198 | 1.00 | 1.42 |

Deployed regressor: **RandomForest** (best mean validation MAE). Deployed classifier: **LightGBM classifier**, isotonic-calibrated, threshold **0.364** (max F2 on validation PR curve). GRU: trained; evaluated on 24,786/25,154 test rows (rows with a full 24 h window).
<!-- RESULTS:END -->

**Discussion.** Tree models reduce the +3 h error compared with persistence and linear trend, mostly during rises (error analysis: dry hours MAE is small, heavy-rain hours larger). The calibrated LightGBM classifier detected the test flood episodes with several hours of warning in the simulation, at the cost of some false alarm episodes; the threshold was set to favour recall. The 80% prediction interval covers close to its nominal rate. An end-to-end run (simulator -> MQTT -> API -> model -> alerts) produced warnings about 3 hours before the danger level at two stations, trilingual alerts, and recovery to safe.

## 5. Conclusion
A reproducible, leakage-checked early-warning pipeline from sensor to multilingual alert was built and tested (43 automated tests). Because all data are synthetic, the accuracy figures are only a demonstration. The next step is to train on real ERA5/GloFAS rainfall, install sensors, record real levels and re-validate before any real use.

## 6. Limitations
Synthetic data; only 10 flood episodes in the test year, so lead-time and recall estimates are uncertain; hourly resolution; no dam-release input; firmware and Docker images not executed in the build sandbox; Groq and Telegram integrations tested with mocks.
