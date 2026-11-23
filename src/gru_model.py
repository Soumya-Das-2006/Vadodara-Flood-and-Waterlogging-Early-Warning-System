"""GRU sequence model (TensorFlow/Keras, CPU) for the model comparison. Optional: skipped if TF missing."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from .config import HORIZONS, SEED

T = 24
CH = ["level_ratio", "rain_s", "soil_moisture", "fc1_s", "fc_missing"]


def make_sequences(df):
    """Returns X (N,T,5), S (N,3) station one-hot, Y (N,3) level ratio targets, keys DataFrame."""
    xs, ss, ys, keys = [], [], [], []
    stations = sorted(df["station"].unique())
    for si, (st, g) in enumerate(df.groupby("station")):
        g = g.sort_values("timestamp")
        full = g.set_index("timestamp").reindex(pd.date_range(g.timestamp.min(), g.timestamp.max(), freq="h"))
        arr = np.stack([full["level_ratio"].values, full["rain_mmh"].values / 20.0,
                        full["soil_moisture"].values,
                        np.nan_to_num(full["fc_rain_next_1h"].values / 20.0),
                        full["fc_rain_next_1h"].isna().astype(float).values], axis=1).astype("float32")
        W = sliding_window_view(arr, T, axis=0).transpose(0, 2, 1)     # (n-T+1, T, 5)
        end_idx = np.arange(T - 1, len(arr))
        present = full["level_in_3h"].notna().values[end_idx] if "level_in_3h" in full else np.ones(len(end_idx), bool)
        ok = ~np.isnan(W).any(axis=(1, 2)) & present
        Wk = W[ok]
        ts = full.index[end_idx][ok]
        danger = full["danger_cm"].values[end_idx][ok]
        y = np.stack([full[f"level_in_{h}h"].values[end_idx][ok] / danger for h in HORIZONS], axis=1)
        xs.append(Wk); ys.append(y.astype("float32"))
        oh = np.zeros((len(Wk), len(stations)), "float32"); oh[:, stations.index(st)] = 1
        ss.append(oh)
        keys.append(pd.DataFrame({"timestamp": ts, "station": st, "danger_cm": danger}))
    return np.concatenate(xs), np.concatenate(ss), np.concatenate(ys), pd.concat(keys, ignore_index=True)


def build_model(n_st):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    x = tf.keras.Input((T, len(CH))); s = tf.keras.Input((n_st,))
    h = tf.keras.layers.GRU(48)(x)
    h = tf.keras.layers.Concatenate()([h, s])
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    out = tf.keras.layers.Dense(len(HORIZONS))(h)
    m = tf.keras.Model([x, s], out)
    m.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss=tf.keras.losses.Huber(0.5))
    return m


def fit_predict(train, val, test, epochs=6, verbose=0):
    """Train on train, early-stop on val, return predictions (cm) for test as DataFrame keyed by timestamp/station."""
    import tensorflow as tf
    Xtr, Str, Ytr, _ = make_sequences(train)
    Xva, Sva, Yva, _ = make_sequences(val)
    Xte, Ste, Yte, kte = make_sequences(test)
    m = build_model(Str.shape[1])
    cb = [tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor="val_loss")]
    m.fit([Xtr, Str], Ytr, validation_data=([Xva, Sva], Yva), epochs=epochs, batch_size=512,
          callbacks=cb, verbose=verbose)
    pva = m.predict([Xva, Sva], batch_size=2048, verbose=0)
    pte = m.predict([Xte, Ste], batch_size=2048, verbose=0)
    _, _, _, kva = make_sequences(val)
    def frame(p, k):
        f = k.copy()
        for j, h in enumerate(HORIZONS):
            f[f"gru_{h}h"] = np.clip(p[:, j] * k["danger_cm"].values, 0, None)
        return f.drop(columns="danger_cm")
    out = (frame(pva, kva), frame(pte, kte))
    del Xtr, Xva, Xte, m
    tf.keras.backend.clear_session()
    import gc; gc.collect()
    return out[0], out[1], None
