"""
ETD-XAI Enterprise v2.0  —  Electricity Theft Detection using Explainable AI
============================================================================
A single-file, production-quality Streamlit application.

Predictions come ONLY from the active CNN-LSTM Keras model
(assets/cnnlstm_final.keras) via tensorflow.keras.models.load_model() and
model.predict(). There are NO fallback / mock / surrogate / rule-based models.
Ground-truth FLAG columns are used ONLY for evaluation metrics, never prediction.

Run locally:        streamlit run app.py
Deploy (free):      Streamlit Community Cloud — main file path: app.py

Author : ETD-XAI Enterprise  ·  License: MIT
"""
from __future__ import annotations

import io
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as scipy_stats
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"
DEFAULT_MODEL = ASSETS / "cnnlstm_final.keras"
SAMPLE_DATASET = ASSETS / "sample_dataset.csv"
LOGO = ASSETS / "logo.png"

# Writable data dir. On Streamlit Community Cloud the repo mount
# (/mount/src/...) is READ-ONLY, so the SQLite DB and any uploads must live in
# a writable location (a temp dir by default). Override with ETD_DATA_DIR /
# DATABASE_PATH to point at a persistent disk in other hosts.
import tempfile  # noqa: E402
DATA_DIR = Path(os.environ.get("ETD_DATA_DIR", str(Path(tempfile.gettempdir()) / "etd_xai")))
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    DATA_DIR = Path(tempfile.gettempdir())
DB_PATH = Path(os.environ.get("DATABASE_PATH", str(DATA_DIR / "etd_xai.db")))
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

APP_VERSION = "2.0.0"

# Exact message shown when no model is available — prediction stops completely.
NO_MODEL_MSG = "No active CNN-LSTM model loaded."

st.set_page_config(page_title="ETD-XAI Enterprise", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Feature engineering (EXACT training preprocessing)
# ═════════════════════════════════════════════════════════════════════════════
def scale_sequences(readings: np.ndarray) -> np.ndarray:
    """Per-row min-max scale each sequence to [0,1] — matches training CELL 8.
    Feeding raw kWh saturates the model → everything predicted Normal."""
    readings = np.asarray(readings, dtype=np.float32)
    scaled = np.zeros_like(readings)
    for i in range(len(readings)):
        mn, mx = readings[i].min(), readings[i].max()
        if mx > mn:
            scaled[i] = (readings[i] - mn) / (mx - mn)
    return scaled


def _features_for_row(row: np.ndarray) -> list:
    """59 statistical features — verbatim from training CELL 7."""
    row = row.astype(np.float32)
    n = len(row)
    mean = np.mean(row); std = np.std(row); mx = np.max(row); mn = np.min(row)
    median = np.median(row)
    skew = float(scipy_stats.skew(row)); kurt = float(scipy_stats.kurtosis(row))
    cv = std / (mean + 1e-9)
    p10, p25, p75, p90 = np.percentile(row, [10, 25, 75, 90])
    iqr = p75 - p25
    zero_ratio = np.mean(row == 0); neg_ratio = np.mean(row < 0)
    near_zero = np.mean(row < 0.01); low_cons_ratio = np.mean(row < mean * 0.1)
    drop_ratio = np.mean(np.diff(row) < -std)
    t = np.arange(n)
    slope = np.polyfit(t, row, 1)[0]
    resid = row - np.polyval(np.polyfit(t, row, 1), t)
    resid_std = np.std(resid)
    energy = np.sum(row ** 2) / n
    hist, _ = np.histogram(row, bins=30, density=True)
    ent = entropy(hist + 1e-9)
    runs, cnt = [], 0
    for v in row:
        if v == 0:
            cnt += 1
        else:
            if cnt > 0: runs.append(cnt)
            cnt = 0
    max_zero_run = max(runs) if runs else 0
    n_zero_runs = len(runs)
    if n >= 48:
        n_days = n // 48
        days = row[:n_days * 48].reshape(n_days, 48)
        dm = np.mean(days, axis=1); ds = np.std(days, axis=1)
        day_cons = np.mean(days[:, :24]); night_cons = np.mean(days[:, 24:])
        dn_ratio = day_cons / (night_cons + 1e-9)
        day_cv = np.std(dm) / (np.mean(dm) + 1e-9)
        theft_days = np.mean(dm < np.mean(dm) * 0.5)
        day_chg = np.abs(np.diff(dm))
        max_day_chg = np.max(day_chg) if len(day_chg) > 0 else 0
        mean_day_chg = np.mean(day_chg) if len(day_chg) > 0 else 0
        dm_mean, dm_std = np.mean(dm), np.std(dm)
        dm_max, dm_min = np.max(dm), np.min(dm); ds_mean = np.mean(ds)
    else:
        dn_ratio = day_cv = theft_days = 0
        max_day_chg = mean_day_chg = 0
        dm_mean = dm_std = dm_max = dm_min = ds_mean = 0
    ac1 = np.corrcoef(row[:-1], row[1:])[0, 1] if n > 1 else 0
    ac48 = np.corrcoef(row[:-48], row[48:])[0, 1] if n > 48 else 0
    ac7d = np.corrcoef(row[:-336], row[336:])[0, 1] if n > 336 else 0
    fft_v = np.abs(np.fft.rfft(row))
    fft_mean = np.mean(fft_v); fft_std = np.std(fft_v); fft_max = np.max(fft_v)
    dominant_freq = np.argmax(fft_v[1:]) + 1
    if n >= 100:
        mean_change = np.mean(row[n // 2:]) - np.mean(row[:n // 2])
        std_change = np.std(row[n // 2:]) - np.std(row[:n // 2])
    else:
        mean_change = std_change = 0.0
    diffs = np.diff(row)
    max_drop = np.min(diffs) if len(diffs) > 0 else 0
    max_rise = np.max(diffs) if len(diffs) > 0 else 0
    n_big_drops = np.sum(diffs < -2 * std); n_big_rises = np.sum(diffs > 2 * std)
    below_median = np.mean(row < median); above_median = np.mean(row > median)
    quarters = np.array_split(row, 4)
    q_means = [np.mean(q) for q in quarters]; q_stds = [np.std(q) for q in quarters]
    q_trend = q_means[-1] - q_means[0]; q_var = np.std(q_means)
    return [mean, std, mx, mn, median, skew, kurt, cv, p10, p25, p75, p90, iqr,
            zero_ratio, neg_ratio, near_zero, low_cons_ratio, drop_ratio,
            slope, resid_std, energy, ent, max_zero_run, n_zero_runs,
            dn_ratio, day_cv, theft_days, max_day_chg, mean_day_chg,
            dm_mean, dm_std, dm_max, dm_min, ds_mean, ac1, ac48, ac7d,
            fft_mean, fft_std, fft_max, dominant_freq, mean_change, std_change,
            max_drop, max_rise, n_big_drops, n_big_rises, below_median, above_median,
            q_means[0], q_means[1], q_means[2], q_means[3],
            q_stds[0], q_stds[1], q_stds[2], q_stds[3], q_trend, q_var]


FEATURE_NAMES = [
    "mean", "std", "max", "min", "median", "skew", "kurtosis", "cv",
    "p10", "p25", "p75", "p90", "iqr", "zero_ratio", "neg_ratio", "near_zero",
    "low_cons_ratio", "drop_ratio", "slope", "resid_std", "energy", "entropy",
    "max_zero_run", "n_zero_runs", "day_night_ratio", "day_cv", "theft_days",
    "max_day_chg", "mean_day_chg", "daymean_mean", "daymean_std", "daymean_max",
    "daymean_min", "daystd_mean", "autocorr_1", "autocorr_48", "autocorr_7d",
    "fft_mean", "fft_std", "fft_max", "dominant_freq", "mean_change", "std_change",
    "max_drop", "max_rise", "n_big_drops", "n_big_rises", "below_median",
    "above_median", "q1_mean", "q2_mean", "q3_mean", "q4_mean", "q1_std",
    "q2_std", "q3_std", "q4_std", "q_trend", "q_var",
]


def extract_features(readings: np.ndarray) -> np.ndarray:
    feats = np.array([_features_for_row(r) for r in readings], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


class FeaturePipeline:
    """StandardScaler fitted per uploaded batch (training scaler was not saved)."""
    def __init__(self):
        self._scaler: Optional[StandardScaler] = None
        self._fitted = False

    def fit_transform(self, readings: np.ndarray) -> np.ndarray:
        raw = extract_features(readings)
        self._scaler = StandardScaler()
        out = np.nan_to_num(self._scaler.fit_transform(raw).astype(np.float32))
        self._fitted = True
        return out

    def transform(self, readings: np.ndarray) -> np.ndarray:
        raw = extract_features(readings)
        if self._fitted and self._scaler is not None:
            out = self._scaler.transform(raw).astype(np.float32)
        else:
            out = np.zeros_like(raw)
        return np.nan_to_num(out)

    def reset(self):
        self._scaler = None; self._fitted = False


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Sequence length-mapping strategies
# ═════════════════════════════════════════════════════════════════════════════
STRATEGIES = ["last_n", "truncate", "pad", "interpolate", "sliding_window"]
STRATEGY_LABELS = {
    "last_n": "Last N readings", "truncate": "Truncate (first N)",
    "pad": "Pad with zeros", "interpolate": "Interpolate / resample",
    "sliding_window": "Sliding window (aggregated)",
}


def resize_row(row: np.ndarray, target_len: int, strategy: str) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32)
    L = len(row)
    if L == target_len:
        return row
    if strategy == "interpolate":
        return np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, L), row).astype(np.float32)
    if strategy == "last_n":
        return row[-target_len:] if L >= target_len else np.concatenate([np.zeros(target_len - L, np.float32), row])
    if strategy in ("truncate", "pad"):
        return row[:target_len] if L >= target_len else np.concatenate([row, np.zeros(target_len - L, np.float32)])
    raise ValueError(f"Unknown strategy: {strategy}")


def resize_sequences(seq_2d: np.ndarray, target_len: int, strategy: str) -> np.ndarray:
    seq_2d = np.asarray(seq_2d, dtype=np.float32)
    if seq_2d.shape[1] == target_len:
        return seq_2d
    return np.vstack([resize_row(r, target_len, strategy) for r in seq_2d]).astype(np.float32)


def windows_for_row(row: np.ndarray, target_len: int, stride: int = 0) -> list:
    row = np.asarray(row, dtype=np.float32)
    L = len(row)
    if L <= target_len:
        return [resize_row(row, target_len, "last_n")]
    stride = stride if stride > 0 else max(1, target_len // 2)
    windows, start = [], 0
    while start + target_len <= L:
        windows.append(row[start:start + target_len]); start += stride
    if (L - target_len) % stride != 0:
        windows.append(row[L - target_len:])
    return windows


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Exclusive CNN-LSTM engine (TensorFlow/Keras only)
# ═════════════════════════════════════════════════════════════════════════════
PIPELINE = FeaturePipeline()
_TF = None


def tf():
    """Lazy TF import + Keras 3 compatibility shim for the saved model."""
    global _TF
    if _TF is None:
        import tensorflow as _t
        _t.get_logger().setLevel("ERROR")
        try:
            import keras
            _orig = keras.layers.Dense.from_config.__func__

            @classmethod
            def _compat(cls, config):
                config = dict(config); config.pop("quantization_config", None)
                dt = config.get("dtype")
                if isinstance(dt, dict):
                    config["dtype"] = dt.get("config", {}).get("name", "float32")
                return _orig(cls, config)
            keras.layers.Dense.from_config = _compat
        except Exception:
            pass
        _TF = _t
    return _TF


class Engine:
    """Process-global model state. Loaded once per server via st.cache_resource."""
    model = None
    name = ""
    path = ""
    upload_time = ""
    input_shape = ()
    output_shape = ()
    total_params = 0
    summary = ""
    is_dual = False
    stat_size = 0
    seq_len = None       # None => variable length
    seq_channels = 1
    last_prediction: dict = {}


E = Engine()


def _rank(shape) -> int:
    r = getattr(shape, "rank", None)
    return r if r is not None else len(shape)


def load_model(path: str, name: Optional[str] = None) -> dict:
    name = name or Path(path).name
    if Path(name).suffix.lower() not in (".keras", ".h5"):
        raise ValueError("Unsupported format. Use .keras or .h5")
    model = tf().keras.models.load_model(path)

    classes = {l.__class__.__name__.lower() for l in model.layers}
    if not (any("conv" in c for c in classes) or any(k in c for c in classes for k in ("lstm", "gru", "rnn"))):
        raise ValueError("Rejected: model has no Conv/LSTM/GRU/RNN layers — not a temporal sequence model.")

    seq_input = next((i for i in model.inputs if _rank(i.shape) == 3), None)
    if seq_input is None:
        raise ValueError("No 3-D sequence input (expected shape (None, T, C)).")
    seq_shape = tuple(seq_input.shape[1:])
    E.seq_len = int(seq_shape[0]) if seq_shape[0] is not None else None
    E.seq_channels = int(seq_shape[1]) if len(seq_shape) > 1 and seq_shape[1] else 1

    E.is_dual, E.stat_size = False, 0
    if len(model.inputs) == 2:
        for inp in model.inputs:
            if _rank(inp.shape) == 2:
                E.is_dual, E.stat_size = True, int(inp.shape[-1]); break

    buf = io.StringIO(); model.summary(print_fn=lambda x: buf.write(x + "\n"))
    E.model = model
    E.name = name
    E.path = path
    E.upload_time = datetime.now().isoformat()
    E.input_shape = tuple(model.input_shape) if isinstance(model.input_shape, (list, tuple)) else (model.input_shape,)
    E.output_shape = tuple(model.output_shape)
    E.total_params = int(model.count_params())
    E.summary = buf.getvalue()
    return model_info()


def auto_load_default() -> bool:
    if E.model is not None:
        return True
    saved = get_setting("active_model_path")
    for cand in (saved, str(DEFAULT_MODEL)):
        if cand and Path(cand).exists():
            try:
                load_model(cand, Path(cand).name)
                return True
            except Exception:
                continue
    return False


def unload_model():
    if E.model is not None:
        try:
            tf().keras.backend.clear_session()
        except Exception:
            pass
    E.model = None; E.name = ""; E.seq_len = None; E.last_prediction = {}
    PIPELINE.reset()


def is_loaded() -> bool:
    return E.model is not None


def model_info() -> dict:
    if E.model is None:
        return {"loaded": False}
    import keras as _k
    return {
        "loaded": True, "name": E.name, "path": E.path, "upload_time": E.upload_time,
        "input_shape": str(E.input_shape), "output_shape": str(E.output_shape),
        "total_params": E.total_params, "total_params_fmt": f"{E.total_params:,}",
        "is_dual": E.is_dual, "stat_size": E.stat_size, "seq_len": E.seq_len,
        "is_variable": E.seq_len is None, "seq_channels": E.seq_channels,
        "summary": E.summary, "tf_version": tf().__version__,
        "keras_version": getattr(_k, "__version__", "unknown"),
        "architecture": "CNN-LSTM" if E.is_dual else "Sequence model",
    }


def check_compatibility(uploaded_len: int) -> dict:
    if E.model is None:
        return {"compatible": False, "reason": NO_MODEL_MSG}
    T = E.seq_len
    if T is None:
        return {"compatible": True, "needs_prep": False,
                "reason": "Model accepts variable-length sequences — sent as-is."}
    if uploaded_len == T:
        return {"compatible": True, "needs_prep": False,
                "reason": f"Uploaded length {uploaded_len} matches the model exactly."}
    return {"compatible": True, "needs_prep": True,
            "reason": f"Uploaded length {uploaded_len} ≠ model length {T}. "
                      f"A length-mapping strategy will be applied."}


def _build_stat(ready_2d, fit_scaler):
    raw = extract_features(ready_2d)
    if raw.shape[1] != E.stat_size:
        raise ValueError(f"Stat-feature mismatch: produced {raw.shape[1]}, model needs "
                         f"{E.stat_size}. Refusing to substitute zeros.")
    return PIPELINE.fit_transform(ready_2d) if fit_scaler else PIPELINE.transform(ready_2d)


def _raw_predict(seq_ready_2d, stat, batch_size):
    """The ONLY inference choke-point — real tensorflow.keras model.predict()."""
    if E.model is None:
        raise RuntimeError(NO_MODEL_MSG)
    seq_scaled = scale_sequences(seq_ready_2d)
    L = seq_scaled.shape[1]
    seq = seq_scaled.reshape(-1, L, E.seq_channels).astype(np.float32)
    if E.is_dual:
        if stat is None:
            raise RuntimeError("Model requires stat_input but none was provided.")
        inputs = {"sequence_input": seq, "stat_input": stat.astype(np.float32)}
        in_shape = f"[{seq.shape}, {stat.shape}]"
    else:
        inputs = seq; in_shape = str(seq.shape)
    out = E.model.predict(inputs, verbose=0, batch_size=batch_size)
    probs = out.flatten().astype(np.float32)
    import keras as _k
    raw0 = float(probs[0]) if len(probs) else float("nan")
    E.last_prediction = {
        "active_model": E.name, "engine": "TensorFlow / Keras",
        "tf_version": tf().__version__, "keras_version": getattr(_k, "__version__", "unknown"),
        "input_shape": in_shape, "output_shape": str(tuple(out.shape)),
        "raw_output": round(raw0, 6), "predicted_label": "Theft" if raw0 >= 0.5 else "Normal",
        "n_rows": int(len(probs)), "timestamp": datetime.now().isoformat(),
    }
    return probs


def predict_sequences(raw_2d, strategy="last_n", threshold=0.5, fit_scaler=True, batch_size=256):
    if E.model is None:
        raise RuntimeError(NO_MODEL_MSG)
    raw_2d = np.asarray(raw_2d, dtype=np.float32)
    T = E.seq_len
    if strategy == "sliding_window" and T is not None and raw_2d.shape[1] > T:
        all_w, idx = [], []
        for ri, row in enumerate(raw_2d):
            for w in windows_for_row(row, T):
                all_w.append(w); idx.append(ri)
        win = np.vstack(all_w).astype(np.float32)
        stat = _build_stat(win, fit_scaler) if E.is_dual else None
        wp = _raw_predict(win, stat, batch_size)
        probs = np.zeros(len(raw_2d), np.float32)
        for ri, p in zip(idx, wp):
            probs[ri] = max(probs[ri], p)
        return probs
    ready = raw_2d if (T is None or raw_2d.shape[1] == T) else resize_sequences(raw_2d, T, strategy)
    stat = _build_stat(ready, fit_scaler) if E.is_dual else None
    return _raw_predict(ready, stat, batch_size)


def classify(prob: float, threshold: float = 0.5) -> dict:
    pred = 1 if prob >= threshold else 0
    conf = prob if pred == 1 else (1.0 - prob)
    risk = round(prob * 100, 2)
    level = "High" if risk >= 75 else "Medium" if risk >= 40 else "Low"
    return {"probability": round(float(prob), 6), "prediction": pred,
            "confidence": round(float(conf), 6), "risk_score": risk,
            "risk_level": level, "status": "Theft" if pred == 1 else "Normal"}


def predict_one(readings, strategy="last_n", threshold=0.5) -> dict:
    if E.model is None:
        raise RuntimeError(NO_MODEL_MSG)
    r = np.asarray(readings, dtype=np.float32).flatten().reshape(1, -1)
    prob = float(predict_sequences(r, strategy, threshold, fit_scaler=False)[0])
    res = classify(prob, threshold)
    res.update({"model_name": E.name, "uploaded_len": int(r.shape[1]), "model_len": E.seq_len,
                "strategy_used": "none" if (E.seq_len is None or r.shape[1] == E.seq_len) else strategy})
    return res


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Dataset ingestion + evaluation metrics
# ═════════════════════════════════════════════════════════════════════════════
ID_COLS = {"cons_no", "customer_id", "id", "customer", "consumer_no", "meter_id", "user_id"}
FLAG_COLS = {"flag", "label", "target", "theft", "is_theft", "class", "y"}


def read_table(file) -> pd.DataFrame:
    name = getattr(file, "name", str(file)).lower()
    return pd.read_excel(file) if name.endswith((".xlsx", ".xls")) else pd.read_csv(file)


def inspect(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    lower = {c: str(c).strip().lower() for c in cols}
    id_col = next((c for c in cols if lower[c] in ID_COLS), None)
    flag_col = next((c for c in cols if lower[c] in FLAG_COLS), None)
    reading_cols = [c for c in cols if c not in (id_col, flag_col)
                    and pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.5]
    return {"id_col": id_col, "flag_col": flag_col, "reading_cols": reading_cols,
            "n_readings": len(reading_cols), "n_rows": len(df),
            "has_flag": flag_col is not None, "columns": [str(c) for c in cols]}


def build_matrix(df, info) -> Tuple[np.ndarray, list, Optional[np.ndarray]]:
    readings = df[info["reading_cols"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    ids = (df[info["id_col"]].astype(str).tolist() if info["id_col"]
           else [f"CUST_{i+1:06d}" for i in range(len(df))])
    flags = None
    if info["flag_col"]:  # ground truth — used ONLY for metrics below
        flags = pd.to_numeric(df[info["flag_col"]], errors="coerce").fillna(0).astype(int).to_numpy()
    return readings, ids, flags


def compute_metrics(flags, preds, probs) -> dict:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    out = {"accuracy": float(accuracy_score(flags, preds)),
           "precision_val": float(precision_score(flags, preds, zero_division=0)),
           "recall_val": float(recall_score(flags, preds, zero_division=0)),
           "f1_score": float(f1_score(flags, preds, zero_division=0)),
           "confusion_matrix": confusion_matrix(flags, preds, labels=[0, 1]).tolist()}
    try:
        out["roc_auc"] = float(roc_auc_score(flags, probs))
    except Exception:
        out["roc_auc"] = None
    return out


def run_batch(df, info, strategy="last_n", threshold=0.5) -> dict:
    readings, ids, flags = build_matrix(df, info)
    probs = predict_sequences(readings, strategy, threshold, fit_scaler=True)
    rows = []
    for i, (cid, p) in enumerate(zip(ids, probs)):
        rows.append({"customer_id": cid, **classify(float(p), threshold),
                     "flag": int(flags[i]) if flags is not None else None})
    preds = np.array([r["prediction"] for r in rows])
    theft = int(preds.sum())
    res = {"rows": rows, "total_rows": len(rows), "theft_rows": theft,
           "normal_rows": len(rows) - theft, "theft_rate": round(theft / max(len(rows), 1), 4),
           "avg_risk": round(float(np.mean([r["risk_score"] for r in rows])), 2),
           "has_flag": flags is not None, "n_readings": info["n_readings"], "metrics": None}
    if flags is not None:
        res["metrics"] = compute_metrics(flags, preds, probs)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Explainable AI (SHAP with integrated-gradients fallback)
# ═════════════════════════════════════════════════════════════════════════════
def shap_or_ig(readings: np.ndarray, background: Optional[np.ndarray] = None) -> Optional[dict]:
    """Per-timestep attribution. Tries SHAP GradientExplainer, falls back to
    integrated gradients (both use the REAL model gradients — no surrogate)."""
    if not is_loaded():
        return None
    r = np.asarray(readings, dtype=np.float32).flatten()
    T = E.seq_len or len(r)
    if len(r) != T:
        r = resize_row(r, T, "last_n")
    seq = scale_sequences(r.reshape(1, -1)).reshape(1, T, E.seq_channels).astype(np.float32)
    stat = (PIPELINE.transform(r.reshape(1, -1)) if (E.is_dual and PIPELINE._fitted)
            else (np.zeros((1, E.stat_size), np.float32) if E.is_dual else None))

    # --- Try SHAP ---
    try:
        import shap  # noqa
        bg_seq = np.zeros_like(seq)
        if E.is_dual:
            expl = shap.GradientExplainer(E.model, [bg_seq, np.zeros_like(stat)])
            sv = expl.shap_values([seq, stat])
            seq_sv = sv[0][0] if isinstance(sv, list) else sv[0]
        else:
            expl = shap.GradientExplainer(E.model, bg_seq)
            sv = expl.shap_values(seq)
            seq_sv = sv[0] if isinstance(sv, list) else sv
        imp = np.abs(np.asarray(seq_sv)).reshape(T, -1).sum(axis=1)
        s = imp.sum()
        return {"method": "SHAP (GradientExplainer)",
                "timestep_importance": (imp / s).tolist() if s > 0 else imp.tolist()}
    except Exception:
        pass

    # --- Integrated gradients fallback ---
    try:
        _t = tf()
        seq_tf = _t.convert_to_tensor(seq)
        baseline = _t.zeros_like(seq_tf)
        stat_tf = _t.convert_to_tensor(stat) if E.is_dual else None
        grads = []
        for a in _t.linspace(0.0, 1.0, 32):
            interp = baseline + a * (seq_tf - baseline)
            with _t.GradientTape() as tape:
                tape.watch(interp)
                out = (E.model({"sequence_input": interp, "stat_input": stat_tf})
                       if E.is_dual else E.model(interp))
                out = _t.reduce_sum(out)
            grads.append(tape.gradient(out, interp))
        ig = (seq_tf - baseline) * _t.reduce_mean(_t.stack(grads), axis=0)
        imp = _t.reduce_sum(_t.abs(ig), axis=-1).numpy().flatten()
        s = imp.sum()
        return {"method": "Integrated Gradients",
                "timestep_importance": (imp / s).tolist() if s > 0 else imp.tolist()}
    except Exception:
        return None


def risk_factors(readings: np.ndarray, top: int = 6) -> list:
    r = np.asarray(readings, dtype=np.float32).flatten()
    f = dict(zip(FEATURE_NAMES, extract_features(r.reshape(1, -1))[0]))
    out = []
    if f["zero_ratio"] > 0.15: out.append(("High proportion of zero readings", f["zero_ratio"], "↑ theft"))
    if f["max_zero_run"] >= 3: out.append(("Long consecutive zero-consumption run", f["max_zero_run"], "↑ theft"))
    if f["drop_ratio"] > 0.2: out.append(("Frequent sharp consumption drops", f["drop_ratio"], "↑ theft"))
    if f["cv"] > 1.0: out.append(("Very high consumption variability", f["cv"], "↑ theft"))
    if f["slope"] < 0: out.append(("Declining consumption trend", f["slope"], "↑ theft"))
    if f["low_cons_ratio"] > 0.3: out.append(("Many abnormally low readings", f["low_cons_ratio"], "↑ theft"))
    if f["q_trend"] < 0: out.append(("Downward quarter-over-quarter trend", f["q_trend"], "↑ theft"))
    if not out: out.append(("Stable, regular consumption pattern", f["cv"], "↓ normal"))
    return out[:top]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Persistence (SQLite + settings)
# ═════════════════════════════════════════════════════════════════════════════
_DDL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS uploads(
  id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, upload_time TEXT,
  total_rows INT, theft_rows INT, normal_rows INT, avg_risk REAL, theft_rate REAL,
  has_flag INT, threshold REAL, n_readings INT, strategy TEXT,
  accuracy REAL, precision_val REAL, recall_val REAL, f1_score REAL, roc_auc REAL,
  confusion_matrix TEXT);
CREATE TABLE IF NOT EXISTS predictions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, upload_id INT, customer_id TEXT,
  probability REAL, prediction INT, confidence REAL, risk_score REAL, status TEXT, flag INT);
CREATE TABLE IF NOT EXISTS manual(
  id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT, probability REAL,
  prediction INT, confidence REAL, risk_score REAL, status TEXT, readings TEXT,
  predicted_at TEXT, threshold REAL, model_name TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
"""


@contextmanager
def _conn():
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    try:
        yield c; c.commit()
    finally:
        c.close()


def init_db():
    with _conn() as c:
        c.executescript(_DDL)


def set_setting(k, v):
    with _conn() as c:
        c.execute("INSERT INTO settings(key,value) VALUES(?,?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, json.dumps(v)))


def get_setting(k, default=None):
    try:
        with _conn() as c:
            row = c.execute("SELECT value FROM settings WHERE key=?", (k,)).fetchone()
        return json.loads(row["value"]) if row else default
    except Exception:
        return default


def save_upload(**kw) -> int:
    cm = kw.get("confusion_matrix")
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO uploads(filename,upload_time,total_rows,theft_rows,normal_rows,
               avg_risk,theft_rate,has_flag,threshold,n_readings,strategy,
               accuracy,precision_val,recall_val,f1_score,roc_auc,confusion_matrix)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (kw["filename"], datetime.now().isoformat(), kw["total_rows"], kw["theft_rows"],
             kw["normal_rows"], kw.get("avg_risk"), kw.get("theft_rate"),
             int(kw.get("has_flag", False)), kw.get("threshold", 0.5), kw.get("n_readings"),
             kw.get("strategy"), kw.get("accuracy"), kw.get("precision_val"),
             kw.get("recall_val"), kw.get("f1_score"), kw.get("roc_auc"),
             json.dumps(cm) if cm is not None else None))
        return cur.lastrowid


def save_predictions_bulk(uid, rows):
    recs = [(uid, r["customer_id"], r["probability"], r["prediction"], r["confidence"],
             r["risk_score"], r["status"], r.get("flag")) for r in rows]
    with _conn() as c:
        c.executemany("INSERT INTO predictions(upload_id,customer_id,probability,prediction,"
                      "confidence,risk_score,status,flag) VALUES(?,?,?,?,?,?,?,?)", recs)


def latest_upload_id():
    with _conn() as c:
        row = c.execute("SELECT id FROM uploads ORDER BY id DESC LIMIT 1").fetchone()
    return row["id"] if row else None


def get_upload(uid):
    with _conn() as c:
        row = c.execute("SELECT * FROM uploads WHERE id=?", (uid,)).fetchone()
    if not row:
        return None
    d = dict(row)
    if d.get("confusion_matrix"):
        try:
            d["confusion_matrix"] = json.loads(d["confusion_matrix"])
        except Exception:
            d["confusion_matrix"] = None
    return d


def all_uploads():
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM uploads ORDER BY id DESC").fetchall()]


def predictions_df(uid) -> pd.DataFrame:
    with _conn() as c:
        rows = c.execute("SELECT customer_id,probability,prediction,confidence,risk_score,status,flag "
                         "FROM predictions WHERE upload_id=? ORDER BY risk_score DESC", (uid,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def save_manual(**kw):
    with _conn() as c:
        c.execute("""INSERT INTO manual(customer_id,probability,prediction,confidence,risk_score,
                     status,readings,predicted_at,threshold,model_name,source)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                  (kw.get("customer_id"), kw["probability"], kw["prediction"], kw["confidence"],
                   kw["risk_score"], kw["status"], json.dumps(kw.get("readings", [])),
                   datetime.now().isoformat(), kw.get("threshold", 0.5),
                   kw.get("model_name"), kw.get("source", "manual")))


def get_manual(limit=500):
    with _conn() as c:
        rows = c.execute("SELECT * FROM manual ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def counts():
    with _conn() as c:
        p = c.execute("SELECT COUNT(*) x FROM predictions").fetchone()["x"]
        m = c.execute("SELECT COUNT(*) x FROM manual").fetchone()["x"]
    return {"predictions": p, "manual": m}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Reports (CSV / Excel / PDF)
# ═════════════════════════════════════════════════════════════════════════════
def to_csv(df): return df.to_csv(index=False).encode("utf-8")


def to_excel(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Predictions")
    return buf.getvalue()


def to_pdf(title, info, summary, metrics, df):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except Exception:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    s = getSampleStyleSheet(); el = []
    el += [Paragraph(f"<b>{title}</b>", s["Title"]),
           Paragraph("Electricity Theft Detection using Explainable AI", s["Italic"]),
           Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", s["Normal"]),
           Spacer(1, .6 * cm)]

    def tbl(t, pairs):
        el.append(Paragraph(f"<b>{t}</b>", s["Heading2"]))
        table = Table([[str(k), str(v)] for k, v in pairs], colWidths=[7 * cm, 9 * cm])
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
        el += [table, Spacer(1, .5 * cm)]

    tbl("Model Information", [("Active Model", info.get("name", "—")),
        ("Architecture", info.get("architecture", "—")), ("Input Shape", info.get("input_shape", "—")),
        ("Output Shape", info.get("output_shape", "—")), ("Parameters", info.get("total_params_fmt", "—")),
        ("TensorFlow", info.get("tf_version", "—"))])
    tbl("Prediction Summary", [("Total Customers", summary.get("total_rows", 0)),
        ("Normal", summary.get("normal_rows", 0)), ("Theft", summary.get("theft_rows", 0)),
        ("Theft Rate", f"{(summary.get('theft_rate') or 0) * 100:.2f}%"),
        ("Avg Risk", summary.get("avg_risk", 0))])
    if metrics:
        tbl("Evaluation Metrics", [("Accuracy", f"{metrics.get('accuracy', 0):.4f}"),
            ("Precision", f"{metrics.get('precision_val', 0):.4f}"),
            ("Recall", f"{metrics.get('recall_val', 0):.4f}"),
            ("F1 Score", f"{metrics.get('f1_score', 0):.4f}"),
            ("ROC-AUC", f"{(metrics.get('roc_auc') or 0):.4f}")])
    if df is not None and len(df):
        el.append(Paragraph("<b>Top 15 Highest-Risk Customers</b>", s["Heading2"]))
        cols = [c for c in ["customer_id", "probability", "risk_score", "status"] if c in df.columns]
        top = df.sort_values("risk_score", ascending=False).head(15)[cols].round(4)
        t = Table([cols] + top.astype(str).values.tolist())
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 7)]))
        el.append(t)
    doc.build(el)
    return buf.getvalue()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — AI Copilot (rule-based, project-scoped, no hallucination)
# ═════════════════════════════════════════════════════════════════════════════
_KB = {
    "cnn-lstm": "**CNN-LSTM** is the hybrid deep model used here. 1-D CNN layers extract local "
                "patterns from the consumption sequence; LSTM layers model temporal dependencies. "
                "It takes the scaled reading sequence (+59 statistical features) and outputs a single "
                "sigmoid theft probability.",
    "accuracy": "**Accuracy** = correct / total. Overall correctness, but misleading on imbalanced "
                "theft data — read it with precision, recall and ROC-AUC.",
    "precision": "**Precision** = TP/(TP+FP). Of customers flagged as theft, how many truly were — "
                 "high precision means few false accusations.",
    "recall": "**Recall** = TP/(TP+FN). Of all real theft cases, how many were caught — high recall "
              "means few thieves slip through.",
    "f1": "**F1** = harmonic mean of precision and recall (2PR/(P+R)) — one balanced number.",
    "roc": "**ROC-AUC** = probability the model ranks a random theft case above a random normal one. "
           "0.5 = random, 1.0 = perfect.",
    "risk": "The **Risk Score** is the theft probability on a 0–100 scale. ≥75 High, 40–74 Medium, "
            "<40 Low. It drives the red/green badge.",
    "threshold": "The **decision threshold** (default 0.5) is the probability cut-off for labelling "
                 "Theft. Lower → more recall (more false alarms); higher → more precision.",
    "shap": "**SHAP** assigns each timestep a contribution to the prediction using the model's own "
            "gradients (GradientExplainer). Here it shows which days pushed the verdict toward theft; "
            "if SHAP is unavailable the app falls back to integrated gradients — both use the real model.",
    "preprocessing": "Each sequence is **per-row min-max scaled to [0,1]** and 59 statistical features "
                     "are extracted + StandardScaler-normalised — exactly as in training.",
    "theft": "A customer is **Theft (Class 1)** when probability ≥ threshold. Typical signatures: long "
             "zero-consumption runs, sudden sustained drops, abnormally low/erratic usage.",
    "normal": "A **Normal (Class 0)** customer shows stable regular consumption; probability below threshold.",
}
SUGGESTIONS = ["Explain the CNN-LSTM model", "What is the Risk Score?", "Explain Recall vs Precision",
               "How does the threshold work?", "Explain SHAP", "Why is a customer classified as theft?"]


def copilot_answer(q: str) -> str:
    q = (q or "").strip().lower()
    if not q:
        return "Ask me about the model, a prediction, or a metric."
    if q in {"hi", "hello", "hey", "salam"}:
        return "Hello! I'm the ETD-XAI Copilot. Ask about the CNN-LSTM model, predictions, or any metric."
    if any(k in q for k in ("which model", "what model", "active model", "model loaded")):
        if is_loaded():
            i = model_info()
            return (f"Active model: **{i['name']}** ({i['architecture']}), loaded via "
                    f"`tensorflow.keras.models.load_model()`. Input {i['input_shape']}, "
                    f"{i['total_params_fmt']} params. Every prediction uses this model only — no fallbacks.")
        return NO_MODEL_MSG
    table = [("cnn-lstm", ("cnn", "lstm", "architecture", "neural", "deep")),
             ("accuracy", ("accuracy",)), ("precision", ("precision",)),
             ("recall", ("recall", "sensitivity")), ("f1", ("f1", "f-1")),
             ("roc", ("roc", "auc")), ("risk", ("risk",)), ("threshold", ("threshold", "cut")),
             ("shap", ("shap", "explain", "xai", "interpret", "gradient", "feature importance")),
             ("preprocessing", ("preprocess", "scal", "normaliz", "feature")),
             ("theft", ("why theft", "theft", "class 1", "steal")), ("normal", ("normal", "class 0"))]
    for key, trig in table:
        if any(t in q for t in trig):
            return _KB[key]
    return ("I only answer questions about **this project** — the CNN-LSTM model, its predictions, "
            "preprocessing, SHAP explanations and the evaluation metrics. Please rephrase around one "
            "of those topics.")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Boot + session state + styling
# ═════════════════════════════════════════════════════════════════════════════
init_db()


@st.cache_resource(show_spinner="Loading CNN-LSTM model…")
def _boot():
    ok = auto_load_default()
    return model_info() if ok else {"loaded": False}


_boot()
if not is_loaded():
    auto_load_default()

ss = st.session_state
ss.setdefault("theme", get_setting("theme", "dark"))
ss.setdefault("threshold", float(get_setting("threshold", 0.5)))
ss.setdefault("strategy", get_setting("strategy", "last_n"))
ss.setdefault("chat", [])
ss.setdefault("manual_text", "")


def _palette() -> dict:
    """Theme tokens for both modes — single source of truth for all UI colors."""
    if ss.theme == "dark":
        return dict(bg="#0b1220", card="#141d2e", card2="#1b2538", text="#e8eefb",
                    sub="#93a4c0", border="#26344d", grid="rgba(255,255,255,.06)",
                    accent="#7c3aed", accent2="#2563eb")
    return dict(bg="#f5f7fb", card="#ffffff", card2="#f1f5f9", text="#0f172a",
                sub="#5a6b86", border="#e2e8f0", grid="rgba(15,23,42,.06)",
                accent="#7c3aed", accent2="#2563eb")


def inject_css():
    p = _palette()
    st.markdown(f"""<style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      html, body, .stApp, [class*="css"] {{ font-family:'Inter',system-ui,sans-serif; }}
      .stApp {{ background:{p['bg']}; color:{p['text']}; }}
      .block-container {{ padding-top:2.2rem; padding-bottom:3.5rem; max-width:1400px; }}
      section[data-testid="stSidebar"] {{ background:{p['card']}; border-right:1px solid {p['border']}; }}
      section[data-testid="stSidebar"] .stRadio label {{ font-weight:500; }}
      h1,h2,h3,h4 {{ color:{p['text']}; letter-spacing:-.01em; }}
      ::-webkit-scrollbar {{ width:9px; height:9px; }}
      ::-webkit-scrollbar-thumb {{ background:{p['border']}; border-radius:8px; }}

      /* KPI cards */
      .kpi {{ position:relative; background:{p['card']}; border:1px solid {p['border']};
              border-radius:16px; padding:18px 20px; overflow:hidden;
              box-shadow:0 4px 18px rgba(2,8,23,.10); transition:transform .18s, box-shadow .18s;
              animation:fade .4s ease both; }}
      .kpi:hover {{ transform:translateY(-4px); box-shadow:0 10px 26px rgba(2,8,23,.18); }}
      .kpi::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:5px;
                      background:linear-gradient(180deg,var(--a1),var(--a2)); }}
      .kpi .top {{ display:flex; justify-content:space-between; align-items:center; }}
      .kpi .label {{ color:{p['sub']}; font-size:.74rem; font-weight:600; text-transform:uppercase;
                     letter-spacing:.06em; }}
      .kpi .icon {{ font-size:1.25rem; opacity:.9; }}
      .kpi .value {{ color:{p['text']}; font-size:1.9rem; font-weight:800; margin-top:6px; line-height:1.1; }}
      .kpi .delta {{ font-size:.78rem; margin-top:3px; font-weight:600; }}

      /* badges */
      .badge {{ display:inline-block; padding:7px 18px; border-radius:999px; font-weight:700; font-size:1rem; }}
      .badge-theft {{ background:rgba(239,68,68,.14); color:#ef4444; border:1px solid rgba(239,68,68,.4); }}
      .badge-normal {{ background:rgba(34,197,94,.14); color:#22c55e; border:1px solid rgba(34,197,94,.4); }}
      .badge.pulse {{ animation:pulse 1.4s ease-in-out infinite; }}

      /* hero / executive header */
      .hero {{ position:relative; background:linear-gradient(120deg,#111c3a 0%,#3b1d7a 55%,#7c3aed 100%);
               border-radius:20px; padding:28px 34px; color:#fff; margin-bottom:22px; overflow:hidden;
               box-shadow:0 12px 34px rgba(76,29,149,.35); animation:fade .5s ease both; }}
      .hero::after {{ content:""; position:absolute; right:-40px; top:-40px; width:220px; height:220px;
                      background:radial-gradient(circle,rgba(255,255,255,.18),transparent 70%); }}
      .hero h1 {{ margin:0; font-size:1.85rem; font-weight:800; color:#fff; }}
      .hero p {{ margin:6px 0 0; opacity:.92; font-size:.98rem; }}

      .pill {{ background:{p['card2']}; border:1px solid {p['border']}; border-radius:8px;
               padding:4px 11px; font-size:.74rem; color:{p['sub']}; font-weight:500; }}
      .sb-group {{ color:{p['sub']}; font-size:.7rem; font-weight:700; text-transform:uppercase;
                   letter-spacing:.08em; margin:10px 2px 2px; }}
      .mcard {{ background:{p['card2']}; border:1px solid {p['border']}; border-radius:14px;
                padding:14px 16px; margin-top:8px; }}
      .mcard .row {{ display:flex; justify-content:space-between; font-size:.8rem; padding:3px 0;
                     color:{p['sub']}; }} .mcard .row b {{ color:{p['text']}; font-weight:600; }}
      .dot {{ height:9px; width:9px; border-radius:50%; display:inline-block; margin-right:6px; }}

      /* callouts */
      .callout {{ border-radius:12px; padding:13px 16px; margin:8px 0; font-weight:500;
                  border:1px solid; animation:fade .3s ease both; }}
      .c-ok {{ background:rgba(34,197,94,.10); border-color:rgba(34,197,94,.35); color:#22c55e; }}
      .c-err {{ background:rgba(239,68,68,.10); border-color:rgba(239,68,68,.35); color:#ef4444; }}
      .c-warn {{ background:rgba(245,158,11,.10); border-color:rgba(245,158,11,.35); color:#f59e0b; }}
      .c-info {{ background:rgba(37,99,235,.10); border-color:rgba(37,99,235,.35); color:{p['accent2']}; }}

      /* skeleton + footer + tables */
      .skel {{ height:96px; border-radius:16px; background:linear-gradient(90deg,{p['card']} 25%,
               {p['card2']} 37%,{p['card']} 63%); background-size:400% 100%;
               animation:shimmer 1.3s infinite; }}
      .footer {{ text-align:center; color:{p['sub']}; font-size:.78rem; margin-top:34px;
                 padding-top:16px; border-top:1px solid {p['border']}; }}
      [data-testid="stDataFrame"] {{ border:1px solid {p['border']}; border-radius:12px; }}
      .stButton>button {{ border-radius:10px; font-weight:600; transition:all .15s; }}
      .stButton>button:hover {{ transform:translateY(-1px); }}
      .stTabs [data-baseweb="tab-list"] {{ gap:6px; }}
      .stTabs [data-baseweb="tab"] {{ border-radius:10px 10px 0 0; font-weight:600; }}

      @keyframes fade {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:none; }} }}
      @keyframes shimmer {{ 0% {{ background-position:100% 0; }} 100% {{ background-position:-100% 0; }} }}
      @keyframes pulse {{ 0%,100% {{ box-shadow:0 0 0 0 currentColor; opacity:1; }}
                          50% {{ box-shadow:0 0 0 6px transparent; opacity:.85; }} }}
    </style>""", unsafe_allow_html=True)


inject_css()
TMPL = "plotly_dark" if ss.theme == "dark" else "plotly_white"


def style_fig(fig, height=320, title=None):
    """Consistent Plotly styling across the whole app."""
    p = _palette()
    fig.update_layout(template=TMPL, height=height, title=title,
                      margin=dict(t=40 if title else 14, b=10, l=10, r=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color=p["text"], size=12),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(gridcolor=p["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=p["grid"], zeroline=False)
    return fig


def kpi(label, value, delta="", color="#3b82f6", icon="📊"):
    a1, a2 = (color, color)
    d = f'<div class="delta" style="color:{color}">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="kpi" style="--a1:{a1};--a2:{a2}">'
        f'<div class="top"><span class="label">{label}</span><span class="icon">{icon}</span></div>'
        f'<div class="value">{value}</div>{d}</div>', unsafe_allow_html=True)


def badge(status, pulse=False):
    cls = "badge-theft" if status == "Theft" else "badge-normal"
    pc = " pulse" if pulse else ""
    return f'<span class="badge {cls}{pc}">{"🔴" if status == "Theft" else "🟢"} {status}</span>'


def callout(kind, msg):
    cls = {"ok": "c-ok", "err": "c-err", "warn": "c-warn", "info": "c-info"}[kind]
    ic = {"ok": "✅", "err": "🚫", "warn": "⚠️", "info": "ℹ️"}[kind]
    st.markdown(f'<div class="callout {cls}">{ic}&nbsp; {msg}</div>', unsafe_allow_html=True)


def risk_gauge(prob: float, threshold: float = 0.5):
    """Power-BI-style radial risk gauge for a single prediction."""
    p = _palette()
    val = prob * 100
    color = "#ef4444" if val >= 75 else "#f59e0b" if val >= 40 else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val, number={"suffix": " /100", "font": {"size": 30}},
        title={"text": "Risk Score", "font": {"size": 14}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": .28},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, 40], "color": "rgba(34,197,94,.18)"},
                         {"range": [40, 75], "color": "rgba(245,158,11,.18)"},
                         {"range": [75, 100], "color": "rgba(239,68,68,.18)"}],
               "threshold": {"line": {"color": p["sub"], "width": 3}, "value": threshold * 100}}))
    fig.update_layout(template=TMPL, height=240, margin=dict(t=40, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color=p["text"]))
    return fig


def skeleton(cols=4):
    cs = st.columns(cols)
    for c in cs:
        c.markdown('<div class="skel"></div>', unsafe_allow_html=True)


def footer():
    st.markdown(
        f'<div class="footer">⚡ <b>ETD-XAI Enterprise</b> v{APP_VERSION} · '
        f'Electricity Theft Detection using Explainable AI · CNN-LSTM (TensorFlow/Keras) · '
        f'© {datetime.now().year} · MIT License</div>', unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def require_model() -> bool:
    if not is_loaded():
        callout("err", f"<b>{NO_MODEL_MSG}</b>")
        callout("info", "Upload a <code>.keras</code> model on the <b>⚙️ Settings</b> page.")
        return False
    return True


def strategy_selector(key):
    labels = [STRATEGY_LABELS[s] for s in STRATEGIES]
    idx = STRATEGIES.index(ss.strategy) if ss.strategy in STRATEGIES else 0
    chosen = st.selectbox("Length-mapping strategy", labels, index=idx, key=key,
                          help="Applied when uploaded length ≠ the model's expected length.")
    sel = STRATEGIES[labels.index(chosen)]
    ss.strategy = sel; set_setting("strategy", sel)
    return sel


def has_gpu():
    try:
        return len(tf().config.list_physical_devices("GPU")) > 0
    except Exception:
        return False


def parse_readings(text):
    if not text or not text.strip():
        return None
    try:
        return [float(p) for p in re.split(r"[,\s;]+", text.strip()) if p != ""]
    except ValueError:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Pages
# ═════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    hero("📊 Executive Dashboard", "Electricity Theft Detection — CNN-LSTM Explainable AI")
    info = model_info()
    uid = latest_upload_id()
    up = get_upload(uid) if uid else None
    c = st.columns(4)
    with c[0]: kpi("Model Status", "Loaded" if info.get("loaded") else "Not Loaded",
                   info.get("name", ""), "#22c55e" if info.get("loaded") else "#ef4444",
                   "🧠" if info.get("loaded") else "⚠️")
    with c[1]: kpi("Engine", "TF / Keras", f"v{info.get('tf_version', '—')}" if info.get("loaded") else "",
                   "#2563eb", "⚙️")
    with c[2]: kpi("Compute", "GPU" if has_gpu() else "CPU", "Inference device", "#7c3aed", "🖥️")
    with c[3]: kpi("Predictions", f"{counts()['predictions']:,}", "stored in SQLite", "#06b6d4", "🗃️")

    if not up:
        callout("info", "No dataset processed yet — open <b>📦 Batch Prediction</b> to score a dataset.")
        return

    st.markdown("### 📈 Prediction Overview")
    c = st.columns(4)
    with c[0]: kpi("Total Customers", f"{up['total_rows']:,}", "in latest run", "#2563eb", "👥")
    with c[1]: kpi("Normal", f"{up['normal_rows']:,}", "Class 0", "#22c55e", "🟢")
    with c[2]: kpi("Theft", f"{up['theft_rows']:,}", "Class 1", "#ef4444", "🔴")
    with c[3]: kpi("Theft Rate", f"{(up['theft_rate'] or 0) * 100:.1f}%", "of customers", "#f59e0b", "📊")

    if up.get("accuracy") is not None:
        st.markdown("### 🎯 Evaluation Metrics (vs ground-truth FLAG)")
        c = st.columns(5)
        icons = ["✅", "🎯", "🔁", "⚖️", "📐"]
        for col, lbl, key, ic in zip(c, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                                     ["accuracy", "precision_val", "recall_val", "f1_score", "roc_auc"], icons):
            with col: kpi(lbl, f"{up[key]:.3f}" if up.get(key) is not None else "—", "", "#7c3aed", ic)

    df = predictions_df(uid)
    if df.empty:
        return
    g = st.columns(2)
    with g[0]:
        fig = go.Figure(go.Pie(labels=["Normal", "Theft"],
                               values=df["status"].value_counts().reindex(["Normal", "Theft"]).fillna(0).values,
                               hole=.6, marker_colors=["#22c55e", "#ef4444"],
                               textfont=dict(size=14)))
        st.plotly_chart(style_fig(fig, title="Prediction Distribution"), use_container_width=True)
    with g[1]:
        fig = px.histogram(df, x="risk_score", nbins=25, color="status",
                           color_discrete_map={"Normal": "#22c55e", "Theft": "#ef4444"})
        st.plotly_chart(style_fig(fig, title="Risk Distribution"), use_container_width=True)
    g = st.columns(2)
    with g[0]:
        fig = px.histogram(df, x="probability", nbins=30, color="status",
                           color_discrete_map={"Normal": "#22c55e", "Theft": "#ef4444"})
        st.plotly_chart(style_fig(fig, title="Probability Distribution"), use_container_width=True)
    with g[1]:
        cm = up.get("confusion_matrix")
        if cm:
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                            x=["Pred Normal", "Pred Theft"], y=["Actual Normal", "Actual Theft"])
            st.plotly_chart(style_fig(fig, title="Confusion Matrix"), use_container_width=True)
        else:
            st.markdown("##### 🔝 Top 10 Highest-Risk")
            st.dataframe(df.head(10)[["customer_id", "risk_score", "status"]],
                        use_container_width=True, hide_index=True)
    st.markdown("##### 🕒 Recent Predictions")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)


def page_manual():
    hero("🔮 Manual Prediction", "Score a single customer with the active CNN-LSTM model.")
    if not require_model():
        return
    info = model_info()
    T = info.get("seq_len")
    left, right = st.columns(2)
    with left:
        cid = st.text_input("Customer ID", value=f"CUST_{datetime.now().strftime('%H%M%S')}")
        n = st.number_input("Number of readings", 2, 2000, int(T or 26), 1,
                            help=f"Model expects {T or 'variable'} readings; any length is auto-mapped.")
        strat = strategy_selector("m_strat")
        thr = st.slider("Decision threshold", 0.0, 1.0, ss.threshold, 0.01)
        d1, d2 = st.columns(2)
        if d1.button("🟢 Demo: Normal", use_container_width=True):
            ss.manual_text = ", ".join(str(int(v)) for v in
                (2000 + 400 * np.sin(np.linspace(0, 6, int(n))) + np.random.randint(-80, 80, int(n))))
        if d2.button("🔴 Demo: Theft", use_container_width=True):
            a = 2200 + np.random.randint(-50, 50, int(n)); a[int(n) // 2:] = np.random.randint(0, 60, int(n) - int(n) // 2)
            ss.manual_text = ", ".join(str(int(v)) for v in a)
        text = st.text_area("Readings (comma/space separated)", value=ss.manual_text, height=120,
                            placeholder="2401, 2500, 2674, ...")
        go_btn = st.button("⚡ Predict", type="primary", use_container_width=True)
    if go_btn:
        raw = parse_readings(text)
        if not raw or len(raw) < 2:
            with right:
                callout("warn", "Enter at least 2 numeric readings.")
            return
        with right:
            with st.spinner("⚡ Running model.predict()…"):
                res = predict_one(raw, strat, thr)
            save_manual(customer_id=cid, **{k: res[k] for k in
                ("probability", "prediction", "confidence", "risk_score", "status")},
                readings=list(map(float, raw)), threshold=thr, model_name=res["model_name"])
            st.markdown(f"### Result · `{cid}`")
            st.markdown(badge(res["status"], pulse=True), unsafe_allow_html=True)
            gcol, kcol = st.columns([1, 1])
            with gcol:
                st.plotly_chart(risk_gauge(res["probability"], thr), use_container_width=True)
            with kcol:
                kpi("Probability", f"{res['probability'] * 100:.1f}%", "theft likelihood", "#2563eb", "📈")
                kpi("Confidence", f"{res['confidence'] * 100:.1f}%", res["risk_level"] + " risk",
                    {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}[res["risk_level"]], "🎯")
            st.caption(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · model **{res['model_name']}** · "
                       f"{res['uploaded_len']}→{res['model_len']} · strategy `{res['strategy_used']}`")
            fig = go.Figure(go.Scatter(y=raw, mode="lines+markers", line=dict(color="#7c3aed", width=2),
                                       fill="tozeroy", fillcolor="rgba(124,58,237,.12)"))
            st.plotly_chart(style_fig(fig, height=240, title="Consumption Sequence (kWh)"),
                            use_container_width=True)
            with st.expander("🧠 Explainable AI", expanded=True):
                ex = shap_or_ig(raw)
                if ex:
                    st.caption(f"Method: {ex['method']}")
                    fig = go.Figure(go.Bar(y=ex["timestep_importance"], marker_color="#7c3aed"))
                    st.plotly_chart(style_fig(fig, height=220, title="Per-timestep importance"),
                                    use_container_width=True)
                st.markdown("**Risk factors:**")
                for name, val, dr in risk_factors(raw):
                    st.markdown(f"- {name} · `{val:.3f}` · {dr}")


def page_batch():
    hero("📦 Batch Prediction", "Upload a dataset and score every customer.")
    if not require_model():
        return
    src = st.radio("Data source", ["Upload file", "Use bundled sample"], horizontal=True)
    file = None
    if src == "Upload file":
        file = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
    elif SAMPLE_DATASET.exists():
        file = str(SAMPLE_DATASET); st.caption(f"Using `{SAMPLE_DATASET.name}`")
    if not file:
        return
    try:
        df = read_table(file)
    except Exception as e:
        st.error(f"Could not read file: {e}"); return
    info = inspect(df)
    c = st.columns(4)
    c[0].metric("Rows", f"{info['n_rows']:,}"); c[1].metric("Reading cols", info["n_readings"])
    c[2].metric("ID column", info["id_col"] or "auto"); c[3].metric("FLAG", "yes ✅" if info["has_flag"] else "no")
    comp = check_compatibility(info["n_readings"])
    (st.success if comp["compatible"] else st.error)(comp["reason"], icon="ℹ️")
    strat = strategy_selector("b_strat")
    thr = st.slider("Decision threshold", 0.0, 1.0, ss.threshold, 0.01, key="b_thr")
    st.dataframe(df.head(8), use_container_width=True)
    c1, c2 = st.columns(2)
    run = c1.button("⚡ Run Predictions", type="primary", use_container_width=True)
    save = c2.checkbox("Save to database", value=True)
    if not run:
        return
    if info["n_readings"] < 2:
        st.error("No usable reading columns (need ≥ 2 numeric)."); return
    prog = st.progress(0, "Preprocessing…")
    try:
        prog.progress(40, "Running model.predict()…")
        result = run_batch(df, info, strat, thr)
        prog.progress(90, "Aggregating…")
    except Exception as e:
        prog.empty(); st.error(f"Prediction failed: {e}"); return
    if save:
        fname = getattr(file, "name", Path(str(file)).name)
        uid = save_upload(filename=fname, total_rows=result["total_rows"], theft_rows=result["theft_rows"],
            normal_rows=result["normal_rows"], avg_risk=result["avg_risk"], theft_rate=result["theft_rate"],
            has_flag=result["has_flag"], threshold=thr, n_readings=result["n_readings"], strategy=strat,
            **(result["metrics"] or {}))
        save_predictions_bulk(uid, result["rows"])
    prog.progress(100, "Done"); prog.empty()
    st.success(f"Scored {result['total_rows']:,} customers — {result['theft_rows']:,} theft / "
               f"{result['normal_rows']:,} normal.", icon="✅")
    rdf = pd.DataFrame([{k: r[k] for k in ("customer_id", "probability", "prediction",
                        "confidence", "risk_score", "status")} for r in result["rows"]])
    m = st.columns(3)
    m[0].metric("Theft detected", f"{result['theft_rows']:,}")
    m[1].metric("Theft rate", f"{result['theft_rate'] * 100:.1f}%")
    m[2].metric("Avg risk", f"{result['avg_risk']:.0f}/100")
    if result["metrics"]:
        mm = result["metrics"]
        st.info(f"Accuracy {mm['accuracy']:.3f} · Precision {mm['precision_val']:.3f} · "
                f"Recall {mm['recall_val']:.3f} · F1 {mm['f1_score']:.3f}"
                + (f" · ROC-AUC {mm['roc_auc']:.3f}" if mm.get("roc_auc") else ""))
    flt = st.selectbox("Filter", ["All", "Theft only", "Normal only"])
    show = rdf if flt == "All" else rdf[rdf.status == flt.split()[0]]
    st.dataframe(show.sort_values("risk_score", ascending=False), use_container_width=True,
                hide_index=True, height=380)
    st.markdown("##### Export")
    e = st.columns(3)
    e[0].download_button("⬇️ CSV", to_csv(rdf), "predictions.csv", "text/csv", use_container_width=True)
    e[1].download_button("⬇️ Excel", to_excel(rdf), "predictions.xlsx", use_container_width=True)
    pdf = to_pdf("ETD-XAI Prediction Report", model_info(), result, result.get("metrics"), rdf)
    if pdf:
        e[2].download_button("⬇️ PDF", pdf, "etd_xai_report.pdf", "application/pdf", use_container_width=True)


def page_history():
    hero("📜 Prediction History", "Every manual prediction, persisted in SQLite.")
    rows = get_manual(500)
    if not rows:
        st.info("No predictions recorded yet."); return
    hdf = pd.DataFrame([{k: r[k] for k in ("customer_id", "probability", "prediction",
        "confidence", "risk_score", "status", "predicted_at", "model_name", "source")} for r in rows])
    f = st.columns(3)
    q = f[0].text_input("Search Customer ID")
    sf = f[1].selectbox("Status", ["All", "Theft", "Normal"])
    src = f[2].selectbox("Source", ["All"] + sorted(hdf["source"].dropna().unique().tolist()))
    show = hdf
    if q: show = show[show.customer_id.astype(str).str.contains(q, case=False, na=False)]
    if sf != "All": show = show[show.status == sf]
    if src != "All": show = show[show.source == src]
    st.dataframe(show, use_container_width=True, hide_index=True, height=460)
    st.download_button("⬇️ Export CSV", to_csv(show), "history.csv", "text/csv")


def page_reports():
    hero("📑 Reports", "Generate a professional PDF / Excel / CSV report.")
    uid = latest_upload_id()
    up = get_upload(uid) if uid else None
    if not up:
        st.info("Run a batch prediction first."); return
    df = predictions_df(uid)
    st.markdown(f"#### Latest run — `{up['filename']}` ({up['upload_time'][:19]})")
    c = st.columns(4)
    c[0].metric("Customers", f"{up['total_rows']:,}"); c[1].metric("Theft", f"{up['theft_rows']:,}")
    c[2].metric("Normal", f"{up['normal_rows']:,}"); c[3].metric("Theft rate", f"{(up['theft_rate'] or 0) * 100:.1f}%")
    metrics = None
    if up.get("accuracy") is not None:
        metrics = {k: up[k] for k in ("accuracy", "precision_val", "recall_val", "f1_score", "roc_auc")}
        metrics["confusion_matrix"] = up.get("confusion_matrix")
    summary = {k: up[k] for k in ("total_rows", "normal_rows", "theft_rows", "theft_rate", "avg_risk")}
    e = st.columns(3)
    e[0].download_button("⬇️ CSV", to_csv(df), "report.csv", "text/csv", use_container_width=True)
    e[1].download_button("⬇️ Excel", to_excel(df), "report.xlsx", use_container_width=True)
    pdf = to_pdf("ETD-XAI Enterprise Report", model_info(), summary, metrics, df)
    if pdf:
        e[2].download_button("⬇️ PDF", pdf, "etd_xai_report.pdf", "application/pdf", use_container_width=True)
    else:
        e[2].caption("Install reportlab for PDF.")


def page_copilot():
    hero("🤖 AI Copilot", "Project-scoped assistant — model, metrics & predictions only.")
    st.caption("Quick questions:")
    cols = st.columns(3)
    for i, sg in enumerate(SUGGESTIONS):
        if cols[i % 3].button(sg, use_container_width=True, key=f"sg{i}"):
            ss.chat.append(("user", sg)); ss.chat.append(("assistant", copilot_answer(sg)))
    for role, msg in ss.chat:
        with st.chat_message(role):
            st.markdown(msg)
    q = st.chat_input("Ask about the model, a metric, or a prediction…")
    if q:
        ss.chat.append(("user", q)); ss.chat.append(("assistant", copilot_answer(q))); st.rerun()


def page_settings():
    hero("⚙️ Settings", "Model & dataset management, defaults, and verification.")
    info = model_info()
    st.markdown("### Active Model")
    if info.get("loaded"):
        st.markdown('<span class="badge badge-normal">🟢 Status: Loaded</span>', unsafe_allow_html=True)
        c = st.columns(3)
        c[0].metric("Active Model Name", info["name"]); c[1].metric("Parameters", info["total_params_fmt"])
        c[2].metric("Architecture", info["architecture"])
        c = st.columns(4)
        c[0].metric("Input Shape", info["input_shape"]); c[1].metric("Output Shape", info["output_shape"])
        c[2].metric("Seq length", info["seq_len"] or "variable"); c[3].metric("Stat features", info["stat_size"])
        c = st.columns(3)
        c[0].metric("TensorFlow", info["tf_version"]); c[1].metric("Keras", info["keras_version"])
        c[2].metric("Upload date", (info.get("upload_time") or "—")[:19].replace("T", " "))
        st.caption("Load: `tensorflow.keras.models.load_model()` · Inference: `model.predict()` · "
                   "Exclusive engine — no fallback models.")
        with st.expander("Model architecture (summary)"):
            st.code(info["summary"], language="text")
    else:
        st.markdown('<span class="badge badge-theft">🔴 Status: Not Loaded</span>', unsafe_allow_html=True)
        st.error(NO_MODEL_MSG, icon="🚫")

    st.divider()
    st.markdown("### Upload / Replace Model")
    up = st.file_uploader("CNN-LSTM model (.keras / .h5)", type=["keras", "h5"])
    if up and st.button("Activate uploaded model", type="primary"):
        path = UPLOAD_DIR / up.name
        path.write_bytes(up.getbuffer())
        try:
            unload_model(); load_model(str(path), up.name)
            set_setting("active_model_path", str(path)); st.cache_resource.clear()
            st.success(f"Activated **{up.name}**.", icon="✅"); st.rerun()
        except Exception as e:
            st.error(f"Rejected: {e}")
    if info.get("loaded") and Path(info["path"]) != DEFAULT_MODEL and st.button("Restore default model"):
        unload_model(); set_setting("active_model_path", None)
        auto_load_default(); st.cache_resource.clear(); st.rerun()

    st.divider()
    st.markdown("### Dataset")
    ds = st.file_uploader("Upload a dataset to persist as the working set", type=["csv", "xlsx", "xls"], key="ds_up")
    if ds and st.button("Save dataset"):
        path = UPLOAD_DIR / ds.name
        path.write_bytes(ds.getbuffer()); set_setting("active_dataset_path", str(path))
        st.success(f"Saved `{ds.name}`. Use it in Batch Prediction.", icon="✅")
    active_ds = get_setting("active_dataset_path")
    st.caption(f"Active dataset: `{Path(active_ds).name if active_ds else 'bundled sample'}`")

    st.divider()
    st.markdown("### Defaults & Theme")
    thr = st.slider("Default decision threshold", 0.0, 1.0, ss.threshold, 0.01)
    if thr != ss.threshold:
        ss.threshold = thr; set_setting("threshold", thr); st.toast(f"Threshold → {thr:.2f}")
    t1, t2 = st.columns(2)
    if t1.button("🌙 Dark theme", use_container_width=True):
        ss.theme = "dark"; set_setting("theme", "dark"); st.rerun()
    if t2.button("☀️ Light theme", use_container_width=True):
        ss.theme = "light"; set_setting("theme", "light"); st.rerun()

    st.divider()
    st.markdown("### Verification Status")
    st.json({"model_loaded": info.get("loaded", False), "active_model": info.get("name"),
             "load_method": "tensorflow.keras.models.load_model(path)", "predict_method": "model.predict(x)",
             "exclusive_engine": True,
             "fallback_models": "none — CNN-LSTM only (no RF/XGBoost/LightGBM/CatBoost/LogReg/DT/SVM/KNN/rule/mock)",
             "ground_truth_use": "evaluation metrics only — never used for prediction",
             "compute": "GPU" if has_gpu() else "CPU", "database": str(DB_PATH),
             "last_prediction": E.last_prediction or None})


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Sidebar navigation + router
# ═════════════════════════════════════════════════════════════════════════════
# Grouped navigation — page → (group, function)
NAV = {
    "📊 Dashboard": ("Overview", page_dashboard),
    "🔮 Manual Prediction": ("Predict", page_manual),
    "📦 Batch Prediction": ("Predict", page_batch),
    "📜 History": ("Insights", page_history),
    "📑 Reports": ("Insights", page_reports),
    "🤖 AI Copilot": ("Insights", page_copilot),
    "⚙️ Settings": ("System", page_settings),
}
GROUP_ORDER = ["Overview", "Predict", "Insights", "System"]

with st.sidebar:
    cols = st.columns([1, 3])
    if LOGO.exists():
        cols[0].image(str(LOGO), width=54)
    cols[1].markdown(f"### ⚡ ETD-XAI\n<span class='pill'>Enterprise v{APP_VERSION}</span>",
                     unsafe_allow_html=True)

    info = model_info()
    online = info.get("loaded")
    dot = "#22c55e" if online else "#ef4444"
    st.markdown(
        f"<div class='mcard'>"
        f"<div class='row'><span><span class='dot' style='background:{dot}'></span>"
        f"Model</span><b>{'Loaded' if online else 'Not loaded'}</b></div>"
        + (f"<div class='row'><span>Name</span><b>{info['name']}</b></div>"
           f"<div class='row'><span>Architecture</span><b>{info['architecture']}</b></div>"
           f"<div class='row'><span>Input</span><b>{info['input_shape']}</b></div>"
           f"<div class='row'><span>Params</span><b>{info['total_params_fmt']}</b></div>"
           f"<div class='row'><span>TensorFlow</span><b>v{info['tf_version']}</b></div>"
           f"<div class='row'><span>Compute</span><b>{'GPU' if has_gpu() else 'CPU'}</b></div>"
           if online else f"<div class='row'><span>{NO_MODEL_MSG}</span><b></b></div>")
        + "</div>", unsafe_allow_html=True)

    # Grouped radio: build a flat list with group separators rendered above.
    page_keys = list(NAV.keys())
    if "nav_choice" not in ss:
        ss.nav_choice = page_keys[0]
    st.markdown("<div class='sb-group'>Navigation</div>", unsafe_allow_html=True)
    for grp in GROUP_ORDER:
        items = [k for k, (g, _) in NAV.items() if g == grp]
        if not items:
            continue
        st.markdown(f"<div class='sb-group'>{grp}</div>", unsafe_allow_html=True)
        for k in items:
            if st.button(k, use_container_width=True, key=f"nav_{k}",
                         type="primary" if ss.nav_choice == k else "secondary"):
                ss.nav_choice = k
                st.rerun()

    st.divider()
    cc = counts()
    st.caption(f"🗃️ SQLite · {cc['predictions']} preds · {cc['manual']} manual")

NAV[ss.nav_choice][1]()
footer()
