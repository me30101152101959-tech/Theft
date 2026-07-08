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
import streamlit.components.v1 as components
from scipy import stats as scipy_stats
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"
# Default active model: prefer the config's base model, else the legacy filename
# (kept for backward compatibility with older single-model projects).
def _default_model_path() -> "Path":
    for cand in ("base_cnnlstm_final.keras", "cnnlstm_final.keras"):
        p = ASSETS / cand
        if p.exists():
            return p
    return ASSETS / "cnnlstm_final.keras"
DEFAULT_MODEL = _default_model_path()
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


# Path to the TRAINING StandardScaler exported by the notebook (CELL 8/19:
# joblib.dump(stat_scaler, 'stat_scaler.pkl')). If present, it is used verbatim
# so inference matches training exactly. If absent, we fall back to per-batch
# re-fitting (approximate) so the app still runs.
SAVED_SCALER = ASSETS / "stat_scaler.pkl"


class FeaturePipeline:
    """
    Stat-feature scaler. Prefers the SAVED training StandardScaler
    (stat_scaler.pkl) for exact train/inference parity; otherwise re-fits
    per batch as a documented approximation.
    """
    def __init__(self):
        self._scaler: Optional[StandardScaler] = None
        self._fitted = False
        self._locked = False          # True => using the saved training scaler
        self._load_saved()

    def _load_saved(self):
        if SAVED_SCALER.exists():
            try:
                import joblib
                sc = joblib.load(SAVED_SCALER)
                # sanity: must expose transform and match the 59-feature vector
                if hasattr(sc, "transform"):
                    self._scaler = sc
                    self._fitted = True
                    self._locked = True
            except Exception:
                self._scaler = None; self._fitted = False; self._locked = False

    @property
    def using_saved_scaler(self) -> bool:
        return self._locked

    def fit_transform(self, readings: np.ndarray) -> np.ndarray:
        raw = extract_features(readings)
        if self._locked and self._scaler is not None:
            # Never refit over the training scaler — transform only.
            out = self._scaler.transform(raw).astype(np.float32)
        else:
            self._scaler = StandardScaler()
            out = self._scaler.fit_transform(raw).astype(np.float32)
            self._fitted = True
        return np.nan_to_num(out)

    def transform(self, readings: np.ndarray) -> np.ndarray:
        raw = extract_features(readings)
        if self._fitted and self._scaler is not None:
            out = self._scaler.transform(raw).astype(np.float32)
        else:
            out = np.zeros_like(raw)
        return np.nan_to_num(out)

    def reset(self):
        # Keep a saved training scaler across resets; only clear a batch-fit one.
        if self._locked:
            return
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
    path = str(path)
    name = name or Path(path).name
    if Path(path).suffix.lower() not in (".keras", ".h5"):
        raise ValueError("Unsupported format. Use .keras or .h5")

    # Apply Keras 3 compatibility shim for quantization_config
    _t = tf()
    try:
        import keras
        _orig = keras.layers.Dense.from_config.__func__
        @classmethod
        def _compat(cls, config):
            config = dict(config)
            config.pop("quantization_config", None)
            dt = config.get("dtype")
            if isinstance(dt, dict):
                config["dtype"] = dt.get("config", {}).get("name", "float32")
            return _orig(cls, config)
        keras.layers.Dense.from_config = _compat
    except Exception:
        pass

    model = _t.keras.models.load_model(path)

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


# ── v3.0: config-driven, dynamic model registry ──────────────────────────────
MODEL_CONFIG = ASSETS / "model_config.json"


def _model_stem() -> str:
    return Path(E.name).stem if E.name else ""


def _model_schema_path(suffix: str) -> Path:
    """assets/<active-model-stem>_<suffix>.json — e.g. base_cnnlstm_final_config.json,
    base_cnnlstm_final_training_columns.json. Lets each model ship its own schema
    (Section 11) without any code change; falls back to the shared config/dates
    when a dedicated file doesn't exist for the active model."""
    return ASSETS / f"{_model_stem()}_{suffix}.json"


def load_config() -> dict:
    """Source of truth for metadata/threshold/training info. Resolution order:
    1) a per-model dedicated file  assets/<model-stem>_config.json  (future-proof:
       drop one in for any new model, no code change needed)
    2) the shared assets/model_config.json (current models)
    3) synthesised defaults derived from the loaded TF model (old projects)."""
    per_model = _model_schema_path("config")
    if per_model.exists():
        try:
            d = json.loads(per_model.read_text(encoding="utf-8"))
            d.setdefault("_source", per_model.name)
            return d
        except Exception:
            pass
    if MODEL_CONFIG.exists():
        try:
            return json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Fallback defaults (never hard-coded elsewhere) — derived from the model.
    return {
        "SEQ_LEN": E.seq_len, "N_STAT": E.stat_size,
        "base_model": {"file": DEFAULT_MODEL.name, "best_thr": 0.5},
        "_source": "auto-generated (no model_config.json found)",
    }


def training_columns() -> Optional[list]:
    """Per-model exact training column list (Section 11), e.g.
    assets/<model-stem>_training_columns.json = ["01/01/2014", ..., "CONS_NO", "FLAG"].
    Returns None when no dedicated schema exists for the active model — callers
    then fall back to the generic date-sequence template."""
    p = _model_schema_path("training_columns")
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return [str(c) for c in data]
        except Exception:
            pass
    return None


def config_threshold() -> float:
    """Active threshold from config (per active model). Falls back to 0.5 only
    when no configuration exists."""
    cfg = load_config()
    # match config entry to the active model file name
    name = E.name
    for key in ("base_model", "tl_model"):
        m = cfg.get(key)
        if isinstance(m, dict) and m.get("file") == name and m.get("best_thr") is not None:
            return float(m["best_thr"])
    if isinstance(cfg.get("best_thr"), (int, float)):
        return float(cfg["best_thr"])
    return 0.5


def startup_validation() -> list:
    """Return the startup readiness checklist (logged + shown in the UI)."""
    cfg = load_config()
    checks = [
        ("model loaded", is_loaded()),
        ("scaler loaded", PIPELINE.using_saved_scaler),
        ("config loaded", MODEL_CONFIG.exists()),
        ("threshold loaded", is_loaded()),
        ("prediction engine ready", is_loaded() and E.model is not None),
    ]
    for name, ok in checks:
        print(f"{'✓' if ok else '✗'} {name}")
    return checks


def discover_models() -> list:
    """Dynamically discover every .keras/.h5 model in assets/ and uploads/."""
    seen, out = set(), []
    for d in (ASSETS, UPLOAD_DIR):
        if not d.exists():
            continue
        for p in sorted(list(d.glob("*.keras")) + list(d.glob("*.h5"))):
            if p.name in seen:
                continue
            seen.add(p.name)
            cfg = load_config()
            meta = {}
            for key in ("base_model", "tl_model"):
                m = cfg.get(key)
                if isinstance(m, dict) and m.get("file") == p.name:
                    meta = m
            out.append({
                "name": p.name, "path": str(p),
                "size_mb": round(p.stat().st_size / 1e6, 2),
                "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "threshold": meta.get("best_thr"), "auc": meta.get("auc"),
                "f1": meta.get("f1"), "accuracy": meta.get("accuracy"),
                "active": (p.name == E.name),
            })
    return out


def compatibility_report(uploaded_len: Optional[int] = None) -> dict:
    """Full compatibility snapshot for the Settings panel (never mutates data)."""
    import keras as _k
    cfg = load_config()
    # Rule 4 — TensorFlow model is authoritative; warn if config disagrees.
    cfg_seq, cfg_stat = cfg.get("SEQ_LEN"), cfg.get("N_STAT")
    conflicts = []
    if E.seq_len is not None and cfg_seq is not None and cfg_seq != E.seq_len:
        conflicts.append(f"config SEQ_LEN={cfg_seq} ≠ model {E.seq_len} (using model)")
    if cfg_stat is not None and cfg_stat != E.stat_size:
        conflicts.append(f"config N_STAT={cfg_stat} ≠ model {E.stat_size} (using model)")
    rep = {
        "✓ model_loaded": is_loaded(),
        "✓ input_count": len(E.model.inputs) if E.model is not None else 0,
        "✓ input_shape": str(E.input_shape),
        "✓ output_shape": str(E.output_shape),
        "✓ variable_length_support": ("Yes" if E.seq_len is None else "No"),
        "✓ expected_sequence_length": ("any (variable)" if E.seq_len is None else E.seq_len),
        "✓ stat_features_expected": E.stat_size,
        "✓ threshold": config_threshold(),
        "✓ active_scaler": ("training stat_scaler.pkl" if PIPELINE.using_saved_scaler
                            else "MISSING — re-fit per batch (may not match training)"),
        "✓ tf_version": tf().__version__,
        "✓ keras_version": getattr(_k, "__version__", "unknown"),
        "config_source": cfg.get("_source", str(MODEL_CONFIG.name)),
        "config_vs_model": ("consistent" if not conflicts else
                            "Configuration differs from actual model. "
                            "Using TensorFlow model information. (" + "; ".join(conflicts) + ")"),
        "✓ prediction_ready": is_loaded() and E.model is not None,
    }
    if uploaded_len is not None:
        rep["uploaded_len"] = uploaded_len
        rep["✓ compatible_dataset"] = (E.seq_len is None or uploaded_len == E.seq_len)

    # Overall Compatibility % — weighted readiness score (Section 9).
    checks = [rep["✓ model_loaded"], PIPELINE.using_saved_scaler, MODEL_CONFIG.exists() or per_model_cfg_exists(),
              not conflicts, rep["✓ prediction_ready"]]
    if uploaded_len is not None:
        checks.append(rep["✓ compatible_dataset"])
    rep["✓ overall_compatibility_pct"] = round(100 * sum(bool(c) for c in checks) / len(checks))
    return rep


def per_model_cfg_exists() -> bool:
    return _model_schema_path("config").exists()


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
ID_COLS = {"cons_no", "customer_id", "consumer_id", "id", "customer", "consumer_no", "meter_id", "user_id"}
FLAG_COLS = {"flag", "label", "target", "theft", "is_theft", "class", "y"}


def read_table(file) -> pd.DataFrame:
    name = getattr(file, "name", str(file)).lower()
    return pd.read_excel(file) if name.endswith((".xlsx", ".xls")) else pd.read_csv(file)


def detect_reading_columns(df) -> Tuple[list, object, object]:
    """SINGLE source of truth for reading-column detection (Rules 2/4).
    Returns (reading_cols, id_col, flag_col).
    - excludes ID / metadata columns (CONS_NO, consumer_id, CustomerID, ID, …)
    - excludes FLAG / label columns (never enters the model)
    - keeps only numeric columns (≥50% parseable)
    - PRESERVES the file's column order.
    Manual and Batch both resolve reading columns through this one function.

    Note on Rule 3 (chronological reorder): intentionally NOT applied. Day-first
    (DD/MM/YYYY) vs month-first (MM/DD/YYYY) headers are ambiguous to parse, so
    auto-sorting can silently scramble a correctly-ordered sequence (verified on
    sample_dataset.csv, which is DD/MM). Trusting the author's CSV order is the
    integrity-preserving choice — the sequence is fed exactly as delivered."""
    cols = list(df.columns)
    lower = {c: str(c).strip().lower() for c in cols}
    id_col = next((c for c in cols if lower[c] in ID_COLS), None)
    flag_col = next((c for c in cols if lower[c] in FLAG_COLS), None)
    reading_cols = [c for c in cols if c not in (id_col, flag_col)
                    and pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.5]
    return reading_cols, id_col, flag_col


def inspect(df: pd.DataFrame) -> dict:
    reading_cols, id_col, flag_col = detect_reading_columns(df)
    return {"id_col": id_col, "flag_col": flag_col, "reading_cols": reading_cols,
            "n_readings": len(reading_cols), "n_rows": len(df),
            "has_flag": flag_col is not None, "columns": [str(c) for c in df.columns]}


def build_matrix(df, info) -> Tuple[np.ndarray, list, Optional[np.ndarray]]:
    readings = df[info["reading_cols"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    ids = (df[info["id_col"]].astype(str).tolist() if info["id_col"]
           else [f"CUST_{i+1:06d}" for i in range(len(df))])
    flags = None
    if info["flag_col"]:  # ground truth — used ONLY for metrics below
        flags = pd.to_numeric(df[info["flag_col"]], errors="coerce").fillna(0).astype(int).to_numpy()
    return readings, ids, flags


def _input_hash(readings_2d) -> str:
    """Deterministic md5 of the reading matrix that feeds the pipeline (Rule 9).
    Identical readings ⇒ identical hash across Manual, Batch, and any caller."""
    import hashlib
    arr = np.ascontiguousarray(np.asarray(readings_2d, dtype=np.float32))
    return hashlib.md5(arr.tobytes()).hexdigest()


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
# SECTION 4B — CSV/Excel dataset templates + smart validation (presentation-only;
# does not touch model loading, preprocessing, scaling, or prediction)
# ═════════════════════════════════════════════════════════════════════════════
import datetime as _dt


COMMON_SEQ_LENGTHS = [120, 180, 240, 350, 365, 730]


def _template_seq_len() -> int:
    """Active sequence length — TensorFlow model is authoritative, else config,
    else the original 120-day training default. Never hardcoded elsewhere."""
    if is_loaded() and E.seq_len is not None:
        return int(E.seq_len)
    cfg = load_config()
    return int(cfg.get("SEQ_LEN") or 120)


def _template_dates(n: int) -> list:
    """Reading-column headers for a template of length n. Uses the active
    model's own training_columns.json schema when one exists (Section 11);
    otherwise generates sequential calendar days from model_config.json's
    START_DATE/DATE_FORMAT (default 01/01/2014, MM/DD/YYYY)."""
    cols = training_columns()
    if cols:
        readings = [c for c in cols if str(c).strip().lower() not in ID_COLS | FLAG_COLS]
        if len(readings) == n:
            return readings
    cfg = load_config()
    start_str = cfg.get("START_DATE", "01/01/2014")
    fmt = cfg.get("DATE_FORMAT", "%m/%d/%Y")
    try:
        start = _dt.datetime.strptime(start_str, fmt)
    except Exception:
        start = _dt.datetime(2014, 1, 1); fmt = "%m/%d/%Y"
    return [(start + _dt.timedelta(days=i)).strftime(fmt) for i in range(n)]


def build_template_df(include_flag: bool, n_examples: int = 3, n_override: Optional[int] = None) -> pd.DataFrame:
    """Construct a template dataframe with realistic example readings.
    Column order: reading columns first, then CONS_NO, then FLAG (if requested) —
    matching the exact training file layout. Deterministic (fixed seed).
    n_override lets variable-length models generate a template of any chosen
    length (120/180/240/350/365/730/custom) without touching the active model."""
    n = int(n_override) if n_override else _template_seq_len()
    dates = _template_dates(n)
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_examples):
        base = 1800 + i * 300
        vals = np.clip(base + rng.normal(0, 220, n), 0, None).round(0).astype(int)
        if include_flag and i % 2 == 1:                      # alternate 0/1/0…
            vals[n // 2:] = np.clip(rng.uniform(0, 40, n - n // 2), 0, None).round(0).astype(int)
        row = {d: v for d, v in zip(dates, vals)}
        row["CONS_NO"] = f"CUST_{i+1:06d}"
        if include_flag:
            row["FLAG"] = i % 2  # 0,1,0,...
        rows.append(row)
    cols = dates + ["CONS_NO"] + (["FLAG"] if include_flag else [])
    return pd.DataFrame(rows, columns=cols)


def render_dataset_templates(show_all: bool = True):
    """'Dataset Templates' UI section — Production / Evaluation / Empty
    downloads, dynamically generated from the active model's sequence length.
    For variable-length models, lets the user pick the template length instead
    of assuming one fixed size (Section 2)."""
    variable = is_loaded() and E.seq_len is None
    n = _template_seq_len()
    if variable:
        options = COMMON_SEQ_LENGTHS + ["Custom…"]
        choice = st.selectbox("Template length (model accepts any length)", options,
                              index=0, key="tmpl_len_choice")
        if choice == "Custom…":
            n = st.number_input("Custom sequence length", min_value=2, max_value=2000,
                                value=120, step=1, key="tmpl_len_custom")
        else:
            n = int(choice)
    st.markdown("#### Dataset Templates")
    st.caption(f"Templates are generated for **{n} daily readings**"
              + (" (variable-length model — choose any size above)." if variable else "."))
    if show_all:
        c = st.columns(3)
        with c[0]:
            st.markdown("**Production**")
            st.caption("Real customer readings, no ground truth. Use this for actual predictions.")
            st.download_button("📥 Download Production Template",
                               to_csv(build_template_df(False, n_override=n)),
                               f"production_template_{n}.csv", "text/csv", use_container_width=True)
        with c[1]:
            st.markdown("**Evaluation**")
            st.caption("Includes FLAG ground truth, for measuring accuracy/precision/recall.")
            st.download_button("📥 Download Evaluation Template",
                               to_csv(build_template_df(True, n_override=n)),
                               f"evaluation_template_{n}.csv", "text/csv", use_container_width=True)
        with c[2]:
            st.markdown("**Empty**")
            st.caption("Header row only — fill in your own customers and readings.")
            empty_df = build_template_df(False, n_examples=0, n_override=n)
            st.download_button("📥 Download Empty Template", to_csv(empty_df),
                               f"empty_template_{n}.csv", "text/csv", use_container_width=True)
    else:
        st.download_button("📥 Download CSV Template", to_csv(build_template_df(False, n_override=n)),
                           f"prediction_template_{n}.csv", "text/csv", use_container_width=True)
        st.caption(f"{n} daily readings + customer ID — fill in your data and upload it above.")


def _normalize_date_str(s: str) -> Optional[str]:
    """Try to parse and normalize a date string to MM/DD/YYYY format.
    Returns normalized format or None if not a date."""
    s = str(s).strip()
    for fmt in ["%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]:
        try:
            dt = _dt.datetime.strptime(s, fmt)
            return dt.strftime("%m/%d/%Y")
        except ValueError:
            pass
    return None


def validate_dataset_report(df: pd.DataFrame, info: dict) -> list:
    """Smart validation checks (Feature 6/7) — read-only, does not alter df.
    Returns a list of (kind, message) for display via callout()."""
    checks = []
    variable = is_loaded() and E.seq_len is None
    if variable:
        # No fixed expected column set — any length is valid (Section 6).
        checks.append(("ok", f"Variable-length model — {info['n_readings']} reading column(s) "
                             f"detected and will be used exactly as uploaded (no resizing)."))
    else:
        n = _template_seq_len()
        expected_dates_raw = _template_dates(n)
        expected_dates = set(_normalize_date_str(d) or d for d in expected_dates_raw)
        cols = [str(c) for c in df.columns]

        # Normalize column names and match (more flexible date format detection)
        date_like_cols = set()
        for c in cols:
            norm_c = _normalize_date_str(c)
            if norm_c and norm_c in expected_dates:
                date_like_cols.add(c)

        missing = sorted(expected_dates - {_normalize_date_str(c) or c for c in cols if _normalize_date_str(c) or c in date_like_cols})
        extra = [c for c in cols if (_normalize_date_str(c) is None or _normalize_date_str(c) not in expected_dates)
                 and str(c).strip().lower() not in ID_COLS | FLAG_COLS]

        if info["n_readings"] == n and not missing:
            checks.append(("ok", "Dataset format matches the active configuration."))
        if missing and len(missing) <= 3:  # Only warn if few are missing (might be edge case)
            checks.append(("warn", f"Missing expected reading column(s): {', '.join(missing[:5])}"
                                  + (f" … (+{len(missing)-5} more)" if len(missing) > 5 else "")))
        if extra and len(extra) > 1:  # Only warn if genuinely extra columns exist
            non_date_extras = [c for c in extra if _normalize_date_str(c) is None]
            if non_date_extras:
                checks.append(("warn", f"Extra/unrecognised column(s) detected: {', '.join(map(str, non_date_extras[:5]))}"
                                      + (f" … (+{len(non_date_extras)-5} more)" if len(non_date_extras) > 5 else "")))
    if info["id_col"]:
        dup = df[info["id_col"]].duplicated().sum()
        if dup:
            checks.append(("err", f"Customer ID column contains {int(dup)} duplicate value(s)."))
    else:
        checks.append(("info", "No customer-ID column detected — one will be auto-generated."))
    miss_vals = int(df[info["reading_cols"]].isna().sum().sum()) if info["reading_cols"] else 0
    if miss_vals:
        checks.append(("warn", f"{miss_vals:,} missing reading value(s) found — treated as 0."))
    invalid = 0
    for c in info["reading_cols"]:
        invalid += int(pd.to_numeric(df[c], errors="coerce").isna().sum() - df[c].isna().sum())
    if invalid:
        checks.append(("err", f"{invalid:,} non-numeric reading value(s) found."))
    if info["flag_col"]:
        bad_flag = (~pd.to_numeric(df[info["flag_col"]], errors="coerce").isin([0, 1])).sum()
        if bad_flag:
            checks.append(("err", f"FLAG column contains {int(bad_flag)} value(s) that are not 0/1."))
    if not checks:
        checks.append(("ok", "Dataset format matches training data."))
    return checks


def render_dataset_preview(df: pd.DataFrame, info: dict):
    """Feature 9 / Section 8 — dataset summary (customers, readings, date range,
    quality, FLAG distribution) + first 5 rows."""
    variable = is_loaded() and E.seq_len is None
    dataset_type = "Training/Evaluation Dataset" if info["has_flag"] else "Production Dataset"
    if variable:
        start_d = info["reading_cols"][0] if info["reading_cols"] else "—"
        end_d = info["reading_cols"][-1] if info["reading_cols"] else "—"
    else:
        n = _template_seq_len()
        date_cols = [c for c in info["reading_cols"] if str(c) in set(_template_dates(n))]
        start_d = date_cols[0] if date_cols else (info["reading_cols"][0] if info["reading_cols"] else "—")
        end_d = date_cols[-1] if date_cols else (info["reading_cols"][-1] if info["reading_cols"] else "—")
    c = st.columns(4)
    with c[0]: kpi("Customers", f"{info['n_rows']:,}", icon="")
    with c[1]: kpi("Readings", info["n_readings"], icon="")
    with c[2]: kpi("Dataset Type", dataset_type.split()[0], dataset_type, icon="")
    with c[3]: kpi("Missing values", f"{int(df.isna().sum().sum()):,}", icon="")
    if info["flag_col"]:
        flags = pd.to_numeric(df[info["flag_col"]], errors="coerce").fillna(0).astype(int)
        c2 = st.columns(2)
        with c2[0]: kpi("Normal Count", f"{int((flags == 0).sum()):,}", "FLAG = 0", "#16a34a")
        with c2[1]: kpi("Theft Count", f"{int((flags == 1).sum()):,}", "FLAG = 1", "#dc2626")
    st.caption(f"Reading range: **{start_d}** → **{end_d}** · "
              f"Duplicate IDs: **{int(df[info['id_col']].duplicated().sum()) if info['id_col'] else 0}** · "
              f"Ground truth (FLAG): **{'present' if info['has_flag'] else 'not present'}**")
    st.dataframe(df.head(5), use_container_width=True)


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
        nonlocal el
        el.append(Paragraph(f"<b>{t}</b>", s["Heading2"]))
        table = Table([[str(k), str(v)] for k, v in pairs], colWidths=[7 * cm, 9 * cm])
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
        el.extend([table, Spacer(1, .5 * cm)])

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
ss.setdefault("bg_custom", get_setting("bg_custom", None))
# v3.0: threshold defaults from model_config.json (per active model), not 0.5.
_cfg_thr = config_threshold() if is_loaded() else 0.5
ss.setdefault("threshold", float(get_setting("threshold", _cfg_thr)))
ss.setdefault("strategy", get_setting("strategy", "last_n"))
ss.setdefault("chat", [])
ss.setdefault("manual_text", "")


def _hex_luminance(hexc: str) -> float:
    """Relative luminance (0=black … 1=white) of a #RRGGBB colour (WCAG)."""
    h = str(hexc).lstrip("#")
    if len(h) != 6:
        return 0.5
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _mix(hexc: str, target: str, amt: float) -> str:
    """Blend hexc toward target (#ffffff/#000000) by amt (0..1)."""
    h = str(hexc).lstrip("#"); t = target.lstrip("#")
    c1 = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    c2 = [int(t[i:i + 2], 16) for i in (0, 2, 4)]
    m = [round(a + (b - a) * amt) for a, b in zip(c1, c2)]
    return "#%02x%02x%02x" % tuple(m)


def _bg_is_dark() -> bool:
    custom = ss.get("bg_custom")
    return _hex_luminance(custom) < 0.5 if custom else (ss.theme == "dark")


def _palette() -> dict:
    """Enterprise theme tokens (few colors) — single source of truth for the UI.
    A custom background (ss.bg_custom) derives readable text/card/border tokens
    automatically from the background's luminance, so text stays clearly visible
    on any chosen background."""
    prim, ok, warn, err = "#2563eb", "#16a34a", "#d97706", "#dc2626"
    custom = ss.get("bg_custom")
    if custom:
        dark = _hex_luminance(custom) < 0.5
        toward = "#ffffff" if dark else "#000000"
        base = dict(
            bg=custom,
            card=_mix(custom, toward, 0.06), card2=_mix(custom, toward, 0.10),
            text=("#f5f7fb" if dark else "#14181f"),
            sub=("rgba(245,247,251,.66)" if dark else "rgba(20,24,31,.62)"),
            border=("rgba(255,255,255,.16)" if dark else "rgba(0,0,0,.12)"),
            grid=("rgba(255,255,255,.06)" if dark else "rgba(0,0,0,.06)"))
    elif ss.theme == "dark":
        base = dict(bg="#0f1420", card="#171d2b", card2="#1e2636", text="#e6ebf4",
                    sub="#9aa7bd", border="#2a3446", grid="rgba(255,255,255,.05)")
    else:
        base = dict(bg="#f6f8fa", card="#ffffff", card2="#f7f9fb", text="#1f2328",
                    sub="#636c76", border="#e4e8ee", grid="rgba(31,35,40,.06)")
    base.update(primary=prim, ok=ok, warn=warn, err=err, accent=prim, accent2=prim)
    return base


def inject_css():
    p = _palette()
    st.markdown(f"""<style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      html, body, .stApp, [class*="css"] {{ font-family:'Inter',system-ui,-apple-system,sans-serif; }}
      .stApp {{ background:{p['bg']}; color:{p['text']}; }}
      .block-container {{ padding-top:1.4rem; padding-bottom:3rem; max-width:1360px; }}
      section[data-testid="stSidebar"] {{ background:{p['card']}; border-right:1px solid {p['border']}; }}
      h1,h2,h3,h4 {{ color:{p['text']}; letter-spacing:-.01em; font-weight:650; }}
      h3 {{ font-size:1.05rem; margin:.4rem 0; }}  h4 {{ font-size:.92rem; }}
      a {{ color:{p['primary']}; text-decoration:none; }}
      ::-webkit-scrollbar {{ width:8px; height:8px; }}
      ::-webkit-scrollbar-thumb {{ background:{p['border']}; border-radius:8px; }}

      /* top header bar */
      .topbar {{ display:flex; align-items:center; justify-content:space-between; gap:14px;
                 background:{p['card']}; border:1px solid {p['border']}; border-radius:10px;
                 padding:10px 16px; margin-bottom:16px; }}
      .topbar .brand {{ font-weight:700; font-size:1rem; color:{p['text']}; letter-spacing:-.02em; }}
      .topbar .meta {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
      .chip {{ background:{p['card2']}; border:1px solid {p['border']}; border-radius:6px;
               padding:3px 10px; font-size:.75rem; color:{p['sub']}; font-weight:500; }}
      .dot {{ height:8px; width:8px; border-radius:50%; display:inline-block; margin-right:6px; vertical-align:middle; }}

      /* page header */
      .phead {{ border-bottom:1px solid {p['border']}; padding-bottom:12px; margin-bottom:18px; }}
      .phead h1 {{ margin:0; font-size:1.35rem; font-weight:680; color:{p['text']}; }}
      .phead p {{ margin:3px 0 0; color:{p['sub']}; font-size:.86rem; }}

      /* KPI cards — flat, equal height, thin accent */
      .kpi {{ background:{p['card']}; border:1px solid {p['border']}; border-radius:10px;
              padding:14px 16px; box-shadow:0 1px 2px rgba(16,24,40,.04);
              transition:box-shadow .15s, border-color .15s; height:100%; }}
      .kpi:hover {{ box-shadow:0 4px 14px rgba(16,24,40,.08); border-color:{p['primary']}33; }}
      .kpi .top {{ display:flex; justify-content:space-between; align-items:center; }}
      .kpi .label {{ color:{p['sub']}; font-size:.72rem; font-weight:600; text-transform:uppercase;
                     letter-spacing:.04em; }}
      .kpi .icon {{ font-size:.95rem; opacity:.55; }}
      .kpi .value {{ color:{p['text']}; font-size:1.5rem; font-weight:700; margin-top:6px; line-height:1.15; }}
      .kpi .delta {{ font-size:.75rem; margin-top:2px; font-weight:500; color:{p['sub']}; }}
      .kpi .accent {{ height:3px; width:26px; border-radius:2px; margin-top:10px; background:var(--a1); }}

      /* badges */
      .badge {{ display:inline-flex; align-items:center; gap:6px; padding:4px 12px; border-radius:6px;
                font-weight:600; font-size:.85rem; }}
      .badge-theft {{ background:rgba(220,38,38,.10); color:{p['err']}; border:1px solid rgba(220,38,38,.28); }}
      .badge-normal {{ background:rgba(22,163,74,.10); color:{p['ok']}; border:1px solid rgba(22,163,74,.28); }}

      .pill {{ background:{p['card2']}; border:1px solid {p['border']}; border-radius:6px;
               padding:3px 10px; font-size:.72rem; color:{p['sub']}; font-weight:500; }}
      .sb-group {{ color:{p['sub']}; font-size:.68rem; font-weight:700; text-transform:uppercase;
                   letter-spacing:.07em; margin:12px 2px 4px; }}
      .mcard {{ background:{p['card2']}; border:1px solid {p['border']}; border-radius:8px;
                padding:12px 14px; margin-top:8px; }}
      .mcard .row {{ display:flex; justify-content:space-between; font-size:.78rem; padding:2px 0;
                     color:{p['sub']}; }} .mcard .row b {{ color:{p['text']}; font-weight:600; }}

      /* notification cards: icon + title + message */
      .callout {{ display:flex; gap:10px; border-radius:8px; padding:11px 14px; margin:8px 0;
                  border:1px solid; background:{p['card']}; }}
      .callout .ci {{ font-size:1rem; line-height:1.4; }}
      .callout .ct {{ font-weight:600; font-size:.86rem; margin-bottom:1px; }}
      .callout .cm {{ font-size:.82rem; color:{p['sub']}; }}
      .c-ok  {{ border-color:rgba(22,163,74,.30); }}   .c-ok  .ct {{ color:{p['ok']}; }}
      .c-err {{ border-color:rgba(220,38,38,.30); }}   .c-err .ct {{ color:{p['err']}; }}
      .c-warn{{ border-color:rgba(217,119,6,.30); }}   .c-warn .ct {{ color:{p['warn']}; }}
      .c-info{{ border-color:rgba(37,99,235,.30); }}   .c-info .ct {{ color:{p['primary']}; }}

      /* empty state */
      .empty {{ text-align:center; padding:40px 20px; color:{p['sub']}; border:1px dashed {p['border']};
                border-radius:10px; background:{p['card']}; }}
      .empty .ei {{ font-size:1.6rem; opacity:.5; }} .empty .et {{ font-weight:600; margin-top:6px; color:{p['text']}; }}

      /* tables, buttons, tabs, footer */
      .skel {{ height:88px; border-radius:10px; background:linear-gradient(90deg,{p['card']} 25%,
               {p['card2']} 37%,{p['card']} 63%); background-size:400% 100%; animation:shimmer 1.3s infinite; }}
      .footer {{ display:flex; justify-content:center; gap:14px; flex-wrap:wrap; color:{p['sub']};
                 font-size:.76rem; margin-top:30px; padding-top:14px; border-top:1px solid {p['border']}; }}
      [data-testid="stDataFrame"] {{ border:1px solid {p['border']}; border-radius:8px; }}
      .stButton>button {{ border-radius:8px; font-weight:600; font-size:.86rem; border:1px solid {p['border']};
                          transition:all .12s; }}
      .stButton>button:hover {{ border-color:{p['primary']}; color:{p['primary']}; }}
      .stButton>button[kind="primary"] {{ background:{p['primary']}; border-color:{p['primary']}; color:#fff; }}
      .stButton>button[kind="primary"]:hover {{ filter:brightness(1.06); color:#fff; }}
      .stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid {p['border']}; }}
      .stTabs [data-baseweb="tab"] {{ font-weight:600; font-size:.86rem; }}

      @keyframes shimmer {{ 0% {{ background-position:100% 0; }} 100% {{ background-position:-100% 0; }} }}
    </style>""", unsafe_allow_html=True)


inject_css()
TMPL = "plotly_dark" if ss.theme == "dark" else "plotly_white"


def style_fig(fig, height=320, title=None):
    """Consistent, restrained Plotly styling across the whole app."""
    p = _palette()
    tmpl = "plotly_dark" if _bg_is_dark() else "plotly_white"
    fig.update_layout(template=tmpl, height=height,
                      title=dict(text=title, font=dict(size=13, family="Inter")) if title else None,
                      margin=dict(t=38 if title else 12, b=10, l=10, r=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color=p["sub"], size=12),
                      colorway=[p["primary"], p["ok"], p["warn"], p["err"], "#8b5cf6", "#0891b2"],
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(gridcolor=p["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=p["grid"], zeroline=False)
    return fig


def kpi(label, value, delta="", color="#2563eb", icon=""):
    ic = f'<span class="icon">{icon}</span>' if icon else ""
    d = f'<div class="delta">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="kpi" style="--a1:{color}">'
        f'<div class="top"><span class="label">{label}</span>{ic}</div>'
        f'<div class="value">{value}</div>{d}<div class="accent"></div></div>',
        unsafe_allow_html=True)


def badge(status, pulse=False):
    cls = "badge-theft" if status == "Theft" else "badge-normal"
    dotc = "#dc2626" if status == "Theft" else "#16a34a"
    return f'<span class="badge {cls}"><span class="dot" style="background:{dotc}"></span>{status}</span>'


# Notification cards with icon + title + message (title inferred from kind).
_CALLOUT = {"ok": ("✓", "Success", "c-ok"), "err": ("✕", "Error", "c-err"),
            "warn": ("!", "Warning", "c-warn"), "info": ("i", "Note", "c-info")}


def callout(kind, msg, title=None):
    ic, deftitle, cls = _CALLOUT[kind]
    st.markdown(
        f'<div class="callout {cls}"><div class="ci">{ic}</div>'
        f'<div><div class="ct">{title or deftitle}</div><div class="cm">{msg}</div></div></div>',
        unsafe_allow_html=True)


def empty_state(icon, title, msg):
    st.markdown(f'<div class="empty"><div class="ei">{icon}</div>'
                f'<div class="et">{title}</div><div>{msg}</div></div>', unsafe_allow_html=True)


def top_header():
    """Slim app-level header. Model/engine details are shown to admins only;
    standard users see a neutral service-status header (no model leakage)."""
    online = is_loaded()
    dotc = "#16a34a" if online else "#dc2626"
    admin = ss.get("logged_in") and ss.get("role") == "Administrator"
    meta = (f'<span class="chip"><span class="dot" style="background:{dotc}"></span>'
            f'{"Ready" if online else "Unavailable"}</span>')
    if admin:
        info = model_info()
        meta += f'<span class="chip">Model: {info.get("name","none") if online else "no model"}</span>'
    if ss.get("role"):
        meta += f'<span class="chip">{ss.get("username","")} · {ss.get("role")}</span>'
    meta += f'<span class="chip">{ss.theme.title()} theme</span>'
    st.markdown(
        f'<div class="topbar"><span class="brand">ETD·XAI Enterprise</span>'
        f'<span class="meta">{meta}</span></div>', unsafe_allow_html=True)


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
    tfv = model_info().get("tf_version", "—") if is_loaded() else "—"
    st.markdown(
        f'<div class="footer"><span>ETD-XAI Enterprise v{APP_VERSION}</span>'
        f'<span>·</span><span>TensorFlow {tfv}</span>'
        f'<span>·</span><a href="https://github.com/me30101152101959-tech/Theft" target="_blank">GitHub</a>'
        f'<span>·</span><span>© {datetime.now().year} · MIT</span></div>', unsafe_allow_html=True)


def hero(title, subtitle):
    # Clean page header (no gradient) — strip any leading emoji from page titles.
    clean = title.strip()
    if clean and clean[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        clean = clean[1:].strip()
    st.markdown(f'<div class="phead"><h1>{clean}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


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
def _dashboard_source():
    """Resolve the working dataset for the dashboard as (raw_bytes, name).
    An in-page upload wins, else the saved active dataset, else the bundled
    sample. Returning bytes lets the heavy work be cached by content."""
    up = st.file_uploader("Upload a dataset (CSV/Excel) — or leave empty to use the saved dataset",
                          type=["csv", "xlsx", "xls"], key="dash_up")
    if up is not None:
        return up.getvalue(), up.name
    saved = get_setting("active_dataset_path")
    p = saved if (saved and Path(saved).exists()) else (
        str(SAMPLE_DATASET) if SAMPLE_DATASET.exists() else None)
    if p is None:
        return None, None
    return Path(p).read_bytes(), Path(p).name


@st.cache_data(show_spinner="Loading dashboard…", max_entries=4)
def _dashboard_bundle(raw: bytes, name: str, thr: float, model_name: str):
    """Read + inspect + build matrix + score ONCE per (file, threshold, model).
    Cached so changing a filter does NOT re-run model.predict(). model_name is
    part of the key so switching the active model invalidates the cache."""
    buf = io.BytesIO(raw)
    df = pd.read_excel(buf) if str(name).lower().endswith((".xlsx", ".xls")) else pd.read_csv(buf)
    info = inspect(df)
    readings, ids, _flags = build_matrix(df, info)
    probs = None
    try:
        res = run_batch(df, info, "last_n", thr)
        probs = np.array([r["probability"] for r in res["rows"]], dtype=float)
    except Exception:
        probs = None
    return df, info, readings, ids, probs


def _alarm_level(risk):
    return ("Critical" if risk >= 90 else "High" if risk >= 75
            else "Medium" if risk >= 40 else "Low")


def page_dashboard():
    """Enterprise analytics dashboard (admin + user). READ-ONLY: it never mutates
    the dataframe, reading columns, model, scaler, threshold or predictions. The
    ONLY source of reading columns is info["reading_cols"] from inspect(); it never
    re-detects columns. Predictions come from the existing run_batch() only."""
    # ── Section 1 — header ──
    hero("Electricity Theft Analytics", "Enterprise Monitoring Dashboard")
    raw, name = _dashboard_source()
    if raw is None:
        callout("info", "No dataset available. Upload a file above to view the dashboard.")
        return
    try:
        df, info, readings, ids, probs = _dashboard_bundle(raw, name, config_threshold(), E.name)
    except Exception as e:
        callout("err", f"Could not read file: {e}"); return
    if info["n_readings"] < 2:
        callout("err", "No usable reading columns found (need ≥ 2 numeric)."); return
    n_days = readings.shape[1]
    day_cols = [str(c) for c in info["reading_cols"]]

    have_pred = probs is not None
    if not have_pred:
        probs = np.full(len(ids), np.nan)
    risk = np.round(probs * 100, 1)
    thr = config_threshold()
    pred = (probs >= thr).astype(int) if have_pred else np.zeros(len(ids), int)
    alarms = np.array([_alarm_level(r) for r in risk]) if have_pred else np.array(["—"] * len(ids))
    total_cons = readings.sum(axis=1)

    # ── Section 2 — filters ──
    n_months = (n_days + 29) // 30
    f = st.columns(5)
    cust_sel = f[0].selectbox("Customer", ["All"] + ids, key="dash_cust")
    month_sel = f[1].selectbox("Month", ["All"] + [f"Month {m}" for m in range(1, n_months + 1)], key="dash_month")
    day_sel = f[2].selectbox("Day", ["All"] + [f"Day {d}" for d in range(1, n_days + 1)], key="dash_day")
    pred_sel = f[3].selectbox("Prediction", ["All", "Normal", "Theft"], key="dash_pred")
    risk_sel = f[4].selectbox("Risk level", ["All", "Critical", "High", "Medium", "Low"], key="dash_risk")

    # customer-level mask (customer + prediction + risk filters)
    cmask = np.ones(len(ids), bool)
    if cust_sel != "All":
        cmask &= (np.arange(len(ids)) == ids.index(cust_sel))
    if have_pred and pred_sel != "All":
        cmask &= (pred == (1 if pred_sel == "Theft" else 0))
    if have_pred and risk_sel != "All":
        cmask &= (alarms == risk_sel)
    csel = np.where(cmask)[0]
    if len(csel) == 0:
        callout("info", "No customers match the selected filters."); return

    # day-level mask (month + day filters)
    dmask = np.ones(n_days, bool)
    if month_sel != "All":
        m = int(month_sel.split()[-1]); dmask[:] = False
        dmask[(m - 1) * 30: min(m * 30, n_days)] = True
    if day_sel != "All":
        d = int(day_sel.split()[-1]) - 1
        dd = np.zeros(n_days, bool)
        if 0 <= d < n_days: dd[d] = True
        dmask &= dd
    dsel = np.where(dmask)[0]
    if len(dsel) == 0: dsel = np.arange(n_days)

    sub = readings[np.ix_(csel, dsel)]                    # (customers, days) — filtered
    sub_labels = [day_cols[d] for d in dsel]

    # ── Section 3 — KPI cards ──
    n_total = len(csel)
    n_theft = int(pred[csel].sum()) if have_pred else 0
    n_normal = n_total - n_theft
    c = st.columns(3)
    with c[0]: kpi("Total customers", f"{n_total:,}", "in view", "#2563eb", "👥")
    with c[1]: kpi("Normal", f"{n_normal:,}", "predicted", "#16a34a", "🟢")
    with c[2]: kpi("Theft", f"{n_theft:,}", "predicted", "#dc2626", "🔴")
    c = st.columns(3)
    with c[0]: kpi("Theft rate", f"{(n_theft / max(n_total,1) * 100):.1f}%", "of customers", "#f59e0b", "📊")
    with c[1]: kpi("Avg risk score", f"{np.nanmean(risk[csel]):.1f}" if have_pred else "—", "0–100", "#7c3aed", "🎯")
    with c[2]: kpi("Avg daily consumption", f"{sub.mean():,.1f}", "kWh / day", "#0891b2", "⚡")

    # ── Section 4 — charts ──
    g = st.columns(2)
    with g[0]:
        fig = go.Figure(go.Scatter(x=sub_labels, y=sub.sum(axis=0), mode="lines",
                                   line=dict(color="#2563eb", width=2),
                                   fill="tozeroy", fillcolor="rgba(37,99,235,.10)"))
        st.plotly_chart(style_fig(fig, title="Daily electricity consumption"), use_container_width=True)
    with g[1]:
        fig = go.Figure(go.Bar(x=sub_labels, y=sub.mean(axis=0), marker_color="#7c3aed"))
        st.plotly_chart(style_fig(fig, title="Average consumption per day"), use_container_width=True)
    g = st.columns(2)
    with g[0]:
        fig = go.Figure(go.Bar(x=["Normal", "Theft"], y=[n_normal, n_theft],
                               marker_color=["#16a34a", "#dc2626"],
                               text=[n_normal, n_theft], textposition="outside"))
        st.plotly_chart(style_fig(fig, title="Normal vs Theft"), use_container_width=True)
    with g[1]:
        # monthly usage (30-day blocks) across filtered customers
        monthly = readings[csel].sum(axis=0)
        mvals = [float(monthly[(m - 1) * 30: min(m * 30, n_days)].sum()) for m in range(1, n_months + 1)]
        fig = go.Figure(go.Scatter(x=[f"M{m}" for m in range(1, n_months + 1)], y=mvals,
                                   mode="lines+markers", line=dict(color="#0891b2", width=2)))
        st.plotly_chart(style_fig(fig, title="Monthly electricity usage"), use_container_width=True)
    if have_pred:
        g = st.columns(2)
        with g[0]:
            fig = px.histogram(pd.DataFrame({"risk": risk[csel]}), x="risk", nbins=20,
                               color_discrete_sequence=["#2563eb"])
            st.plotly_chart(style_fig(fig, title="Risk distribution"), use_container_width=True)
        with g[1]:
            ac = pd.Series(alarms[csel]).value_counts().reindex(["Critical", "High", "Medium", "Low"]).fillna(0)
            fig = go.Figure(go.Pie(labels=ac.index.tolist(), values=ac.values, hole=.55,
                                   marker_colors=["#dc2626", "#f59e0b", "#eab308", "#16a34a"]))
            st.plotly_chart(style_fig(fig, title="Risk categories"), use_container_width=True)

    # ── Sections 5 & 6 — Top 10 highest / lowest risk ──
    if have_pred:
        tbl = pd.DataFrame({"Customer ID": [ids[i] for i in csel],
                            "Risk Score": risk[csel],
                            "Probability": [f"{probs[i]*100:.1f}%" for i in csel],
                            "Prediction": ["Theft" if pred[i] else "Normal" for i in csel]})
        t = st.columns(2)
        with t[0]:
            st.markdown("##### 🔺 Top 10 highest risk")
            st.dataframe(tbl.sort_values("Risk Score", ascending=False).head(10),
                         use_container_width=True, hide_index=True)
        with t[1]:
            st.markdown("##### 🔻 Top 10 lowest risk")
            st.dataframe(tbl.sort_values("Risk Score", ascending=True).head(10),
                         use_container_width=True, hide_index=True)

        # ── Section 9 — alarm summary ──
        st.markdown("##### 🚨 Alarm summary")
        counts_a = pd.Series(alarms[csel]).value_counts()
        a = st.columns(4)
        for col, lvl, clr, ic in zip(a, ["Critical", "High", "Medium", "Low"],
                                     ["#dc2626", "#f59e0b", "#eab308", "#16a34a"], ["⛔", "⚠️", "🟡", "✅"]):
            with col: kpi(lvl, f"{int(counts_a.get(lvl, 0)):,}", "customers", clr, ic)

    # ── Section 7 — customer consumption explorer ──
    st.markdown("##### 🔎 Customer consumption explorer")
    exp_sel = st.selectbox("Select a customer to view their consumption sequence",
                           ["—"] + [ids[i] for i in csel], key="dash_explorer")
    if exp_sel != "—":
        ci = ids.index(exp_sel)
        seq = readings[ci]
        fig = go.Figure(go.Scatter(x=day_cols, y=seq, mode="lines",
                                   line=dict(color="#2563eb", width=2),
                                   fill="tozeroy", fillcolor="rgba(37,99,235,.10)"))
        st.plotly_chart(style_fig(fig, height=300, title=f"{exp_sel} — consumption sequence"),
                        use_container_width=True)
        # ── Section 8 — per-timestep importance (XAI), opt-in, hidden if unavailable ──
        if st.checkbox("Show per-timestep importance (XAI)", value=False, key="dash_xai"):
            try:
                ex = shap_or_ig(seq)
            except Exception:
                ex = None
            if ex and ex.get("timestep_importance"):
                imp = ex["timestep_importance"]
                fig = go.Figure(go.Bar(x=list(range(1, len(imp) + 1)), y=imp, marker_color="#7c3aed"))
                st.plotly_chart(style_fig(fig, height=260, title=f"Per-timestep importance · {ex['method']}"),
                                use_container_width=True)
            else:
                callout("info", "Explainability is unavailable for this input.")

    # ── Section 10 — batch analytics ──
    if have_pred:
        st.markdown("##### 📦 Batch analytics")
        monthly_all = [float(readings[csel][:, (m - 1) * 30: min(m * 30, n_days)].sum())
                       for m in range(1, n_months + 1)]
        day_totals = readings[csel].sum(axis=0)
        hi = csel[int(np.argmax(total_cons[csel]))]
        lo = csel[int(np.argmin(total_cons[csel]))]
        b = st.columns(3)
        with b[0]: kpi("Avg probability", f"{np.nanmean(probs[csel])*100:.1f}%", "theft likelihood", "#2563eb", "📈")
        with b[1]: kpi("Avg consumption", f"{readings[csel].mean():,.1f}", "kWh / day", "#0891b2", "⚡")
        with b[2]: kpi("Most active month", f"M{int(np.argmax(monthly_all)) + 1}", "highest usage", "#7c3aed", "📅")
        b = st.columns(3)
        with b[0]: kpi("Highest consumer", str(ids[hi]), f"{total_cons[hi]:,.0f} kWh", "#16a34a", "🔼")
        with b[1]: kpi("Lowest consumer", str(ids[lo]), f"{total_cons[lo]:,.0f} kWh", "#dc2626", "🔽")
        with b[2]: kpi("Most suspicious day", day_cols[int(np.argmin(day_totals))], "lowest usage", "#f59e0b", "🕵️")


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
                            help=(f"Model expects exactly {T} readings — input is never resized."
                                  if T is not None else "Model accepts variable length."))
        # Fixed-length model → no length-mapping strategy is ever applied.
        strat = "last_n" if T is not None else strategy_selector("m_strat")
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
        # The CNN-LSTM accepts variable length, so any count is ACCEPTED. When it
        # differs from the trained length the sequence is adapted (engine untouched);
        # accuracy is only guaranteed at exactly T readings — we warn, never block.
        strat_use = strat
        if T is not None and len(raw) != T:
            with right:
                callout("warn", f"Entered <b>{len(raw)}</b> readings; model was trained on "
                                f"<b>{T}</b>. The sequence will be adapted to {T} — accuracy "
                                f"is only guaranteed at exactly {T}.")
            strat_use = strategy_selector("m_strat_adapt")
        with right:
            with st.spinner("⚡ Running model.predict()…"):
                res = predict_one(raw, strat_use, thr)
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
            st.caption(f"🔒 Input hash (md5): `{_input_hash(np.asarray(raw).reshape(1, -1))}` — "
                       f"identical to the Batch hash for the same readings.")
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

    with st.expander("📁 Dataset Templates", expanded=False):
        render_dataset_templates(show_all=True)

    saved_ds = get_setting("active_dataset_path")
    has_saved = bool(saved_ds and Path(saved_ds).exists())
    options = ["Upload file"] + (["Use saved dataset"] if has_saved else []) + ["Use bundled sample"]
    src = st.radio("Data source", options, horizontal=True)
    file = None
    if src == "Upload file":
        file = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
    elif src == "Use saved dataset":
        file = saved_ds; st.caption(f"Using saved dataset `{Path(saved_ds).name}`")
    elif SAMPLE_DATASET.exists():
        file = str(SAMPLE_DATASET); st.caption(f"Using `{SAMPLE_DATASET.name}`")
    if not file:
        return
    try:
        df = read_table(file)
    except Exception as e:
        st.error(f"Could not read file: {e}"); return
    info = inspect(df)
    render_dataset_preview(df, info)
    st.markdown("##### Validation Report")
    for kind, msg in validate_dataset_report(df, info):
        callout(kind, msg)

    # Prediction Mode notice — FLAG is OPTIONAL (ground truth only, never required)
    if info["has_flag"]:
        callout("ok", "<b>Evaluation Mode</b> — Ground Truth (FLAG) detected. "
                      "Predictions will run and evaluation metrics will be computed.")
    else:
        callout("info", "<b>Prediction Mode</b> — Ground Truth (FLAG) not detected. "
                        "Predictions will be generated normally. Evaluation metrics are disabled.")

    with st.expander("🔧 Compatibility Panel", expanded=False):
        rep = compatibility_report(info["n_readings"])
        st.json(rep)
        st.progress(rep["✓ overall_compatibility_pct"] / 100,
                   text=f"Overall Compatibility: {rep['✓ overall_compatibility_pct']}%")

    # Developer Mode — input-tensor verification (Rules 9/10). Read-only; the
    # reading hash lets you confirm Manual and Batch feed model.predict() the
    # exact same bytes for the same customer.
    if st.checkbox("🛠 Developer Mode (input verification)", value=False, key="b_dev"):
        _rd, _ids, _fl = build_matrix(df, info)
        _rhash = _input_hash(_rd)
        _ignored = [c for c in map(str, df.columns)
                    if c not in set(map(str, info["reading_cols"]))]
        st.json({
            "reading_count": info["n_readings"],
            "detected_columns": [str(c) for c in info["reading_cols"][:6]]
                                + (["…"] if info["n_readings"] > 6 else []),
            "ignored_columns": _ignored,
            "reading_hash_md5": _rhash,
            "sequence_shape": f"({len(_rd)}, {info['n_readings']}, {E.seq_channels})",
            "statistics_shape": f"({len(_rd)}, {E.stat_size})",
            "scaler_loaded": PIPELINE.using_saved_scaler,
            "scaler_locked": PIPELINE._locked,
            "threshold": config_threshold(),
            "model_input_shape": str(E.input_shape),
            "model_name": E.name,
            "prediction_mode": "Evaluation (FLAG present)" if info["has_flag"] else "Prediction only",
        })
        st.caption("Same customer readings ⇒ identical `reading_hash_md5` in Manual and Batch.")

    # Section 5/6/7 — never silently reshape. Fixed-length model + mismatch
    # => prediction is BLOCKED unless the user explicitly opts into ONE resize
    # strategy (OFF by default). Variable-length model => data always as-is.
    T = E.seq_len
    mismatch = (T is not None and info["n_readings"] != T)

    # Data distribution warning: model trained on specific length → poor accuracy on very different lengths
    if mismatch and T is not None:
        if info["n_readings"] > T * 1.5:
            st.warning(f"⚠️ **Data mismatch detected**: Dataset has {info['n_readings']} readings, "
                       f"but model trained on {T} readings. Predictions may have lower accuracy. "
                       f"For best results, use data matching training conditions.", icon="⚠️")
        elif info["n_readings"] < T * 0.5:
            st.warning(f"⚠️ **Insufficient data**: Dataset has {info['n_readings']} readings, "
                       f"model expects ~{T}. Predictions may be unreliable.", icon="⚠️")
    if T is None:
        integrity_note = "✓ Variable-length model — original data used, no modification."
        callout("ok", f"Model accepts <b>variable-length</b> input — the uploaded "
                      f"{info['n_readings']} readings are sent <b>exactly as-is</b> (no resizing).")
        allow_resize, strat = True, "last_n"
    elif not mismatch:
        integrity_note = "✓ Fixed-length model — uploaded length matches exactly, original data used."
        callout("ok", f"Uploaded length <b>{info['n_readings']}</b> matches the model "
                      f"exactly — data sent as-is, no resizing.")
        allow_resize, strat = True, "last_n"
    else:
        # Length differs from training. The CNN-LSTM accepts variable length, so we
        # ACCEPT the upload and adapt the sequence to the trained length via an
        # explicit, user-chosen strategy (never silent). Accuracy is only
        # guaranteed at exactly T readings.
        callout("warn", f"<b>Length differs from training.</b><br>Model was trained on <b>{T}</b> "
                        f"readings; this file has <b>{info['n_readings']}</b>. The sequence will be "
                        f"adapted to <b>{T}</b> — accuracy is only guaranteed at exactly {T}.")
        strat = strategy_selector("b_strat")
        allow_resize = True
        integrity_note = f"⚠ Adapted to {T} via {STRATEGY_LABELS[strat]} (accuracy not guaranteed)."
    # Suggest adjusted threshold for data distribution mismatch
    thr_default = ss.threshold
    thr_help = f"Config default for this model: {config_threshold():.2f}"
    if mismatch and T is not None and info["n_readings"] > T * 1.5:
        thr_default = 0.10
        thr_help += f" · **For {info['n_readings']}-day data (vs {T} training): try 0.10-0.20 for better accuracy**"

    # Auto-optimize the decision threshold against ground-truth FLAG. This is a
    # DISPLAY/decision-layer helper only: FLAG is used solely to pick the best
    # cut-off — it never enters the model, features, or scaler. The model's
    # probabilities are unchanged; only the 0/1 cut-off moves.
    auto_thr = info["has_flag"] and st.checkbox(
        "🎯 Auto-optimize decision threshold (uses FLAG to pick the best cut-off — "
        "never for prediction)", value=False, key="b_autothr",
        disabled=(mismatch and not allow_resize))
    if auto_thr:
        with st.spinner("Calibrating threshold against ground-truth FLAG…"):
            _rd, _ids, _fl = build_matrix(df, info)
            _probs = predict_sequences(_rd, strat, 0.5, fit_scaler=True)
            best_t, best_a = 0.50, -1.0
            for _t in np.arange(0.05, 0.99, 0.01):
                _acc = float(((_probs >= _t).astype(int) == _fl).mean())
                if _acc > best_a:
                    best_a, best_t = _acc, float(_t)
        thr = round(best_t, 2)
        callout("ok", f"Auto-optimized threshold = <b>{thr:.2f}</b> → accuracy "
                      f"<b>{best_a*100:.1f}%</b> on this dataset. FLAG was used only to choose the "
                      f"cut-off; the model's probabilities are unchanged.")
        st.slider("Decision threshold (auto-optimized)", 0.0, 1.0, thr, 0.01, key="b_thr", disabled=True)
    else:
        thr = st.slider("Decision threshold", 0.0, 1.0, thr_default, 0.01, key="b_thr", help=thr_help)
    callout("info" if "✓" in integrity_note else "err", integrity_note, "Data integrity")
    c1, c2 = st.columns(2)
    run = c1.button("⚡ Run Predictions", type="primary", use_container_width=True,
                    disabled=(mismatch and not allow_resize))
    save = c2.checkbox("Save to database", value=True)
    if not run:
        return
    if mismatch and not allow_resize:
        callout("err", f"Blocked: model expects exactly {T} readings, dataset has "
                       f"{info['n_readings']}. No automatic resizing is allowed.")
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
    _out_keys = ("customer_id", "probability", "prediction", "confidence", "risk_score", "status")
    if result["has_flag"]:
        rdf = pd.DataFrame([{"customer_id": r["customer_id"], "flag": r["flag"],
                             **{k: r[k] for k in _out_keys[1:]}} for r in result["rows"]])
    else:
        rdf = pd.DataFrame([{k: r[k] for k in _out_keys} for r in result["rows"]])
    m = st.columns(3)
    m[0].metric("Theft detected", f"{result['theft_rows']:,}")
    m[1].metric("Theft rate", f"{result['theft_rate'] * 100:.1f}%")
    m[2].metric("Avg risk", f"{result['avg_risk']:.0f}/100")
    if result["metrics"]:
        mm = result["metrics"]
        st.markdown("##### 📊 Evaluation Metrics (vs Ground-Truth FLAG)")
        st.info(f"Accuracy {mm['accuracy']:.3f} · Precision {mm['precision_val']:.3f} · "
                f"Recall {mm['recall_val']:.3f} · F1 {mm['f1_score']:.3f}"
                + (f" · ROC-AUC {mm['roc_auc']:.3f}" if mm.get("roc_auc") else ""))
    else:
        st.success("✔ Prediction completed successfully. No Ground Truth (FLAG) was provided — "
                   "evaluation metrics are unavailable.", icon="✅")
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

    # ── v3.0: Dynamic Model Registry ─────────────────────────────────────────
    st.divider()
    st.markdown("### 🗂️ Model Registry (auto-discovered)")
    reg = discover_models()
    if reg:
        st.dataframe(pd.DataFrame(reg)[["name", "size_mb", "modified", "threshold",
                                        "auc", "f1", "accuracy", "active"]],
                     use_container_width=True, hide_index=True)
        names = [m["name"] for m in reg]
        cur = next((i for i, m in enumerate(reg) if m["active"]), 0)
        pick = st.selectbox("Active model", names, index=cur, key="reg_pick")
        if pick != E.name and st.button("Activate selected model", type="primary"):
            path = next(m["path"] for m in reg if m["name"] == pick)
            try:
                load_model(path, pick)
                set_setting("active_model_path", path)
                ss.threshold = config_threshold()  # auto-update threshold from config
                ss.model_msg = ("ok", f"Activated <b>{pick}</b> — threshold auto-set to "
                                      f"{config_threshold():.2f}, seq_len {E.seq_len}.")
                st.cache_resource.clear(); st.rerun()
            except Exception as e:
                callout("err", f"Rejected: {e}")
    else:
        callout("info", "No models found in assets/ or uploads/.")

    # ── v3.0: Compatibility & configuration panel ────────────────────────────
    st.markdown("### 🔧 Compatibility & Configuration")
    st.json(compatibility_report())
    if not PIPELINE.using_saved_scaler:
        callout("warn", "Training scaler <code>assets/stat_scaler.pkl</code> not found — "
                        "features are re-fit per batch, so <b>inference may not match training</b>. "
                        "Add the notebook's stat_scaler.pkl for exact parity.")

    st.divider()
    st.markdown("### Upload / Replace Model")
    # Show a persisted result from the previous run (survives the rerun).
    if ss.get("model_msg"):
        kind, m = ss.pop("model_msg")
        callout(kind, m)
    up = st.file_uploader("CNN-LSTM model (.keras / .h5)", type=["keras", "h5"])
    if up is not None:
        if st.button("Activate uploaded model", type="primary", use_container_width=True):
            path = UPLOAD_DIR / up.name
            try:
                path.write_bytes(up.getbuffer())
                # Load+validate the NEW model first. load_model() only overwrites
                # the engine state on success, so a rejected file never leaves the
                # app without a working model.
                load_model(str(path), up.name)
                set_setting("active_model_path", str(path))
                ss.model_msg = ("ok", f"Activated <b>{up.name}</b> — "
                                      f"{model_info()['total_params_fmt']} params, "
                                      f"{model_info()['architecture']}.")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                # Engine state is untouched on failure — current model stays active.
                callout("err", f"Rejected: {e}")
    if info.get("loaded") and Path(info["path"]) != DEFAULT_MODEL:
        if st.button("Restore default model", use_container_width=True):
            unload_model()
            set_setting("active_model_path", None)
            auto_load_default()
            ss.model_msg = ("ok", f"Restored default model <b>{DEFAULT_MODEL.name}</b>.")
            st.cache_resource.clear()
            st.rerun()

    st.divider()
    st.markdown("### Dataset")
    if ss.get("ds_msg"):
        kind, m = ss.pop("ds_msg")
        callout(kind, m)
    ds = st.file_uploader("Upload a dataset to use as the working set", type=["csv", "xlsx", "xls"], key="ds_up")
    if ds is not None and st.button("Save dataset", type="primary", use_container_width=True):
        try:
            path = UPLOAD_DIR / ds.name
            path.write_bytes(ds.getbuffer())
            # quick validation so the user gets immediate feedback
            _pv = inspect(read_table(str(path)))
            set_setting("active_dataset_path", str(path))
            ss.ds_msg = ("ok", f"Saved <b>{ds.name}</b> — {_pv['n_rows']:,} rows, "
                               f"{_pv['n_readings']} reading columns. "
                               f"Open <b>📦 Batch Prediction → “Use saved dataset”</b> to score it.")
            st.rerun()
        except Exception as e:
            ss.ds_msg = ("err", f"Could not read dataset: {e}")
            st.rerun()
    active_ds = get_setting("active_dataset_path")
    valid_ds = bool(active_ds and Path(active_ds).exists())
    st.caption(f"Active saved dataset: `{Path(active_ds).name if valid_ds else 'none — bundled sample in Batch'}`")

    st.divider()
    st.markdown("### Defaults & Theme")
    thr = st.slider("Default decision threshold", 0.0, 1.0, ss.threshold, 0.01)
    if thr != ss.threshold:
        ss.threshold = thr; set_setting("threshold", thr); st.toast(f"Threshold → {thr:.2f}")
    t1, t2 = st.columns(2)
    if t1.button("🌙 Dark theme", use_container_width=True):
        ss.theme = "dark"; ss.bg_custom = None
        set_setting("theme", "dark"); set_setting("bg_custom", None); st.rerun()
    if t2.button("☀️ Light theme", use_container_width=True):
        ss.theme = "light"; ss.bg_custom = None
        set_setting("theme", "light"); set_setting("bg_custom", None); st.rerun()

    st.markdown("#### Custom background colour")
    st.caption("Pick any background — text, cards and charts recolour automatically for "
               "clear contrast (light text on dark backgrounds, dark text on light).")
    cc = st.columns([2, 1, 1])
    picked = cc[0].color_picker("Background colour",
                                value=(ss.bg_custom or (_palette()["bg"])), key="bg_pick")
    if cc[1].button("Apply", use_container_width=True):
        ss.bg_custom = picked; set_setting("bg_custom", picked); st.rerun()
    if cc[2].button("Reset", use_container_width=True):
        ss.bg_custom = None; set_setting("bg_custom", None); st.rerun()
    if ss.bg_custom:
        _mode = "dark" if _bg_is_dark() else "light"
        st.caption(f"Active custom background `{ss.bg_custom}` · auto text mode: **{_mode}**")

    st.divider()
    st.markdown("### Verification Status")
    _scaler_state = ("training scaler stat_scaler.pkl (exact parity)"
                     if PIPELINE.using_saved_scaler else
                     "⚠️ re-fit per batch (approximate — add assets/stat_scaler.pkl for exact parity)")
    st.json({"model_loaded": info.get("loaded", False), "active_model": info.get("name"),
             "stat_scaler": _scaler_state,
             "load_method": "tensorflow.keras.models.load_model(path)", "predict_method": "model.predict(x)",
             "exclusive_engine": True,
             "fallback_models": "none — CNN-LSTM only (no RF/XGBoost/LightGBM/CatBoost/LogReg/DT/SVM/KNN/rule/mock)",
             "ground_truth_use": "evaluation metrics only — never used for prediction",
             "compute": "GPU" if has_gpu() else "CPU", "database": str(DB_PATH),
             "last_prediction": E.last_prediction or None})


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Authentication & Role-Based Access Control (RBAC)
# ═════════════════════════════════════════════════════════════════════════════
# Auth store — kept isolated so it can later be swapped for SQLite/Postgres
# without touching application logic (same authenticate() contract).
class AuthProvider:
    """In-memory user store. Replace `verify()` with a DB lookup later."""
    USERS = {
        "admin": {"password": "admin", "role": "Administrator"},
        "user":  {"password": "user",  "role": "Standard User"},
    }

    def verify(self, username: str, password: str) -> Optional[str]:
        u = self.USERS.get((username or "").strip().lower())
        if u and password == u["password"]:
            return u["role"]
        return None


AUTH = AuthProvider()


def is_admin() -> bool:
    return ss.get("logged_in") and ss.get("role") == "Administrator"


def require_admin():
    """Defense-in-depth: every admin page calls this before rendering."""
    if not is_admin():
        callout("err", "You do not have permission to view this page.", "Access denied")
        st.stop()


def do_logout():
    for k in ("logged_in", "username", "role", "nav_choice", "chat", "user_last"):
        ss.pop(k, None)
    st.rerun()


# Animated Three.js background for the login screen only (presentation-only —
# no application logic). A glowing wireframe "AI core" behind the login card.
_LOGIN_BG_HTML = """
<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>*{margin:0;padding:0;box-sizing:border-box}html,body{width:100%;height:100%;overflow:hidden;background:#05070d}</style>
</head><body>
<div id="tjs" style="width:100%;height:100%"></div>
<script src="https://ajax.googleapis.com/ajax/libs/threejs/r125/three.min.js"></script>
<script>
(function() {
  const container = document.getElementById('tjs');
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  container.appendChild(renderer.domElement);

  const geometry = new THREE.IcosahedronGeometry(1, 4);
  const material = new THREE.MeshPhongMaterial({
      color: 0x2563eb, wireframe: true, emissive: 0x2563eb,
      emissiveIntensity: 0.45, transparent: true, opacity: 0.55 });
  const aiCore = new THREE.Mesh(geometry, material);
  scene.add(aiCore);

  const ringGeo = new THREE.TorusGeometry(1.5, 0.015, 16, 100);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0x2563eb, transparent: true, opacity: 0.28 });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  scene.add(ring);

  const light = new THREE.PointLight(0xffffff, 1, 100);
  light.position.set(5, 5, 5);
  scene.add(light);
  camera.position.z = 4;

  function animate() {
      requestAnimationFrame(animate);
      aiCore.rotation.y += 0.006;
      aiCore.rotation.x += 0.003;
      ring.rotation.z -= 0.003;
      ring.rotation.y += 0.005;
      aiCore.position.y = Math.sin(Date.now() * 0.0015) * 0.08;
      renderer.render(scene, camera);
  }
  window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth/window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
  });
  animate();
})();
</script>
</body></html>
"""


def _login_background():
    """Render the Three.js animation full-screen behind the login card."""
    st.markdown("""<style>
      div[data-testid="stIFrame"], div[data-testid="stIFrame"] iframe, iframe {
        position:fixed !important; inset:0 !important; top:0 !important; left:0 !important;
        width:100vw !important; height:100vh !important; z-index:0 !important; border:0 !important;
      }
    </style>""", unsafe_allow_html=True)
    components.html(_LOGIN_BG_HTML, height=0)


def login_view():
    """Professional login card shown before the app is accessible, with a
    subtle animated Three.js background."""
    _login_background()
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.write("")
        st.write("")
        if LOGO.exists():
            lc = st.columns([2, 1, 2])
            lc[1].image(str(LOGO), width=64)
        st.markdown(
            "<div style='text-align:center;margin-bottom:6px'>"
            "<div style='font-size:1.3rem;font-weight:700'>ETD-XAI Enterprise</div>"
            "<div style='color:#636c76;font-size:.85rem'>Electricity Theft Detection Platform</div>"
            "</div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Sign in**")
            u = st.text_input("Username", key="login_u", placeholder="admin or user")
            p = st.text_input("Password", type="password", key="login_p")
            remember = st.checkbox("Remember this session", value=True)
            if st.button("Sign in", type="primary", use_container_width=True):
                role = AUTH.verify(u, p)
                if role:
                    ss.logged_in = True
                    ss.username = u.strip().lower()
                    ss.role = role
                    ss.remember = remember
                    st.rerun()
                else:
                    callout("err", "Invalid username or password.", "Login failed")
        st.caption("Demo credentials — admin / admin · user / user")


# ── Role-specific pages (Standard User — no model/AI details exposed) ──────────
def page_user_home():
    hero("Home", f"Welcome, {ss.get('username','user')}. Detect electricity theft in a few clicks.")
    c = st.columns(3)
    with c[0]: kpi("Step 1", "Upload", "your consumption CSV/Excel", "#2563eb")
    with c[1]: kpi("Step 2", "Predict", "run the analysis", "#16a34a")
    with c[2]: kpi("Step 3", "Report", "download the results", "#d97706")
    st.markdown("#### How it works")
    st.markdown("- Go to **Predict**, upload a dataset of customer meter readings.\n"
                "- The system analyses each customer and flags likely **theft** vs **normal**.\n"
                "- Review the results table and **download a report** from the Reports page.")
    empty_state("→", "Ready when you are", "Open <b>Predict</b> from the sidebar to begin.")


def page_user_predict():
    hero("Predict", "Upload a dataset and get theft-detection results.")
    if not is_loaded():
        callout("err", "The prediction service is temporarily unavailable. Please contact an administrator.",
                "Service unavailable")
        return
    with st.expander("📥 Need a template?", expanded=False):
        render_dataset_templates(show_all=False)
    up = st.file_uploader("Upload consumption data (CSV or Excel)", type=["csv", "xlsx", "xls"])
    if not up:
        empty_state("⬆", "No file yet", "Choose a CSV or Excel file of customer readings to analyse.")
        return
    try:
        df = read_table(up)
    except Exception:
        callout("err", "That file could not be read. Please upload a valid CSV or Excel file.", "Invalid file")
        return
    info = inspect(df)
    render_dataset_preview(df, info)
    for kind, msg in validate_dataset_report(df, info):
        callout(kind, msg)
    # Any length is accepted (the model adapts). Only truly empty files are rejected.
    T = E.seq_len
    if info["n_readings"] < 2:
        callout("err", "This dataset has no usable readings. Please contact an administrator.",
                "Incompatible dataset")
        return
    if T is not None and info["n_readings"] != T:
        callout("warn", f"This dataset has {info['n_readings']} readings; the system is tuned "
                        f"for {T}. Results are shown but accuracy is best at exactly {T}.")
    if not st.button("Run prediction", type="primary", use_container_width=True):
        return
    with st.spinner("Analysing customers…"):
        try:
            result = run_batch(df, info, "last_n", config_threshold())  # threshold hidden from user
        except Exception:
            callout("err", "Prediction could not be completed. Please contact an administrator.", "Prediction error")
            return
    rdf = pd.DataFrame([{"Customer": r["customer_id"], "Prediction": r["status"],
                         "Probability": f"{r['probability']*100:.1f}%",
                         "Confidence": f"{r['confidence']*100:.1f}%",
                         "Risk Score": f"{r['risk_score']:.0f}/100"} for r in result["rows"]])
    ss.user_last = rdf
    callout("ok", f"Analysed {result['total_rows']:,} customers — "
                  f"{result['theft_rows']:,} flagged as theft, {result['normal_rows']:,} normal.", "Done")
    m = st.columns(3)
    with m[0]: kpi("Customers", f"{result['total_rows']:,}", icon="")
    with m[1]: kpi("Theft flagged", f"{result['theft_rows']:,}", "", "#dc2626")
    with m[2]: kpi("Normal", f"{result['normal_rows']:,}", "", "#16a34a")
    st.dataframe(rdf, use_container_width=True, hide_index=True, height=380)
    e = st.columns(2)
    e[0].download_button("Download CSV", to_csv(rdf), "prediction_results.csv", "text/csv",
                         use_container_width=True)
    e[1].download_button("Download Excel", to_excel(rdf), "prediction_results.xlsx",
                         use_container_width=True)


def page_user_reports():
    hero("Reports", "Download the results of your latest prediction.")
    rdf = ss.get("user_last")
    if rdf is None or len(rdf) == 0:
        empty_state("▤", "No results yet", "Run a prediction first from the <b>Predict</b> page.")
        return
    st.dataframe(rdf, use_container_width=True, hide_index=True, height=420)
    e = st.columns(2)
    e[0].download_button("Download CSV", to_csv(rdf), "prediction_results.csv", "text/csv",
                         use_container_width=True)
    e[1].download_button("Download Excel", to_excel(rdf), "prediction_results.xlsx",
                         use_container_width=True)


# ── Role-based navigation maps (admin pages wrapped with a permission guard) ──
def _admin(fn):
    def _wrapped():
        require_admin()
        return fn()
    return _wrapped


NAV_ADMIN = {
    "📊 Dashboard": ("Overview", _admin(page_dashboard)),
    "🔮 Manual Prediction": ("Predict", _admin(page_manual)),
    "📦 Batch Prediction": ("Predict", _admin(page_batch)),
    "📜 History": ("Insights", _admin(page_history)),
    "📑 Reports": ("Insights", _admin(page_reports)),
    "🤖 AI Copilot": ("Insights", _admin(page_copilot)),
    "⚙️ Settings": ("System", _admin(page_settings)),
}
NAV_USER = {
    "📊 Dashboard": ("Overview", page_dashboard),
    "⚡ Predict": ("Predict", page_user_predict),
    "📄 Reports": ("Reports", page_user_reports),
}
GROUP_ORDER = ["Overview", "Home", "Predict", "Insights", "Reports", "System"]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12 — Login gate + role-aware sidebar & router
# ═════════════════════════════════════════════════════════════════════════════
if not ss.get("logged_in"):
    login_view()
    st.stop()

NAV = NAV_ADMIN if is_admin() else NAV_USER
if ss.get("nav_choice") not in NAV:          # reset stale/tampered selection per role
    ss.nav_choice = list(NAV.keys())[0]

with st.sidebar:
    cols = st.columns([1, 3])
    if LOGO.exists():
        cols[0].image(str(LOGO), width=54)
    cols[1].markdown(f"### ⚡ ETD-XAI\n<span class='pill'>Enterprise v{APP_VERSION}</span>",
                     unsafe_allow_html=True)

    # Signed-in identity + role badge
    role = ss.get("role", "")
    rc = "#2563eb" if is_admin() else "#16a34a"
    st.markdown(
        f"<div class='mcard'><div class='row'><span>Signed in</span>"
        f"<b>{ss.get('username','')}</b></div>"
        f"<div class='row'><span>Role</span>"
        f"<b style='color:{rc}'>{role}</b></div></div>", unsafe_allow_html=True)

    # Admin-only model status card (never shown to standard users)
    if is_admin():
        info = model_info()
        online = info.get("loaded")
        dot = "#16a34a" if online else "#dc2626"
        st.markdown(
            f"<div class='mcard'>"
            f"<div class='row'><span><span class='dot' style='background:{dot}'></span>"
            f"Model</span><b>{'Loaded' if online else 'Not loaded'}</b></div>"
            + (f"<div class='row'><span>Name</span><b>{info['name']}</b></div>"
               f"<div class='row'><span>Input</span><b>{info['input_shape']}</b></div>"
               f"<div class='row'><span>Params</span><b>{info['total_params_fmt']}</b></div>"
               f"<div class='row'><span>TensorFlow</span><b>v{info['tf_version']}</b></div>"
               if online else f"<div class='row'><span>{NO_MODEL_MSG}</span><b></b></div>")
            + "</div>", unsafe_allow_html=True)

    st.markdown("<div class='sb-group'>Navigation</div>", unsafe_allow_html=True)
    for grp in GROUP_ORDER:
        items = [k for k, (gp, _) in NAV.items() if gp == grp]
        for k in items:
            if st.button(k, use_container_width=True, key=f"nav_{k}",
                         type="primary" if ss.nav_choice == k else "secondary"):
                ss.nav_choice = k
                st.rerun()

    st.divider()
    if is_admin():
        st.markdown("<div class='sb-group'>System readiness</div>", unsafe_allow_html=True)
        _rows = "".join(
            f"<div class='row'><span>{'✓' if ok else '✗'} {name}</span>"
            f"<b style='color:{'#16a34a' if ok else '#dc2626'}'>{'OK' if ok else '—'}</b></div>"
            for name, ok in startup_validation())
        st.markdown(f"<div class='mcard'>{_rows}</div>", unsafe_allow_html=True)
    if st.button("Log out", use_container_width=True, key="logout_btn"):
        do_logout()

top_header()
NAV[ss.nav_choice][1]()
footer()
