"""
Model Service  (dynamic, model-driven)
======================================
Loads any Keras CNN/LSTM model and runs inference. ALL dimensions are
discovered at runtime from the model and the uploaded data — nothing is
hardcoded (no 26, no 59, no fixed reshape).

Discovered from the model on load:
  • seq_len_expected   : sequence length the model wants (int, or None=variable)
  • seq_channels       : channels of the sequence input (usually 1)
  • is_dual_input      : whether a second statistical input exists
  • stat_input_size    : number of statistical features the model wants

Rules still enforced:
  • Real TF inference only — never fake / fallback / zero-substituted predictions
  • FLAG is never used during prediction
  • Model registry persisted in SQLite (survives restarts)
"""

from __future__ import annotations
import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from services.feature_service import pipeline, scale_sequences, extract_features
from services import preprocessing

logger = logging.getLogger(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Exact message required when no valid CNN-LSTM model is available.
NO_MODEL_MSG = "No valid CNN-LSTM model is currently loaded. Please load a valid .keras model."

# Holds the verification details of the most recent model.predict() call,
# exposed via /api/model/status so the frontend can prove inference is real.
last_prediction: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Keras compatibility shim (strips quantization_config / flattens DTypePolicy)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import keras
    _orig_dense_from_config = keras.layers.Dense.from_config.__func__

    @classmethod  # type: ignore[misc]
    def _compat_dense_from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        dtype = config.get("dtype")
        if isinstance(dtype, dict):
            config["dtype"] = dtype.get("config", {}).get("name", "float32")
        return _orig_dense_from_config(cls, config)

    keras.layers.Dense.from_config = _compat_dense_from_config
    logger.debug("Keras compat shim applied.")
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
class ModelState:
    model:            Optional[tf.keras.Model] = None
    model_name:       str   = ""
    model_path:       str   = ""
    upload_time:      str   = ""
    input_shape:      tuple = ()
    output_shape:     tuple = ()
    total_params:     int   = 0
    summary_text:     str   = ""
    is_dual_input:    bool  = False
    stat_input_size:  int   = 0
    seq_len_expected: Optional[int] = None    # None => variable length
    seq_channels:     int   = 1


state = ModelState()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _capture_summary(model: tf.keras.Model) -> str:
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    return buf.getvalue()


def _rank(shape) -> int:
    r = getattr(shape, "rank", None)
    return r if r is not None else len(shape)


def _detect_dual_input(model: tf.keras.Model) -> Tuple[bool, int]:
    """Detect a 2-D statistical input alongside the 3-D sequence input."""
    if len(model.inputs) == 2:
        for inp in model.inputs:
            if _rank(inp.shape) == 2:
                return True, int(inp.shape[-1])
        # fallback: name-based
        for inp in model.inputs:
            if "stat" in inp.name.lower():
                return True, int(inp.shape[-1])
    return False, 0


def _detect_sequence_input(model: tf.keras.Model):
    """Return the 3-D sequence input tensor (or None)."""
    for inp in model.inputs:
        if _rank(inp.shape) == 3:
            return inp
    return None


def _validate_architecture(model: tf.keras.Model, filename: str) -> None:
    layer_classes = {l.__class__.__name__.lower() for l in model.layers}
    has_conv = any("conv" in c for c in layer_classes)
    has_rnn  = any(k in c for c in layer_classes for k in ("lstm", "gru", "rnn"))
    if not (has_conv or has_rnn):
        raise ValueError(
            "Rejected: the model contains no Conv or recurrent (LSTM/GRU/RNN) layers. "
            "A temporal sequence model is required."
        )


def _load_from_path(model_path: str, filename: str) -> None:
    """Load model file and populate state with dynamically discovered dimensions."""
    global state
    model = tf.keras.models.load_model(model_path)
    _validate_architecture(model, filename)

    seq_input = _detect_sequence_input(model)
    if seq_input is None:
        raise ValueError(
            "No 3-D sequence input found. Expected a model whose sequence input "
            "has shape (None, T, C)."
        )

    seq_shape = tuple(seq_input.shape[1:])           # (T, C) — T may be None
    seq_len   = seq_shape[0]                          # int or None
    seq_chan  = seq_shape[1] if len(seq_shape) > 1 and seq_shape[1] else 1

    is_dual, stat_size = _detect_dual_input(model)

    state.model            = model
    state.model_name       = filename
    state.model_path       = model_path
    state.upload_time      = datetime.utcnow().isoformat()
    state.input_shape      = tuple(model.input_shape) if isinstance(model.input_shape, (list, tuple)) else (model.input_shape,)
    state.output_shape     = tuple(model.output_shape)
    state.total_params     = model.count_params()
    state.summary_text     = _capture_summary(model)
    state.is_dual_input    = is_dual
    state.stat_input_size  = stat_size
    state.seq_len_expected = int(seq_len) if seq_len is not None else None
    state.seq_channels     = int(seq_chan)

    logger.info(
        "Model loaded: %s | seq_len=%s | channels=%d | dual=%s | stat=%d | params=%d",
        filename, state.seq_len_expected, state.seq_channels,
        is_dual, stat_size, state.total_params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API — loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str, filename: str) -> dict:
    ext = Path(filename).suffix.lower()
    if ext not in (".keras", ".h5"):
        raise ValueError(f"Unsupported format: {ext}. Use .keras or .h5")

    logger.info("Loading model from %s", model_path)
    _load_from_path(model_path, filename)

    import database as db
    db.register_model(
        model_name      = filename,
        model_path      = model_path,
        loaded_at       = state.upload_time,
        input_shape     = str(state.input_shape),
        output_shape    = str(state.output_shape),
        total_params    = state.total_params,
        is_dual_input   = state.is_dual_input,
        stat_input_size = state.stat_input_size,
    )
    logger.info("Model registered in SQLite.")
    return get_model_info()


def auto_load_on_startup() -> bool:
    import database as db
    record = db.get_active_model()
    if not record:
        logger.info("Startup: no model in SQLite registry — waiting for upload.")
        return False
    path = record["model_path"]
    if not Path(path).exists():
        logger.warning("Startup: model path '%s' no longer on disk.", path)
        return False
    try:
        _load_from_path(path, record["model_name"])
        logger.info("Startup: auto-loaded model from SQLite registry.")
        return True
    except Exception as exc:
        logger.error("Startup: auto-load failed — %s", exc)
        return False


def unload_model() -> None:
    global state
    if state.model is not None:
        del state.model
        state.model = None
    state = ModelState()
    pipeline.reset()
    tf.keras.backend.clear_session()
    logger.info("Model unloaded.")


def get_model_info() -> dict:
    if state.model is None:
        return {"loaded": False}
    import keras as _keras
    return {
        "loaded":           True,
        "model_name":       state.model_name,
        "model_path":       state.model_path,
        "upload_time":      state.upload_time,
        "input_shape":      str(state.input_shape),
        "output_shape":     str(state.output_shape),
        "total_params":     state.total_params,
        "total_params_fmt": f"{state.total_params:,}",
        "is_dual_input":    state.is_dual_input,
        "stat_input_size":  state.stat_input_size,
        "seq_len_expected": state.seq_len_expected,        # None => variable
        "is_variable_length": state.seq_len_expected is None,
        "seq_channels":     state.seq_channels,
        "summary":          state.summary_text,
        "tf_version":       tf.__version__,
        "keras_version":    getattr(_keras, "__version__", "unknown"),
        "architecture":     "CNN-LSTM" if state.is_dual_input else "Sequence model",
    }


def is_model_loaded() -> bool:
    return state.model is not None


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility check
# ─────────────────────────────────────────────────────────────────────────────
def check_compatibility(uploaded_len: int, strategy: str = "last_n") -> dict:
    """Compare uploaded sequence length to what the model expects."""
    if state.model is None:
        return {"compatible": False, "reason": "No model loaded."}

    T = state.seq_len_expected
    if T is None:
        return {
            "compatible": True,
            "expected_len": None,
            "uploaded_len": uploaded_len,
            "variable_length": True,
            "preprocessing_needed": False,
            "reason": "Model accepts variable-length sequences — full series sent as-is.",
        }

    if uploaded_len == T:
        return {
            "compatible": True,
            "expected_len": T,
            "uploaded_len": uploaded_len,
            "variable_length": False,
            "preprocessing_needed": False,
            "reason": "Uploaded sequence length matches the model exactly.",
        }

    # Mismatch — solvable with preprocessing
    if uploaded_len > T:
        suggestion = "last_n (recent readings) or interpolate (resample whole series)"
    else:
        suggestion = "pad (zero-fill to model length) or interpolate"

    return {
        "compatible": True,            # solvable via preprocessing
        "expected_len": T,
        "uploaded_len": uploaded_len,
        "variable_length": False,
        "preprocessing_needed": True,
        "selected_strategy": strategy,
        "reason": f"Uploaded length {uploaded_len} != model length {T}. "
                  f"Will apply '{strategy}' preprocessing.",
        "suggested_strategy": suggestion,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction core — dynamic
# ─────────────────────────────────────────────────────────────────────────────
def _model_ready(raw_2d: np.ndarray, strategy: str) -> np.ndarray:
    """Resize raw (N,L) to the model's expected length (N,T). Variable-length → as-is."""
    T = state.seq_len_expected
    if T is None or raw_2d.shape[1] == T:
        return raw_2d.astype(np.float32)
    return preprocessing.resize_sequences(raw_2d, T, strategy)


def _build_stat(raw_ready_2d: np.ndarray, fit_scaler: bool) -> np.ndarray:
    """Compute + scale statistical features. Raises if count != model expectation."""
    raw_feats = extract_features(raw_ready_2d)
    if raw_feats.shape[1] != state.stat_input_size:
        raise ValueError(
            f"Statistical-feature mismatch: the feature extractor produced "
            f"{raw_feats.shape[1]} features but the model requires "
            f"{state.stat_input_size}. Cannot generate the exact features this "
            f"model was trained on — refusing to substitute zeros."
        )
    return pipeline.fit_transform(raw_ready_2d) if fit_scaler else pipeline.transform(raw_ready_2d)


def _raw_predict(seq_ready_2d: np.ndarray, stat: Optional[np.ndarray], batch_size: int) -> np.ndarray:
    """
    The ONLY inference choke-point. Scales the sequence per-row, assembles inputs,
    and runs tensorflow.keras model.predict(). Logs full verification afterwards.
    There is no fallback path — if this fails, prediction fails.
    """
    if state.model is None:
        raise RuntimeError(NO_MODEL_MSG)

    seq_scaled = scale_sequences(seq_ready_2d)
    L = seq_scaled.shape[1]
    seq = seq_scaled.reshape(-1, L, state.seq_channels).astype(np.float32)

    if state.is_dual_input:
        if stat is None:
            raise RuntimeError("Model requires stat_input but none was provided.")
        inputs = {"sequence_input": seq, "stat_input": stat.astype(np.float32)}
        in_shape = f"[{seq.shape}, {stat.shape}]"
    else:
        inputs = seq
        in_shape = str(seq.shape)

    out = state.model.predict(inputs, verbose=0, batch_size=batch_size)
    probs = out.flatten().astype(np.float32)

    _log_verification(in_shape, out.shape, probs)
    return probs


def _log_verification(input_shape: str, output_shape, probs: np.ndarray) -> None:
    """Record + log the verification trail for this model.predict() call."""
    import keras as _keras
    raw0  = float(probs[0]) if len(probs) else float("nan")
    label = "Theft" if raw0 >= 0.5 else "Normal"
    ts    = datetime.utcnow().isoformat()

    last_prediction.clear()
    last_prediction.update({
        "active_model":    state.model_name,
        "engine":          "TensorFlow / Keras",
        "tf_version":      tf.__version__,
        "keras_version":   getattr(_keras, "__version__", "unknown"),
        "input_shape":     input_shape,
        "output_shape":    str(tuple(output_shape)),
        "raw_output":      round(raw0, 6),
        "predicted_label": label,
        "n_rows":          int(len(probs)),
        "timestamp":       ts,
    })

    logger.info(
        "PREDICTION VERIFIED | Active Model: %s | Engine: TensorFlow/Keras | "
        "TF %s | Keras %s | Input Shape: %s | Output Shape: %s | "
        "Raw Output: %.6f | Predicted Label: %s | rows=%d | %s",
        state.model_name, tf.__version__, getattr(_keras, "__version__", "unknown"),
        input_shape, tuple(output_shape), raw0, label, len(probs), ts,
    )


def predict_sequences(
    raw_2d: np.ndarray,
    strategy: str = "last_n",
    threshold: float = 0.5,
    fit_scaler: bool = True,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Predict an (N, L) raw-readings array. Handles dynamic length + all strategies.
    Returns (N,) probabilities. PROOF: real model.predict() on every row.
    """
    if state.model is None:
        raise RuntimeError(NO_MODEL_MSG)

    raw_2d = np.asarray(raw_2d, dtype=np.float32)
    T = state.seq_len_expected

    # ── Sliding window (only when uploaded longer than model length) ──────────
    if strategy == "sliding_window" and T is not None and raw_2d.shape[1] > T:
        all_windows, row_index = [], []
        for ri, row in enumerate(raw_2d):
            wins = preprocessing.windows_for_row(row, T)
            for w in wins:
                all_windows.append(w)
                row_index.append(ri)
        win_arr = np.vstack(all_windows).astype(np.float32)

        stat = _build_stat(win_arr, fit_scaler) if state.is_dual_input else None
        win_probs = _raw_predict(win_arr, stat, batch_size)

        # aggregate per customer by MAX (theft in any window flags the customer)
        probs = np.zeros(len(raw_2d), dtype=np.float32)
        counts = np.zeros(len(raw_2d), dtype=np.int32)
        for ri, p in zip(row_index, win_probs):
            probs[ri] = max(probs[ri], p)
            counts[ri] += 1
        logger.info("Sliding-window: %d windows across %d customers", len(win_arr), len(raw_2d))
        return probs

    # ── Standard path (resize once, predict once) ─────────────────────────────
    ready = _model_ready(raw_2d, strategy)
    stat  = _build_stat(ready, fit_scaler) if state.is_dual_input else None
    return _raw_predict(ready, stat, batch_size)


def classify(prob: float, threshold: float) -> dict:
    pred = 1 if prob >= threshold else 0
    conf = prob if pred == 1 else (1.0 - prob)
    risk = round(prob * 100, 2)
    level = "High" if risk >= 75 else "Medium" if risk >= 40 else "Low"
    return {
        "probability":  round(float(prob), 6),
        "prediction":   pred,
        "confidence":   round(float(conf), 6),
        "risk_score":   risk,
        "risk_level":   level,
        "status":       "Theft" if pred == 1 else "Normal",
    }


def predict_one(readings: np.ndarray, strategy: str = "last_n", threshold: float = 0.5) -> dict:
    """
    Predict ONE customer of ANY length. Resizes to the model length if needed.
    Reuses the scaler fitted on the last batch (single-sample fit is meaningless).
    """
    if state.model is None:
        raise RuntimeError(NO_MODEL_MSG)

    r = np.asarray(readings, dtype=np.float32).flatten().reshape(1, -1)
    probs = predict_sequences(r, strategy=strategy, threshold=threshold, fit_scaler=False)
    prob  = float(probs[0])

    result = classify(prob, threshold)
    T = state.seq_len_expected
    result.update({
        "label":          "Electricity Theft Detected" if result["prediction"] == 1 else "Normal Customer",
        "threshold_used": threshold,
        "model_name":     state.model_name,
        "uploaded_len":   int(r.shape[1]),
        "model_len":      T,
        "strategy_used":  "none" if (T is None or r.shape[1] == T) else strategy,
        "predict_proof":  f"model.predict(x) x.shape=(1,{T if T else r.shape[1]},{state.seq_channels}) out={round(prob,6)}",
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible wrappers (older callers)
# ─────────────────────────────────────────────────────────────────────────────
def predict_batch(readings, stat_feats=None, batch_size=256, threshold=0.5) -> np.ndarray:
    """Compat shim: callers that already computed stat_feats. Prefer predict_sequences()."""
    if stat_feats is not None and state.is_dual_input:
        return _raw_predict(np.asarray(readings, np.float32), stat_feats, batch_size)
    return predict_sequences(readings, strategy="last_n", threshold=threshold,
                             fit_scaler=True, batch_size=batch_size)


def predict_single(readings: np.ndarray, threshold: float = 0.5) -> dict:
    """Compat shim → predict_one()."""
    return predict_one(readings, strategy="last_n", threshold=threshold)
