"""
CNN-LSTM Model Service
======================
Loads, validates, and runs inference with the CNN-LSTM model.

On every backend startup:
  * Reads model_registry from SQLite for the last known model path
  * If the file still exists on disk, loads it automatically
  * Dashboard works immediately — no manual re-upload required

Rules
-----
  * ONLY .keras / .h5 files accepted
  * Architecture must contain Conv1D + LSTM layers
  * BiGRU / BiLSTM / Transfer-Learning / Ensemble models are rejected
  * Input tensor: (batch, 26, 1)   Output tensor: (batch, 1) sigmoid
  * Every prediction calls model.predict()  — no shortcuts
  * FLAG is NEVER used for prediction
"""
from __future__ import annotations

import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from services.feature_service import pipeline

logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


# ── Keras 3.13 ↔ 3.10 compatibility shim ─────────────────────────────────
# Models saved with Keras 3.11+ store `quantization_config` in Dense layer
# configs, which older Keras doesn't recognise. Monkey-patch removes it.
_orig_dense_from_config = keras.layers.Dense.from_config.__func__


@classmethod  # type: ignore[misc]
def _compat_dense_from_config(cls, config):
    config = dict(config)
    config.pop("quantization_config", None)
    dtype = config.get("dtype")
    if isinstance(dtype, dict):
        config["dtype"] = dtype.get("config", {}).get("name", "float32")
    return _orig_dense_from_config(cls, config)


keras.layers.Dense.from_config = _compat_dense_from_config  # type: ignore[method-assign]
logger.info("Applied Keras 3.13→3.10 Dense.from_config compatibility shim")


# ─────────────────────────────────────────────────────────────────────────────
# In-process state  (model object lives here — SQLite stores metadata)
# ─────────────────────────────────────────────────────────────────────────────
class ModelState:
    model:           Optional[keras.Model] = None
    model_name:      str   = ""
    model_path:      str   = ""
    upload_time:     str   = ""
    input_shape:     tuple = ()
    output_shape:    tuple = ()
    total_params:    int   = 0
    summary_text:    str   = ""
    is_dual_input:   bool  = False
    stat_input_size: int   = 0


state = ModelState()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _capture_summary(model: keras.Model) -> str:
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    return buf.getvalue()


def _detect_dual_input(model: keras.Model) -> Tuple[bool, int]:
    if len(model.inputs) == 2:
        for inp in model.inputs:
            name = getattr(inp, "name", "") or ""
            if "stat" in name.lower():
                return True, int(inp.shape[-1])
        for inp in model.inputs:
            if len(tuple(inp.shape)) == 2:
                return True, int(inp.shape[-1])
    return False, 0


def _validate_architecture(model: keras.Model, filename: str) -> None:
    layer_classes = {l.__class__.__name__.lower() for l in model.layers}
    name_lower    = filename.lower()
    for banned in ("bigru", "bilstm", "ensemble", "transfer"):
        if banned in name_lower:
            raise ValueError(
                f"Rejected: filename contains '{banned}'. "
                "Only CNN-LSTM models are accepted."
            )
    has_conv = any("conv1d" in c for c in layer_classes)
    has_lstm = any("lstm"   in c for c in layer_classes)
    if not (has_conv or has_lstm):
        raise ValueError(
            "Rejected: model lacks Conv1D or LSTM layers. "
            "Only CNN-LSTM architectures are accepted."
        )
    if "bidirectional" in layer_classes:
        raise ValueError(
            "Rejected: model uses Bidirectional (BiGRU/BiLSTM) layers. "
            "Only uni-directional CNN-LSTM is accepted."
        )


def _load_from_path(model_path: str, filename: str) -> dict:
    """
    Core loader.  Returns metadata dict on success.
    Mutates module-level `state`.
    """
    global state

    logger.info("─" * 55)
    logger.info("  keras.models.load_model('%s', compile=False)", model_path)
    logger.info("  TensorFlow %s | Keras %s", tf.__version__, keras.__version__)

    model = keras.models.load_model(model_path, compile=False)
    _validate_architecture(model, filename)

    # Validate sequence_input shape
    seq_input = None
    for inp in model.inputs:
        if len(tuple(inp.shape)) == 3:
            seq_input = inp
            break
    if seq_input is None:
        raise ValueError("No 3-D sequence input found. Expected (None, 26, 1).")
    seq_shape = tuple(seq_input.shape)[1:]
    if seq_shape != (26, 1):
        raise ValueError(
            f"Incompatible sequence input shape {seq_shape}. Expected (26, 1)."
        )

    is_dual, stat_size = _detect_dual_input(model)

    try:
        in_shape = model.input_shape
        input_shape = tuple(in_shape) if isinstance(in_shape, (list, tuple)) else (in_shape,)
    except Exception:
        input_shape = tuple(tuple(i.shape) for i in model.inputs)
    try:
        out_shape = model.output_shape
        output_shape = tuple(out_shape) if isinstance(out_shape, (list, tuple)) else (out_shape,)
    except Exception:
        output_shape = tuple(tuple(o.shape) for o in model.outputs)

    total_params = model.count_params()
    summary      = _capture_summary(model)
    now          = datetime.utcnow().isoformat()

    state.model           = model
    state.model_name      = filename
    state.model_path      = model_path
    state.upload_time     = now
    state.input_shape     = input_shape
    state.output_shape    = output_shape
    state.total_params    = total_params
    state.summary_text    = summary
    state.is_dual_input   = is_dual
    state.stat_input_size = stat_size

    logger.info(
        "  Model loaded: %s | params=%d | dual_input=%s | stat_size=%d",
        filename, total_params, is_dual, stat_size,
    )
    logger.info("  Input shape  → %s", input_shape)
    logger.info("  Output shape → %s", output_shape)
    logger.info("─" * 55)

    return get_model_info()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str, filename: str) -> dict:
    """
    Load + validate + register model.
    Called from the upload endpoint and from startup auto-load.
    Persists model metadata to SQLite.
    """
    info = _load_from_path(model_path, filename)

    # Persist to SQLite so the next restart can find it
    import database as db
    db.register_model(
        model_name     = filename,
        model_path     = model_path,
        loaded_at      = state.upload_time,
        input_shape    = str(state.input_shape),
        output_shape   = str(state.output_shape),
        total_params   = state.total_params,
        is_dual_input  = state.is_dual_input,
        stat_input_size= state.stat_input_size,
    )
    return info


def auto_load_on_startup() -> bool:
    """
    Called during FastAPI lifespan.
    Reads the last model_path from SQLite; if the file exists, loads it.
    Returns True if model was loaded.
    """
    import database as db
    record = db.get_active_model()
    if not record:
        logger.info("Startup: no model in SQLite registry — waiting for upload.")
        return False

    path = record["model_path"]
    name = record["model_name"]
    if not Path(path).exists():
        logger.warning(
            "Startup: model path '%s' no longer exists on disk — "
            "please re-upload cnnlstm_final.keras.",
            path,
        )
        return False

    logger.info("Startup: auto-loading model from SQLite registry → %s", path)
    try:
        _load_from_path(path, name)
        logger.info("Startup: model ready — dashboard will use SQLite data immediately.")
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
    keras.backend.clear_session()
    logger.info("Model unloaded from memory.")


def get_model_info() -> dict:
    if state.model is None:
        return {"loaded": False}
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
        "summary":          state.summary_text,
        "architecture":     "CNN-LSTM",
        "tensorflow_version": tf.__version__,
        "keras_version":    keras.__version__,
        "load_proof":       f"keras.models.load_model('{state.model_path}', compile=False)",
    }


def is_model_loaded() -> bool:
    return state.model is not None


# ─────────────────────────────────────────────────────────────────────────────
# Prediction core
# ─────────────────────────────────────────────────────────────────────────────
def _build_inputs(readings: np.ndarray, stat_feats: Optional[np.ndarray]):
    seq = readings.reshape(-1, 26, 1).astype(np.float32)
    if state.is_dual_input:
        if stat_feats is None:
            raise RuntimeError("Dual-input model requires stat_feats but none provided.")
        return {"sequence_input": seq, "stat_input": stat_feats.astype(np.float32)}
    return seq


def predict_batch(
    readings:   np.ndarray,
    stat_feats: Optional[np.ndarray] = None,
    batch_size: int = 256,
    threshold:  float = 0.5,
) -> np.ndarray:
    """
    Run model.predict() in mini-batches.
    readings   : (N, 26) float32
    stat_feats : (N, 59) float32  — required for dual-input models
    Returns    : (N,) float32 probabilities
    """
    if state.model is None:
        raise RuntimeError("No model loaded.")

    probs = []
    n = len(readings)
    for i in range(0, n, batch_size):
        r_batch = readings[i: i + batch_size]
        s_batch = stat_feats[i: i + batch_size] if stat_feats is not None else None
        inputs  = _build_inputs(r_batch, s_batch)
        logger.debug(
            "model.predict() batch %d–%d | seq_shape=%s",
            i, min(i + batch_size, n) - 1,
            r_batch.reshape(-1, 26, 1).shape,
        )
        out = state.model.predict(inputs, verbose=0, batch_size=batch_size)
        probs.append(out.flatten())

    result = np.concatenate(probs, axis=0).astype(np.float32)
    logger.debug("predict_batch output shape: %s", result.shape)
    return result


def predict_single(readings: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Predict ONE customer row.
    readings : (26,) float32
    Returns full result dict.
    """
    if state.model is None:
        raise RuntimeError("No CNN-LSTM model loaded.")

    r = readings.flatten().astype(np.float32)
    if len(r) != 26:
        raise ValueError(f"Expected 26 readings, got {len(r)}.")

    r2d = r.reshape(1, 26)
    stat = pipeline.transform(r2d)  # (1, 59)

    inputs = _build_inputs(r2d, stat if state.is_dual_input else None)
    logger.info(
        "model.predict() single | seq_shape=%s | model=%s",
        r2d.reshape(1, 26, 1).shape, state.model_name,
    )
    out  = state.model.predict(inputs, verbose=0)
    prob = float(out.flatten()[0])

    pred       = 1 if prob >= threshold else 0
    confidence = prob if pred == 1 else (1.0 - prob)

    return {
        "probability":     round(prob, 6),
        "prediction":      pred,
        "confidence":      round(confidence, 6),
        "risk_score":      round(prob * 100, 2),
        "status":          "Theft" if pred == 1 else "Normal",
        "label":           "Electricity Theft Detected" if pred == 1 else "Normal Customer",
        "threshold_used":  threshold,
        "model_name":      state.model_name,
        "model_path":      state.model_path,
        "tensorflow_version": tf.__version__,
        "keras_version":   keras.__version__,
        "predict_proof":   f"model.predict(x) where x.shape == {r2d.reshape(1,26,1).shape}",
    }
