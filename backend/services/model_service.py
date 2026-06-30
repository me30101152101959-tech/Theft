"""
CNN-LSTM Model Service
======================
Loads, validates, and runs inference with the uploaded CNN-LSTM model.

Rules enforced:
  ▸ ONLY .keras / .h5 formats accepted
  ▸ Input must be (None,26,1) for sequence_input
  ▸ No BiGRU / BiLSTM / Ensemble / Transfer-Learning models accepted
  ▸ No fake / fallback predictions — real TF inference only
  ▸ FLAG is NEVER used during prediction
  ▸ Model registry stored in SQLite (survives restarts)
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

from services.feature_service import pipeline

logger = logging.getLogger(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# Keras 3 / Keras 2 compatibility shim
# (strips quantization_config and flattens DTypePolicy dicts)
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
    model:           Optional[tf.keras.Model] = None
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
def _capture_summary(model: tf.keras.Model) -> str:
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    return buf.getvalue()


def _detect_dual_input(model: tf.keras.Model) -> Tuple[bool, int]:
    if len(model.inputs) == 2:
        for inp in model.inputs:
            if "stat" in inp.name.lower():
                return True, int(inp.shape[-1])
    return False, 0


def _validate_architecture(model: tf.keras.Model, filename: str) -> None:
    layer_classes = {l.__class__.__name__.lower() for l in model.layers}
    name_lower    = filename.lower()

    for banned in ("bigru", "bilstm", "ensemble", "transfer"):
        if banned in name_lower:
            raise ValueError(
                f"Rejected: filename contains '{banned}'. Only CNN-LSTM models are accepted."
            )

    has_conv = any("conv1d" in c for c in layer_classes)
    has_lstm = any("lstm" in c for c in layer_classes)

    if not (has_conv or has_lstm):
        raise ValueError(
            "Rejected: no Conv1D or LSTM layers found. Only CNN-LSTM architectures are accepted."
        )

    if "bidirectional" in layer_classes:
        raise ValueError(
            "Rejected: Bidirectional layer detected (BiGRU/BiLSTM). "
            "Only CNN-LSTM models are accepted."
        )


def _load_from_path(model_path: str, filename: str) -> None:
    """Internal: load model file and update state (no DB write)."""
    global state
    model = tf.keras.models.load_model(model_path)
    _validate_architecture(model, filename)

    seq_input = None
    for inp in model.inputs:
        if inp.shape.rank == 3:
            seq_input = inp
            break

    if seq_input is None:
        raise ValueError("No 3-D sequence input found. Expected shape (None,26,1).")

    seq_shape = tuple(seq_input.shape[1:])
    if seq_shape != (26, 1):
        raise ValueError(f"Incompatible input shape {seq_shape}. Expected (26,1).")

    is_dual, stat_size = _detect_dual_input(model)

    state.model          = model
    state.model_name     = filename
    state.model_path     = model_path
    state.upload_time    = datetime.utcnow().isoformat()
    state.input_shape    = tuple(model.input_shape) if isinstance(model.input_shape, (list, tuple)) else (model.input_shape,)
    state.output_shape   = tuple(model.output_shape)
    state.total_params   = model.count_params()
    state.summary_text   = _capture_summary(model)
    state.is_dual_input  = is_dual
    state.stat_input_size = stat_size

    logger.info("Model loaded: %s | dual=%s | params=%d", filename, is_dual, state.total_params)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str, filename: str) -> dict:
    """
    Load, validate, register in SQLite, and update in-memory state.
    Raises ValueError on any validation failure.
    """
    ext = Path(filename).suffix.lower()
    if ext not in (".keras", ".h5"):
        raise ValueError(f"Unsupported format: {ext}. Use .keras or .h5")

    logger.info("Loading model from %s", model_path)
    _load_from_path(model_path, filename)

    # Register in SQLite
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
    """
    Called from FastAPI lifespan.
    Reads latest model from SQLite registry; loads file if still on disk.
    """
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
    return {
        "loaded":          True,
        "model_name":      state.model_name,
        "model_path":      state.model_path,
        "upload_time":     state.upload_time,
        "input_shape":     str(state.input_shape),
        "output_shape":    str(state.output_shape),
        "total_params":    state.total_params,
        "total_params_fmt": f"{state.total_params:,}",
        "is_dual_input":   state.is_dual_input,
        "stat_input_size": state.stat_input_size,
        "summary":         state.summary_text,
        "architecture":    "CNN-LSTM",
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
            raise RuntimeError("Model requires stat_input but no features provided.")
        return {"sequence_input": seq, "stat_input": stat_feats.astype(np.float32)}
    return seq


def predict_batch(
    readings:   np.ndarray,
    stat_feats: Optional[np.ndarray] = None,
    batch_size: int   = 256,
    threshold:  float = 0.5,
) -> np.ndarray:
    """
    Run batch CNN-LSTM prediction.
    readings   : (N, 26) float32
    stat_feats : (N, 59) float32 — required for dual-input models
    Returns    : (N,) float32 probabilities [0,1]
    PROOF      : model.predict(inputs, verbose=0, batch_size=256)  inputs.shape=(N,26,1)
    """
    if state.model is None:
        raise RuntimeError("No model loaded.")

    n      = len(readings)
    probs  = []

    for i in range(0, n, batch_size):
        r_batch = readings[i: i + batch_size]
        s_batch = stat_feats[i: i + batch_size] if stat_feats is not None else None
        inputs  = _build_inputs(r_batch, s_batch)
        out     = state.model.predict(inputs, verbose=0, batch_size=batch_size)
        probs.append(out.flatten())

    return np.concatenate(probs, axis=0).astype(np.float32)


def predict_single(readings: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Predict ONE customer.
    readings : (26,) float32
    Returns  : dict with probability, prediction, confidence, risk_score.
    PROOF    : model.predict(x) where x.shape == (1, 26, 1)
    """
    if state.model is None:
        raise RuntimeError("No model loaded.")

    r = readings.flatten().astype(np.float32)
    if len(r) != 26:
        raise ValueError(f"Expected 26 readings, got {len(r)}.")

    r2d  = r.reshape(1, 26)
    stat = pipeline.transform(r2d)
    inputs = _build_inputs(r2d, stat if state.is_dual_input else None)
    out  = state.model.predict(inputs, verbose=0)
    prob = float(out.flatten()[0])

    prediction = 1 if prob >= threshold else 0
    confidence = prob if prediction == 1 else (1.0 - prob)
    risk_score = round(prob * 100, 2)

    risk_level = (
        "High"   if risk_score >= 75 else
        "Medium" if risk_score >= 40 else
        "Low"
    )

    return {
        "probability":   round(prob, 6),
        "prediction":    prediction,
        "confidence":    round(confidence, 6),
        "risk_score":    risk_score,
        "risk_level":    risk_level,
        "status":        "Theft" if prediction == 1 else "Normal",
        "label":         "Electricity Theft Detected" if prediction == 1 else "Normal Customer",
        "threshold_used": threshold,
        "model_name":    state.model_name,
        "predict_proof": f"model.predict(x) x.shape=(1,26,1) out={round(prob,6)}",
    }
