"""
CNN-LSTM Prediction Engine  (Streamlit, exclusive)
===================================================
Loads ONLY a Keras CNN-LSTM model via tensorflow.keras.models.load_model()
and runs real model.predict() inference. No fallback / mock / surrogate models.

All dimensions (sequence length T, channels, stat-feature count) are discovered
from the loaded model at runtime — nothing is hardcoded.

Preprocessing is the exact training-time pipeline (see core/features.py):
  • sequence  : per-row min-max scaling to [0,1]
  • stat input: 59 statistical features → StandardScaler
"""
from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

logger = logging.getLogger("etd_engine")

# Exact message required when no valid CNN-LSTM model is available.
# The application STOPS all prediction and shows this — it never falls back to
# any other model or simulated/random/rule-based output.
NO_MODEL_MSG = "No active CNN-LSTM model loaded."

# Default bundled model (auto-loaded so the app works immediately on deploy).
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "assets" / "cnnlstm_final.keras"


# ─────────────────────────────────────────────────────────────────────────────
# Lazy TF import + Keras compatibility shim (Keras 3 ← model saved on 2.x)
# ─────────────────────────────────────────────────────────────────────────────
_TF = None


def _tf():
    """Import TensorFlow lazily so the app boots fast and TF errors are catchable."""
    global _TF
    if _TF is None:
        import tensorflow as tf  # noqa: WPS433
        tf.get_logger().setLevel("ERROR")
        try:
            import keras  # noqa: WPS433
            _orig = keras.layers.Dense.from_config.__func__

            @classmethod  # type: ignore[misc]
            def _compat(cls, config):
                config = dict(config)
                config.pop("quantization_config", None)
                dtype = config.get("dtype")
                if isinstance(dtype, dict):
                    config["dtype"] = dtype.get("config", {}).get("name", "float32")
                return _orig(cls, config)

            keras.layers.Dense.from_config = _compat
        except Exception:  # pragma: no cover
            pass
        _TF = tf
    return _TF


# ─────────────────────────────────────────────────────────────────────────────
# Model state (held in a singleton; Streamlit keeps it via cache_resource)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelState:
    model: object = None
    model_name: str = ""
    model_path: str = ""
    upload_time: str = ""
    input_shape: tuple = ()
    output_shape: tuple = ()
    total_params: int = 0
    summary_text: str = ""
    is_dual_input: bool = False
    stat_input_size: int = 0
    seq_len_expected: Optional[int] = None  # None => variable length
    seq_channels: int = 1
    last_prediction: dict = field(default_factory=dict)


state = ModelState()


# ─────────────────────────────────────────────────────────────────────────────
# Introspection helpers
# ─────────────────────────────────────────────────────────────────────────────
def _capture_summary(model) -> str:
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    return buf.getvalue()


def _rank(shape) -> int:
    r = getattr(shape, "rank", None)
    return r if r is not None else len(shape)


def _detect_dual_input(model) -> Tuple[bool, int]:
    if len(model.inputs) == 2:
        for inp in model.inputs:
            if _rank(inp.shape) == 2:
                return True, int(inp.shape[-1])
        for inp in model.inputs:
            if "stat" in inp.name.lower():
                return True, int(inp.shape[-1])
    return False, 0


def _detect_sequence_input(model):
    for inp in model.inputs:
        if _rank(inp.shape) == 3:
            return inp
    return None


def _validate_architecture(model) -> None:
    classes = {l.__class__.__name__.lower() for l in model.layers}
    has_conv = any("conv" in c for c in classes)
    has_rnn = any(k in c for c in classes for k in ("lstm", "gru", "rnn"))
    if not (has_conv or has_rnn):
        raise ValueError(
            "Rejected: model has no Conv or recurrent (LSTM/GRU/RNN) layers — "
            "a temporal sequence model is required."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str, filename: Optional[str] = None) -> dict:
    """Load a .keras / .h5 CNN-LSTM model and populate the engine state."""
    tf = _tf()
    filename = filename or Path(model_path).name
    ext = Path(filename).suffix.lower()
    if ext not in (".keras", ".h5"):
        raise ValueError(f"Unsupported format '{ext}'. Use .keras or .h5")

    model = tf.keras.models.load_model(model_path)
    _validate_architecture(model)

    seq_input = _detect_sequence_input(model)
    if seq_input is None:
        raise ValueError("No 3-D sequence input found (expected shape (None, T, C)).")

    seq_shape = tuple(seq_input.shape[1:])
    seq_len = seq_shape[0]
    seq_chan = seq_shape[1] if len(seq_shape) > 1 and seq_shape[1] else 1
    is_dual, stat_size = _detect_dual_input(model)

    state.model = model
    state.model_name = filename
    state.model_path = model_path
    state.upload_time = datetime.utcnow().isoformat()
    state.input_shape = (
        tuple(model.input_shape)
        if isinstance(model.input_shape, (list, tuple))
        else (model.input_shape,)
    )
    state.output_shape = tuple(model.output_shape)
    state.total_params = int(model.count_params())
    state.summary_text = _capture_summary(model)
    state.is_dual_input = is_dual
    state.stat_input_size = stat_size
    state.seq_len_expected = int(seq_len) if seq_len is not None else None
    state.seq_channels = int(seq_chan)

    logger.info(
        "Model loaded: %s | seq_len=%s | channels=%d | dual=%s | stat=%d | params=%d",
        filename, state.seq_len_expected, state.seq_channels,
        is_dual, stat_size, state.total_params,
    )
    return get_model_info()


def auto_load_default() -> bool:
    """Load the bundled model so a freshly-deployed app works with no upload."""
    if state.model is not None:
        return True
    if DEFAULT_MODEL.exists():
        try:
            load_model(str(DEFAULT_MODEL), DEFAULT_MODEL.name)
            return True
        except Exception as exc:  # pragma: no cover
            logger.error("auto_load_default failed: %s", exc)
    return False


def unload_model() -> None:
    from core.features import pipeline
    if state.model is not None:
        try:
            _tf().keras.backend.clear_session()
        except Exception:
            pass
    state.model = None
    state.model_name = ""
    state.seq_len_expected = None
    state.last_prediction = {}
    pipeline.reset()


def is_model_loaded() -> bool:
    return state.model is not None


def get_model_info() -> dict:
    if state.model is None:
        return {"loaded": False}
    tf = _tf()
    import keras as _keras
    return {
        "loaded": True,
        "model_name": state.model_name,
        "model_path": state.model_path,
        "upload_time": state.upload_time,
        "input_shape": str(state.input_shape),
        "output_shape": str(state.output_shape),
        "total_params": state.total_params,
        "total_params_fmt": f"{state.total_params:,}",
        "is_dual_input": state.is_dual_input,
        "stat_input_size": state.stat_input_size,
        "seq_len_expected": state.seq_len_expected,
        "is_variable_length": state.seq_len_expected is None,
        "seq_channels": state.seq_channels,
        "summary": state.summary_text,
        "tf_version": tf.__version__,
        "keras_version": getattr(_keras, "__version__", "unknown"),
        "architecture": "CNN-LSTM" if state.is_dual_input else "Sequence model",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility check
# ─────────────────────────────────────────────────────────────────────────────
def check_compatibility(uploaded_len: int, strategy: str = "last_n") -> dict:
    if state.model is None:
        return {"compatible": False, "reason": "No model loaded."}
    T = state.seq_len_expected
    if T is None:
        return {"compatible": True, "expected_len": None, "uploaded_len": uploaded_len,
                "variable_length": True, "preprocessing_needed": False,
                "reason": "Model accepts variable-length sequences — sent as-is."}
    if uploaded_len == T:
        return {"compatible": True, "expected_len": T, "uploaded_len": uploaded_len,
                "variable_length": False, "preprocessing_needed": False,
                "reason": "Uploaded length matches the model exactly."}
    suggestion = ("last_n or interpolate" if uploaded_len > T else "pad or interpolate")
    return {"compatible": True, "expected_len": T, "uploaded_len": uploaded_len,
            "variable_length": False, "preprocessing_needed": True,
            "selected_strategy": strategy, "suggested_strategy": suggestion,
            "reason": f"Uploaded length {uploaded_len} != model length {T}. "
                      f"Will apply '{strategy}' preprocessing."}


# ─────────────────────────────────────────────────────────────────────────────
# Prediction core
# ─────────────────────────────────────────────────────────────────────────────
def _model_ready(raw_2d: np.ndarray, strategy: str) -> np.ndarray:
    from core import preprocessing
    T = state.seq_len_expected
    if T is None or raw_2d.shape[1] == T:
        return raw_2d.astype(np.float32)
    return preprocessing.resize_sequences(raw_2d, T, strategy)


def _build_stat(raw_ready_2d: np.ndarray, fit_scaler: bool) -> np.ndarray:
    from core.features import pipeline, extract_features
    raw_feats = extract_features(raw_ready_2d)
    if raw_feats.shape[1] != state.stat_input_size:
        raise ValueError(
            f"Statistical-feature mismatch: extractor produced {raw_feats.shape[1]} "
            f"features but the model requires {state.stat_input_size}. "
            f"Refusing to substitute zeros."
        )
    return pipeline.fit_transform(raw_ready_2d) if fit_scaler else pipeline.transform(raw_ready_2d)


def _raw_predict(seq_ready_2d: np.ndarray, stat: Optional[np.ndarray], batch_size: int) -> np.ndarray:
    """The ONLY inference choke-point. Real tensorflow.keras model.predict()."""
    from core.features import scale_sequences
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
    tf = _tf()
    import keras as _keras
    raw0 = float(probs[0]) if len(probs) else float("nan")
    label = "Theft" if raw0 >= 0.5 else "Normal"
    state.last_prediction = {
        "active_model": state.model_name,
        "engine": "TensorFlow / Keras",
        "tf_version": tf.__version__,
        "keras_version": getattr(_keras, "__version__", "unknown"),
        "input_shape": input_shape,
        "output_shape": str(tuple(output_shape)),
        "raw_output": round(raw0, 6),
        "predicted_label": label,
        "n_rows": int(len(probs)),
        "timestamp": datetime.utcnow().isoformat(),
    }


def predict_sequences(raw_2d, strategy="last_n", threshold=0.5,
                      fit_scaler=True, batch_size=256) -> np.ndarray:
    """Predict an (N, L) raw-readings array. Returns (N,) probabilities."""
    from core import preprocessing
    if state.model is None:
        raise RuntimeError(NO_MODEL_MSG)

    raw_2d = np.asarray(raw_2d, dtype=np.float32)
    T = state.seq_len_expected

    if strategy == "sliding_window" and T is not None and raw_2d.shape[1] > T:
        all_windows, row_index = [], []
        for ri, row in enumerate(raw_2d):
            for w in preprocessing.windows_for_row(row, T):
                all_windows.append(w)
                row_index.append(ri)
        win_arr = np.vstack(all_windows).astype(np.float32)
        stat = _build_stat(win_arr, fit_scaler) if state.is_dual_input else None
        win_probs = _raw_predict(win_arr, stat, batch_size)
        probs = np.zeros(len(raw_2d), dtype=np.float32)
        for ri, p in zip(row_index, win_probs):
            probs[ri] = max(probs[ri], p)
        return probs

    ready = _model_ready(raw_2d, strategy)
    stat = _build_stat(ready, fit_scaler) if state.is_dual_input else None
    return _raw_predict(ready, stat, batch_size)


def classify(prob: float, threshold: float) -> dict:
    pred = 1 if prob >= threshold else 0
    conf = prob if pred == 1 else (1.0 - prob)
    risk = round(prob * 100, 2)
    level = "High" if risk >= 75 else "Medium" if risk >= 40 else "Low"
    return {
        "probability": round(float(prob), 6),
        "prediction": pred,
        "confidence": round(float(conf), 6),
        "risk_score": risk,
        "risk_level": level,
        "status": "Theft" if pred == 1 else "Normal",
    }


def predict_one(readings, strategy="last_n", threshold=0.5) -> dict:
    """Predict ONE customer of any length."""
    if state.model is None:
        raise RuntimeError(NO_MODEL_MSG)
    r = np.asarray(readings, dtype=np.float32).flatten().reshape(1, -1)
    probs = predict_sequences(r, strategy=strategy, threshold=threshold, fit_scaler=False)
    prob = float(probs[0])
    result = classify(prob, threshold)
    T = state.seq_len_expected
    result.update({
        "label": "Electricity Theft Detected" if result["prediction"] == 1 else "Normal Customer",
        "threshold_used": threshold,
        "model_name": state.model_name,
        "uploaded_len": int(r.shape[1]),
        "model_len": T,
        "strategy_used": "none" if (T is None or r.shape[1] == T) else strategy,
    })
    return result
