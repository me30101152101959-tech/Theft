"""
Explainable AI for the CNN-LSTM model.
Uses integrated-gradients over the scaled sequence input (real gradients from the
loaded model — no surrogate). Falls back gracefully if gradients are unavailable.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from core import engine
from core.features import scale_sequences, extract_features

# Names matching the 59-feature extractor in core/features.py
FEATURE_NAMES = [
    "mean", "std", "max", "min", "median", "skew", "kurtosis", "cv",
    "p10", "p25", "p75", "p90", "iqr",
    "zero_ratio", "neg_ratio", "near_zero", "low_cons_ratio", "drop_ratio",
    "slope", "resid_std", "energy", "entropy",
    "max_zero_run", "n_zero_runs",
    "day_night_ratio", "day_cv", "theft_days", "max_day_chg", "mean_day_chg",
    "daymean_mean", "daymean_std", "daymean_max", "daymean_min", "daystd_mean",
    "autocorr_1", "autocorr_48", "autocorr_7d",
    "fft_mean", "fft_std", "fft_max", "dominant_freq",
    "mean_change", "std_change", "max_drop", "max_rise", "n_big_drops", "n_big_rises",
    "below_median", "above_median",
    "q1_mean", "q2_mean", "q3_mean", "q4_mean",
    "q1_std", "q2_std", "q3_std", "q4_std", "q_trend", "q_var",
]


def integrated_gradients(readings: np.ndarray, steps: int = 32) -> Optional[dict]:
    """
    Per-timestep attribution for a single sequence via integrated gradients.
    Returns {'timestep_importance': [...], 'baseline_prob', 'prob'} or None.
    """
    if not engine.is_model_loaded():
        return None
    tf = engine._tf()

    r = np.asarray(readings, dtype=np.float32).flatten()
    T = engine.state.seq_len_expected or len(r)
    if len(r) != T:
        from core import preprocessing
        r = preprocessing.resize_row(r, T, "last_n")

    seq = scale_sequences(r.reshape(1, -1)).reshape(1, T, engine.state.seq_channels)
    seq_tf = tf.convert_to_tensor(seq, dtype=tf.float32)
    baseline = tf.zeros_like(seq_tf)

    if engine.state.is_dual_input:
        feats = extract_features(r.reshape(1, -1))
        from core.features import pipeline
        stat = pipeline.transform(r.reshape(1, -1)) if pipeline._fitted else np.zeros_like(feats)
        stat_tf = tf.convert_to_tensor(stat.astype(np.float32))

    alphas = tf.linspace(0.0, 1.0, steps)
    grads = []
    try:
        for a in alphas:
            interp = baseline + a * (seq_tf - baseline)
            with tf.GradientTape() as tape:
                tape.watch(interp)
                if engine.state.is_dual_input:
                    out = engine.state.model({"sequence_input": interp, "stat_input": stat_tf})
                else:
                    out = engine.state.model(interp)
                out = tf.reduce_sum(out)
            grads.append(tape.gradient(out, interp))
        avg_grad = tf.reduce_mean(tf.stack(grads), axis=0)
        ig = (seq_tf - baseline) * avg_grad
        importance = tf.reduce_sum(tf.abs(ig), axis=-1).numpy().flatten()
    except Exception:
        return None

    s = importance.sum()
    importance = (importance / s).tolist() if s > 0 else importance.tolist()
    prob = float(engine.state.model(
        {"sequence_input": seq_tf, "stat_input": stat_tf} if engine.state.is_dual_input else seq_tf
    ).numpy().flatten()[0])

    return {"timestep_importance": importance, "prob": prob}


def risk_factors(readings: np.ndarray, top: int = 6) -> list:
    """Human-readable risk factors derived from the statistical features of the series."""
    r = np.asarray(readings, dtype=np.float32).flatten()
    feats = extract_features(r.reshape(1, -1))[0]
    named = dict(zip(FEATURE_NAMES, feats))
    factors = []

    if named.get("zero_ratio", 0) > 0.15:
        factors.append(("High proportion of zero readings", named["zero_ratio"], "↑ theft"))
    if named.get("max_zero_run", 0) >= 3:
        factors.append(("Long consecutive zero-consumption run", named["max_zero_run"], "↑ theft"))
    if named.get("drop_ratio", 0) > 0.2:
        factors.append(("Frequent sharp consumption drops", named["drop_ratio"], "↑ theft"))
    if named.get("cv", 0) > 1.0:
        factors.append(("Very high consumption variability", named["cv"], "↑ theft"))
    if named.get("slope", 0) < 0:
        factors.append(("Declining consumption trend", named["slope"], "↑ theft"))
    if named.get("low_cons_ratio", 0) > 0.3:
        factors.append(("Many abnormally low readings", named["low_cons_ratio"], "↑ theft"))
    if named.get("q_trend", 0) < 0:
        factors.append(("Downward quarter-over-quarter trend", named["q_trend"], "↑ theft"))
    if not factors:
        factors.append(("Stable, regular consumption pattern", named.get("cv", 0), "↓ normal"))

    return factors[:top]
