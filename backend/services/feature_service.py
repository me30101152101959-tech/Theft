"""
Feature Engineering Service for CNN-LSTM Model
================================================
The CNN-LSTM model expects TWO inputs:
  1. sequence_input : (N, 26, 1)  — raw time-series readings
  2. stat_input     : (N, 59)     — 59 statistical features

This module computes the 59 features, scales them with a MinMaxScaler
fitted on each dataset upload, and returns both tensors.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from scipy import stats as scipy_stats
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 59-feature extractor  (per customer row)
# ─────────────────────────────────────────────
def _features_for_row(r: np.ndarray) -> np.ndarray:
    """Compute 59 statistical features from one 26-reading array."""
    r = r.astype(np.float64)
    eps = 1e-8

    diff1 = np.diff(r)              # 25 values
    diff2 = np.diff(diff1)          # 24 values
    r_abs = np.abs(r)

    # ── Group 1: Basic stats (8) ──────────────────────────────────────
    g1 = [
        np.mean(r),
        np.std(r),
        np.min(r),
        np.max(r),
        np.median(r),
        np.sum(r),
        np.ptp(r),       # range
        np.var(r),
    ]

    # ── Group 2: Shape descriptors (4) ───────────────────────────────
    g2 = [
        float(scipy_stats.skew(r)),
        float(scipy_stats.kurtosis(r)),
        float(np.percentile(r, 25)),
        float(np.percentile(r, 75)),
    ]

    # ── Group 3: Percentiles (7) ──────────────────────────────────────
    g3 = [float(np.percentile(r, p)) for p in [10, 20, 30, 40, 60, 70, 80]]

    # ── Group 4: First-difference stats (8) ──────────────────────────
    g4 = [
        np.mean(diff1),
        np.std(diff1),
        np.min(diff1),
        np.max(diff1),
        np.sum(np.abs(diff1)),
        np.mean(np.abs(diff1)),
        float(np.sum(diff1 > 0)),
        float(np.sum(diff1 < 0)),
    ]

    # ── Group 5: Energy / signal power (4) ───────────────────────────
    g5 = [
        np.sqrt(np.mean(r ** 2)),          # RMS
        np.sum(r ** 2),                    # energy
        np.mean(r_abs),                    # mean absolute
        float(np.sum(r == 0)),             # zero readings
    ]

    # ── Group 6: Weekly-segment means (4) ────────────────────────────
    g6 = [np.mean(r[i * 7:(i + 1) * 7]) for i in range(4)]

    # ── Group 7: Weekly-segment std (4) ──────────────────────────────
    g7 = [np.std(r[i * 7:(i + 1) * 7]) for i in range(4)]

    # ── Group 8: Half-period ratios (4) ──────────────────────────────
    h1, h2 = r[:13], r[13:]
    g8 = [
        np.mean(h2) / (np.mean(h1) + eps),
        np.std(h2) / (np.std(h1) + eps),
        np.max(r) / (np.mean(r) + eps),
        np.min(r_abs) / (np.mean(r_abs) + eps),
    ]

    # ── Group 9: Consumption patterns (3) ────────────────────────────
    mean_r = np.mean(r)
    g9 = [
        float(np.sum(r > mean_r)),
        float(np.sum(r < mean_r)),
        np.mean(np.abs(r - mean_r)),    # MAD
    ]

    # ── Group 10: Lag-autocorrelation (2) ────────────────────────────
    if np.std(r) > eps:
        g10 = [
            float(np.corrcoef(r[:-1], r[1:])[0, 1]),
            float(np.corrcoef(r[:-2], r[2:])[0, 1]),
        ]
    else:
        g10 = [0.0, 0.0]

    # ── Group 11: Second-difference stats (4) ────────────────────────
    g11 = [
        np.mean(diff2),
        np.std(diff2),
        np.min(diff2),
        np.max(diff2),
    ]

    # ── Group 12: Linear trend (2) ────────────────────────────────────
    x = np.arange(26, dtype=np.float64)
    if np.std(r) > eps:
        slope, intercept = np.polyfit(x, r, 1)
        trend = slope * x + intercept
        g12 = [slope, np.std(r - trend)]
    else:
        g12 = [0.0, 0.0]

    # ── Group 13: Remaining misc (5) ──────────────────────────────────
    g13 = [
        float(np.percentile(r, 75) - np.percentile(r, 25)),   # IQR
        float(np.argmax(r)) / 26.0,
        float(np.argmin(r)) / 26.0,
        float(np.sum(np.abs(diff1) > np.std(diff1))),          # volatile steps
        np.max(np.abs(diff1)),                                   # max jump
    ]

    feats = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9 + g10 + g11 + g12 + g13
    assert len(feats) == 59, f"Expected 59 features, got {len(feats)}"

    arr = np.array(feats, dtype=np.float32)
    # Safety: replace nan / inf
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def compute_stat_features(readings: np.ndarray) -> np.ndarray:
    """
    Compute 59 stat features for an (N, 26) readings array.
    Returns float32 array of shape (N, 59).
    """
    return np.vstack([_features_for_row(row) for row in readings]).astype(np.float32)


# ─────────────────────────────────────────────
# Scaler (fitted per upload)
# ─────────────────────────────────────────────
class FeaturePipeline:
    """
    Fits a MinMaxScaler on the uploaded dataset and applies it during
    both batch prediction and single manual prediction.
    """

    def __init__(self):
        self._scaler: Optional[MinMaxScaler] = None
        self._fitted = False

    def fit_transform(self, readings: np.ndarray) -> np.ndarray:
        """Fit scaler on dataset readings and return scaled (N,59)."""
        raw = compute_stat_features(readings)
        self._scaler = MinMaxScaler()
        scaled = self._scaler.fit_transform(raw).astype(np.float32)
        self._fitted = True
        logger.info("FeaturePipeline: scaler fitted on %d samples", len(readings))
        return scaled

    def transform(self, readings: np.ndarray) -> np.ndarray:
        """Transform new readings using the fitted scaler."""
        raw = compute_stat_features(readings)
        if self._fitted and self._scaler is not None:
            scaled = self._scaler.transform(raw).astype(np.float32)
        else:
            # No dataset scaler yet — apply per-sample MinMax so values are [0,1]
            _min = raw.min(axis=1, keepdims=True)
            _max = raw.max(axis=1, keepdims=True)
            _range = np.where((_max - _min) == 0, 1.0, _max - _min)
            scaled = ((raw - _min) / _range).astype(np.float32)
        scaled = np.where(np.isfinite(scaled), scaled, 0.0)
        return scaled

    def reset(self):
        self._scaler = None
        self._fitted = False


# Singleton used by the application
pipeline = FeaturePipeline()
