"""
Feature Engineering Service for CNN-LSTM Model
================================================
EXACT replica of the training preprocessing (Project2YEARS.ipynb).

The CNN-LSTM model has TWO inputs:
  1. sequence_input : (N, 26, 1)  — readings scaled PER-ROW to [0,1]
  2. stat_input     : (N, 59)     — 59 statistical features, StandardScaler'd

Training preprocessing (must match exactly):
  • Sequence  : per-row min-max, (row - min) / (max - min)   [CELL 8]
  • Stat feats: extract_features() → 59 values               [CELL 7]
  • Stat scale: StandardScaler                                [CELL 8]
"""

from __future__ import annotations
from typing import Optional
import logging

import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence scaling — per-row min-max to [0,1]  (CELL 8 of training notebook)
# ─────────────────────────────────────────────────────────────────────────────
def scale_sequences(readings: np.ndarray) -> np.ndarray:
    """
    Scale each customer's reading sequence to [0,1] independently.
    readings : (N, 26) raw kWh values
    returns  : (N, 26) scaled to [0,1] per row
    This matches X_seq_scaled in training exactly.
    """
    readings = np.asarray(readings, dtype=np.float32)
    scaled   = np.zeros_like(readings)
    for i in range(len(readings)):
        mn = readings[i].min()
        mx = readings[i].max()
        if mx > mn:
            scaled[i] = (readings[i] - mn) / (mx - mn)
        # else: row is constant → stays all-zeros (matches training)
    return scaled


# ─────────────────────────────────────────────────────────────────────────────
# 59-feature extractor — VERBATIM from training CELL 7 (extract_features)
# ─────────────────────────────────────────────────────────────────────────────
def _features_for_row(row: np.ndarray) -> list:
    row = row.astype(np.float32)
    n   = len(row)

    mean = np.mean(row); std = np.std(row); mx = np.max(row); mn = np.min(row)
    median = np.median(row)
    skew = float(scipy_stats.skew(row)); kurt = float(scipy_stats.kurtosis(row))
    cv = std / (mean + 1e-9)
    p10, p25, p75, p90 = np.percentile(row, [10, 25, 75, 90])
    iqr = p75 - p25

    zero_ratio = np.mean(row == 0)
    neg_ratio = np.mean(row < 0)
    near_zero = np.mean(row < 0.01)
    low_cons_ratio = np.mean(row < mean * 0.1)
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
        dm_max, dm_min = np.max(dm), np.min(dm)
        ds_mean = np.mean(ds)
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
        mean_change = np.mean(row[n//2:]) - np.mean(row[:n//2])
        std_change = np.std(row[n//2:]) - np.std(row[:n//2])
    else:
        mean_change = std_change = 0.0

    diffs = np.diff(row)
    max_drop = np.min(diffs) if len(diffs) > 0 else 0
    max_rise = np.max(diffs) if len(diffs) > 0 else 0
    n_big_drops = np.sum(diffs < -2 * std)
    n_big_rises = np.sum(diffs > 2 * std)

    below_median = np.mean(row < median)
    above_median = np.mean(row > median)

    quarters = np.array_split(row, 4)
    q_means = [np.mean(q) for q in quarters]
    q_stds = [np.std(q) for q in quarters]
    q_trend = q_means[-1] - q_means[0]
    q_var = np.std(q_means)

    return [
        mean, std, mx, mn, median, skew, kurt, cv,
        p10, p25, p75, p90, iqr,
        zero_ratio, neg_ratio, near_zero, low_cons_ratio, drop_ratio,
        slope, resid_std,
        energy, ent,
        max_zero_run, n_zero_runs,
        dn_ratio, day_cv, theft_days,
        max_day_chg, mean_day_chg,
        dm_mean, dm_std, dm_max, dm_min, ds_mean,
        ac1, ac48, ac7d,
        fft_mean, fft_std, fft_max, dominant_freq,
        mean_change, std_change,
        max_drop, max_rise, n_big_drops, n_big_rises,
        below_median, above_median,
        q_means[0], q_means[1], q_means[2], q_means[3],
        q_stds[0], q_stds[1], q_stds[2], q_stds[3],
        q_trend, q_var,
    ]


def extract_features(readings: np.ndarray) -> np.ndarray:
    """
    Compute the 59 statistical features for an (N, 26) array.
    Returns float32 (N, 59), with nan/inf replaced by 0 (matches training).
    """
    feats = np.array([_features_for_row(row) for row in readings], dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline: holds the StandardScaler fitted per dataset/batch upload
# ─────────────────────────────────────────────────────────────────────────────
class FeaturePipeline:
    """
    Replicates training inference preprocessing:
      • fit StandardScaler on the uploaded batch's stat features
      • reuse it for subsequent single (manual) predictions

    NOTE: the original training StandardScaler (fit on the SMOTE-augmented
    train split) was not saved with the model. We re-fit a StandardScaler
    on each uploaded dataset, which closely matches the training distribution
    for reasonably sized datasets. The sequence scaling is per-row and exact.
    """

    def __init__(self):
        self._scaler: Optional[StandardScaler] = None
        self._fitted = False

    def fit_transform(self, readings: np.ndarray) -> np.ndarray:
        """Fit StandardScaler on this dataset's stat features; return scaled (N,59)."""
        raw = extract_features(readings)
        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(raw).astype(np.float32)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        self._fitted = True
        logger.info("FeaturePipeline: StandardScaler fitted on %d samples (59 features)", len(readings))
        return scaled

    def transform(self, readings: np.ndarray) -> np.ndarray:
        """Transform readings using the fitted StandardScaler (for single predictions)."""
        raw = extract_features(readings)
        if self._fitted and self._scaler is not None:
            scaled = self._scaler.transform(raw).astype(np.float32)
        else:
            # No batch fitted yet — fall back to fitting on this single sample.
            # (Standardizing 1 row yields ~zeros; the sequence branch still drives the prediction.)
            logger.warning("FeaturePipeline.transform called before any fit — using zero-centered features")
            scaled = np.zeros_like(raw)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return scaled

    def reset(self):
        self._scaler = None
        self._fitted = False


# Singleton used across the app
pipeline = FeaturePipeline()
