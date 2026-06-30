"""
Sequence Preprocessing Strategies
==================================
Maps an uploaded sequence of arbitrary length L to the length T the model
expects. NOTHING here is hardcoded — T is discovered from the loaded model and
L is discovered from the uploaded dataset.

Strategies
----------
  last_n        : keep the most recent T readings (left-pad with zeros if L < T)
  truncate      : keep the first T readings  (right-pad with zeros if L < T)
  pad           : zero-pad to length T       (truncate first-T if L > T)
  interpolate   : linearly resample the whole series to T points
  sliding_window: slide a T-wide window across the series; predictions on all
                  windows are aggregated per customer (handled in the predict layer)

If the model's sequence length is variable (None), no resizing is performed.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

STRATEGIES = ["last_n", "truncate", "pad", "interpolate", "sliding_window"]

STRATEGY_LABELS = {
    "last_n":         "Last N Readings",
    "truncate":       "Truncation (first N)",
    "pad":            "Padding (zeros)",
    "interpolate":    "Interpolation / Resampling",
    "sliding_window": "Sliding Window (aggregated)",
}


def resize_row(row: np.ndarray, target_len: int, strategy: str) -> np.ndarray:
    """Resize a single 1-D sequence to target_len using the chosen strategy."""
    row = np.asarray(row, dtype=np.float32)
    L = len(row)
    if L == target_len:
        return row

    if strategy == "interpolate":
        x_old = np.linspace(0.0, 1.0, num=L)
        x_new = np.linspace(0.0, 1.0, num=target_len)
        return np.interp(x_new, x_old, row).astype(np.float32)

    if strategy == "last_n":
        if L >= target_len:
            return row[-target_len:]
        return np.concatenate([np.zeros(target_len - L, dtype=np.float32), row])

    if strategy in ("truncate", "pad"):
        # truncate = first N; pad = keep + zero-pad — both converge to
        # "first target_len values, right-zero-padded if shorter"
        if L >= target_len:
            return row[:target_len]
        return np.concatenate([row, np.zeros(target_len - L, dtype=np.float32)])

    raise ValueError(f"Unknown preprocessing strategy: {strategy}")


def resize_sequences(seq_2d: np.ndarray, target_len: int, strategy: str) -> np.ndarray:
    """
    Resize an (N, L) array to (N, target_len) for the non-windowing strategies.
    For sliding_window use windows_for_row() in the predict layer instead.
    """
    seq_2d = np.asarray(seq_2d, dtype=np.float32)
    if seq_2d.shape[1] == target_len:
        return seq_2d
    out = np.vstack([resize_row(r, target_len, strategy) for r in seq_2d])
    logger.info(
        "Resized sequences %s -> %s using strategy '%s'",
        seq_2d.shape, out.shape, strategy,
    )
    return out.astype(np.float32)


def windows_for_row(row: np.ndarray, target_len: int, stride: int = 0) -> List[np.ndarray]:
    """
    Produce all T-wide windows of a sequence for the sliding_window strategy.
    If L < target_len the row is left-padded (single window).
    Returns a list of (target_len,) arrays.
    """
    row = np.asarray(row, dtype=np.float32)
    L = len(row)
    if L <= target_len:
        return [resize_row(row, target_len, "last_n")]
    if stride <= 0:
        stride = max(1, target_len // 2)
    windows = []
    start = 0
    while start + target_len <= L:
        windows.append(row[start:start + target_len])
        start += stride
    # ensure the final window reaches the end of the series
    if (L - target_len) % stride != 0:
        windows.append(row[L - target_len:])
    return windows
