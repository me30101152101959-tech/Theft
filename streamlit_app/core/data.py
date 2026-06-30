"""
Dataset ingestion + evaluation metrics.
Auto-detects ID / FLAG / reading columns in CSV or Excel files (any number of
readings), runs the CNN-LSTM engine, and computes evaluation metrics when a
ground-truth FLAG column is present.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ID_COLS = {"cons_no", "customer_id", "id", "customer", "consumer_no", "meter_id", "user_id"}
FLAG_COLS = {"flag", "label", "target", "theft", "is_theft", "class", "y"}


def read_table(file) -> pd.DataFrame:
    """Read CSV or Excel from an uploaded file / path."""
    name = getattr(file, "name", str(file)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)


def inspect(df: pd.DataFrame) -> dict:
    """Detect id / flag / reading columns. Returns column roles + reading matrix shape."""
    cols = list(df.columns)
    lower = {c: str(c).strip().lower() for c in cols}

    id_col = next((c for c in cols if lower[c] in ID_COLS), None)
    flag_col = next((c for c in cols if lower[c] in FLAG_COLS), None)

    reading_cols = []
    for c in cols:
        if c in (id_col, flag_col):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.5:  # >=50% numeric → reading column
            reading_cols.append(c)

    return {
        "id_col": id_col,
        "flag_col": flag_col,
        "reading_cols": reading_cols,
        "n_readings": len(reading_cols),
        "n_rows": len(df),
        "has_flag": flag_col is not None,
        "columns": [str(c) for c in cols],
    }


def build_matrix(df: pd.DataFrame, info: dict) -> Tuple[np.ndarray, list, Optional[np.ndarray]]:
    """Return (readings (N,L), ids list, flags (N,) or None)."""
    rc = info["reading_cols"]
    readings = df[rc].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if info["id_col"]:
        ids = df[info["id_col"]].astype(str).tolist()
    else:
        ids = [f"CUST_{i+1:06d}" for i in range(len(df))]
    flags = None
    if info["flag_col"]:
        flags = pd.to_numeric(df[info["flag_col"]], errors="coerce").fillna(0).astype(int).to_numpy()
    return readings, ids, flags


def compute_metrics(flags: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    out = {
        "accuracy": float(accuracy_score(flags, preds)),
        "precision_val": float(precision_score(flags, preds, zero_division=0)),
        "recall_val": float(recall_score(flags, preds, zero_division=0)),
        "f1_score": float(f1_score(flags, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(flags, preds, labels=[0, 1]).tolist(),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(flags, probs))
    except Exception:
        out["roc_auc"] = None
    return out


def run_batch(df: pd.DataFrame, info: dict, engine, strategy="last_n",
              threshold=0.5, fit_scaler=True) -> dict:
    """Run the engine on a whole dataframe; return rows + metrics + aggregates."""
    readings, ids, flags = build_matrix(df, info)
    probs = engine.predict_sequences(readings, strategy=strategy, threshold=threshold,
                                     fit_scaler=fit_scaler)
    rows = []
    for i, (cid, p) in enumerate(zip(ids, probs)):
        cls = engine.classify(float(p), threshold)
        rows.append({
            "customer_id": cid, **cls,
            "flag": int(flags[i]) if flags is not None else None,
            "readings": readings[i].tolist(),
            "predicted_at": datetime.utcnow().isoformat(),
        })
    preds = np.array([r["prediction"] for r in rows])
    theft = int(preds.sum())
    result = {
        "rows": rows,
        "total_rows": len(rows),
        "theft_rows": theft,
        "normal_rows": len(rows) - theft,
        "theft_rate": round(theft / max(len(rows), 1), 4),
        "avg_risk": round(float(np.mean([r["risk_score"] for r in rows])), 2),
        "has_flag": flags is not None,
        "n_readings": info["n_readings"],
        "metrics": None,
    }
    if flags is not None:
        result["metrics"] = compute_metrics(flags, preds, probs)
    return result
