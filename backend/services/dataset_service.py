"""
Dataset Service  —  SQLite-backed
==================================
Handles CSV ingestion, CNN-LSTM batch prediction, and persisting ALL results
to SQLite.  After this module completes, the data is durable across restarts.

Rules
-----
  * FLAG column is NEVER used for prediction — only for evaluation
  * Every prediction calls model.predict() via model_service.predict_batch()
  * All results are written to SQLite via database.save_predictions_bulk()
  * No data lives in Python variables after this function returns
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report,
)

import database as db
from services.model_service import predict_batch, is_model_loaded, state as model_state
from services.feature_service import pipeline

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CSV validation
# ─────────────────────────────────────────────────────────────────────────────
def _detect_reading_cols(df: pd.DataFrame) -> list:
    skip = {"CONS_NO", "FLAG"}
    cols = [c for c in df.columns if c.strip().upper() not in skip]
    if len(cols) < 26:
        raise ValueError(
            f"Need at least 26 reading columns; found {len(cols)}. "
            "Ensure CONS_NO and FLAG columns are named correctly."
        )
    return cols[:26]


def _validate(df: pd.DataFrame) -> dict:
    upper_cols = {c.strip().upper() for c in df.columns}
    if "CONS_NO" not in upper_cols:
        raise ValueError("Dataset must contain a 'CONS_NO' column.")

    col_map = {c: c.strip().upper() if c.strip().upper() in ("CONS_NO", "FLAG") else c
               for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    reading_cols = _detect_reading_cols(df)
    for c in reading_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    nan_ct = df[reading_cols].isnull().sum().sum()
    if nan_ct > 0:
        logger.warning("%d NaN values — filling with 0", nan_ct)
        df[reading_cols] = df[reading_cols].fillna(0)

    has_flag = "FLAG" in df.columns
    return {"reading_cols": reading_cols, "has_flag": has_flag, "total": len(df)}


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline  (CSV → model.predict() → SQLite)
# ─────────────────────────────────────────────────────────────────────────────
def load_and_predict(filepath: str, filename: str, threshold: float = 0.5) -> dict:
    """
    Full pipeline:
      1. Read CSV
      2. Validate columns
      3. Fit MinMaxScaler on readings
      4. Run model.predict() in batches of 256
      5. Compute evaluation metrics (only if FLAG present)
      6. Write everything to SQLite  ← durable storage
    Returns summary dict.
    """
    if not is_model_loaded():
        raise RuntimeError("No CNN-LSTM model loaded. Upload cnnlstm_final.keras first.")

    logger.info("═" * 55)
    logger.info("  Dataset pipeline started: %s", filename)
    logger.info("  Model: %s  |  threshold: %.2f", model_state.model_name, threshold)
    logger.info("═" * 55)

    df = pd.read_csv(filepath)
    meta = _validate(df)
    reading_cols: list = meta["reading_cols"]
    has_flag: bool = meta["has_flag"]

    readings = df[reading_cols].values.astype(np.float32)   # (N, 26)
    n = len(readings)
    logger.info("Rows: %d  |  Reading cols: %s", n, reading_cols[:3])

    # ── Feature engineering ──────────────────────────────────────────
    logger.info("Computing 59 statistical features and fitting MinMaxScaler...")
    stat_feats = pipeline.fit_transform(readings)           # (N, 59)
    logger.info("stat_feats.shape = %s", stat_feats.shape)

    # ── CNN-LSTM inference ───────────────────────────────────────────
    logger.info(
        "model.predict() input shape → (%d, 26, 1) | model: %s",
        n, model_state.model_name,
    )
    probs = predict_batch(
        readings=readings,
        stat_feats=stat_feats if model_state.is_dual_input else None,
        threshold=threshold,
    )
    logger.info("model.predict() complete | output shape → %s", probs.shape)

    # ── Build prediction rows ────────────────────────────────────────
    now = datetime.utcnow().isoformat()
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        prob = float(probs[i])
        pred = 1 if prob >= threshold else 0
        conf = prob if pred == 1 else (1.0 - prob)
        risk = round(prob * 100, 2)
        rows.append({
            "customer_id": str(row["CONS_NO"]),
            "probability": round(prob, 6),
            "prediction": pred,
            "confidence": round(conf, 6),
            "risk_score": risk,
            "status": "Theft" if pred == 1 else "Normal",
            "flag": int(row["FLAG"]) if has_flag else None,
            "readings": [float(row[c]) for c in reading_cols],
            "predicted_at": now,
        })

    theft_count  = sum(1 for r in rows if r["prediction"] == 1)
    normal_count = n - theft_count
    avg_conf = float(np.mean([r["confidence"] for r in rows]))
    avg_risk = float(np.mean([r["risk_score"]  for r in rows]))
    theft_rate = round(theft_count / n, 6) if n else 0.0

    # ── Evaluation metrics ───────────────────────────────────────────
    accuracy = precision_val = recall_val = f1 = roc_auc = None
    roc_fpr = roc_tpr = pr_prec = pr_rec = confusion_mat = None

    if has_flag:
        y_true = df["FLAG"].values.astype(int)
        y_pred = (probs >= threshold).astype(int)

        accuracy     = round(float(accuracy_score(y_true, y_pred)), 6)
        precision_val= round(float(precision_score(y_true, y_pred, zero_division=0)), 6)
        recall_val   = round(float(recall_score(y_true, y_pred, zero_division=0)), 6)
        f1           = round(float(f1_score(y_true, y_pred, zero_division=0)), 6)
        try:
            roc_auc = round(float(roc_auc_score(y_true, probs)), 6)
        except Exception:
            roc_auc = 0.0

        cm = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, probs)
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, probs)

        confusion_mat = cm.tolist()
        roc_fpr = fpr.tolist()
        roc_tpr = tpr.tolist()
        pr_prec = prec_arr.tolist()
        pr_rec  = rec_arr.tolist()

        logger.info(
            "Evaluation  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
            accuracy, precision_val, recall_val, f1, roc_auc,
        )

    # ── Persist to SQLite ────────────────────────────────────────────
    logger.info("Writing %d rows to SQLite...", n)
    upload_id = db.save_upload(
        filename=filename,
        upload_time=now,
        total_rows=n,
        theft_rows=theft_count,
        normal_rows=normal_count,
        avg_confidence=round(avg_conf, 6),
        avg_risk=round(avg_risk, 6),
        theft_rate=theft_rate,
        has_flag=has_flag,
        threshold=threshold,
        accuracy=accuracy,
        precision_val=precision_val,
        recall_val=recall_val,
        f1_score=f1,
        roc_auc=roc_auc,
        roc_fpr=roc_fpr,
        roc_tpr=roc_tpr,
        pr_precision=pr_prec,
        pr_recall=pr_rec,
        confusion_matrix=confusion_mat,
    )
    db.save_predictions_bulk(upload_id, rows)
    logger.info(
        "SQLite write complete: upload_id=%d  theft=%d  normal=%d",
        upload_id, theft_count, normal_count,
    )

    return {
        "upload_id": upload_id,
        "total": n,
        "theft": theft_count,
        "normal": normal_count,
        "avg_confidence": round(avg_conf, 4),
        "avg_risk": round(avg_risk, 4),
        "theft_rate": round(theft_rate, 4),
        "has_flag": has_flag,
        "accuracy": accuracy,
        "precision": precision_val,
        "recall": recall_val,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "dataset_name": filename,
        "upload_time": now,
        "model_used": model_state.model_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# State checks  (read from SQLite — never from RAM)
# ─────────────────────────────────────────────────────────────────────────────
def is_dataset_loaded() -> bool:
    """Returns True if any upload exists in SQLite."""
    return db.has_any_upload()
