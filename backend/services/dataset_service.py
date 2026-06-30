"""
Dataset Service
===============
Handles CSV ingestion, validation, batch prediction, and SQLite persistence.

Rules:
  ▸ FLAG is NEVER used during prediction — only for evaluation
  ▸ All predictions come from the uploaded CNN-LSTM model via model.predict()
  ▸ No mock data ever generated
  ▸ All results written to SQLite immediately after prediction
  ▸ All read functions query SQLite (survives server restarts)
"""

from __future__ import annotations
from typing import Optional

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)

import database as db
from services.model_service import predict_batch, is_model_loaded, state as model_state
from services.feature_service import pipeline

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Validators
# ─────────────────────────────────────────────
ID_COLS   = {"CONS_NO", "CUSTOMER_ID", "ID", "CUSTOMER", "METER_ID"}
FLAG_COLS = {"FLAG", "LABEL", "TARGET", "THEFT", "Y"}


def validate_dataset(df: pd.DataFrame) -> dict:
    """Dynamically detect id column, flag column, and ALL numeric reading columns."""
    id_col   = next((c for c in df.columns if c.strip().upper() in ID_COLS), None)
    flag_col = next((c for c in df.columns if c.strip().upper() in FLAG_COLS), None)

    reading_cols = []
    for c in df.columns:
        if c in (id_col, flag_col):
            continue
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().mean() >= 0.5:
            df[c] = coerced.fillna(0.0)
            reading_cols.append(c)

    if len(reading_cols) < 2:
        raise ValueError(
            f"Need at least 2 numeric reading columns; found {len(reading_cols)}. "
            f"Columns: {list(df.columns)}"
        )

    nan_count = int(df[reading_cols].isnull().sum().sum())
    if nan_count > 0:
        df[reading_cols] = df[reading_cols].fillna(0)

    return {
        "id_col":       id_col,
        "flag_col":     flag_col,
        "reading_cols": reading_cols,
        "has_flag":     flag_col is not None,
        "total":        len(df),
        "nan_count":    nan_count,
        "seq_len":      len(reading_cols),
    }


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def load_and_predict(
    filepath:  str,
    filename:  str,
    threshold: float = 0.5,
    strategy:  str   = "last_n",
) -> dict:
    """
    Load CSV → validate → run real model predictions → write ALL to SQLite.
    Sequence length is detected dynamically and resized to the model length
    via the chosen strategy. Returns summary dict.
    """
    if not is_model_loaded():
        raise RuntimeError("No model loaded. Upload a model first.")

    logger.info("Loading dataset: %s", filename)
    df   = pd.read_csv(filepath)
    meta = validate_dataset(df)

    reading_cols = meta["reading_cols"]
    has_flag     = meta["has_flag"]
    id_col       = meta["id_col"]
    flag_col     = meta["flag_col"]
    readings     = df[reading_cols].values.astype(np.float32)
    n            = len(readings)

    logger.info(
        "Predicting %d customers | detected_seq_len=%d | model_seq_len=%s | strategy=%s",
        n, meta["seq_len"], model_state.seq_len_expected, strategy,
    )
    from services.model_service import predict_sequences
    probs = predict_sequences(readings, strategy=strategy, threshold=threshold, fit_scaler=True)

    now = datetime.utcnow().isoformat()

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        prob = float(probs[i])
        pred = 1 if prob >= threshold else 0
        conf = prob if pred == 1 else (1.0 - prob)
        risk = round(prob * 100, 2)
        rows.append({
            "customer_id": str(row[id_col]) if id_col else f"ROW_{i+1}",
            "probability": round(prob, 6),
            "prediction":  pred,
            "confidence":  round(conf, 6),
            "risk_score":  risk,
            "status":      "Theft" if pred == 1 else "Normal",
            "flag":        int(row[flag_col]) if flag_col else None,
            "readings":    [float(row[c]) for c in reading_cols],
            "predicted_at": now,
        })

    # Evaluation (FLAG only used here — never for prediction)
    accuracy = precision_val = recall_val = f1 = roc_auc = None
    roc_fpr = roc_tpr = pr_prec = pr_rec = conf_mat = None

    if has_flag:
        y_true = df[flag_col].values.astype(int)
        y_pred = (probs >= threshold).astype(int)
        accuracy      = round(float(accuracy_score(y_true, y_pred)), 6)
        precision_val = round(float(precision_score(y_true, y_pred, zero_division=0)), 6)
        recall_val    = round(float(recall_score(y_true, y_pred, zero_division=0)), 6)
        f1            = round(float(f1_score(y_true, y_pred, zero_division=0)), 6)
        try:
            roc_auc = round(float(roc_auc_score(y_true, probs)), 6)
        except Exception:
            roc_auc = 0.0
        cm           = confusion_matrix(y_true, y_pred)
        fpr, tpr, _  = roc_curve(y_true, probs)
        pp, rr, _    = precision_recall_curve(y_true, probs)
        conf_mat     = cm.tolist()
        roc_fpr      = fpr.tolist()
        roc_tpr      = tpr.tolist()
        pr_prec      = pp.tolist()
        pr_rec       = rr.tolist()

    theft   = sum(1 for r in rows if r["prediction"] == 1)
    normal  = n - theft
    avg_conf = float(np.mean([r["confidence"]  for r in rows]))
    avg_risk = float(np.mean([r["risk_score"]  for r in rows]))

    upload_id = db.save_upload(
        filename       = filename,
        upload_time    = now,
        total_rows     = n,
        theft_rows     = theft,
        normal_rows    = normal,
        avg_confidence = round(avg_conf, 6),
        avg_risk       = round(avg_risk, 6),
        theft_rate     = round(theft / n, 6) if n else 0.0,
        has_flag       = has_flag,
        threshold      = threshold,
        accuracy       = accuracy,
        precision_val  = precision_val,
        recall_val     = recall_val,
        f1_score       = f1,
        roc_auc        = roc_auc,
        roc_fpr        = roc_fpr,
        roc_tpr        = roc_tpr,
        pr_precision   = pr_prec,
        pr_recall      = pr_rec,
        confusion_matrix = conf_mat,
    )
    db.save_predictions_bulk(upload_id, rows)

    logger.info(
        "SQLite: upload_id=%d  theft=%d  normal=%d  total=%d",
        upload_id, theft, normal, n,
    )

    metrics = {}
    if has_flag:
        metrics = {
            "accuracy":  accuracy,
            "precision": precision_val,
            "recall":    recall_val,
            "f1_score":  f1,
            "roc_auc":   roc_auc,
        }

    return {
        "total":        n,
        "theft":        theft,
        "normal":       normal,
        "avg_confidence": round(avg_conf, 4),
        "avg_risk":     round(avg_risk, 4),
        "has_flag":     has_flag,
        "metrics":      metrics,
        "dataset_name": filename,
        "upload_time":  now,
        "model_used":   model_state.model_name,
        "upload_id":    upload_id,
    }


# ─────────────────────────────────────────────
# Read functions — ALL from SQLite
# ─────────────────────────────────────────────
def is_dataset_loaded() -> bool:
    return db.has_any_upload()


def get_dashboard_stats() -> dict:
    uid = db.get_latest_upload_id()
    if uid is None:
        return {}
    row = db.get_upload_summary(uid)
    if not row:
        return {}
    return {
        "total_customers":     row["total_rows"],
        "processed_customers": row["total_rows"],
        "predicted_theft":     row["theft_rows"],
        "predicted_normal":    row["normal_rows"],
        "avg_confidence":      row["avg_confidence"] or 0.0,
        "avg_risk_score":      row["avg_risk"] or 0.0,
        "theft_rate":          row["theft_rate"] or 0.0,
        "dataset_name":        row["filename"],
        "upload_time":         row["upload_time"],
        "has_flag":            bool(row["has_flag"]),
        "threshold":           row["threshold"],
        "accuracy":            row["accuracy"],
        "precision":           row["precision_val"],
        "recall":              row["recall_val"],
        "f1_score":            row["f1_score"],
        "roc_auc":             row["roc_auc"],
        "data_source":         "SQLite — etd_xai.db",
    }


def get_customers_paginated(
    page: int = 1,
    page_size: int = 50,
    search: str = "",
    status_filter: str = "",
    sort_by: str = "risk_score",
    sort_dir: str = "desc",
) -> dict:
    uid = db.get_latest_upload_id()
    if uid is None:
        return {"data": [], "total": 0, "page": 1, "page_size": page_size, "total_pages": 0}
    return db.get_customers_paginated(
        upload_id     = uid,
        page          = page,
        page_size     = page_size,
        search        = search,
        status_filter = status_filter,
        sort_by       = sort_by,
        sort_dir      = sort_dir,
    )


def get_chart_data() -> dict:
    uid = db.get_latest_upload_id()
    if uid is None:
        return {}
    return db.get_chart_data(uid)


def get_all_customers_for_export() -> list:
    uid = db.get_latest_upload_id()
    if uid is None:
        return []
    return db.get_all_customers_for_export(uid)


def get_customer_by_id(customer_id: str) -> Optional[dict]:
    uid = db.get_latest_upload_id()
    if uid is None:
        return None
    return db.get_customer_by_id(uid, customer_id)


def re_predict_with_threshold(threshold: float) -> dict:
    uid = db.get_latest_upload_id()
    if uid is None:
        raise RuntimeError("No dataset in SQLite.")
    db.update_predictions_threshold(uid, threshold)
    return get_dashboard_stats()
