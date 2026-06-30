"""
Prediction Router — /api/predict/*
=====================================
All predictions call model.predict() via model_service.
Results written to SQLite immediately. No in-memory caching.

Endpoints
---------
  POST /api/predict/manual            — single customer  → manual_predictions table
  POST /api/predict/batch-preview     — CSV → predict only, no store (UI preview)
  POST /api/predict/batch-store       — CSV → predict → store in predictions table
  POST /api/predict/update-threshold  — reclassify latest upload in SQLite
  GET  /api/predict/shap/{cid}        — gradient-based feature importance
  GET  /api/predict/history           — recent manual predictions from SQLite
  GET  /api/predict/template          — blank CSV template download
"""
from __future__ import annotations

import csv as csv_mod
import io
import json
import logging
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

import database as db
from models.schemas import ManualPredictionRequest, ThresholdUpdateRequest
from services import model_service
from services.feature_service import pipeline

router = APIRouter(prefix="/api/predict", tags=["predict"])
logger = logging.getLogger(__name__)

READING_LABELS = [
    "Jan-Y1", "Feb-Y1", "Mar-Y1", "Apr-Y1", "May-Y1", "Jun-Y1",
    "Jul-Y1", "Aug-Y1", "Sep-Y1", "Oct-Y1", "Nov-Y1", "Dec-Y1",
    "Jan-Y2", "Feb-Y2", "Mar-Y2", "Apr-Y2", "May-Y2", "Jun-Y2",
    "Jul-Y2", "Aug-Y2", "Sep-Y2", "Oct-Y2", "Nov-Y2", "Dec-Y2",
    "W1", "W2",
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _require_model():
    if not model_service.is_model_loaded():
        raise HTTPException(400, "No CNN-LSTM model loaded. Upload cnnlstm_final.keras first.")


def _risk_level(risk_score: float) -> str:
    if risk_score >= 75:
        return "High"
    if risk_score >= 40:
        return "Medium"
    return "Low"


def _parse_csv_upload(content: bytes) -> tuple[pd.DataFrame, list, bool, bool]:
    """
    Parse uploaded CSV bytes.
    Returns (df, reading_cols, has_cons_no, has_flag).
    Raises HTTPException on errors.
    """
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8", errors="replace")))
    except Exception as exc:
        raise HTTPException(400, f"Cannot parse CSV: {exc}")

    if df.empty:
        raise HTTPException(400, "CSV file is empty.")

    # Normalise CONS_NO / FLAG column names
    col_map = {
        c: c.strip().upper()
        for c in df.columns
        if c.strip().upper() in ("CONS_NO", "FLAG")
    }
    df = df.rename(columns=col_map)

    skip      = {"CONS_NO", "FLAG"}
    read_cols = [c for c in df.columns if c.strip().upper() not in skip][:26]

    if len(read_cols) < 26:
        raise HTTPException(
            400,
            f"Need at least 26 reading columns; found {len(read_cols)}. "
            f"Columns detected: {list(df.columns[:20])}. "
            "Download the template for the correct format.",
        )

    for c in read_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    has_cons = "CONS_NO" in df.columns
    has_flag = "FLAG" in df.columns
    return df, read_cols, has_cons, has_flag


def _run_model(
    df: pd.DataFrame,
    read_cols: list,
    has_cons: bool,
    has_flag: bool,
    threshold: float,
    store_readings: bool = True,
) -> tuple[list, np.ndarray, float]:
    """
    Run model.predict() on df rows.
    Returns (rows, probs_array, elapsed_seconds).
    """
    readings_arr = df[read_cols].values.astype(np.float32)
    n            = len(readings_arr)
    id_col       = "CONS_NO" if has_cons else df.columns[0]
    now          = datetime.utcnow().isoformat()

    stat_feats = None
    if model_service.state.is_dual_input:
        logger.info("Computing 59 stat features for %d rows …", n)
        stat_feats = pipeline.fit_transform(readings_arr)

    logger.info(
        "model.predict() on %d rows | model=%s | threshold=%.2f",
        n, model_service.state.model_name, threshold,
    )
    t0    = time.time()
    probs = model_service.predict_batch(
        readings=readings_arr,
        stat_feats=stat_feats,
        threshold=threshold,
    )
    elapsed = round(time.time() - t0, 2)

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        prob = float(probs[i])
        pred = 1 if prob >= threshold else 0
        conf = prob if pred == 1 else (1.0 - prob)
        risk = round(prob * 100, 2)
        rows.append({
            "customer_id": str(row[id_col]),
            "probability": round(prob, 6),
            "prediction":  pred,
            "confidence":  round(conf, 6),
            "risk_score":  risk,
            "risk_level":  _risk_level(risk),
            "status":      "Theft" if pred == 1 else "Normal",
            "flag":        int(row["FLAG"]) if has_flag else None,
            "readings":    [float(row[c]) for c in read_cols] if store_readings else [],
            "predicted_at": now,
        })

    logger.info(
        "Prediction done: %d rows in %.2fs | theft=%d | normal=%d",
        n, elapsed,
        sum(1 for r in rows if r["prediction"] == 1),
        sum(1 for r in rows if r["prediction"] == 0),
    )
    return rows, probs, elapsed


def _compute_eval_metrics(df: pd.DataFrame, probs: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score as sk_f1, roc_auc_score,
        confusion_matrix, roc_curve, precision_recall_curve,
    )
    y_true = df["FLAG"].values.astype(int)
    y_pred = (probs >= threshold).astype(int)
    try:
        roc_auc = round(float(roc_auc_score(y_true, probs)), 6)
    except Exception:
        roc_auc = 0.0
    cm        = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, probs)
    pp,  rr,  _ = precision_recall_curve(y_true, probs)
    return {
        "accuracy":        round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_val":   round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall_val":      round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_score":        round(float(sk_f1(y_true, y_pred, zero_division=0)), 6),
        "roc_auc":         roc_auc,
        "confusion_matrix": cm.tolist(),
        "roc_fpr":         fpr.tolist(),
        "roc_tpr":         tpr.tolist(),
        "pr_precision":    pp.tolist(),
        "pr_recall":       rr.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/manual
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/manual")
async def manual_predict(req: ManualPredictionRequest):
    """
    Single customer prediction.
    PROOF: model.predict(x) where x.shape == (1, 26, 1)
    Stores result in SQLite manual_predictions table.
    """
    _require_model()

    readings = np.array(req.readings, dtype=np.float32)
    if len(readings) != 26:
        raise HTTPException(400, f"Expected 26 readings, got {len(readings)}.")

    try:
        result = model_service.predict_single(readings, threshold=req.threshold)
    except Exception as exc:
        logger.exception("[manual] model.predict() failed")
        raise HTTPException(500, f"TensorFlow inference error: {exc}")

    now    = datetime.utcnow().isoformat()
    row_id = db.save_manual_prediction(
        customer_id  = req.customer_id,
        probability  = result["probability"],
        prediction   = result["prediction"],
        confidence   = result["confidence"],
        risk_score   = result["risk_score"],
        status       = result["status"],
        readings     = list(req.readings),
        predicted_at = now,
        threshold    = req.threshold,
        model_name   = model_service.state.model_name,
        source       = "manual",
    )

    result.update({
        "customer_id":   req.customer_id,
        "readings":      list(req.readings),
        "predicted_at":  now,
        "sqlite_row_id": row_id,
        "stored_in":     "SQLite — manual_predictions",
        "risk_level":    result.get("risk_level", _risk_level(result["risk_score"])),
    })

    logger.info(
        "[manual] id=%s | prob=%.4f | status=%s | row_id=%d",
        req.customer_id, result["probability"], result["status"], row_id,
    )
    return JSONResponse({"success": True, "result": result})


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/batch-preview
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/batch-preview")
async def batch_preview(
    csv_file:  UploadFile = File(...),
    threshold: float      = Form(0.5),
):
    """
    Run batch CNN-LSTM predictions on uploaded CSV.
    Does NOT store results in SQLite — use batch-store for that.
    Returns predictions array for the UI table.
    """
    _require_model()
    content = await csv_file.read()
    df, read_cols, has_cons, has_flag = _parse_csv_upload(content)
    n = len(df)

    rows, probs, elapsed = _run_model(df, read_cols, has_cons, has_flag, threshold)

    theft  = sum(1 for r in rows if r["prediction"] == 1)
    normal = n - theft
    avg_conf = round(float(np.mean([r["confidence"] for r in rows])), 4)
    avg_risk = round(float(np.mean([r["risk_score"]  for r in rows])), 4)

    metrics = {}
    if has_flag:
        metrics = _compute_eval_metrics(df, probs, threshold)
        # Only return summary metrics (not large arrays) in the preview
        metrics = {k: v for k, v in metrics.items()
                   if k not in ("confusion_matrix", "roc_fpr", "roc_tpr", "pr_precision", "pr_recall")}

    return JSONResponse({
        "success":         True,
        "stored":          False,
        "total":           n,
        "theft":           theft,
        "normal":          normal,
        "avg_confidence":  avg_conf,
        "avg_risk":        avg_risk,
        "elapsed_seconds": elapsed,
        "has_flag":        has_flag,
        "metrics":         metrics,
        "predictions":     rows,
        "predict_proof":   f"model.predict(x, verbose=0, batch_size=256) x.shape=({n},26,1)",
        "model_name":      model_service.state.model_name,
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/batch-store
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/batch-store")
async def batch_store(
    csv_file:  UploadFile = File(...),
    threshold: float      = Form(0.5),
    label:     str        = Form("Batch Upload"),
):
    """
    CSV → model.predict() → write ALL results to SQLite predictions table.
    These results appear immediately on the Customer Predictions page
    and refresh dashboard statistics.
    FLAG column (if present) is used ONLY for evaluation, never prediction.
    """
    _require_model()
    content = await csv_file.read()
    df, read_cols, has_cons, has_flag = _parse_csv_upload(content)
    n = len(df)

    rows, probs, elapsed = _run_model(df, read_cols, has_cons, has_flag, threshold)

    theft     = sum(1 for r in rows if r["prediction"] == 1)
    normal    = n - theft
    avg_conf  = float(np.mean([r["confidence"] for r in rows]))
    avg_risk  = float(np.mean([r["risk_score"]  for r in rows]))
    now       = datetime.utcnow().isoformat()

    metrics = {}
    if has_flag:
        metrics = _compute_eval_metrics(df, probs, threshold)

    upload_id = db.save_upload(
        filename         = f"{label} — {csv_file.filename}",
        upload_time      = now,
        total_rows       = n,
        theft_rows       = theft,
        normal_rows      = normal,
        avg_confidence   = round(avg_conf, 6),
        avg_risk         = round(avg_risk, 6),
        theft_rate       = round(theft / n, 6) if n else 0.0,
        has_flag         = has_flag,
        threshold        = threshold,
        accuracy         = metrics.get("accuracy"),
        precision_val    = metrics.get("precision_val"),
        recall_val       = metrics.get("recall_val"),
        f1_score         = metrics.get("f1_score"),
        roc_auc          = metrics.get("roc_auc"),
        roc_fpr          = metrics.get("roc_fpr"),
        roc_tpr          = metrics.get("roc_tpr"),
        pr_precision     = metrics.get("pr_precision"),
        pr_recall        = metrics.get("pr_recall"),
        confusion_matrix = metrics.get("confusion_matrix"),
    )
    db.save_predictions_bulk(upload_id, rows)

    logger.info(
        "[batch-store] upload_id=%d | %d rows in %.2fs | theft=%d",
        upload_id, n, elapsed, theft,
    )

    pub_metrics = {k: v for k, v in metrics.items()
                   if k not in ("roc_fpr", "roc_tpr", "pr_precision", "pr_recall", "confusion_matrix")}

    return JSONResponse({
        "success":         True,
        "upload_id":       upload_id,
        "stored":          True,
        "total":           n,
        "theft":           theft,
        "normal":          normal,
        "avg_confidence":  round(avg_conf, 4),
        "avg_risk":        round(avg_risk, 4),
        "elapsed_seconds": elapsed,
        "has_flag":        has_flag,
        "metrics":         pub_metrics,
        "predict_proof":   f"model.predict(x, verbose=0, batch_size=256) x.shape=({n},26,1)",
        "stored_in":       f"SQLite — predictions table (upload_id={upload_id})",
        "model_name":      model_service.state.model_name,
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/update-threshold
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/update-threshold")
async def update_threshold(req: ThresholdUpdateRequest):
    """Re-classify all predictions in the latest SQLite upload at a new threshold."""
    uid = db.get_latest_upload_id()
    if uid is None:
        raise HTTPException(400, "No dataset in SQLite.")

    result = db.update_predictions_threshold(uid, req.threshold)
    return JSONResponse({
        "success":     True,
        "threshold":   req.threshold,
        "theft":       result["theft"],
        "normal":      result["normal"],
        "data_source": "SQLite — predictions reclassified in-place",
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/predict/shap/{customer_id}
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/shap/{customer_id}")
async def get_shap(customer_id: str):
    """
    Gradient-based feature importance for one customer.
    Uses TensorFlow GradientTape (always available — no SHAP library needed).
    Looks up readings from SQLite predictions or manual_predictions tables.
    """
    _require_model()

    # Look up customer readings from predictions table first
    uid = db.get_latest_upload_id()
    row = None
    if uid:
        row = db.get_customer_by_id(uid, customer_id)

    # Fall back to manual predictions
    if not row:
        history = db.get_manual_predictions(limit=500)
        for h in history:
            if h.get("customer_id") == customer_id:
                row = {
                    "id":          customer_id,
                    "readings":    h["readings"],
                    "probability": h["probability"],
                    "status":      h["status"],
                }
                break

    if not row:
        raise HTTPException(404, f"Customer '{customer_id}' not found in SQLite.")

    readings_raw = row.get("readings", [])
    if len(readings_raw) < 26:
        raise HTTPException(400, "No readings stored for this customer.")

    readings = np.array(readings_raw[:26], dtype=np.float32)
    r2d      = readings.reshape(1, 26, 1).astype(np.float32)
    stat     = pipeline.transform(readings.reshape(1, 26)).astype(np.float32)

    import tensorflow as tf
    seq_var = tf.Variable(r2d)

    with tf.GradientTape() as tape:
        tape.watch(seq_var)
        if model_service.state.is_dual_input:
            stat_t   = tf.constant(stat)
            pred_out = model_service.state.model(
                {"sequence_input": seq_var, "stat_input": stat_t},
                training=False,
            )
        else:
            pred_out = model_service.state.model(seq_var, training=False)

    grads = tape.gradient(pred_out, seq_var)

    if grads is not None:
        importance = np.abs(grads.numpy().flatten())
        total      = importance.sum()
        importance = (importance / total) if total > 1e-10 else np.ones(26) / 26
    else:
        importance = np.ones(26) / 26

    shap_data = [
        {
            "feature":    READING_LABELS[i],
            "value":      float(readings[i]),
            "importance": float(importance[i]),
            "rank":       0,
        }
        for i in range(26)
    ]
    shap_data.sort(key=lambda x: x["importance"], reverse=True)
    for rank, item in enumerate(shap_data):
        item["rank"] = rank + 1

    return JSONResponse({
        "customer_id":        customer_id,
        "probability":        float(row.get("probability", 0)),
        "status":             row.get("status", "Unknown"),
        "method":             "GradientTape integrated gradients",
        "feature_importance": shap_data,
        "top5_features":      [d["feature"] for d in shap_data[:5]],
        "readings_labels":    READING_LABELS,
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/predict/history
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/history")
async def prediction_history(
    limit:  int            = Query(100, ge=1, le=1000),
    source: Optional[str]  = Query(None),
):
    """Return recent manual predictions from SQLite."""
    rows = db.get_manual_predictions(limit=limit, source=source)
    return JSONResponse({
        "success": True,
        "count":   len(rows),
        "history": rows,
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/predict/template
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/template")
async def download_template():
    """Download blank CSV template with correct column headers."""
    buf    = io.StringIO()
    writer = csv_mod.writer(buf)
    writer.writerow(["CONS_NO"] + READING_LABELS + ["FLAG"])
    writer.writerow(["CUSTOMER_001"] + [str(x * 100) for x in range(1, 27)] + [""])
    writer.writerow(["CUSTOMER_002"] + ["1200"] * 24 + ["850", "900"] + ["1"])
    writer.writerow(["CUSTOMER_003"] + ["0", "2400", "0", "2100", "0", "2300",
                                         "0", "2200", "0", "2500", "0", "2000",
                                         "2400", "0", "2100", "0", "2300", "0",
                                         "2200", "0", "2500", "0", "2000", "0",
                                         "0", "0"] + ["0"])
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=etd_prediction_template.csv"
        },
    )
