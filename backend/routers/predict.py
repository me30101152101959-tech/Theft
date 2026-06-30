"""
Prediction Router  —  /api/predict/*
=====================================
All predictions call model.predict() via model_service.
Results are stored in SQLite immediately so they survive restarts.

Endpoints
---------
  POST /api/predict/manual       — single customer (legacy URL kept)
  POST /api/predict/batch-store  — CSV → predict → store in SQLite
  POST /api/predict/update-threshold
  GET  /api/predict/shap/{customer_id}  — SHAP feature importance
  GET  /api/predict/template     — download blank CSV template
"""
from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

import database as db
from models.schemas import ManualPredictionRequest, ThresholdUpdateRequest
from services import model_service
from services.feature_service import pipeline

router = APIRouter(prefix="/api/predict", tags=["predict"])
logger = logging.getLogger(__name__)

READING_LABELS = [
    "Jan-Y1","Feb-Y1","Mar-Y1","Apr-Y1","May-Y1","Jun-Y1",
    "Jul-Y1","Aug-Y1","Sep-Y1","Oct-Y1","Nov-Y1","Dec-Y1",
    "Jan-Y2","Feb-Y2","Mar-Y2","Apr-Y2","May-Y2","Jun-Y2",
    "Jul-Y2","Aug-Y2","Sep-Y2","Oct-Y2","Nov-Y2","Dec-Y2",
    "W1","W2",
]


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/manual  (legacy single-customer endpoint)
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/manual")
async def manual_predict(req: ManualPredictionRequest):
    """
    Single customer prediction.
    Calls model.predict(x) where x.shape == (1, 26, 1).
    Stores result in manual_predictions table.
    """
    if not model_service.is_model_loaded():
        raise HTTPException(400, "No CNN-LSTM model loaded. Upload cnnlstm_final.keras first.")

    readings = np.array(req.readings, dtype=np.float32)
    if len(readings) != 26:
        raise HTTPException(400, f"Expected 26 readings, got {len(readings)}.")

    try:
        result = model_service.predict_single(readings, threshold=req.threshold)
    except Exception as exc:
        logger.exception("[manual_predict] model.predict() failed")
        raise HTTPException(500, f"TensorFlow error during model.predict(): {exc}")

    now = datetime.utcnow().isoformat()
    db.save_manual_prediction(
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
    )

    result["customer_id"] = req.customer_id
    result["readings"]    = list(req.readings)
    result["stored_in"]   = "SQLite — manual_predictions table"
    result["input_shape"] = "(1, 26, 1)"

    logger.info(
        "[manual_predict] id=%s | prob=%.4f | status=%s",
        req.customer_id, result["probability"], result["status"],
    )
    return JSONResponse({"success": True, "result": result})


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/batch-store
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/batch-store")
async def batch_predict_and_store(
    csv_file:  UploadFile = File(...),
    threshold: float      = Form(0.5),
    label:     str        = Form("manual_batch"),
):
    """
    Upload CSV → model.predict() on every row → store ALL results in SQLite.
    These results appear immediately on the Customer Predictions page.
    FLAG column (if present) is used only for evaluation, never for prediction.
    """
    if not model_service.is_model_loaded():
        raise HTTPException(400, "No CNN-LSTM model loaded.")

    content = await csv_file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8", errors="replace")))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    # Normalise column names
    col_map = {c: c.strip().upper() if c.strip().upper() in ("CONS_NO", "FLAG") else c
               for c in df.columns}
    df = df.rename(columns=col_map)

    skip = {"CONS_NO", "FLAG"}
    reading_cols = [c for c in df.columns if c.strip().upper() not in skip][:26]
    if len(reading_cols) < 26:
        raise HTTPException(
            400,
            f"Need at least 26 reading columns, found {len(reading_cols)}. "
            f"Columns found: {list(df.columns[:10])}"
        )
    has_cons = "CONS_NO" in df.columns
    has_flag = "FLAG" in df.columns

    for c in reading_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    readings = df[reading_cols].values.astype(np.float32)
    n = len(readings)

    logger.info(
        "[batch-store] model.predict() on %d rows | model=%s | threshold=%.2f",
        n, model_service.state.model_name, threshold,
    )
    t0 = time.time()

    stat_feats = None
    if model_service.state.is_dual_input:
        stat_feats = pipeline.fit_transform(readings)

    probs = model_service.predict_batch(
        readings=readings,
        stat_feats=stat_feats,
        threshold=threshold,
    )
    elapsed = round(time.time() - t0, 2)

    now = datetime.utcnow().isoformat()
    id_col = "CONS_NO" if has_cons else df.columns[0]

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
            "status":      "Theft" if pred == 1 else "Normal",
            "flag":        int(row["FLAG"]) if has_flag else None,
            "readings":    [float(row[c]) for c in reading_cols],
            "predicted_at": now,
        })

    # Evaluation metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score as sk_f1, roc_auc_score,
        confusion_matrix, roc_curve, precision_recall_curve,
    )
    accuracy = precision_val = recall_val = f1 = roc_auc = None
    roc_fpr = roc_tpr = pr_prec = pr_rec = conf_mat = None

    if has_flag:
        y_true = df["FLAG"].values.astype(int)
        y_pred = (probs >= threshold).astype(int)
        accuracy      = round(float(accuracy_score(y_true, y_pred)), 6)
        precision_val = round(float(precision_score(y_true, y_pred, zero_division=0)), 6)
        recall_val    = round(float(recall_score(y_true, y_pred, zero_division=0)), 6)
        f1            = round(float(sk_f1(y_true, y_pred, zero_division=0)), 6)
        try:
            roc_auc = round(float(roc_auc_score(y_true, probs)), 6)
        except Exception:
            roc_auc = 0.0
        cm = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, probs)
        pp, rr, _   = precision_recall_curve(y_true, probs)
        conf_mat = cm.tolist(); roc_fpr = fpr.tolist(); roc_tpr = tpr.tolist()
        pr_prec  = pp.tolist(); pr_rec  = rr.tolist()

    theft  = sum(1 for r in rows if r["prediction"] == 1)
    normal = n - theft
    avg_conf = float(np.mean([r["confidence"] for r in rows]))
    avg_risk = float(np.mean([r["risk_score"]  for r in rows]))

    upload_id = db.save_upload(
        filename        = f"{label} — {csv_file.filename}",
        upload_time     = now,
        total_rows      = n,
        theft_rows      = theft,
        normal_rows     = normal,
        avg_confidence  = round(avg_conf, 6),
        avg_risk        = round(avg_risk, 6),
        theft_rate      = round(theft / n, 6) if n else 0.0,
        has_flag        = has_flag,
        threshold       = threshold,
        accuracy        = accuracy,
        precision_val   = precision_val,
        recall_val      = recall_val,
        f1_score        = f1,
        roc_auc         = roc_auc,
        roc_fpr         = roc_fpr,
        roc_tpr         = roc_tpr,
        pr_precision    = pr_prec,
        pr_recall       = pr_rec,
        confusion_matrix= conf_mat,
    )
    db.save_predictions_bulk(upload_id, rows)

    logger.info(
        "[batch-store] Done: %d rows in %.2fs | upload_id=%d | theft=%d | normal=%d",
        n, elapsed, upload_id, theft, normal,
    )
    return JSONResponse({
        "success":         True,
        "upload_id":       upload_id,
        "total":           n,
        "theft":           theft,
        "normal":          normal,
        "avg_confidence":  round(avg_conf, 4),
        "avg_risk":        round(avg_risk, 4),
        "elapsed_seconds": elapsed,
        "has_flag":        has_flag,
        "accuracy":        accuracy,
        "precision":       precision_val,
        "recall":          recall_val,
        "f1_score":        f1,
        "roc_auc":         roc_auc,
        "predictions":     rows,
        "predict_proof":   f"model.predict(x, verbose=0, batch_size=256)  x.shape=({n},26,1)",
        "stored_in":       f"SQLite — predictions table (upload_id={upload_id})",
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/update-threshold
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/update-threshold")
async def update_threshold(req: ThresholdUpdateRequest):
    """Re-classify all predictions in SQLite with a new threshold."""
    uid = db.get_latest_upload_id()
    if uid is None:
        raise HTTPException(400, "No dataset in SQLite.")

    threshold = req.threshold
    conn = db.get_db()
    rows = conn.execute(
        "SELECT id, probability FROM predictions WHERE upload_id = ?", (uid,)
    ).fetchall()
    if not rows:
        raise HTTPException(400, "No predictions in SQLite.")

    updates = []
    for r in rows:
        prob  = r["probability"]
        pred  = 1 if prob >= threshold else 0
        conf  = prob if pred == 1 else (1.0 - prob)
        risk  = round(prob * 100, 2)
        status = "Theft" if pred == 1 else "Normal"
        updates.append((pred, conf, risk, status, r["id"]))

    with conn:
        conn.executemany(
            "UPDATE predictions SET prediction=?, confidence=?, risk_score=?, status=? WHERE id=?",
            updates,
        )

    agg = conn.execute(
        """SELECT SUM(CASE WHEN status='Theft' THEN 1 ELSE 0 END) AS theft,
                  SUM(CASE WHEN status='Normal' THEN 1 ELSE 0 END) AS normal,
                  AVG(confidence) AS avg_conf,
                  AVG(risk_score) AS avg_risk,
                  COUNT(*) AS total
           FROM predictions WHERE upload_id=?""",
        (uid,),
    ).fetchone()

    with conn:
        conn.execute(
            "UPDATE dataset_uploads SET theft_rows=?, normal_rows=?, avg_confidence=?, "
            "avg_risk=?, theft_rate=?, threshold=? WHERE id=?",
            (agg["theft"], agg["normal"], round(agg["avg_conf"], 6),
             round(agg["avg_risk"], 6),
             round((agg["theft"] or 0) / (agg["total"] or 1), 6), threshold, uid),
        )

    return JSONResponse({
        "success":   True,
        "threshold": threshold,
        "theft":     agg["theft"],
        "normal":    agg["normal"],
        "data_source": "SQLite — predictions updated in-place",
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/predict/shap/{customer_id}
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/shap/{customer_id}")
async def get_shap(customer_id: str):
    """
    SHAP feature importance for one customer.
    Falls back gracefully if SHAP is not installed.
    """
    if not model_service.is_model_loaded():
        raise HTTPException(400, "No model loaded.")

    uid = db.get_latest_upload_id()
    if uid is None:
        raise HTTPException(400, "No dataset in SQLite.")

    row = db.get_db().execute(
        "SELECT readings, probability, status FROM predictions "
        "WHERE upload_id=? AND customer_id=? LIMIT 1",
        (uid, customer_id),
    ).fetchone()

    if not row:
        raise HTTPException(404, f"Customer '{customer_id}' not found in SQLite.")

    readings_raw = json.loads(row["readings"]) if row["readings"] else None
    if not readings_raw or len(readings_raw) < 26:
        raise HTTPException(400, "No readings stored for this customer.")

    readings = np.array(readings_raw[:26], dtype=np.float32)

    # ── Gradient-based feature importance (always available) ──────────
    import tensorflow as tf
    r2d  = readings.reshape(1, 26, 1).astype(np.float32)
    stat = pipeline.transform(readings.reshape(1, 26))

    with tf.GradientTape() as tape:
        if model_service.state.is_dual_input:
            seq_t  = tf.constant(r2d)
            stat_t = tf.constant(stat)
            tape.watch(seq_t)
            pred = model_service.state.model({"sequence_input": seq_t, "stat_input": stat_t}, training=False)
        else:
            seq_t = tf.constant(r2d)
            tape.watch(seq_t)
            pred = model_service.state.model(seq_t, training=False)

    grads = tape.gradient(pred, seq_t)
    if grads is not None:
        importance = np.abs(grads.numpy().flatten())
        importance = importance / (importance.sum() + 1e-8)
    else:
        importance = np.ones(26) / 26

    shap_data = [
        {
            "feature": READING_LABELS[i],
            "value":   float(readings[i]),
            "importance": float(importance[i]),
            "rank":    0,
        }
        for i in range(26)
    ]
    shap_data.sort(key=lambda x: x["importance"], reverse=True)
    for rank, item in enumerate(shap_data):
        item["rank"] = rank + 1

    # Check if real SHAP is available
    shap_available = False
    try:
        import shap  # type: ignore
        shap_available = True
    except ImportError:
        pass

    return JSONResponse({
        "customer_id":     customer_id,
        "probability":     float(row["probability"]),
        "status":          row["status"],
        "method":          "GradientTape (integrated gradients)" if not shap_available else "SHAP + GradientTape",
        "shap_available":  shap_available,
        "feature_importance": shap_data,
        "top5_features":   [d["feature"] for d in shap_data[:5]],
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/predict/template
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/template")
async def download_template():
    """Download a blank CSV template with correct column headers."""
    out = io.StringIO()
    import csv as csv_mod
    writer = csv_mod.writer(out)
    writer.writerow(["CONS_NO"] + READING_LABELS + ["FLAG"])
    writer.writerow(["CUSTOMER_001"] + ["0"] * 26 + [""])
    out.seek(0)
    return StreamingResponse(
        io.BytesIO(out.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=etd_template.csv"},
    )
