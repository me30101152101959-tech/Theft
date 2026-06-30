"""
Prediction Router — /api/predict/*  (dynamic, model-driven)
===========================================================
Sequence length and statistical-feature count are discovered from the model
and dataset — nothing hardcoded. The user chooses a preprocessing strategy
when the dataset length differs from the model length.

Endpoints
---------
  POST /api/predict/manual            — single customer (any length) → SQLite
  POST /api/predict/validate-dataset  — inspect CSV + model compatibility (no predict)
  POST /api/predict/batch-preview     — CSV → predict (no store)
  POST /api/predict/batch-store       — CSV → predict → store in SQLite
  POST /api/predict/update-threshold  — reclassify latest upload
  GET  /api/predict/shap/{cid}        — gradient feature importance
  GET  /api/predict/history           — recent manual predictions
  GET  /api/predict/strategies        — list available preprocessing strategies
  GET  /api/predict/template          — blank CSV template (model length)
"""
from __future__ import annotations

import csv as csv_mod
import io
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
from services import model_service, preprocessing
from services.feature_service import scale_sequences

router = APIRouter(prefix="/api/predict", tags=["predict"])
logger = logging.getLogger(__name__)

ID_COLS   = {"CONS_NO", "CUSTOMER_ID", "ID", "CUSTOMER", "METER_ID"}
FLAG_COLS = {"FLAG", "LABEL", "TARGET", "THEFT", "Y"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _require_model():
    if not model_service.is_model_loaded():
        raise HTTPException(400, model_service.NO_MODEL_MSG)


def _risk_level(risk: float) -> str:
    return "High" if risk >= 75 else "Medium" if risk >= 40 else "Low"


def _inspect_csv(content: bytes) -> dict:
    """
    Parse CSV and dynamically detect: id column, flag column, and ALL reading
    columns (everything numeric that is not id/flag). Returns a rich info dict.
    """
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8", errors="replace")))
    except Exception as exc:
        raise HTTPException(400, f"Cannot parse CSV: {exc}")
    if df.empty:
        raise HTTPException(400, "CSV file is empty.")

    # Identify id / flag columns by name (case-insensitive)
    id_col = next((c for c in df.columns if c.strip().upper() in ID_COLS), None)
    flag_col = next((c for c in df.columns if c.strip().upper() in FLAG_COLS), None)

    # Reading columns = everything else, coerced to numeric
    reading_cols = [c for c in df.columns if c not in (id_col, flag_col)]
    # keep only columns that are (mostly) numeric
    numeric_reading_cols = []
    for c in reading_cols:
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().mean() >= 0.5:      # at least half numeric → it's a reading column
            df[c] = coerced.fillna(0.0)
            numeric_reading_cols.append(c)

    seq_len = len(numeric_reading_cols)
    if seq_len < 2:
        raise HTTPException(
            400,
            f"Could not detect numeric reading columns (found {seq_len}). "
            f"Provide a CSV with an ID column and ≥2 consumption columns."
        )

    missing = int(pd.to_numeric(df[numeric_reading_cols].stack(), errors="coerce").isna().sum())
    dup = int(df[id_col].duplicated().sum()) if id_col else 0

    return {
        "df":            df,
        "id_col":        id_col,
        "flag_col":      flag_col,
        "reading_cols":  numeric_reading_cols,
        "seq_len":       seq_len,
        "n_customers":   len(df),
        "missing":       missing,
        "duplicates":    dup,
        "has_flag":      flag_col is not None,
    }


def _build_rows(df, info, probs, threshold, store_readings=True):
    id_col, flag_col, reading_cols = info["id_col"], info["flag_col"], info["reading_cols"]
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
            "risk_level":  _risk_level(risk),
            "status":      "Theft" if pred == 1 else "Normal",
            "flag":        int(row[flag_col]) if flag_col else None,
            "readings":    [float(row[c]) for c in reading_cols] if store_readings else [],
            "predicted_at": now,
        })
    return rows


def _eval_metrics(df, info, probs, threshold) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score as sk_f1, roc_auc_score,
        confusion_matrix, roc_curve, precision_recall_curve,
    )
    y_true = df[info["flag_col"]].values.astype(int)
    y_pred = (probs >= threshold).astype(int)
    try:
        roc_auc = round(float(roc_auc_score(y_true, probs)), 6)
    except Exception:
        roc_auc = 0.0
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, probs)
    pp, rr, _ = precision_recall_curve(y_true, probs)
    return {
        "accuracy":         round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_val":    round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall_val":       round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_score":         round(float(sk_f1(y_true, y_pred, zero_division=0)), 6),
        "roc_auc":          roc_auc,
        "confusion_matrix": cm.tolist(),
        "roc_fpr":          fpr.tolist(),
        "roc_tpr":          tpr.tolist(),
        "pr_precision":     pp.tolist(),
        "pr_recall":        rr.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/predict/strategies
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/strategies")
async def list_strategies():
    return JSONResponse({
        "strategies": [
            {"value": s, "label": preprocessing.STRATEGY_LABELS[s]}
            for s in preprocessing.STRATEGIES
        ],
        "default": "last_n",
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/validate-dataset
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/validate-dataset")
async def validate_dataset(
    csv_file: UploadFile = File(...),
    strategy: str        = Form("last_n"),
):
    """Inspect the CSV and report compatibility with the loaded model. No prediction."""
    _require_model()
    content = await csv_file.read()
    info    = _inspect_csv(content)
    df      = info["df"]

    compat  = model_service.check_compatibility(info["seq_len"], strategy)

    preview = []
    for _, row in df.head(20).iterrows():
        rec = {}
        if info["id_col"]:
            rec["customer_id"] = str(row[info["id_col"]])
        for c in info["reading_cols"][:8]:
            rec[c] = float(row[c])
        if info["flag_col"]:
            rec["FLAG"] = int(row[info["flag_col"]])
        preview.append(rec)

    return JSONResponse({
        "success":          True,
        "dataset_name":     csv_file.filename,
        "n_customers":      info["n_customers"],
        "n_reading_cols":   info["seq_len"],
        "detected_seq_len": info["seq_len"],
        "missing_values":   info["missing"],
        "duplicate_customers": info["duplicates"],
        "id_column":        info["id_col"],
        "ground_truth_column": info["flag_col"],
        "feature_columns":  info["reading_cols"][:30],
        "status":           "Ready" if compat["compatible"] else "Incompatible",
        "compatibility":    compat,
        "preview":          preview,
        "preview_columns":  (["customer_id"] if info["id_col"] else []) + info["reading_cols"][:8]
                            + (["FLAG"] if info["flag_col"] else []),
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/manual
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/manual")
async def manual_predict(req: ManualPredictionRequest):
    """Single customer of ANY length. Resized to the model length via chosen strategy."""
    _require_model()
    readings = np.array(req.readings, dtype=np.float32)
    if len(readings) < 2:
        raise HTTPException(400, "Provide at least 2 readings.")

    try:
        result = model_service.predict_one(readings, strategy=req.strategy, threshold=req.threshold)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("[manual] prediction failed")
        raise HTTPException(500, f"Inference error: {exc}")

    now = datetime.utcnow().isoformat()
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
    })
    logger.info("[manual] id=%s | prob=%.4f | %s | row=%d",
                req.customer_id, result["probability"], result["status"], row_id)
    return JSONResponse({"success": True, "result": result})


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/batch-preview
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/batch-preview")
async def batch_preview(
    csv_file:  UploadFile = File(...),
    threshold: float      = Form(0.5),
    strategy:  str        = Form("last_n"),
):
    """Predict CSV rows (no store). Dynamic length + chosen strategy."""
    _require_model()
    content = await csv_file.read()
    info    = _inspect_csv(content)
    df      = info["df"]
    raw     = df[info["reading_cols"]].values.astype(np.float32)

    t0 = time.time()
    try:
        probs = model_service.predict_sequences(raw, strategy=strategy, threshold=threshold, fit_scaler=True)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    elapsed = round(time.time() - t0, 2)

    rows = _build_rows(df, info, probs, threshold, store_readings=False)
    theft = sum(r["prediction"] for r in rows)

    metrics = {}
    if info["has_flag"]:
        m = _eval_metrics(df, info, probs, threshold)
        metrics = {k: v for k, v in m.items()
                   if k not in ("confusion_matrix", "roc_fpr", "roc_tpr", "pr_precision", "pr_recall")}

    return JSONResponse({
        "success":          True,
        "stored":           False,
        "total":            len(rows),
        "theft":            theft,
        "normal":           len(rows) - theft,
        "avg_confidence":   round(float(np.mean([r["confidence"] for r in rows])), 4),
        "avg_risk":         round(float(np.mean([r["risk_score"] for r in rows])), 4),
        "elapsed_seconds":  elapsed,
        "has_flag":         info["has_flag"],
        "metrics":          metrics,
        "predictions":      rows,
        "detected_seq_len": info["seq_len"],
        "model_seq_len":    model_service.state.seq_len_expected,
        "strategy_used":    strategy,
        "predict_proof":    f"model.predict() on {len(rows)} rows | uploaded_len={info['seq_len']} "
                            f"model_len={model_service.state.seq_len_expected} strategy={strategy}",
        "model_name":       model_service.state.model_name,
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/batch-store
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/batch-store")
async def batch_store(
    csv_file:  UploadFile = File(...),
    threshold: float      = Form(0.5),
    strategy:  str        = Form("last_n"),
    label:     str        = Form("Batch Upload"),
):
    """CSV → predict → store ALL in SQLite (appears on Customer Predictions + dashboard)."""
    _require_model()
    content = await csv_file.read()
    info    = _inspect_csv(content)
    df      = info["df"]
    raw     = df[info["reading_cols"]].values.astype(np.float32)

    t0 = time.time()
    try:
        probs = model_service.predict_sequences(raw, strategy=strategy, threshold=threshold, fit_scaler=True)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    elapsed = round(time.time() - t0, 2)

    rows  = _build_rows(df, info, probs, threshold, store_readings=True)
    theft = sum(r["prediction"] for r in rows)
    now   = datetime.utcnow().isoformat()
    n     = len(rows)

    metrics = _eval_metrics(df, info, probs, threshold) if info["has_flag"] else {}

    upload_id = db.save_upload(
        filename         = f"{label} — {csv_file.filename}",
        upload_time      = now,
        total_rows       = n,
        theft_rows       = theft,
        normal_rows      = n - theft,
        avg_confidence   = round(float(np.mean([r["confidence"] for r in rows])), 6),
        avg_risk         = round(float(np.mean([r["risk_score"] for r in rows])), 6),
        theft_rate       = round(theft / n, 6) if n else 0.0,
        has_flag         = info["has_flag"],
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
    logger.info("[batch-store] upload_id=%d | %d rows in %.2fs | theft=%d", upload_id, n, elapsed, theft)

    pub = {k: v for k, v in metrics.items()
           if k not in ("roc_fpr", "roc_tpr", "pr_precision", "pr_recall", "confusion_matrix")}

    return JSONResponse({
        "success":          True,
        "upload_id":        upload_id,
        "stored":           True,
        "total":            n,
        "theft":            theft,
        "normal":           n - theft,
        "avg_confidence":   round(float(np.mean([r["confidence"] for r in rows])), 4),
        "avg_risk":         round(float(np.mean([r["risk_score"] for r in rows])), 4),
        "elapsed_seconds":  elapsed,
        "has_flag":         info["has_flag"],
        "metrics":          pub,
        "detected_seq_len": info["seq_len"],
        "model_seq_len":    model_service.state.seq_len_expected,
        "strategy_used":    strategy,
        "predict_proof":    f"model.predict() on {n} rows | strategy={strategy}",
        "stored_in":        f"SQLite — predictions (upload_id={upload_id})",
        "model_name":       model_service.state.model_name,
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/update-threshold
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/update-threshold")
async def update_threshold(req: ThresholdUpdateRequest):
    uid = db.get_latest_upload_id()
    if uid is None:
        raise HTTPException(400, "No dataset in SQLite.")
    result = db.update_predictions_threshold(uid, req.threshold)
    return JSONResponse({
        "success": True, "threshold": req.threshold,
        "theft": result["theft"], "normal": result["normal"],
        "data_source": "SQLite — predictions reclassified in-place",
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/predict/shap/{customer_id}
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/shap/{customer_id}")
async def get_shap(customer_id: str):
    """Gradient-based feature importance over the model's sequence input."""
    _require_model()
    uid = db.get_latest_upload_id()
    row = db.get_customer_by_id(uid, customer_id) if uid else None
    if not row:
        for h in db.get_manual_predictions(limit=500):
            if h.get("customer_id") == customer_id:
                row = {"id": customer_id, "readings": h["readings"],
                       "probability": h["probability"], "status": h["status"]}
                break
    if not row:
        raise HTTPException(404, f"Customer '{customer_id}' not found in SQLite.")

    readings_raw = row.get("readings", [])
    if len(readings_raw) < 2:
        raise HTTPException(400, "No readings stored for this customer.")

    raw = np.array(readings_raw, dtype=np.float32).reshape(1, -1)
    # Resize to model length, then scale per-row
    ready = model_service._model_ready(raw, "last_n")
    T = ready.shape[1]
    seq_scaled = scale_sequences(ready)

    import tensorflow as tf
    seq_var = tf.Variable(seq_scaled.reshape(1, T, model_service.state.seq_channels).astype(np.float32))
    with tf.GradientTape() as tape:
        tape.watch(seq_var)
        if model_service.state.is_dual_input:
            stat = model_service._build_stat(ready, fit_scaler=False)
            out = model_service.state.model(
                {"sequence_input": seq_var, "stat_input": tf.constant(stat.astype(np.float32))},
                training=False)
        else:
            out = model_service.state.model(seq_var, training=False)
    grads = tape.gradient(out, seq_var)

    if grads is not None:
        imp = np.abs(grads.numpy().flatten())
        s = imp.sum()
        imp = imp / s if s > 1e-10 else np.ones(T) / T
    else:
        imp = np.ones(T) / T

    labels = [f"T{i+1}" for i in range(T)]
    shap_data = sorted(
        [{"feature": labels[i], "value": float(ready[0][i]), "importance": float(imp[i]), "rank": 0}
         for i in range(T)],
        key=lambda x: x["importance"], reverse=True,
    )
    for rank, item in enumerate(shap_data):
        item["rank"] = rank + 1

    return JSONResponse({
        "customer_id":        customer_id,
        "probability":        float(row.get("probability", 0)),
        "status":             row.get("status", "Unknown"),
        "method":             "GradientTape integrated gradients",
        "feature_importance": shap_data,
        "top5_features":      [d["feature"] for d in shap_data[:5]],
        "readings_labels":    labels,
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/predict/history
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/history")
async def prediction_history(limit: int = Query(100, ge=1, le=1000), source: Optional[str] = Query(None)):
    rows = db.get_manual_predictions(limit=limit, source=source)
    return JSONResponse({"success": True, "count": len(rows), "history": rows})


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/predict/template
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/template")
async def download_template():
    """Blank CSV template sized to the model's expected sequence length (or 26 if variable)."""
    T = model_service.state.seq_len_expected if model_service.is_model_loaded() else None
    n = T if T else 26
    labels = [f"T{i+1}" for i in range(n)]
    buf = io.StringIO()
    w = csv_mod.writer(buf)
    w.writerow(["CONS_NO"] + labels + ["FLAG"])
    w.writerow(["CUSTOMER_001"] + [str((i + 1) * 100) for i in range(n)] + [""])
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=etd_prediction_template.csv"},
    )
