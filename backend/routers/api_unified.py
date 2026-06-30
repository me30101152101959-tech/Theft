"""
Unified API Router  —  all endpoints backed by SQLite
======================================================
Every GET endpoint reads from SQLite.
Every POST/upload writes to SQLite immediately after inference.
No Python variables are used as the data source.

Endpoints
---------
  GET  /api/dashboard       — KPIs + chart data from SQLite
  GET  /api/customers       — paginated customer list from SQLite
  POST /api/load-model      — upload + load CNN-LSTM, register in SQLite
  POST /api/upload          — upload CSV → predict → store in SQLite
  POST /api/predict         — predict single customer (stores to manual_predictions)
  POST /api/predict-batch   — predict CSV rows, return results (no storage)
  GET  /api/system/status   — debug panel: backend, DB, model, shapes, counts
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

import database as db
from services import model_service
from services.dataset_service import load_and_predict
from services.feature_service import pipeline

router = APIRouter(prefix="/api", tags=["unified"])
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/system/status
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/system/status")
async def system_status():
    """
    Debug panel.
    Returns backend + database + model + prediction counts — all from SQLite.
    """
    model_info   = model_service.get_model_info()
    upload_rec   = db.get_dashboard_from_db() or {}
    pred_count   = db.get_prediction_count()
    db_upload_id = db.get_latest_upload_id()
    active_model = db.get_active_model() or {}

    # Locate model file on disk
    model_path_on_disk = ""
    for candidate in [
        "uploads/model/cnnlstm_final.keras",
        "uploads/models/cnnlstm_final.keras",
    ]:
        if Path(candidate).exists():
            model_path_on_disk = str(Path(candidate).resolve())
            break

    return JSONResponse({
        # System
        "backend_online":      True,
        "database_online":     True,
        "sqlite_path":         str(db.DB_PATH.resolve()),
        # Model
        "model_loaded":        model_info.get("loaded", False),
        "model_name":          model_info.get("model_name", ""),
        "model_path_memory":   model_info.get("model_path", ""),
        "model_path_sqlite":   active_model.get("model_path", ""),
        "model_path_on_disk":  model_path_on_disk,
        "model_architecture":  model_info.get("architecture", "CNN-LSTM"),
        "tensorflow_version":  tf.__version__,
        "keras_version":       keras.__version__,
        "input_shape":         model_info.get("input_shape", ""),
        "output_shape":        model_info.get("output_shape", ""),
        "total_params":        model_info.get("total_params", 0),
        "total_params_fmt":    model_info.get("total_params_fmt", ""),
        "is_dual_input":       model_info.get("is_dual_input", False),
        "stat_input_size":     model_info.get("stat_input_size", 0),
        "load_proof":          model_info.get("load_proof", ""),
        # Dataset / predictions
        "dataset_loaded":      db.has_any_upload(),
        "latest_upload_id":    db_upload_id,
        "dataset_name":        upload_rec.get("filename", ""),
        "dataset_rows":        upload_rec.get("total_rows", 0),
        "prediction_count":    pred_count,
        "last_upload_time":    upload_rec.get("upload_time", ""),
        "data_source":         "SQLite — etd_xai.db",
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/load-model
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/load-model")
async def load_model_endpoint(model_file: UploadFile = File(...)):
    """
    Upload cnnlstm_final.keras → load with keras.models.load_model() → register in SQLite.
    File is saved to uploads/model/ so startup auto-load can find it next time.
    """
    ext = Path(model_file.filename).suffix.lower()
    if ext not in (".keras", ".h5"):
        raise HTTPException(400, f"Invalid extension: '{ext}'. Use .keras or .h5")

    dest = UPLOAD_DIR / "model" / model_file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[load-model] Saving uploaded file → %s", dest)
    with open(dest, "wb") as f:
        shutil.copyfileobj(model_file.file, f)

    logger.info(
        "[load-model] keras.models.load_model('%s', compile=False)  "
        "TF=%s  Keras=%s",
        dest, tf.__version__, keras.__version__,
    )
    try:
        info = model_service.load_model(str(dest), model_file.filename)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(exc))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.exception("[load-model] TensorFlow exception")
        raise HTTPException(500, f"TensorFlow error loading model: {exc}")

    logger.info(
        "[load-model] SUCCESS — %s | params=%s | input=%s | output=%s",
        info["model_name"], info["total_params_fmt"],
        info["input_shape"], info["output_shape"],
    )
    return JSONResponse({
        "success":   True,
        "message":   f"CNN-LSTM loaded — {info['total_params_fmt']} parameters",
        "model_info": info,
        "load_proof": info["load_proof"],
        "sqlite_registered": True,
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/upload
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/upload")
async def upload_dataset(
    dataset_file: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    """
    Upload CSV → model.predict() → store ALL results in SQLite.
    FLAG is used ONLY for evaluation metrics, never for prediction.
    """
    if not model_service.is_model_loaded():
        raise HTTPException(400, "No CNN-LSTM model loaded. Upload cnnlstm_final.keras first.")
    if Path(dataset_file.filename).suffix.lower() != ".csv":
        raise HTTPException(400, "Dataset must be a .csv file.")

    dest = UPLOAD_DIR / "dataset" / dataset_file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f:
        shutil.copyfileobj(dataset_file.file, f)

    logger.info(
        "[upload] Dataset: %s | threshold=%.2f | model=%s",
        dataset_file.filename, threshold, model_service.state.model_name,
    )
    try:
        t0      = time.time()
        summary = load_and_predict(str(dest), dataset_file.filename, threshold)
        elapsed = round(time.time() - t0, 2)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("[upload] Pipeline failure")
        raise HTTPException(500, f"Prediction pipeline error: {exc}")

    logger.info(
        "[upload] Done in %.2fs — upload_id=%s | theft=%d | normal=%d | SQLite=✓",
        elapsed, summary["upload_id"], summary["theft"], summary["normal"],
    )
    return JSONResponse({
        "success":        True,
        "summary":        summary,
        "elapsed_seconds": elapsed,
        "predict_proof":  "model.predict(x, verbose=0, batch_size=256)  x.shape=(N,26,1)",
        "storage":        "All results written to SQLite — dashboard ready.",
        "flag_note":      "FLAG used only for evaluation. Not used in prediction.",
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/dashboard
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/dashboard")
async def get_dashboard():
    """
    Returns ALL dashboard data read from SQLite.
    Source: SELECT * FROM dataset_uploads + aggregates FROM predictions.
    Never reads Python in-memory variables.
    """
    upload = db.get_dashboard_from_db()

    if upload is None:
        return JSONResponse({
            "ready":        False,
            "model_loaded": model_service.is_model_loaded(),
            "model_name":   model_service.get_model_info().get("model_name", ""),
            "message":      "No dataset in SQLite. Upload cnnlstm_final.keras then your CSV.",
            "data_source":  "SQLite — etd_xai.db",
        })

    uid  = upload["id"]
    info = model_service.get_model_info()

    charts = db.get_chart_data_from_db(uid)

    return JSONResponse({
        "ready":             True,
        # ── KPIs from SQLite ──────────────────────────────────────────
        "total_customers":    upload["total_rows"],
        "processed_customers":upload["total_rows"],
        "predicted_theft":    upload["theft_rows"],
        "predicted_normal":   upload["normal_rows"],
        "avg_confidence":     upload["avg_confidence"],
        "avg_risk_score":     upload["avg_risk"],
        "theft_rate":         upload["theft_rate"],
        "dataset_name":       upload["filename"],
        "upload_time":        upload["upload_time"],
        "has_flag":           bool(upload["has_flag"]),
        # ── Evaluation metrics (only if FLAG was present) ─────────────
        "accuracy":           upload["accuracy"],
        "precision":          upload["precision_val"],
        "recall":             upload["recall_val"],
        "f1_score":           upload["f1_score"],
        "roc_auc":            upload["roc_auc"],
        # ── Charts (SQL aggregates) ───────────────────────────────────
        "charts":             charts,
        # ── Model info ────────────────────────────────────────────────
        "model_name":         info.get("model_name", ""),
        "model_architecture": "CNN-LSTM",
        "model_params":       info.get("total_params_fmt", ""),
        "input_shape":        info.get("input_shape", ""),
        "output_shape":       info.get("output_shape", ""),
        "is_dual_input":      info.get("is_dual_input", False),
        # ── Proof ─────────────────────────────────────────────────────
        "data_source":        "SQLite — etd_xai.db",
        "predict_proof":      "model.predict(x, verbose=0, batch_size=256)  x.shape=(N,26,1)",
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/customers
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/customers")
async def get_customers(
    page:          int = Query(1,    ge=1),
    page_size:     int = Query(50,   ge=1, le=500),
    search:        str = Query(""),
    status_filter: str = Query(""),
    sort_by:       str = Query("risk_score"),
    sort_dir:      str = Query("desc"),
):
    """
    Paginated customer predictions — reads from SQLite predictions table.
    SQL: SELECT ... FROM predictions WHERE upload_id=? ORDER BY ? LIMIT ? OFFSET ?
    """
    uid = db.get_latest_upload_id()
    if uid is None:
        raise HTTPException(400, "No dataset in SQLite. Upload a CSV first.")

    result = db.get_customers_from_db(
        upload_id=uid,
        page=page, page_size=page_size,
        search=search, status_filter=status_filter,
        sort_by=sort_by, sort_dir=sort_dir,
    )
    result["data_source"] = "SQLite — predictions table"
    return JSONResponse(result)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/predict")
async def predict_single(payload: dict):
    """
    Predict ONE customer.
    Calls model.predict(x) where x.shape == (1, 26, 1).
    Result is stored in manual_predictions table (does NOT affect evaluation data).
    """
    if not model_service.is_model_loaded():
        raise HTTPException(400, "No CNN-LSTM model loaded. Upload cnnlstm_final.keras first.")

    customer_id  = str(payload.get("customer_id", "MANUAL"))
    readings_raw = payload.get("readings", [])
    threshold    = float(payload.get("threshold", 0.5))

    if len(readings_raw) != 26:
        raise HTTPException(400, f"Expected 26 readings, got {len(readings_raw)}.")

    try:
        readings = np.array(readings_raw, dtype=np.float32)
        result   = model_service.predict_single(readings, threshold=threshold)
    except Exception as exc:
        logger.exception("[predict] model.predict() failed")
        raise HTTPException(500, f"TensorFlow error: {exc}")

    # Persist to manual_predictions (separate from dataset evaluation)
    now = datetime.utcnow().isoformat()
    db.save_manual_prediction(
        customer_id  = customer_id,
        probability  = result["probability"],
        prediction   = result["prediction"],
        confidence   = result["confidence"],
        risk_score   = result["risk_score"],
        status       = result["status"],
        readings     = readings_raw,
        predicted_at = now,
        threshold    = threshold,
        model_name   = model_service.state.model_name,
    )

    result["customer_id"] = customer_id
    result["readings"]    = readings_raw
    result["stored_in"]   = "SQLite — manual_predictions table"

    logger.info(
        "[predict] customer=%s | prob=%.4f | status=%s | proof=%s",
        customer_id, result["probability"], result["status"], result["predict_proof"],
    )
    return JSONResponse({"success": True, "result": result})


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict-batch
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/predict-batch")
async def predict_batch_endpoint(
    csv_file:  UploadFile = File(...),
    threshold: float      = Form(0.5),
):
    """
    Upload CSV → model.predict() on every row → return results.
    Does NOT persist to dataset_uploads or predictions tables.
    Use /api/upload to persist.
    """
    if not model_service.is_model_loaded():
        raise HTTPException(400, "No CNN-LSTM model loaded.")

    content = await csv_file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8", errors="replace")))

    skip = {"CONS_NO", "FLAG"}
    reading_cols = [c for c in df.columns if c.strip().upper() not in skip][:26]
    if len(reading_cols) < 26:
        raise HTTPException(400, f"Need ≥26 reading columns, found {len(reading_cols)}.")

    for c in reading_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    readings = df[reading_cols].values.astype(np.float32)
    n = len(readings)

    stat_feats = None
    if model_service.state.is_dual_input:
        stat_feats = pipeline.fit_transform(readings)

    logger.info(
        "[predict-batch] model.predict(x) x.shape=(%d,26,1) model=%s",
        n, model_service.state.model_name,
    )
    t0   = time.time()
    probs = model_service.predict_batch(
        readings=readings, stat_feats=stat_feats, threshold=threshold
    )
    elapsed = round(time.time() - t0, 2)

    id_col = next(
        (c for c in df.columns if c.strip().upper() == "CONS_NO"),
        df.columns[0],
    )
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        prob = float(probs[i])
        pred = 1 if prob >= threshold else 0
        conf = prob if pred == 1 else (1.0 - prob)
        results.append({
            "id":          str(row[id_col]),
            "probability": round(prob, 6),
            "prediction":  pred,
            "confidence":  round(conf, 6),
            "risk_score":  round(prob * 100, 2),
            "status":      "Theft" if pred == 1 else "Normal",
        })

    theft = sum(1 for r in results if r["prediction"] == 1)
    return JSONResponse({
        "success":         True,
        "total":           n,
        "theft":           theft,
        "normal":          n - theft,
        "elapsed_seconds": elapsed,
        "predictions":     results,
        "predict_proof":   f"model.predict(x, verbose=0, batch_size=256)  x.shape=({n},26,1)",
    })
