"""
Upload Router  —  /api/upload/*  (legacy URLs, SQLite-backed)
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import database as db
from services import model_service
from services.dataset_service import load_and_predict

router = APIRouter(prefix="/api/upload", tags=["upload"])
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/model")
async def upload_model(model_file: UploadFile = File(...)):
    """Load CNN-LSTM model and register in SQLite."""
    ext = Path(model_file.filename).suffix.lower()
    if ext not in (".keras", ".h5"):
        raise HTTPException(400, f"Invalid model format: {ext}. Use .keras or .h5")

    dest = UPLOAD_DIR / "model" / model_file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(model_file.file, f)

    try:
        info = model_service.load_model(str(dest), model_file.filename)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(exc))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.exception("[upload/model] load failed")
        raise HTTPException(500, f"TensorFlow error: {exc}")

    return JSONResponse({
        "success":    True,
        "model_info": info,
        "load_proof": info.get("load_proof", ""),
        "sqlite_registered": True,
    })


@router.post("/dataset")
async def upload_dataset(
    dataset_file: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    """Upload CSV → predict → store ALL results in SQLite."""
    if not model_service.is_model_loaded():
        raise HTTPException(400, "Upload cnnlstm_final.keras first.")
    if Path(dataset_file.filename).suffix.lower() != ".csv":
        raise HTTPException(400, "Dataset must be a .csv file.")

    dest = UPLOAD_DIR / "dataset" / dataset_file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(dataset_file.file, f)

    try:
        summary = load_and_predict(str(dest), dataset_file.filename, threshold)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("[upload/dataset] pipeline failed")
        raise HTTPException(500, f"Prediction pipeline error: {exc}")

    return JSONResponse({
        "success": True,
        "summary": summary,
        "storage": "All results written to SQLite — etd_xai.db",
    })


@router.post("/reset-model")
async def reset_model():
    model_service.unload_model()
    return JSONResponse({"success": True, "message": "Model unloaded from memory."})


@router.get("/status")
async def upload_status():
    return JSONResponse({
        "model_loaded":    model_service.is_model_loaded(),
        "dataset_loaded":  db.has_any_upload(),
        "prediction_count": db.get_prediction_count(),
        "model_info":      model_service.get_model_info(),
        "data_source":     "SQLite — etd_xai.db",
    })
