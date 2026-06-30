"""Upload Router — /api/upload/*"""
import os, shutil, logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from services import model_service, dataset_service

router = APIRouter(prefix="/api/upload", tags=["upload"])
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_MODEL_EXT = {".keras", ".h5"}
ALLOWED_DATASET_EXT = {".csv"}


def _save_file(upload: UploadFile, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return str(dest)


@router.post("/model")
async def upload_model(model_file: UploadFile = File(...)):
    """
    Upload a CNN-LSTM model (.keras or .h5).
    Validates architecture and input shape before accepting.
    """
    ext = Path(model_file.filename).suffix.lower()
    if ext not in ALLOWED_MODEL_EXT:
        raise HTTPException(400, f"Invalid model format: {ext}. Use .keras or .h5")

    dest = UPLOAD_DIR / "model" / model_file.filename
    path = _save_file(model_file, dest)

    try:
        info = model_service.load_model(path, model_file.filename)
    except ValueError as e:
        os.remove(path)
        raise HTTPException(400, str(e))
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        raise HTTPException(500, f"Failed to load model: {e}")

    return JSONResponse({"success": True, "model_info": info})


@router.post("/dataset")
async def upload_dataset(
    dataset_file: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    """
    Upload a customer CSV dataset and run predictions.
    Model must be loaded before this endpoint is called.
    """
    if not model_service.is_model_loaded():
        raise HTTPException(400, "Upload a CNN-LSTM model first before uploading the dataset.")

    ext = Path(dataset_file.filename).suffix.lower()
    if ext not in ALLOWED_DATASET_EXT:
        raise HTTPException(400, "Invalid dataset format. Use .csv")

    dest = UPLOAD_DIR / "dataset" / dataset_file.filename
    path = _save_file(dataset_file, dest)

    try:
        summary = dataset_service.load_and_predict(path, dataset_file.filename, threshold)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Dataset processing failed")
        raise HTTPException(500, f"Processing failed: {e}")

    return JSONResponse({"success": True, "summary": summary})


@router.post("/reset-model")
async def reset_model():
    """Unload the current model (new model upload is required before predictions)."""
    model_service.unload_model()
    return JSONResponse({"success": True, "message": "Model unloaded."})


@router.get("/status")
async def upload_status():
    return JSONResponse({
        "model_loaded": model_service.is_model_loaded(),
        "dataset_loaded": dataset_service.is_dataset_loaded(),
        "model_info": model_service.get_model_info(),
    })
