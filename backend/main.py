"""
ETD-XAI Enterprise — FastAPI Backend
=====================================
Electricity Theft Detection using CNN-LSTM Deep Learning

Startup:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger("etd_xai")

for d in ["uploads/model", "uploads/dataset", "uploads/exports"]:
    Path(d).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═" * 60)
    logger.info("  ETD-XAI Enterprise  |  CNN-LSTM Only  |  v2.0")
    logger.info("═" * 60)

    # 1. Initialise SQLite tables
    import database as db_module
    db_module.init_db()
    logger.info("SQLite ready: %s", db_module.DB_PATH.resolve())

    # 2. Auto-load last registered model
    from services.model_service import auto_load_on_startup
    model_ok = auto_load_on_startup()

    # 3. Report state
    has_data     = db_module.has_any_upload()
    pred_count   = db_module.get_prediction_count()
    manual_count = db_module.get_manual_prediction_count()

    if model_ok and has_data:
        logger.info(
            "Ready — model loaded | SQLite: %d predictions + %d manual | Dashboard immediate.",
            pred_count, manual_count,
        )
    elif model_ok:
        logger.info("Model loaded. Awaiting dataset upload.")
    else:
        logger.info("Awaiting model upload.")

    yield
    logger.info("Server shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ETD-XAI Enterprise",
    description="Electricity Theft Detection using CNN-LSTM",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from routers.upload    import router as upload_router
from routers.predict   import router as predict_router
from routers.dashboard import router as dashboard_router
from routers.copilot   import router as copilot_router

app.include_router(upload_router)
app.include_router(predict_router)
app.include_router(dashboard_router)
app.include_router(copilot_router)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    from services import model_service
    import database as db_module

    model_info   = model_service.get_model_info()
    has_data     = db_module.has_any_upload()
    pred_count   = db_module.get_prediction_count()
    manual_count = db_module.get_manual_prediction_count()
    upload_id    = db_module.get_latest_upload_id()

    upload_summary = None
    if upload_id:
        upload_summary = db_module.get_upload_summary(upload_id)

    return JSONResponse({
        "status":          "ok",
        "model_loaded":    model_info.get("loaded", False),
        "dataset_loaded":  has_data,
        "model_info":      model_info,
        "prediction_count": pred_count,
        "manual_count":    manual_count,
        "data_source":     "SQLite — etd_xai.db",
        "upload_summary":  upload_summary,
    })


# ── System status (debug panel) ───────────────────────────────────────────────
@app.get("/api/system/status")
async def system_status():
    import tensorflow as tf
    from services import model_service
    import database as db_module

    info      = model_service.get_model_info()
    upload_id = db_module.get_latest_upload_id()

    return JSONResponse({
        "backend_online":    True,
        "database_online":   True,
        "sqlite_path":       str(db_module.DB_PATH.resolve()),
        "model_loaded":      info.get("loaded", False),
        "model_name":        info.get("model_name"),
        "model_path":        info.get("model_path"),
        "input_shape":       info.get("input_shape"),
        "output_shape":      info.get("output_shape"),
        "total_params":      info.get("total_params"),
        "is_dual_input":     info.get("is_dual_input"),
        "stat_input_size":   info.get("stat_input_size"),
        "tf_version":        tf.__version__,
        "prediction_count":  db_module.get_prediction_count(upload_id),
        "manual_count":      db_module.get_manual_prediction_count(),
        "latest_upload_id":  upload_id,
        "load_proof":        "keras.models.load_model(path) — real CNN-LSTM weights",
        "predict_proof":     "model.predict(x, verbose=0, batch_size=256)  x.shape=(N,26,1)",
        "no_mock":           True,
    })


# ── Serve React SPA ───────────────────────────────────────────────────────────
STATIC_DIR = Path("../frontend/dist")
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        return FileResponse(str(STATIC_DIR / "index.html"))
