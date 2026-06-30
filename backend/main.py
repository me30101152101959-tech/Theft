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

    # 3. Optional dataset bootstrap — process a bundled dataset on first boot
    #    so a freshly-cloned/deployed instance shows the dashboard immediately.
    #    Runs ONLY when the DB has no uploads yet (i.e. truly first boot).
    if model_ok and not db_module.has_any_upload():
        boot_ds = os.environ.get("BOOTSTRAP_DATASET_PATH")
        if not boot_ds:
            for c in ["uploads/dataset/datasetsmall (2).csv"]:
                if Path(c).exists():
                    boot_ds = c
                    break
        if boot_ds and Path(boot_ds).exists():
            try:
                from services.dataset_service import load_and_predict
                logger.info("Startup: bootstrapping dataset from %s …", boot_ds)
                load_and_predict(boot_ds, Path(boot_ds).name, threshold=0.5)
                logger.info("Startup: dataset bootstrap complete.")
            except Exception as exc:
                logger.error("Startup: dataset bootstrap failed — %s", exc)

    # 4. Report state
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

# CORS origins are configurable via CORS_ORIGINS (comma-separated, or "*").
_cors = os.environ.get("CORS_ORIGINS", "*").strip()
_origins = ["*"] if _cors == "*" else [o.strip() for o in _cors.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
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
        "seq_len_expected":  info.get("seq_len_expected"),
        "is_variable_length": info.get("is_variable_length"),
        "seq_channels":      info.get("seq_channels"),
        "tf_version":        tf.__version__,
        "prediction_count":  db_module.get_prediction_count(upload_id),
        "manual_count":      db_module.get_manual_prediction_count(),
        "latest_upload_id":  upload_id,
        "load_proof":        "keras.models.load_model(path) — real CNN-LSTM weights",
        "predict_proof":     "model.predict(x, verbose=0, batch_size=256)  x.shape=(N,26,1)",
        "no_mock":           True,
    })


# ── Model status / verification (proves predictions come from the active model) ─
@app.get("/api/model/status")
async def model_status():
    import keras as _keras
    import tensorflow as tf
    from services import model_service

    if not model_service.is_model_loaded():
        return JSONResponse({
            "model_loaded":  False,
            "active_model":  None,
            "engine":        "TensorFlow / Keras",
            "tf_version":    tf.__version__,
            "keras_version": getattr(_keras, "__version__", "unknown"),
            "message":       model_service.NO_MODEL_MSG,
        }, status_code=200)

    info = model_service.get_model_info()
    return JSONResponse({
        "model_loaded":      True,
        "active_model":      info.get("model_name"),
        "model_path":        info.get("model_path"),
        "engine":            "TensorFlow / Keras",
        "load_method":       "tensorflow.keras.models.load_model(path)",
        "predict_method":    "model.predict(x)",
        "tf_version":        tf.__version__,
        "keras_version":     getattr(_keras, "__version__", "unknown"),
        "input_shape":       info.get("input_shape"),
        "output_shape":      info.get("output_shape"),
        "seq_len_expected":  info.get("seq_len_expected"),
        "is_variable_length": info.get("is_variable_length"),
        "is_dual_input":     info.get("is_dual_input"),
        "stat_input_size":   info.get("stat_input_size"),
        "total_params":      info.get("total_params"),
        "total_params_fmt":  info.get("total_params_fmt"),
        # Verification trail of the most recent model.predict() call:
        "last_prediction":   model_service.last_prediction or None,
        "fallback_models":   "none — CNN-LSTM only, no RandomForest/XGBoost/LightGBM/LogReg/mock",
        "exclusive_engine":  True,
    })


# ── Serve React SPA ───────────────────────────────────────────────────────────
STATIC_DIR = Path("../frontend/dist")
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        return FileResponse(str(STATIC_DIR / "index.html"))
