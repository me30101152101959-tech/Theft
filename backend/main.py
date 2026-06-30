"""
ETD-XAI Enterprise  —  FastAPI Backend
=======================================
Electricity Theft Detection using CNN-LSTM Deep Learning

Architecture
------------
  * SQLite (etd_xai.db) is the single source of truth
  * All prediction results are persisted to SQLite immediately
  * Dashboard reads only from SQLite — survives restarts
  * CNN-LSTM model is auto-loaded from disk on startup if registered

Start
-----
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
)
logger = logging.getLogger("etd_xai")

# ── Ensure directories exist ──────────────────────────────────────────────────
for d in ("uploads/model", "uploads/dataset", "uploads/exports"):
    Path(d).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan  (startup + shutdown)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═" * 60)
    logger.info("  ETD-XAI Enterprise  |  CNN-LSTM + SQLite  |  v2.0")
    logger.info("═" * 60)

    # 1. Initialise SQLite (creates tables if they don't exist)
    import database as db_module
    db_module.init_db()

    # 2. Auto-load CNN-LSTM model from last registered path
    from services.model_service import auto_load_on_startup
    model_ready = auto_load_on_startup()

    # 3. Report readiness
    has_data = db_module.has_any_upload()
    pred_count = db_module.get_prediction_count()

    if model_ready and has_data:
        logger.info(
            "  Ready — model loaded | SQLite has %d predictions | "
            "Dashboard will work immediately.",
            pred_count,
        )
    elif model_ready:
        logger.info("  Model loaded | No dataset in SQLite yet | Upload a CSV to predict.")
    else:
        logger.info("  Waiting for cnnlstm_final.keras upload.")

    logger.info("═" * 60)
    yield
    logger.info("Server shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ETD-XAI Enterprise",
    description=(
        "Electricity Theft Detection using CNN-LSTM Deep Learning. "
        "SQLite-backed — all predictions survive backend restarts."
    ),
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
from routers.api_unified import router as unified_router
from routers.upload      import router as upload_router
from routers.predict     import router as predict_router
from routers.dashboard   import router as dashboard_router
from routers.copilot     import router as copilot_router

app.include_router(unified_router)   # /api/dashboard  /api/customers  /api/load-model  /api/upload  /api/predict
app.include_router(upload_router)    # legacy: /api/upload/model  /api/upload/dataset
app.include_router(predict_router)   # legacy: /api/predict/manual
app.include_router(dashboard_router) # legacy: /api/dashboard/stats  /api/dashboard/charts
app.include_router(copilot_router)   # /api/copilot/ask


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    from services import model_service
    import database as db_module

    upload = db_module.get_dashboard_from_db()
    return {
        "status":         "ok",
        "model_loaded":   model_service.is_model_loaded(),
        "dataset_loaded": db_module.has_any_upload(),
        "prediction_count": db_module.get_prediction_count(),
        "dataset_name":   upload["filename"] if upload else "",
        "model_info":     model_service.get_model_info(),
        "data_source":    "SQLite — etd_xai.db",
    }


# ── Serve React SPA ───────────────────────────────────────────────────────────
STATIC_DIR = Path("../frontend/dist")
if STATIC_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(STATIC_DIR / "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        return FileResponse(str(STATIC_DIR / "index.html"))
