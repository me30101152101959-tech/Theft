"""
Dashboard Router  —  /api/dashboard/*  (legacy URLs, now SQLite-backed)
All reads come from SQLite.  No Python variables used as data source.
"""
from __future__ import annotations

import csv
import io
import json
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

import database as db
from services import model_service

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])
logger = logging.getLogger(__name__)


def _require_data():
    uid = db.get_latest_upload_id()
    if uid is None:
        raise HTTPException(400, "No dataset in SQLite. Upload a CSV first.")
    return uid


@router.get("/stats")
async def get_stats():
    """Dashboard KPIs — reads from dataset_uploads table in SQLite."""
    _require_data()
    upload = db.get_dashboard_from_db()
    return JSONResponse({
        "total_customers":     upload["total_rows"],
        "processed_customers": upload["total_rows"],
        "predicted_theft":     upload["theft_rows"],
        "predicted_normal":    upload["normal_rows"],
        "avg_confidence":      upload["avg_confidence"],
        "avg_risk_score":      upload["avg_risk"],
        "theft_rate":          upload["theft_rate"],
        "dataset_name":        upload["filename"],
        "upload_time":         upload["upload_time"],
        "has_flag":            bool(upload["has_flag"]),
        "accuracy":            upload["accuracy"],
        "precision":           upload["precision_val"],
        "recall":              upload["recall_val"],
        "f1_score":            upload["f1_score"],
        "roc_auc":             upload["roc_auc"],
        "data_source":         "SQLite — dataset_uploads table",
    })


@router.get("/charts")
async def get_charts():
    """Chart data — built from SQL aggregates on predictions table."""
    uid = _require_data()
    return JSONResponse(db.get_chart_data_from_db(uid))


@router.get("/customers")
async def get_customers(
    page:          int = Query(1,    ge=1),
    page_size:     int = Query(50,   ge=1, le=500),
    search:        str = Query(""),
    status_filter: str = Query(""),
    sort_by:       str = Query("risk_score"),
    sort_dir:      str = Query("desc"),
):
    """Paginated customer list — SELECT from predictions WHERE upload_id=?"""
    uid = _require_data()
    result = db.get_customers_from_db(
        upload_id=uid,
        page=page, page_size=page_size,
        search=search, status_filter=status_filter,
        sort_by=sort_by, sort_dir=sort_dir,
    )
    result["data_source"] = "SQLite — predictions table"
    return JSONResponse(result)


@router.get("/customer/{customer_id}")
async def get_customer(customer_id: str):
    uid = _require_data()
    rows = db.get_db().execute(
        "SELECT * FROM predictions WHERE upload_id=? AND customer_id=? LIMIT 1",
        (uid, customer_id),
    ).fetchall()
    if not rows:
        raise HTTPException(404, f"Customer '{customer_id}' not found in SQLite.")
    return JSONResponse(dict(rows[0]))


@router.get("/model-info")
async def get_model_info():
    return JSONResponse(model_service.get_model_info())


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/export/csv")
async def export_csv():
    uid = _require_data()
    rows = db.get_db().execute(
        "SELECT customer_id, prediction, probability, confidence, risk_score, status, flag "
        "FROM predictions WHERE upload_id=?",
        (uid,),
    ).fetchall()
    out = io.StringIO()
    fields = ["customer_id", "prediction", "probability", "confidence",
              "risk_score", "status", "flag"]
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(dict(r))
    out.seek(0)
    return StreamingResponse(
        io.BytesIO(out.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=etd_predictions.csv"},
    )


@router.get("/export/json")
async def export_json():
    uid = _require_data()
    rows = db.get_db().execute(
        "SELECT customer_id, prediction, probability, confidence, risk_score, status, flag "
        "FROM predictions WHERE upload_id=?",
        (uid,),
    ).fetchall()
    data = json.dumps({"customers": [dict(r) for r in rows]}, indent=2)
    return StreamingResponse(
        io.BytesIO(data.encode()),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=etd_predictions.json"},
    )


@router.get("/report/summary")
async def summary_report():
    _require_data()
    upload = db.get_dashboard_from_db()
    info   = model_service.get_model_info()

    lines = [
        "═" * 60,
        "  ETD-XAI Enterprise — Executive Summary",
        "═" * 60,
        f"  Data Source    : SQLite — etd_xai.db",
        f"  Model          : {info.get('model_name', 'N/A')}",
        f"  Architecture   : CNN-LSTM",
        f"  Parameters     : {info.get('total_params_fmt', 'N/A')}",
        f"  Dataset        : {upload['filename']}",
        f"  Upload Time    : {upload['upload_time']}",
        "─" * 60,
        f"  Total Customers  : {upload['total_rows']:,}",
        f"  Predicted Theft  : {upload['theft_rows']:,}",
        f"  Predicted Normal : {upload['normal_rows']:,}",
        f"  Theft Rate       : {(upload['theft_rate'] or 0)*100:.2f}%",
        f"  Avg Confidence   : {(upload['avg_confidence'] or 0)*100:.2f}%",
        f"  Avg Risk Score   : {upload['avg_risk'] or 0:.2f}/100",
    ]
    if upload.get("accuracy"):
        lines += [
            "─" * 60,
            "  EVALUATION METRICS (FLAG ground truth)",
            "─" * 60,
            f"  Accuracy   : {upload['accuracy']*100:.2f}%",
            f"  Precision  : {(upload['precision_val'] or 0)*100:.2f}%",
            f"  Recall     : {(upload['recall_val'] or 0)*100:.2f}%",
            f"  F1 Score   : {(upload['f1_score'] or 0)*100:.2f}%",
            f"  ROC-AUC    : {(upload['roc_auc'] or 0)*100:.2f}%",
        ]
    lines.append("═" * 60)
    return StreamingResponse(
        io.BytesIO("\n".join(lines).encode()),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=etd_summary_report.txt"},
    )
