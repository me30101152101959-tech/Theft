"""Dashboard Router — /api/dashboard/*"""
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import io, csv, json
from services import dataset_service, model_service

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/stats")
async def get_stats():
    if not dataset_service.is_dataset_loaded():
        raise HTTPException(400, "No dataset loaded.")
    return JSONResponse(dataset_service.get_dashboard_stats())


@router.get("/charts")
async def get_charts():
    if not dataset_service.is_dataset_loaded():
        raise HTTPException(400, "No dataset loaded.")
    return JSONResponse(dataset_service.get_chart_data())


@router.get("/customers")
async def get_customers(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    search: str = Query(""),
    status_filter: str = Query(""),
    sort_by: str = Query("risk_score"),
    sort_dir: str = Query("desc"),
):
    if not dataset_service.is_dataset_loaded():
        raise HTTPException(400, "No dataset loaded.")
    result = dataset_service.get_customers_paginated(
        page=page,
        page_size=page_size,
        search=search,
        status_filter=status_filter,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )
    return JSONResponse(result)


@router.get("/customer/{customer_id}")
async def get_customer(customer_id: str):
    c = dataset_service.get_customer_by_id(customer_id)
    if not c:
        raise HTTPException(404, f"Customer '{customer_id}' not found.")
    return JSONResponse(c)


@router.get("/model-info")
async def get_model_info():
    return JSONResponse(model_service.get_model_info())


# ─────────────────────────────────────────────
# Export endpoints
# ─────────────────────────────────────────────
@router.get("/export/csv")
async def export_csv():
    if not dataset_service.is_dataset_loaded():
        raise HTTPException(400, "No dataset loaded.")

    customers = dataset_service.get_all_customers_for_export()
    output = io.StringIO()
    fieldnames = ["id", "prediction", "probability", "confidence",
                  "risk_score", "status", "flag"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(customers)

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=etd_predictions.csv"},
    )


@router.get("/export/json")
async def export_json():
    if not dataset_service.is_dataset_loaded():
        raise HTTPException(400, "No dataset loaded.")
    customers = dataset_service.get_all_customers_for_export()
    data = json.dumps({"customers": customers}, indent=2)
    return StreamingResponse(
        io.BytesIO(data.encode()),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=etd_predictions.json"},
    )


@router.get("/report/summary")
async def summary_report():
    """Generate a text executive summary."""
    stats = dataset_service.get_dashboard_stats()
    if not stats:
        raise HTTPException(400, "No data available.")

    model_info = model_service.get_model_info()
    lines = [
        "═" * 60,
        "  ETD-XAI Enterprise — Executive Summary",
        "═" * 60,
        f"  Model          : {model_info.get('model_name', 'N/A')}",
        f"  Architecture   : CNN-LSTM",
        f"  Parameters     : {model_info.get('total_params_fmt', 'N/A')}",
        f"  Dataset        : {stats.get('dataset_name', 'N/A')}",
        f"  Upload Time    : {stats.get('upload_time', 'N/A')}",
        "─" * 60,
        "  PREDICTION RESULTS",
        "─" * 60,
        f"  Total Customers    : {stats.get('total_customers', 0):,}",
        f"  Predicted Theft    : {stats.get('predicted_theft', 0):,}",
        f"  Predicted Normal   : {stats.get('predicted_normal', 0):,}",
        f"  Theft Rate         : {stats.get('theft_rate', 0)*100:.2f}%",
        f"  Avg Confidence     : {stats.get('avg_confidence', 0)*100:.2f}%",
        f"  Avg Risk Score     : {stats.get('avg_risk_score', 0):.2f}/100",
    ]

    if stats.get("accuracy"):
        lines += [
            "─" * 60,
            "  EVALUATION METRICS (vs Ground Truth FLAG)",
            "─" * 60,
            f"  Accuracy   : {stats.get('accuracy', 0)*100:.2f}%",
            f"  Precision  : {stats.get('precision', 0)*100:.2f}%",
            f"  Recall     : {stats.get('recall', 0)*100:.2f}%",
            f"  F1 Score   : {stats.get('f1_score', 0)*100:.2f}%",
            f"  ROC-AUC    : {stats.get('roc_auc', 0)*100:.2f}%",
        ]

    lines.append("═" * 60)
    text = "\n".join(lines)

    return StreamingResponse(
        io.BytesIO(text.encode()),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=etd_summary_report.txt"},
    )
