"""
Report / export helpers: CSV, Excel, and PDF.
PDF uses reportlab if available; otherwise a clear message is returned.
"""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame, sheet="Predictions") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name=sheet)
    return buf.getvalue()


def pdf_report(title: str, model_info: dict, summary: dict,
               metrics: dict | None, df: pd.DataFrame | None = None) -> bytes | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle)
    except Exception:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    el = []

    el.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    el.append(Paragraph("Electricity Theft Detection using Explainable AI", styles["Italic"]))
    el.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    el.append(Spacer(1, 0.6 * cm))

    def table(title_txt, pairs):
        el.append(Paragraph(f"<b>{title_txt}</b>", styles["Heading2"]))
        data = [[str(k), str(v)] for k, v in pairs]
        t = Table(data, colWidths=[7 * cm, 9 * cm])
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ]))
        el.append(t)
        el.append(Spacer(1, 0.5 * cm))

    table("Model Information", [
        ("Active Model", model_info.get("model_name", "—")),
        ("Architecture", model_info.get("architecture", "—")),
        ("Input Shape", model_info.get("input_shape", "—")),
        ("Output Shape", model_info.get("output_shape", "—")),
        ("Parameters", model_info.get("total_params_fmt", "—")),
        ("TensorFlow", model_info.get("tf_version", "—")),
    ])

    table("Prediction Summary", [
        ("Total Customers", summary.get("total_rows", 0)),
        ("Normal", summary.get("normal_rows", 0)),
        ("Theft", summary.get("theft_rows", 0)),
        ("Theft Rate", f"{summary.get('theft_rate', 0) * 100:.2f}%"),
        ("Avg Risk Score", summary.get("avg_risk", 0)),
    ])

    if metrics:
        table("Evaluation Metrics", [
            ("Accuracy", f"{metrics.get('accuracy', 0):.4f}"),
            ("Precision", f"{metrics.get('precision_val', 0):.4f}"),
            ("Recall", f"{metrics.get('recall_val', 0):.4f}"),
            ("F1 Score", f"{metrics.get('f1_score', 0):.4f}"),
            ("ROC-AUC", f"{metrics.get('roc_auc') or 0:.4f}"),
        ])
        cm = metrics.get("confusion_matrix")
        if cm:
            el.append(Paragraph("<b>Confusion Matrix</b>", styles["Heading2"]))
            data = [["", "Pred Normal", "Pred Theft"],
                    ["Actual Normal", cm[0][0], cm[0][1]],
                    ["Actual Theft", cm[1][0], cm[1][1]]]
            t = Table(data, colWidths=[4 * cm] * 3)
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]))
            el.append(t)
            el.append(Spacer(1, 0.5 * cm))

    if df is not None and len(df):
        el.append(Paragraph("<b>Top 15 Highest-Risk Customers</b>", styles["Heading2"]))
        top = df.sort_values("risk_score", ascending=False).head(15)
        cols = [c for c in ["customer_id", "probability", "risk_score", "status"] if c in top.columns]
        data = [cols] + top[cols].round(4).astype(str).values.tolist()
        t = Table(data)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]))
        el.append(t)

    doc.build(el)
    return buf.getvalue()
