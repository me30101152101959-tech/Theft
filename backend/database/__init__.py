"""
database  —  SQLite single source of truth
===========================================
All prediction results, upload metadata, evaluation metrics, and model
registration are stored here.  The dashboard NEVER reads Python variables.
After any backend restart the data is immediately available via SQL.

Schema
------
  model_registry     — one row per loaded model (most recent = active)
  dataset_uploads    — one row per CSV upload, includes aggregate stats
  predictions        — one row per customer, references dataset_uploads.id
  manual_predictions — one row per /api/predict call

Usage
-----
  import database as db
  db.init_db()
  summary = db.get_dashboard_from_db()
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path("etd_xai.db")
logger  = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set to sqlite3.Row."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Schema  (idempotent — safe to call on every startup)
# ─────────────────────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS model_registry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT    NOT NULL,
    model_path      TEXT    NOT NULL,
    loaded_at       TEXT    NOT NULL,
    input_shape     TEXT,
    output_shape    TEXT,
    total_params    INTEGER,
    is_dual_input   INTEGER DEFAULT 0,
    stat_input_size INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS dataset_uploads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT    NOT NULL,
    upload_time     TEXT    NOT NULL,
    total_rows      INTEGER NOT NULL DEFAULT 0,
    theft_rows      INTEGER NOT NULL DEFAULT 0,
    normal_rows     INTEGER NOT NULL DEFAULT 0,
    avg_confidence  REAL,
    avg_risk        REAL,
    theft_rate      REAL,
    has_flag        INTEGER DEFAULT 0,
    threshold       REAL    DEFAULT 0.5,
    accuracy        REAL,
    precision_val   REAL,
    recall_val      REAL,
    f1_score        REAL,
    roc_auc         REAL,
    roc_fpr         TEXT,
    roc_tpr         TEXT,
    pr_precision    TEXT,
    pr_recall       TEXT,
    confusion_matrix TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id    INTEGER NOT NULL,
    customer_id  TEXT    NOT NULL,
    probability  REAL    NOT NULL,
    prediction   INTEGER NOT NULL,
    confidence   REAL    NOT NULL,
    risk_score   REAL    NOT NULL,
    status       TEXT    NOT NULL,
    flag         INTEGER,
    readings     TEXT,
    predicted_at TEXT    NOT NULL,
    FOREIGN KEY (upload_id) REFERENCES dataset_uploads(id)
);

CREATE INDEX IF NOT EXISTS idx_pred_upload ON predictions(upload_id);
CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_pred_cid    ON predictions(customer_id);

CREATE TABLE IF NOT EXISTS manual_predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id  TEXT,
    probability  REAL    NOT NULL,
    prediction   INTEGER NOT NULL,
    confidence   REAL    NOT NULL,
    risk_score   REAL    NOT NULL,
    status       TEXT    NOT NULL,
    readings     TEXT    NOT NULL,
    predicted_at TEXT    NOT NULL,
    threshold    REAL    DEFAULT 0.5,
    model_name   TEXT
);
"""


def init_db() -> None:
    """Create all tables.  Safe to call multiple times."""
    with get_db() as conn:
        conn.executescript(_DDL)
    logger.info("SQLite ready: %s", DB_PATH.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────
def register_model(
    model_name: str,
    model_path: str,
    loaded_at: str,
    input_shape: str,
    output_shape: str,
    total_params: int,
    is_dual_input: bool,
    stat_input_size: int,
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO model_registry
              (model_name, model_path, loaded_at, input_shape, output_shape,
               total_params, is_dual_input, stat_input_size)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (model_name, model_path, loaded_at,
             input_shape, output_shape,
             total_params, int(is_dual_input), stat_input_size),
        )
        return cur.lastrowid


def get_active_model() -> Optional[dict]:
    """Most-recently registered model, or None."""
    row = get_db().execute(
        "SELECT * FROM model_registry ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Upload + predictions
# ─────────────────────────────────────────────────────────────────────────────
def save_upload(
    filename: str,
    upload_time: str,
    total_rows: int,
    theft_rows: int,
    normal_rows: int,
    avg_confidence: float,
    avg_risk: float,
    theft_rate: float,
    has_flag: bool,
    threshold: float,
    accuracy: Optional[float],
    precision_val: Optional[float],
    recall_val: Optional[float],
    f1_score: Optional[float],
    roc_auc: Optional[float],
    roc_fpr: Optional[list],
    roc_tpr: Optional[list],
    pr_precision: Optional[list],
    pr_recall: Optional[list],
    confusion_matrix: Optional[list],
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO dataset_uploads
              (filename, upload_time, total_rows, theft_rows, normal_rows,
               avg_confidence, avg_risk, theft_rate, has_flag, threshold,
               accuracy, precision_val, recall_val, f1_score, roc_auc,
               roc_fpr, roc_tpr, pr_precision, pr_recall, confusion_matrix)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                filename, upload_time, total_rows, theft_rows, normal_rows,
                avg_confidence, avg_risk, theft_rate, int(has_flag), threshold,
                accuracy, precision_val, recall_val, f1_score, roc_auc,
                json.dumps(roc_fpr)         if roc_fpr         else None,
                json.dumps(roc_tpr)         if roc_tpr         else None,
                json.dumps(pr_precision)    if pr_precision     else None,
                json.dumps(pr_recall)       if pr_recall        else None,
                json.dumps(confusion_matrix)if confusion_matrix else None,
            ),
        )
        return cur.lastrowid


def save_predictions_bulk(upload_id: int, rows: list) -> None:
    """Bulk-insert prediction rows into SQLite."""
    with get_db() as conn:
        conn.executemany(
            """
            INSERT INTO predictions
              (upload_id, customer_id, probability, prediction, confidence,
               risk_score, status, flag, readings, predicted_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    upload_id,
                    r["customer_id"],
                    r["probability"],
                    r["prediction"],
                    r["confidence"],
                    r["risk_score"],
                    r["status"],
                    r.get("flag"),
                    json.dumps(r["readings"]) if r.get("readings") else None,
                    r["predicted_at"],
                )
                for r in rows
            ],
        )


def save_manual_prediction(
    customer_id: str,
    probability: float,
    prediction: int,
    confidence: float,
    risk_score: float,
    status: str,
    readings: list,
    predicted_at: str,
    threshold: float,
    model_name: str,
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO manual_predictions
              (customer_id, probability, prediction, confidence, risk_score,
               status, readings, predicted_at, threshold, model_name)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                customer_id, probability, prediction, confidence, risk_score,
                status, json.dumps(readings), predicted_at, threshold, model_name,
            ),
        )
        return cur.lastrowid


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard helpers  (ALL reads go through here — never Python variables)
# ─────────────────────────────────────────────────────────────────────────────
def get_latest_upload_id() -> Optional[int]:
    row = get_db().execute(
        "SELECT id FROM dataset_uploads ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["id"] if row else None


def get_upload_summary(upload_id: int) -> Optional[dict]:
    row = get_db().execute(
        "SELECT * FROM dataset_uploads WHERE id = ?", (upload_id,)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    for key in ("roc_fpr", "roc_tpr", "pr_precision", "pr_recall", "confusion_matrix"):
        if d.get(key):
            d[key] = json.loads(d[key])
    return d


def get_dashboard_from_db() -> Optional[dict]:
    """Complete dashboard row from SQLite, or None when nothing uploaded."""
    uid = get_latest_upload_id()
    return get_upload_summary(uid) if uid else None


def get_customers_from_db(
    upload_id: int,
    page: int = 1,
    page_size: int = 50,
    search: str = "",
    status_filter: str = "",
    sort_by: str = "risk_score",
    sort_dir: str = "desc",
) -> dict:
    allowed = {"risk_score", "probability", "confidence", "customer_id", "status", "prediction"}
    if sort_by not in allowed:
        sort_by = "risk_score"
    order = "DESC" if sort_dir == "desc" else "ASC"

    where_parts = ["upload_id = ?"]
    params: list = [upload_id]
    if search:
        where_parts.append("customer_id LIKE ?")
        params.append(f"%{search}%")
    if status_filter in ("Theft", "Normal"):
        where_parts.append("status = ?")
        params.append(status_filter)

    where_sql = " AND ".join(where_parts)

    count_row = get_db().execute(
        f"SELECT COUNT(*) AS n FROM predictions WHERE {where_sql}", params
    ).fetchone()
    total = count_row["n"] if count_row else 0

    offset = (page - 1) * page_size
    rows = get_db().execute(
        f"""
        SELECT customer_id, probability, prediction, confidence,
               risk_score, status, flag
        FROM predictions
        WHERE {where_sql}
        ORDER BY {sort_by} {order}
        LIMIT ? OFFSET ?
        """,
        params + [page_size, offset],
    ).fetchall()

    return {
        "data":        [dict(r) for r in rows],
        "total":       total,
        "page":        page,
        "page_size":   page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
    }


def get_prediction_count() -> int:
    row = get_db().execute("SELECT COUNT(*) AS n FROM predictions").fetchone()
    return row["n"] if row else 0


def has_any_upload() -> bool:
    row = get_db().execute("SELECT id FROM dataset_uploads LIMIT 1").fetchone()
    return row is not None


def get_chart_data_from_db(upload_id: int) -> dict:
    """All chart data from SQL queries — no Python lists held in RAM."""
    db = get_db()

    scatter_rows = db.execute(
        """
        SELECT risk_score, confidence, status, customer_id
        FROM predictions WHERE upload_id = ?
        ORDER BY id LIMIT 5000
        """,
        (upload_id,),
    ).fetchall()

    risks    = [r["risk_score"]  for r in scatter_rows]
    confs    = [r["confidence"]  for r in scatter_rows]
    statuses = [r["status"]      for r in scatter_rows]
    cids     = [r["customer_id"] for r in scatter_rows]

    pie_row = db.execute(
        """
        SELECT
          SUM(CASE WHEN status='Normal' THEN 1 ELSE 0 END) AS normal_count,
          SUM(CASE WHEN status='Theft'  THEN 1 ELSE 0 END) AS theft_count
        FROM predictions WHERE upload_id = ?
        """,
        (upload_id,),
    ).fetchone()

    top10_high = db.execute(
        "SELECT customer_id, risk_score, probability FROM predictions "
        "WHERE upload_id=? ORDER BY risk_score DESC LIMIT 10",
        (upload_id,),
    ).fetchall()

    top10_low = db.execute(
        "SELECT customer_id, risk_score FROM predictions "
        "WHERE upload_id=? ORDER BY risk_score ASC LIMIT 10",
        (upload_id,),
    ).fetchall()

    summary = get_upload_summary(upload_id) or {}

    def _trim(cid: str) -> str:
        return (cid[:12] + "…") if len(cid) > 12 else cid

    return {
        "risk_distribution": {"values": risks, "labels": statuses},
        "pie": {
            "labels": ["Normal", "Theft"],
            "values": [pie_row["normal_count"] or 0, pie_row["theft_count"] or 0],
        },
        "top10_high": {
            "ids":   [_trim(r["customer_id"]) for r in top10_high],
            "risks": [r["risk_score"]  for r in top10_high],
            "probs": [r["probability"] for r in top10_high],
        },
        "top10_low": {
            "ids":   [_trim(r["customer_id"]) for r in top10_low],
            "risks": [r["risk_score"]  for r in top10_low],
        },
        "scatter": {
            "risk":       risks,
            "confidence": confs,
            "status":     statuses,
            "ids":        cids,
        },
        "confusion": summary.get("confusion_matrix") or [],
        "roc": {
            "fpr": summary.get("roc_fpr") or [],
            "tpr": summary.get("roc_tpr") or [],
        },
        "pr": {
            "precision": summary.get("pr_precision") or [],
            "recall":    summary.get("pr_recall")    or [],
        },
    }
