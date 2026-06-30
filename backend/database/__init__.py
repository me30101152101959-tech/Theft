"""
ETD-XAI  —  SQLite Persistence Layer
======================================
Single source of truth for all predictions, model registry, and upload history.
After any server restart, dashboard and customer predictions are rebuilt from this DB.

Tables
------
  model_registry    — tracks uploaded CNN-LSTM models
  dataset_uploads   — one row per CSV upload (aggregate stats + metrics)
  predictions       — one row per customer per upload
  manual_predictions — one row per manual / batch-predict-store call
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("etd_xai.db")

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────
_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

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
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    filename       TEXT    NOT NULL,
    upload_time    TEXT    NOT NULL,
    total_rows     INTEGER NOT NULL DEFAULT 0,
    theft_rows     INTEGER NOT NULL DEFAULT 0,
    normal_rows    INTEGER NOT NULL DEFAULT 0,
    avg_confidence REAL,
    avg_risk       REAL,
    theft_rate     REAL,
    has_flag       INTEGER DEFAULT 0,
    threshold      REAL    DEFAULT 0.5,
    accuracy       REAL,
    precision_val  REAL,
    recall_val     REAL,
    f1_score       REAL,
    roc_auc        REAL,
    roc_fpr        TEXT,
    roc_tpr        TEXT,
    pr_precision   TEXT,
    pr_recall      TEXT,
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

CREATE INDEX IF NOT EXISTS idx_pred_upload  ON predictions(upload_id);
CREATE INDEX IF NOT EXISTS idx_pred_cust    ON predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_pred_status  ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_pred_risk    ON predictions(risk_score);

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
    model_name   TEXT,
    source       TEXT    DEFAULT 'manual'
);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Connection helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _conn():
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables. Safe to call multiple times."""
    with _conn() as conn:
        conn.executescript(_DDL)
    logger.info("SQLite DB initialised at %s", DB_PATH.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────
def register_model(
    model_name: str,
    model_path: str,
    loaded_at: str,
    input_shape: str = "",
    output_shape: str = "",
    total_params: int = 0,
    is_dual_input: bool = False,
    stat_input_size: int = 0,
) -> int:
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO model_registry
               (model_name,model_path,loaded_at,input_shape,output_shape,
                total_params,is_dual_input,stat_input_size)
               VALUES (?,?,?,?,?,?,?,?)""",
            (model_name, model_path, loaded_at, input_shape, output_shape,
             total_params, int(is_dual_input), stat_input_size),
        )
        return cur.lastrowid


def get_active_model() -> Optional[dict]:
    """Return the most-recently registered model (latest row)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM model_registry ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset uploads
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
    accuracy=None,
    precision_val=None,
    recall_val=None,
    f1_score=None,
    roc_auc=None,
    roc_fpr=None,
    roc_tpr=None,
    pr_precision=None,
    pr_recall=None,
    confusion_matrix=None,
) -> int:
    def _enc(x):
        return json.dumps(x) if x is not None else None

    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO dataset_uploads
               (filename,upload_time,total_rows,theft_rows,normal_rows,
                avg_confidence,avg_risk,theft_rate,has_flag,threshold,
                accuracy,precision_val,recall_val,f1_score,roc_auc,
                roc_fpr,roc_tpr,pr_precision,pr_recall,confusion_matrix)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (filename, upload_time, total_rows, theft_rows, normal_rows,
             avg_confidence, avg_risk, theft_rate, int(has_flag), threshold,
             accuracy, precision_val, recall_val, f1_score, roc_auc,
             _enc(roc_fpr), _enc(roc_tpr), _enc(pr_precision), _enc(pr_recall),
             _enc(confusion_matrix)),
        )
        upload_id = cur.lastrowid

    logger.info(
        "SQLite: saved upload id=%d  file=%s  rows=%d",
        upload_id, filename, total_rows,
    )
    return upload_id


def get_latest_upload_id() -> Optional[int]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT id FROM dataset_uploads ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row["id"] if row else None


def get_upload_summary(upload_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM dataset_uploads WHERE id=?", (upload_id,)
        ).fetchone()
    if not row:
        return None

    def _dec(x):
        return json.loads(x) if x else None

    d = dict(row)
    d["roc_fpr"]         = _dec(d.get("roc_fpr"))
    d["roc_tpr"]         = _dec(d.get("roc_tpr"))
    d["pr_precision"]    = _dec(d.get("pr_precision"))
    d["pr_recall"]       = _dec(d.get("pr_recall"))
    d["confusion_matrix"]= _dec(d.get("confusion_matrix"))
    return d


def has_any_upload() -> bool:
    with get_db() as conn:
        row = conn.execute("SELECT id FROM dataset_uploads LIMIT 1").fetchone()
    return row is not None


def get_all_uploads() -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,filename,upload_time,total_rows,theft_rows,normal_rows,theft_rate,threshold "
            "FROM dataset_uploads ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Predictions (bulk)
# ─────────────────────────────────────────────────────────────────────────────
def save_predictions_bulk(upload_id: int, rows: list[dict]) -> None:
    """
    Bulk-insert N prediction rows.  Each dict must have:
      customer_id, probability, prediction, confidence, risk_score,
      status, flag (or None), readings (list), predicted_at.
    """
    records = [
        (
            upload_id,
            r["customer_id"],
            r["probability"],
            r["prediction"],
            r["confidence"],
            r["risk_score"],
            r["status"],
            r.get("flag"),
            json.dumps(r.get("readings", [])),
            r["predicted_at"],
        )
        for r in rows
    ]
    with _conn() as conn:
        conn.executemany(
            """INSERT INTO predictions
               (upload_id,customer_id,probability,prediction,confidence,
                risk_score,status,flag,readings,predicted_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            records,
        )
    logger.info("SQLite: bulk-inserted %d predictions for upload_id=%d", len(records), upload_id)


def get_prediction_count(upload_id: Optional[int] = None) -> int:
    with get_db() as conn:
        if upload_id is not None:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM predictions WHERE upload_id=?",
                (upload_id,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()
    return row["c"] if row else 0


def get_customers_paginated(
    upload_id: int,
    page: int = 1,
    page_size: int = 50,
    search: str = "",
    status_filter: str = "",
    sort_by: str = "risk_score",
    sort_dir: str = "desc",
) -> dict:
    allowed_sort = {"risk_score", "probability", "confidence", "customer_id", "status", "predicted_at"}
    if sort_by not in allowed_sort:
        sort_by = "risk_score"
    order = "DESC" if sort_dir.lower() == "desc" else "ASC"

    conditions = ["upload_id = ?"]
    params: list = [upload_id]

    if search:
        conditions.append("customer_id LIKE ?")
        params.append(f"%{search}%")
    if status_filter in ("Theft", "Normal"):
        conditions.append("status = ?")
        params.append(status_filter)

    where = " AND ".join(conditions)

    with get_db() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM predictions WHERE {where}", params
        ).fetchone()["c"]

        offset = (page - 1) * page_size
        rows = conn.execute(
            f"""SELECT customer_id,probability,prediction,confidence,
                       risk_score,status,flag,readings,predicted_at
                FROM predictions
                WHERE {where}
                ORDER BY {sort_by} {order}
                LIMIT ? OFFSET ?""",
            params + [page_size, offset],
        ).fetchall()

    data = []
    for r in rows:
        d = dict(r)
        d["id"] = d.pop("customer_id")
        try:
            d["readings"] = json.loads(d["readings"]) if d["readings"] else []
        except Exception:
            d["readings"] = []
        data.append(d)

    return {
        "data": data,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
    }


def get_customer_by_id(upload_id: int, customer_id: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            """SELECT customer_id,probability,prediction,confidence,
                      risk_score,status,flag,readings,predicted_at
               FROM predictions
               WHERE upload_id=? AND customer_id=? LIMIT 1""",
            (upload_id, customer_id),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["id"] = d.pop("customer_id")
    try:
        d["readings"] = json.loads(d["readings"]) if d["readings"] else []
    except Exception:
        d["readings"] = []
    return d


def get_all_customers_for_export(upload_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT customer_id AS id,probability,prediction,
                      confidence,risk_score,status,flag,predicted_at
               FROM predictions WHERE upload_id=?
               ORDER BY risk_score DESC""",
            (upload_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def update_predictions_threshold(upload_id: int, threshold: float) -> dict:
    """Re-classify all predictions in an upload with a new threshold."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, probability FROM predictions WHERE upload_id=?",
            (upload_id,),
        ).fetchall()

        updates = []
        for r in rows:
            prob = r["probability"]
            pred = 1 if prob >= threshold else 0
            conf = prob if pred == 1 else (1.0 - prob)
            risk = round(prob * 100, 2)
            status = "Theft" if pred == 1 else "Normal"
            updates.append((pred, conf, risk, status, r["id"]))

        conn.executemany(
            "UPDATE predictions SET prediction=?,confidence=?,risk_score=?,status=? WHERE id=?",
            updates,
        )

        agg = conn.execute(
            """SELECT
                   SUM(CASE WHEN status='Theft'  THEN 1 ELSE 0 END) AS theft,
                   SUM(CASE WHEN status='Normal' THEN 1 ELSE 0 END) AS normal,
                   AVG(confidence) AS avg_conf,
                   AVG(risk_score) AS avg_risk,
                   COUNT(*) AS total
               FROM predictions WHERE upload_id=?""",
            (upload_id,),
        ).fetchone()

        conn.execute(
            """UPDATE dataset_uploads
               SET theft_rows=?,normal_rows=?,avg_confidence=?,
                   avg_risk=?,theft_rate=?,threshold=?
               WHERE id=?""",
            (
                agg["theft"], agg["normal"],
                round(float(agg["avg_conf"] or 0), 6),
                round(float(agg["avg_risk"] or 0), 6),
                round((agg["theft"] or 0) / max(agg["total"] or 1, 1), 6),
                threshold,
                upload_id,
            ),
        )

    return {"theft": agg["theft"], "normal": agg["normal"]}


def get_chart_data(upload_id: int) -> dict:
    """Build chart-ready data from SQL — no Python lists held in RAM."""
    with get_db() as conn:
        # Pie data
        pie = conn.execute(
            """SELECT status, COUNT(*) AS cnt
               FROM predictions WHERE upload_id=?
               GROUP BY status""",
            (upload_id,),
        ).fetchall()

        # Risk distribution (sample up to 5000 for scatter)
        scatter = conn.execute(
            """SELECT risk_score, confidence, status, customer_id
               FROM predictions WHERE upload_id=?
               ORDER BY RANDOM() LIMIT 5000""",
            (upload_id,),
        ).fetchall()

        # Top 10 high risk
        top10h = conn.execute(
            """SELECT customer_id, risk_score, probability
               FROM predictions WHERE upload_id=?
               ORDER BY risk_score DESC LIMIT 10""",
            (upload_id,),
        ).fetchall()

        # Top 10 low risk
        top10l = conn.execute(
            """SELECT customer_id, risk_score
               FROM predictions WHERE upload_id=?
               ORDER BY risk_score ASC LIMIT 10""",
            (upload_id,),
        ).fetchall()

        # Risk histogram buckets
        hist = conn.execute(
            """SELECT CAST(risk_score/5 AS INT)*5 AS bucket, COUNT(*) AS cnt
               FROM predictions WHERE upload_id=?
               GROUP BY bucket ORDER BY bucket""",
            (upload_id,),
        ).fetchall()

        # Upload row for roc/pr/confusion
        upload = conn.execute(
            "SELECT roc_fpr,roc_tpr,pr_precision,pr_recall,confusion_matrix,has_flag "
            "FROM dataset_uploads WHERE id=?",
            (upload_id,),
        ).fetchone()

    pie_data = {r["status"]: r["cnt"] for r in pie}

    def _short(cid: str) -> str:
        return (cid[:14] + "…") if len(cid) > 14 else cid

    def _dec(x):
        try:
            return json.loads(x) if x else None
        except Exception:
            return None

    return {
        "pie": {
            "labels": ["Normal", "Theft"],
            "values": [pie_data.get("Normal", 0), pie_data.get("Theft", 0)],
        },
        "risk_distribution": {
            "values": [r["risk_score"] for r in scatter],
            "labels": [r["status"] for r in scatter],
        },
        "scatter": {
            "risk":       [r["risk_score"]  for r in scatter],
            "confidence": [r["confidence"]  for r in scatter],
            "status":     [r["status"]      for r in scatter],
            "ids":        [r["customer_id"] for r in scatter],
        },
        "top10_high": {
            "ids":   [_short(r["customer_id"]) for r in top10h],
            "risks": [r["risk_score"]          for r in top10h],
            "probs": [r["probability"]         for r in top10h],
        },
        "top10_low": {
            "ids":   [_short(r["customer_id"]) for r in top10l],
            "risks": [r["risk_score"]          for r in top10l],
        },
        "histogram": {
            "buckets": [r["bucket"] for r in hist],
            "counts":  [r["cnt"]    for r in hist],
        },
        "confusion": _dec(upload["confusion_matrix"]) if upload else None,
        "roc": {
            "fpr": _dec(upload["roc_fpr"]),
            "tpr": _dec(upload["roc_tpr"]),
        } if upload else {},
        "pr": {
            "precision": _dec(upload["pr_precision"]),
            "recall":    _dec(upload["pr_recall"]),
        } if upload else {},
        "has_flag": bool(upload["has_flag"]) if upload else False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Manual predictions
# ─────────────────────────────────────────────────────────────────────────────
def save_manual_prediction(
    customer_id: Optional[str],
    probability: float,
    prediction: int,
    confidence: float,
    risk_score: float,
    status: str,
    readings: list,
    predicted_at: str,
    threshold: float = 0.5,
    model_name: Optional[str] = None,
    source: str = "manual",
) -> int:
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO manual_predictions
               (customer_id,probability,prediction,confidence,risk_score,
                status,readings,predicted_at,threshold,model_name,source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                customer_id, probability, prediction, confidence, risk_score,
                status, json.dumps(readings), predicted_at, threshold, model_name, source,
            ),
        )
        return cur.lastrowid


def get_manual_predictions(limit: int = 100, source: Optional[str] = None) -> list[dict]:
    with get_db() as conn:
        if source:
            rows = conn.execute(
                """SELECT * FROM manual_predictions WHERE source=?
                   ORDER BY id DESC LIMIT ?""",
                (source, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM manual_predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["readings"] = json.loads(d["readings"]) if d["readings"] else []
        except Exception:
            d["readings"] = []
        result.append(d)
    return result


def get_manual_prediction_count() -> int:
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM manual_predictions").fetchone()
    return row["c"] if row else 0
