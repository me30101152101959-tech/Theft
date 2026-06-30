"""
SQLite persistence for the Streamlit app.
Stores dataset uploads, per-customer predictions, manual/batch predictions,
and user settings. Survives restarts; path configurable via DATABASE_PATH.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.environ.get("DATABASE_PATH", str(Path(__file__).resolve().parent.parent / "etd_xai.db")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS dataset_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    upload_time TEXT NOT NULL,
    total_rows INTEGER DEFAULT 0,
    theft_rows INTEGER DEFAULT 0,
    normal_rows INTEGER DEFAULT 0,
    avg_risk REAL,
    theft_rate REAL,
    has_flag INTEGER DEFAULT 0,
    threshold REAL DEFAULT 0.5,
    accuracy REAL, precision_val REAL, recall_val REAL, f1_score REAL, roc_auc REAL,
    confusion_matrix TEXT, n_readings INTEGER, strategy TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id INTEGER NOT NULL,
    customer_id TEXT NOT NULL,
    probability REAL NOT NULL,
    prediction INTEGER NOT NULL,
    confidence REAL NOT NULL,
    risk_score REAL NOT NULL,
    status TEXT NOT NULL,
    flag INTEGER,
    readings TEXT,
    predicted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_upload ON predictions(upload_id);
CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(status);

CREATE TABLE IF NOT EXISTS manual_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,
    probability REAL NOT NULL,
    prediction INTEGER NOT NULL,
    confidence REAL NOT NULL,
    risk_score REAL NOT NULL,
    status TEXT NOT NULL,
    readings TEXT NOT NULL,
    predicted_at TEXT NOT NULL,
    threshold REAL DEFAULT 0.5,
    model_name TEXT,
    source TEXT DEFAULT 'manual'
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as c:
        c.executescript(_DDL)


# ── settings ────────────────────────────────────────────────────────────────
def set_setting(key: str, value) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )


def get_setting(key: str, default=None):
    with _connect() as c:
        row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


# ── uploads ───────────────────────────────────────────────────────────────--
def save_upload(**kw) -> int:
    cm = kw.get("confusion_matrix")
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO dataset_uploads
               (filename,upload_time,total_rows,theft_rows,normal_rows,avg_risk,
                theft_rate,has_flag,threshold,accuracy,precision_val,recall_val,
                f1_score,roc_auc,confusion_matrix,n_readings,strategy)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (kw["filename"], kw.get("upload_time", datetime.utcnow().isoformat()),
             kw["total_rows"], kw["theft_rows"], kw["normal_rows"], kw.get("avg_risk"),
             kw.get("theft_rate"), int(kw.get("has_flag", False)), kw.get("threshold", 0.5),
             kw.get("accuracy"), kw.get("precision_val"), kw.get("recall_val"),
             kw.get("f1_score"), kw.get("roc_auc"),
             json.dumps(cm) if cm is not None else None,
             kw.get("n_readings"), kw.get("strategy")),
        )
        return cur.lastrowid


def save_predictions_bulk(upload_id: int, rows: list) -> None:
    records = [
        (upload_id, r["customer_id"], r["probability"], r["prediction"], r["confidence"],
         r["risk_score"], r["status"], r.get("flag"),
         json.dumps(r.get("readings", [])), r.get("predicted_at", datetime.utcnow().isoformat()))
        for r in rows
    ]
    with _conn() as c:
        c.executemany(
            """INSERT INTO predictions
               (upload_id,customer_id,probability,prediction,confidence,risk_score,
                status,flag,readings,predicted_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            records,
        )


def get_latest_upload_id() -> Optional[int]:
    with _connect() as c:
        row = c.execute("SELECT id FROM dataset_uploads ORDER BY id DESC LIMIT 1").fetchone()
    return row["id"] if row else None


def get_upload(upload_id: int) -> Optional[dict]:
    with _connect() as c:
        row = c.execute("SELECT * FROM dataset_uploads WHERE id=?", (upload_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    if d.get("confusion_matrix"):
        try:
            d["confusion_matrix"] = json.loads(d["confusion_matrix"])
        except Exception:
            d["confusion_matrix"] = None
    return d


def get_all_uploads() -> list:
    with _connect() as c:
        rows = c.execute("SELECT * FROM dataset_uploads ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def has_any_upload() -> bool:
    return get_latest_upload_id() is not None


def get_predictions_df(upload_id: int):
    import pandas as pd
    with _connect() as c:
        rows = c.execute(
            "SELECT customer_id,probability,prediction,confidence,risk_score,status,flag "
            "FROM predictions WHERE upload_id=? ORDER BY risk_score DESC", (upload_id,)
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


# ── manual predictions ────────────────────────────────────────────────────--
def save_manual(**kw) -> int:
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO manual_predictions
               (customer_id,probability,prediction,confidence,risk_score,status,
                readings,predicted_at,threshold,model_name,source)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (kw.get("customer_id"), kw["probability"], kw["prediction"], kw["confidence"],
             kw["risk_score"], kw["status"], json.dumps(kw.get("readings", [])),
             kw.get("predicted_at", datetime.utcnow().isoformat()),
             kw.get("threshold", 0.5), kw.get("model_name"), kw.get("source", "manual")),
        )
        return cur.lastrowid


def get_manual(limit: int = 200) -> list:
    with _connect() as c:
        rows = c.execute(
            "SELECT * FROM manual_predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["readings"] = json.loads(d["readings"]) if d["readings"] else []
        except Exception:
            d["readings"] = []
        out.append(d)
    return out


def counts() -> dict:
    with _connect() as c:
        p = c.execute("SELECT COUNT(*) AS x FROM predictions").fetchone()["x"]
        m = c.execute("SELECT COUNT(*) AS x FROM manual_predictions").fetchone()["x"]
    return {"predictions": p, "manual": m}
