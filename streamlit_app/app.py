"""
ETD-XAI Enterprise v2.0  —  Streamlit Application
==================================================
Electricity Theft Detection using Explainable Artificial Intelligence.

Single-file multipage Streamlit app. Predictions come ONLY from the active
CNN-LSTM model (cnnlstm_final.keras) via real tensorflow.keras model.predict().

Run locally:   streamlit run app.py
Deploy free:   push to GitHub → share.streamlit.io (main file: streamlit_app/app.py)
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import engine, db, data, explain, copilot, reports
from core.preprocessing import STRATEGIES, STRATEGY_LABELS

# ─────────────────────────────────────────────────────────────────────────────
# Page config + boot
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ETD-XAI Enterprise",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

db.init_db()


@st.cache_resource(show_spinner="Loading CNN-LSTM model…")
def _boot_model():
    """Load the bundled model once per server process."""
    ok = engine.auto_load_default()
    return engine.get_model_info() if ok else {"loaded": False}


_boot_model()
# engine.state is process-global; cache_resource guarantees it's loaded once.
if not engine.is_model_loaded():
    engine.auto_load_default()

if "theme" not in st.session_state:
    st.session_state.theme = db.get_setting("theme", "dark")
if "threshold" not in st.session_state:
    st.session_state.threshold = float(db.get_setting("threshold", 0.5))
if "strategy" not in st.session_state:
    st.session_state.strategy = db.get_setting("strategy", "last_n")
if "chat" not in st.session_state:
    st.session_state.chat = []


# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    dark = st.session_state.theme == "dark"
    if dark:
        bg, card, text, sub, border = "#0f172a", "#1e293b", "#f1f5f9", "#94a3b8", "#334155"
    else:
        bg, card, text, sub, border = "#f8fafc", "#ffffff", "#0f172a", "#64748b", "#e2e8f0"
    st.markdown(f"""
    <style>
      .stApp {{ background:{bg}; color:{text}; }}
      section[data-testid="stSidebar"] {{ background:{card}; border-right:1px solid {border}; }}
      .kpi {{ background:{card}; border:1px solid {border}; border-radius:14px;
              padding:18px 20px; box-shadow:0 2px 10px rgba(0,0,0,.08); }}
      .kpi .label {{ color:{sub}; font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; }}
      .kpi .value {{ color:{text}; font-size:2rem; font-weight:700; margin-top:4px; }}
      .kpi .delta {{ font-size:.8rem; margin-top:2px; }}
      .badge {{ display:inline-block; padding:6px 16px; border-radius:999px;
                font-weight:700; font-size:1rem; }}
      .badge-theft {{ background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5; }}
      .badge-normal {{ background:#dcfce7; color:#15803d; border:1px solid #86efac; }}
      .hero {{ background:linear-gradient(135deg,#1e3a8a 0%,#7c3aed 100%);
               border-radius:16px; padding:26px 30px; color:#fff; margin-bottom:18px; }}
      .hero h1 {{ margin:0; font-size:1.8rem; }}
      .hero p {{ margin:4px 0 0; opacity:.9; }}
      .pill {{ background:{card}; border:1px solid {border}; border-radius:8px;
               padding:4px 10px; font-size:.8rem; color:{sub}; }}
    </style>
    """, unsafe_allow_html=True)


inject_css()


def kpi(label, value, delta="", color="#3b82f6"):
    d = f'<div class="delta" style="color:{color}">{delta}</div>' if delta else ""
    st.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                f'<div class="value">{value}</div>{d}</div>', unsafe_allow_html=True)


def status_badge(status: str) -> str:
    cls = "badge-theft" if status == "Theft" else "badge-normal"
    icon = "🔴" if status == "Theft" else "🟢"
    return f'<span class="badge {cls}">{icon} {status}</span>'


PLOT_TMPL = "plotly_dark" if st.session_state.theme == "dark" else "plotly_white"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ETD-XAI")
    st.caption("Enterprise v2.0 · CNN-LSTM")

    info = engine.get_model_info()
    if info.get("loaded"):
        st.success(f"Model: **{info['model_name']}**", icon="✅")
        st.markdown(
            f"<span class='pill'>Input {info['input_shape']}</span> "
            f"<span class='pill'>{info['total_params_fmt']} params</span>",
            unsafe_allow_html=True,
        )
    else:
        st.error("No model loaded", icon="⚠️")

    PAGES = [
        "📊 Dashboard", "🔮 Manual Prediction", "📦 Batch Prediction",
        "🗂️ Dataset Manager", "📜 History", "📑 Reports",
        "🤖 AI Copilot", "⚙️ Settings",
    ]
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.divider()
    tcol1, tcol2 = st.columns(2)
    if tcol1.button("🌙 Dark", use_container_width=True):
        st.session_state.theme = "dark"; db.set_setting("theme", "dark"); st.rerun()
    if tcol2.button("☀️ Light", use_container_width=True):
        st.session_state.theme = "light"; db.set_setting("theme", "light"); st.rerun()

    c = db.counts()
    st.caption(f"SQLite · {c['predictions']} preds · {c['manual']} manual")


def require_model() -> bool:
    if not engine.is_model_loaded():
        st.error(engine.NO_MODEL_MSG, icon="🚫")
        st.info("Go to **⚙️ Settings** to upload a `.keras` model, or restart to load the bundled one.")
        return False
    return True


def strategy_selector(key: str):
    labels = [STRATEGY_LABELS[s] for s in STRATEGIES]
    idx = STRATEGIES.index(st.session_state.strategy) if st.session_state.strategy in STRATEGIES else 0
    chosen = st.selectbox("Length-mapping strategy", labels, index=idx, key=key,
                          help="Used when the uploaded sequence length differs from the model's expected length.")
    sel = STRATEGIES[labels.index(chosen)]
    st.session_state.strategy = sel
    db.set_setting("strategy", sel)
    return sel


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ═════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown('<div class="hero"><h1>📊 Executive Dashboard</h1>'
                '<p>Electricity Theft Detection — CNN-LSTM Explainable AI</p></div>',
                unsafe_allow_html=True)

    info = engine.get_model_info()
    uid = db.get_latest_upload_id()
    up = db.get_upload(uid) if uid else None

    # System status row
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Model", "Active ✅" if info.get("loaded") else "None", info.get("model_name", ""))
    with c2: kpi("Engine", "TF / Keras", f"v{info.get('tf_version','—')}" if info.get("loaded") else "")
    with c3: kpi("Compute", "GPU" if _has_gpu() else "CPU", "")
    with c4: kpi("Database", "SQLite ✅", f"{db.counts()['predictions']} rows")

    if not up:
        st.info("No dataset processed yet. Go to **📦 Batch Prediction** or **🗂️ Dataset Manager** to load one.", icon="📥")
        return

    st.markdown("### Prediction Overview")
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Total Customers", f"{up['total_rows']:,}")
    with k2: kpi("Normal", f"{up['normal_rows']:,}", "Class 0", "#15803d")
    with k3: kpi("Theft", f"{up['theft_rows']:,}", "Class 1", "#b91c1c")
    with k4: kpi("Theft Rate", f"{(up['theft_rate'] or 0)*100:.1f}%", "", "#f59e0b")

    if up.get("accuracy") is not None:
        st.markdown("### Evaluation Metrics (vs ground-truth FLAG)")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: kpi("Accuracy", f"{up['accuracy']:.3f}")
        with m2: kpi("Precision", f"{up['precision_val']:.3f}")
        with m3: kpi("Recall", f"{up['recall_val']:.3f}")
        with m4: kpi("F1 Score", f"{up['f1_score']:.3f}")
        with m5: kpi("ROC-AUC", f"{up['roc_auc']:.3f}" if up.get("roc_auc") else "—")

    df = db.get_predictions_df(uid)
    if df.empty:
        return

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### Prediction Distribution")
        pie = df["status"].value_counts().reindex(["Normal", "Theft"]).fillna(0)
        fig = go.Figure(go.Pie(labels=["Normal", "Theft"], values=pie.values, hole=.55,
                               marker_colors=["#22c55e", "#ef4444"]))
        fig.update_layout(template=PLOT_TMPL, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        st.markdown("##### Risk Score Histogram")
        fig = px.histogram(df, x="risk_score", nbins=25, color="status",
                           color_discrete_map={"Normal": "#22c55e", "Theft": "#ef4444"})
        fig.update_layout(template=PLOT_TMPL, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        st.markdown("##### Probability Distribution")
        fig = px.histogram(df, x="probability", nbins=30, color="status",
                           color_discrete_map={"Normal": "#22c55e", "Theft": "#ef4444"})
        fig.update_layout(template=PLOT_TMPL, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with g4:
        cm = up.get("confusion_matrix")
        if cm:
            st.markdown("##### Confusion Matrix")
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            x=["Pred Normal", "Pred Theft"], y=["Actual Normal", "Actual Theft"])
            fig.update_layout(template=PLOT_TMPL, height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("##### Top 10 Highest-Risk Customers")
            top = df.head(10)[["customer_id", "risk_score", "status"]]
            st.dataframe(top, use_container_width=True, hide_index=True)

    st.markdown("##### Recent Predictions")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)


def _has_gpu() -> bool:
    try:
        return len(engine._tf().config.list_physical_devices("GPU")) > 0
    except Exception:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Manual Prediction
# ═════════════════════════════════════════════════════════════════════════════
def page_manual():
    st.markdown('<div class="hero"><h1>🔮 Manual Prediction</h1>'
                '<p>Enter a single customer\'s readings and get an instant CNN-LSTM verdict.</p></div>',
                unsafe_allow_html=True)
    if not require_model():
        return

    info = engine.get_model_info()
    T = info.get("seq_len_expected")
    default_len = T or 26

    left, right = st.columns([1, 1])
    with left:
        cust_id = st.text_input("Customer ID", value=f"CUST_{datetime.utcnow().strftime('%H%M%S')}")
        n = st.number_input("Number of readings", min_value=2, max_value=2000,
                            value=int(default_len), step=1,
                            help=f"Model expects {T if T else 'variable'} readings; "
                                 f"any length is auto-mapped via the chosen strategy.")
        strat = strategy_selector("manual_strat")
        threshold = st.slider("Decision threshold", 0.0, 1.0, st.session_state.threshold, 0.01)

        st.caption("Paste comma/space-separated readings, or use a demo profile:")
        d1, d2 = st.columns(2)
        if d1.button("🟢 Demo: Normal", use_container_width=True):
            st.session_state.manual_text = ", ".join(
                str(int(v)) for v in (2000 + 400 * np.sin(np.linspace(0, 6, int(n))) +
                                      np.random.randint(-80, 80, int(n))))
        if d2.button("🔴 Demo: Theft", use_container_width=True):
            arr = 2200 + np.random.randint(-50, 50, int(n))
            arr[int(n) // 2:] = np.random.randint(0, 60, int(n) - int(n) // 2)  # sudden drop to ~0
            st.session_state.manual_text = ", ".join(str(int(v)) for v in arr)

        text = st.text_area("Readings", value=st.session_state.get("manual_text", ""),
                            height=120, key="manual_text_area",
                            placeholder="2401, 2500, 2674, 2432, ...")
        go_btn = st.button("⚡ Predict", type="primary", use_container_width=True)

    if go_btn:
        raw = _parse_readings(text)
        if raw is None or len(raw) < 2:
            right.error("Please enter at least 2 numeric readings.")
            return
        with right:
            with st.spinner("Running model.predict()…"):
                res = engine.predict_one(raw, strategy=strat, threshold=threshold)
            db.save_manual(customer_id=cust_id, probability=res["probability"],
                           prediction=res["prediction"], confidence=res["confidence"],
                           risk_score=res["risk_score"], status=res["status"],
                           readings=list(map(float, raw)), threshold=threshold,
                           model_name=res["model_name"], source="manual")
            _render_single_result(cust_id, raw, res)


def _parse_readings(text: str):
    if not text or not text.strip():
        return None
    import re
    parts = re.split(r"[,\s;]+", text.strip())
    try:
        return [float(p) for p in parts if p != ""]
    except ValueError:
        return None


def _render_single_result(cust_id, raw, res):
    st.markdown(f"### Result for `{cust_id}`")
    st.markdown(status_badge(res["status"]), unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a: kpi("Probability", f"{res['probability']*100:.1f}%")
    with b: kpi("Confidence", f"{res['confidence']*100:.1f}%")
    with c: kpi("Risk", f"{res['risk_score']:.0f}/100",
                res["risk_level"], {"High": "#b91c1c", "Medium": "#f59e0b", "Low": "#15803d"}[res["risk_level"]])
    st.caption(f"Model **{res['model_name']}** · uploaded {res['uploaded_len']} → model "
               f"{res['model_len']} · strategy `{res['strategy_used']}`")

    fig = go.Figure(go.Scatter(y=raw, mode="lines+markers",
                               line=dict(color="#3b82f6"), name="kWh"))
    fig.update_layout(template=PLOT_TMPL, height=240, margin=dict(t=20, b=10),
                      title="Consumption Sequence", yaxis_title="kWh")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🧠 Explainable AI", expanded=True):
        ig = explain.integrated_gradients(raw)
        if ig:
            imp = ig["timestep_importance"]
            fig = go.Figure(go.Bar(y=imp, marker_color="#7c3aed"))
            fig.update_layout(template=PLOT_TMPL, height=220, margin=dict(t=20, b=10),
                              title="Per-timestep importance (integrated gradients)")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Risk factors:**")
        for name, val, dir_ in explain.risk_factors(raw):
            st.markdown(f"- {name}  ·  `{val:.3f}`  ·  {dir_}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Batch Prediction
# ═════════════════════════════════════════════════════════════════════════════
def page_batch():
    st.markdown('<div class="hero"><h1>📦 Batch Prediction</h1>'
                '<p>Upload a CSV / Excel dataset, validate it, and score every customer.</p></div>',
                unsafe_allow_html=True)
    if not require_model():
        return

    src = st.radio("Data source", ["Upload file", "Use bundled sample dataset"], horizontal=True)
    file = None
    if src == "Upload file":
        file = st.file_uploader("CSV or Excel (.csv / .xlsx)", type=["csv", "xlsx", "xls"])
    else:
        sample = Path(__file__).parent / "assets" / "sample_dataset.csv"
        if sample.exists():
            file = str(sample)
            st.caption(f"Using bundled sample: `{sample.name}`")

    if not file:
        return

    try:
        df = data.read_table(file)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        return

    info = data.inspect(df)
    col = st.columns(4)
    col[0].metric("Rows", f"{info['n_rows']:,}")
    col[1].metric("Reading columns", info["n_readings"])
    col[2].metric("ID column", info["id_col"] or "auto")
    col[3].metric("Ground-truth FLAG", "yes ✅" if info["has_flag"] else "no")

    comp = engine.check_compatibility(info["n_readings"])
    (st.success if comp["compatible"] else st.error)(comp["reason"], icon="ℹ️")

    strat = strategy_selector("batch_strat")
    threshold = st.slider("Decision threshold", 0.0, 1.0, st.session_state.threshold, 0.01, key="batch_thr")

    st.dataframe(df.head(8), use_container_width=True)

    c1, c2 = st.columns(2)
    run = c1.button("⚡ Run Predictions", type="primary", use_container_width=True)
    save = c2.checkbox("Save results to database (Dashboard + History)", value=True)

    if not run:
        return
    if info["n_readings"] < 2:
        st.error("No usable reading columns detected (need ≥ 2 numeric columns).")
        return

    prog = st.progress(0, "Preprocessing…")
    try:
        prog.progress(30, "Running model.predict()…")
        result = data.run_batch(df, info, engine, strategy=strat, threshold=threshold)
        prog.progress(80, "Aggregating…")
    except Exception as exc:
        prog.empty()
        st.error(f"Prediction failed: {exc}")
        return

    if save:
        fname = getattr(file, "name", Path(str(file)).name)
        uid = db.save_upload(
            filename=fname, total_rows=result["total_rows"], theft_rows=result["theft_rows"],
            normal_rows=result["normal_rows"], avg_risk=result["avg_risk"],
            theft_rate=result["theft_rate"], has_flag=result["has_flag"], threshold=threshold,
            n_readings=result["n_readings"], strategy=strat,
            **(result["metrics"] or {}),
        )
        db.save_predictions_bulk(uid, result["rows"])
    prog.progress(100, "Done")
    prog.empty()

    st.success(f"Scored {result['total_rows']:,} customers — "
               f"{result['theft_rows']:,} theft / {result['normal_rows']:,} normal.", icon="✅")

    rdf = pd.DataFrame([{k: r[k] for k in
                        ("customer_id", "probability", "prediction", "confidence", "risk_score", "status")}
                       for r in result["rows"]])
    st.session_state.last_batch = rdf

    m1, m2, m3 = st.columns(3)
    m1.metric("Theft detected", f"{result['theft_rows']:,}")
    m2.metric("Theft rate", f"{result['theft_rate']*100:.1f}%")
    m3.metric("Avg risk", f"{result['avg_risk']:.0f}/100")
    if result["metrics"]:
        mm = result["metrics"]
        st.info(f"Accuracy {mm['accuracy']:.3f} · Precision {mm['precision_val']:.3f} · "
                f"Recall {mm['recall_val']:.3f} · F1 {mm['f1_score']:.3f} · "
                f"ROC-AUC {mm['roc_auc']:.3f}" if mm.get("roc_auc") else
                f"Accuracy {mm['accuracy']:.3f} · F1 {mm['f1_score']:.3f}")

    flt = st.selectbox("Filter", ["All", "Theft only", "Normal only"])
    show = rdf
    if flt == "Theft only": show = rdf[rdf.status == "Theft"]
    elif flt == "Normal only": show = rdf[rdf.status == "Normal"]
    st.dataframe(show.sort_values("risk_score", ascending=False),
                use_container_width=True, hide_index=True, height=380)

    _export_buttons(rdf, info, engine.get_model_info(), result)


def _export_buttons(rdf, info, model_info, result):
    st.markdown("##### Export")
    e1, e2, e3 = st.columns(3)
    e1.download_button("⬇️ CSV", reports.to_csv_bytes(rdf), "predictions.csv",
                       "text/csv", use_container_width=True)
    e2.download_button("⬇️ Excel", reports.to_excel_bytes(rdf), "predictions.xlsx",
                       use_container_width=True)
    pdf = reports.pdf_report("ETD-XAI Prediction Report", model_info, result,
                             result.get("metrics"), rdf)
    if pdf:
        e3.download_button("⬇️ PDF Report", pdf, "etd_xai_report.pdf",
                           "application/pdf", use_container_width=True)
    else:
        e3.caption("PDF needs `reportlab` (in requirements).")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Dataset Manager
# ═════════════════════════════════════════════════════════════════════════════
def page_dataset():
    st.markdown('<div class="hero"><h1>🗂️ Dataset Manager</h1>'
                '<p>Inspect dataset structure and quality before prediction.</p></div>',
                unsafe_allow_html=True)

    file = st.file_uploader("Upload a dataset to inspect (CSV / Excel)", type=["csv", "xlsx", "xls"])
    if not file:
        ups = db.get_all_uploads()
        if ups:
            st.markdown("##### Previously processed datasets")
            st.dataframe(pd.DataFrame(ups)[["id", "filename", "upload_time", "total_rows",
                                            "theft_rows", "normal_rows", "theft_rate", "n_readings"]],
                        use_container_width=True, hide_index=True)
        return

    try:
        df = data.read_table(file)
    except Exception as exc:
        st.error(f"Read error: {exc}")
        return

    info = data.inspect(df)
    c = st.columns(4)
    c[0].metric("Customers (rows)", f"{info['n_rows']:,}")
    c[1].metric("Reading columns", info["n_readings"])
    c[2].metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    c[3].metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    st.markdown(f"**Detected ID column:** `{info['id_col'] or 'auto-generated'}`  ·  "
                f"**FLAG column:** `{info['flag_col'] or 'none'}`")

    if info["flag_col"]:
        st.markdown("##### Ground-truth distribution")
        dist = pd.to_numeric(df[info["flag_col"]], errors="coerce").value_counts().sort_index()
        fig = go.Figure(go.Bar(x=["Normal (0)", "Theft (1)"],
                               y=[dist.get(0, 0), dist.get(1, 0)],
                               marker_color=["#22c55e", "#ef4444"]))
        fig.update_layout(template=PLOT_TMPL, height=280, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Preview")
    st.dataframe(df.head(15), use_container_width=True)

    comp = engine.check_compatibility(info["n_readings"])
    (st.success if comp["compatible"] else st.error)(comp["reason"], icon="🔎")

    summary = pd.DataFrame({
        "metric": ["rows", "reading_columns", "missing_values", "duplicate_rows",
                   "id_column", "flag_column"],
        "value": [info["n_rows"], info["n_readings"], int(df.isna().sum().sum()),
                  int(df.duplicated().sum()), info["id_col"] or "auto", info["flag_col"] or "none"],
    })
    st.download_button("⬇️ Export dataset summary (CSV)",
                       reports.to_csv_bytes(summary), "dataset_summary.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: History
# ═════════════════════════════════════════════════════════════════════════════
def page_history():
    st.markdown('<div class="hero"><h1>📜 Prediction History</h1>'
                '<p>Every manual and batch prediction, persisted in SQLite.</p></div>',
                unsafe_allow_html=True)

    rows = db.get_manual(limit=500)
    if not rows:
        st.info("No predictions recorded yet.")
        return
    hdf = pd.DataFrame([{k: r[k] for k in
                        ("customer_id", "probability", "prediction", "confidence",
                         "risk_score", "status", "predicted_at", "model_name", "source")}
                       for r in rows])

    f1, f2, f3 = st.columns(3)
    q = f1.text_input("Search Customer ID")
    statf = f2.selectbox("Status", ["All", "Theft", "Normal"])
    srcf = f3.selectbox("Source", ["All"] + sorted(hdf["source"].dropna().unique().tolist()))

    show = hdf
    if q: show = show[show.customer_id.astype(str).str.contains(q, case=False, na=False)]
    if statf != "All": show = show[show.status == statf]
    if srcf != "All": show = show[show.source == srcf]

    st.dataframe(show, use_container_width=True, hide_index=True, height=460)
    st.download_button("⬇️ Export history (CSV)", reports.to_csv_bytes(show),
                       "prediction_history.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Reports
# ═════════════════════════════════════════════════════════════════════════════
def page_reports():
    st.markdown('<div class="hero"><h1>📑 Reports</h1>'
                '<p>Generate a professional PDF / Excel / CSV report from the latest run.</p></div>',
                unsafe_allow_html=True)

    uid = db.get_latest_upload_id()
    up = db.get_upload(uid) if uid else None
    if not up:
        st.info("Run a batch prediction first (📦 Batch Prediction).")
        return

    info = engine.get_model_info()
    df = db.get_predictions_df(uid)

    st.markdown(f"#### Latest run — `{up['filename']}`  ({up['upload_time'][:19]})")
    c = st.columns(4)
    c[0].metric("Customers", f"{up['total_rows']:,}")
    c[1].metric("Theft", f"{up['theft_rows']:,}")
    c[2].metric("Normal", f"{up['normal_rows']:,}")
    c[3].metric("Theft rate", f"{(up['theft_rate'] or 0)*100:.1f}%")

    metrics = None
    if up.get("accuracy") is not None:
        metrics = {k: up[k] for k in ("accuracy", "precision_val", "recall_val", "f1_score", "roc_auc")}
        metrics["confusion_matrix"] = up.get("confusion_matrix")

    summary = {"total_rows": up["total_rows"], "normal_rows": up["normal_rows"],
               "theft_rows": up["theft_rows"], "theft_rate": up["theft_rate"], "avg_risk": up["avg_risk"]}

    e1, e2, e3 = st.columns(3)
    e1.download_button("⬇️ CSV", reports.to_csv_bytes(df), "report.csv", "text/csv", use_container_width=True)
    e2.download_button("⬇️ Excel", reports.to_excel_bytes(df), "report.xlsx", use_container_width=True)
    pdf = reports.pdf_report("ETD-XAI Enterprise Report", info, summary, metrics, df)
    if pdf:
        e3.download_button("⬇️ PDF", pdf, "etd_xai_report.pdf", "application/pdf", use_container_width=True)
    else:
        e3.caption("Install `reportlab` for PDF.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: AI Copilot
# ═════════════════════════════════════════════════════════════════════════════
def page_copilot():
    st.markdown('<div class="hero"><h1>🤖 AI Copilot</h1>'
                '<p>Project-scoped assistant — explains the model, predictions and metrics.</p></div>',
                unsafe_allow_html=True)

    st.caption("Quick questions:")
    cols = st.columns(3)
    for i, sugg in enumerate(copilot.SUGGESTIONS):
        if cols[i % 3].button(sugg, use_container_width=True, key=f"sg{i}"):
            st.session_state.chat.append(("user", sugg))
            st.session_state.chat.append(("assistant", copilot.answer(sugg)))

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)

    q = st.chat_input("Ask about the model, a metric, or a prediction…")
    if q:
        st.session_state.chat.append(("user", q))
        st.session_state.chat.append(("assistant", copilot.answer(q)))
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Settings
# ═════════════════════════════════════════════════════════════════════════════
def page_settings():
    st.markdown('<div class="hero"><h1>⚙️ Settings</h1>'
                '<p>Model management, defaults, and system status.</p></div>',
                unsafe_allow_html=True)

    info = engine.get_model_info()
    st.markdown("### Active Model")
    if info.get("loaded"):
        st.markdown(
            '<span class="badge badge-normal">🟢 Status: Loaded</span>',
            unsafe_allow_html=True,
        )
        c = st.columns(3)
        c[0].metric("Active Model Name", info["model_name"])
        c[1].metric("Parameters", info["total_params_fmt"])
        c[2].metric("Architecture", info["architecture"])
        c = st.columns(4)
        c[0].metric("Input shape", info["input_shape"])
        c[1].metric("Output shape", info["output_shape"])
        c[2].metric("Seq length", info["seq_len_expected"] or "variable")
        c[3].metric("Stat features", info["stat_input_size"])
        c = st.columns(3)
        c[0].metric("TensorFlow", info["tf_version"])
        c[1].metric("Keras", info["keras_version"])
        c[2].metric("Upload date", (info.get("upload_time") or "—")[:19].replace("T", " "))
        st.caption("Load method: `tensorflow.keras.models.load_model()` · "
                   "Inference: `model.predict()` · Exclusive engine — no fallback models.")
        with st.expander("Model architecture (summary)"):
            st.code(info["summary"], language="text")
    else:
        st.markdown(
            '<span class="badge badge-theft">🔴 Status: Not Loaded</span>',
            unsafe_allow_html=True,
        )
        st.error(engine.NO_MODEL_MSG, icon="🚫")

    st.divider()
    st.markdown("### Upload / Activate a New Model")
    up = st.file_uploader("CNN-LSTM model (.keras / .h5)", type=["keras", "h5"])
    if up and st.button("Activate uploaded model", type="primary"):
        dest = Path(__file__).parent / "uploads"
        dest.mkdir(exist_ok=True)
        path = dest / up.name
        path.write_bytes(up.getbuffer())
        try:
            engine.unload_model()
            engine.load_model(str(path), up.name)
            st.cache_resource.clear()
            st.success(f"Activated **{up.name}**.", icon="✅")
            st.rerun()
        except Exception as exc:
            st.error(f"Rejected: {exc}")

    st.divider()
    st.markdown("### Defaults")
    thr = st.slider("Default decision threshold", 0.0, 1.0, st.session_state.threshold, 0.01)
    if thr != st.session_state.threshold:
        st.session_state.threshold = thr
        db.set_setting("threshold", thr)
        st.toast(f"Default threshold set to {thr:.2f}")

    st.divider()
    st.markdown("### System / Verification Status")
    st.json({
        "model_loaded": info.get("loaded", False),
        "active_model": info.get("model_name"),
        "load_method": "tensorflow.keras.models.load_model(path)",
        "predict_method": "model.predict(x)",
        "exclusive_engine": True,
        "fallback_models": "none — CNN-LSTM only (no RandomForest/XGBoost/LightGBM/LogReg/mock)",
        "compute": "GPU" if _has_gpu() else "CPU",
        "database": str(db.DB_PATH),
        "last_prediction": engine.state.last_prediction or None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
ROUTES = {
    "📊 Dashboard": page_dashboard,
    "🔮 Manual Prediction": page_manual,
    "📦 Batch Prediction": page_batch,
    "🗂️ Dataset Manager": page_dataset,
    "📜 History": page_history,
    "📑 Reports": page_reports,
    "🤖 AI Copilot": page_copilot,
    "⚙️ Settings": page_settings,
}
ROUTES[page]()
