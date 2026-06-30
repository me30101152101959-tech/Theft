# ETD-XAI Enterprise v2.0 — Streamlit Edition

**Electricity Theft Detection using Explainable Artificial Intelligence**

A pure **Python + Streamlit** application. No React, no Node.js, no separate
backend — one app you can host **for free** on Streamlit Community Cloud.

Every prediction comes **only** from the trained CNN-LSTM model
(`assets/cnnlstm_final.keras`) via real `tensorflow.keras.models.load_model()` +
`model.predict()`. There are no fallback / mock / surrogate models.

---

## ✨ Features

| Page | What it does |
|------|--------------|
| 📊 **Dashboard** | KPIs, accuracy/precision/recall/F1/ROC-AUC, confusion matrix, distribution & risk charts, model/DB/compute status |
| 🔮 **Manual Prediction** | Single customer, any number of readings, status badge, consumption chart, integrated-gradients XAI + risk factors |
| 📦 **Batch Prediction** | Upload CSV/Excel, auto-detect columns, validate, score everyone, export CSV/Excel/PDF |
| 🗂️ **Dataset Manager** | Inspect rows/missing/duplicates, ground-truth distribution, compatibility check |
| 📜 **History** | Every prediction, searchable, persisted in SQLite |
| 📑 **Reports** | Professional PDF / Excel / CSV report of the latest run |
| 🤖 **AI Copilot** | Project-scoped Q&A (model, metrics, predictions) — never hallucinates |
| ⚙️ **Settings** | Active-model info, upload/activate model, defaults, verification status |

Auto-detects datasets of **26 / 59 / 90 / 120 / 365** (or any) readings and maps
them to the model's expected length via 5 strategies (last_n, truncate, pad,
interpolate, sliding_window). Dark / light theme.

---

## 🚀 Run locally

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
# opens http://localhost:8501
```

The bundled model and a sample dataset are included, so it works immediately.

---

## ☁️ Deploy FREE on Streamlit Community Cloud

1. Push this repo to GitHub (already done).
2. Go to **https://share.streamlit.io** → sign in with GitHub → **New app**.
3. Fill in:
   - **Repository:** `me30101152101959-tech/Theft`
   - **Branch:** `feat/dynamic-exclusive-prediction-engine` (or `main` after merge)
   - **Main file path:** `streamlit_app/app.py`
4. Click **Advanced settings** → set **Python version = 3.11**.
5. **Deploy**. First build takes ~5 min (TensorFlow). You get a public URL like
   `https://<your-app>.streamlit.app` — fully online, not local.

> Streamlit Cloud reads `streamlit_app/requirements.txt` automatically.
> The 2.9 MB `.keras` model is committed in `assets/`, so no upload is needed.

### Alternative free / cheap hosts
- **Hugging Face Spaces** (Streamlit SDK) — point it at `streamlit_app/app.py`.
- **Render.com** — start command:
  `streamlit run streamlit_app/app.py --server.port $PORT --server.address 0.0.0.0`

---

## 🧠 Model & preprocessing (must match training)

- **Sequence input** `(N, 26, 1)` — per-row min-max scaled to **[0,1]**.
- **Stat input** `(N, 59)` — 59 engineered features → **StandardScaler**.
- Skipping this scaling makes the model predict everything as *Normal*.

Output: single sigmoid probability → **Class 0 (Normal, green)** /
**Class 1 (Theft, red)**. Only ever two classes.

---

## 🗃️ Persistence

SQLite (`etd_xai.db`, path override via `DATABASE_PATH`) stores uploads,
predictions, manual predictions and settings.

> Note: on Streamlit Community Cloud the filesystem is **ephemeral** — the DB
> resets when the app restarts/redeploys. For permanent storage point
> `DATABASE_PATH` at a mounted volume (Render disk) or swap in a hosted DB.
