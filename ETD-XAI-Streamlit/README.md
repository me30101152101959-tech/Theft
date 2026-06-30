# ⚡ ETD-XAI Enterprise v2.0

### Electricity Theft Detection using Explainable Artificial Intelligence

A production-quality, single-file **Streamlit** application that detects electricity
theft using **only** a trained **CNN-LSTM** TensorFlow/Keras model. Built to deploy
on **Streamlit Community Cloud** for free, and to run locally with no code changes.

> Every prediction comes directly from the active Keras model via
> `tensorflow.keras.models.load_model()` + `model.predict()`. There are **no**
> fallback / mock / rule-based / random models. The ground-truth `FLAG` column is
> used **only** for evaluation metrics — never for prediction.

---

## ✨ Features

- **Executive Dashboard** — KPIs, Accuracy / Precision / Recall / F1 / ROC-AUC,
  confusion matrix, prediction & risk distributions, recent predictions, system status.
- **Manual Prediction** — single customer, any number of readings, green/red status
  badge, probability · confidence · risk score · prediction time, consumption chart.
- **Batch Prediction** — upload CSV/Excel, auto-detect columns, validate compatibility,
  score everyone, export **CSV / Excel / PDF**.
- **Explainable AI** — **SHAP** (GradientExplainer) per-timestep importance with an
  integrated-gradients fallback, plus human-readable risk factors.
- **AI Copilot** — project-scoped assistant (CNN-LSTM, metrics, SHAP, risk score). No hallucination.
- **Reports** — professional PDF / Excel / CSV of the latest run.
- **Settings** — active model info (input/output shape, params, TF version, upload date,
  status), upload/replace/restore model, persist a dataset, theme & threshold defaults.
- **Persistence** — SQLite + settings; model & preferences remembered across restarts.
- **UI** — dark / light theme, sidebar navigation, KPI cards, interactive Plotly charts,
  progress bars, loading indicators.

Auto-detects datasets with **26 / 59 / 90 / 120 / 365** (or any) readings and maps them
to the model length via 5 strategies: `last_n`, `truncate`, `pad`, `interpolate`,
`sliding_window`.

---

## 🧩 Requirements

Python **3.11+** and the packages in [`requirements.txt`](requirements.txt):
`streamlit`, `tensorflow-cpu`, `numpy`, `pandas`, `plotly`, `scikit-learn`, `shap`,
`openpyxl`, `reportlab`, `scipy`.

---

## 🛠️ Installation & Running Locally

```bash
git clone https://github.com/me30101152101959-tech/Theft.git
cd Theft/ETD-XAI-Streamlit

python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
# opens http://localhost:8501
```

The CNN-LSTM model (`assets/cnnlstm_final.keras`) and a sample dataset are bundled,
so the app works immediately with no upload.

---

## ☁️ Deploying to Streamlit Community Cloud (free)

1. Push this repo to GitHub.
2. Open **https://share.streamlit.io** → sign in with GitHub → **New app**.
3. Set:
   - **Repository:** `me30101152101959-tech/Theft`
   - **Branch:** your branch (e.g. `main`)
   - **Main file path:** `ETD-XAI-Streamlit/app.py`
4. **Advanced settings → Python version = 3.11**.
5. **Deploy.** First build ~5 min (TensorFlow). You get a public `*.streamlit.app` URL.

`requirements.txt` and `.streamlit/config.toml` are picked up automatically.

> The bundled 2.9 MB `.keras` model is committed, so no upload is needed.
> Streamlit Cloud's filesystem is ephemeral — the SQLite history resets on restart.
> For permanent storage set `DATABASE_PATH` to a persistent disk (e.g. on Render).

### Google Colab (optional)

```python
!pip install -r requirements.txt pyngrok -q
!streamlit run app.py &>/dev/null &
from pyngrok import ngrok; print(ngrok.connect(8501))
```

---

## 🧠 Model & preprocessing

- **Sequence input** `(N, 26, 1)` — per-row min-max scaled to **[0,1]**.
- **Stat input** `(N, 59)` — 59 engineered features → **StandardScaler**.
- Output: one sigmoid probability → **Class 0 Normal (🟢)** / **Class 1 Theft (🔴)**.
- Threshold (default 0.50): `p ≥ 0.50 → Theft`, else `Normal`.

> Skipping the [0,1] scaling makes the model predict everything as *Normal* — the
> exact training preprocessing is reproduced in `app.py`.

---

## 📸 Screenshots

_Add screenshots of the Dashboard, Manual Prediction, Batch Prediction and Settings here._

| Dashboard | Manual Prediction | Batch + XAI |
|-----------|-------------------|-------------|
| _(placeholder)_ | _(placeholder)_ | _(placeholder)_ |

---

## 📁 Folder Structure

```text
ETD-XAI-Streamlit/
├── app.py                      # the entire application
├── requirements.txt
├── README.md
├── assets/
│   ├── logo.png
│   ├── cnnlstm_final.keras     # bundled active model
│   └── sample_dataset.csv      # bundled demo dataset
└── .streamlit/
    └── config.toml
```

---

## 👤 Author

**ETD-XAI Enterprise** — graduation project · CNN-LSTM Explainable AI for electricity
theft detection.

## 📄 License

MIT License.
