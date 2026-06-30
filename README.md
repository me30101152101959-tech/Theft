# ETD-XAI Enterprise
## Electricity Theft Detection using CNN-LSTM Deep Learning

---

## 🚀 Quick Start

### Option A — Docker (Recommended, Production)

```bash
git clone <your-repo>
cd etd-xai-enterprise

# Build and launch everything
docker-compose up --build

# Open: http://localhost:80
```

### Option B — Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
# Open: http://localhost:5173
```

---

## 📁 Project Structure

```
etd-xai-enterprise/
├── backend/
│   ├── main.py                   # FastAPI entry point
│   ├── services/
│   │   ├── model_service.py      # CNN-LSTM loading + inference
│   │   ├── dataset_service.py    # Dataset ingestion + batch predict
│   │   └── feature_service.py   # 59 statistical features + MinMaxScaler
│   ├── routers/
│   │   ├── upload.py             # POST /api/upload/model, /dataset
│   │   ├── predict.py            # POST /api/predict/manual
│   │   ├── dashboard.py          # GET /api/dashboard/* + exports
│   │   └── copilot.py            # POST /api/copilot/ask
│   ├── models/schemas.py         # Pydantic schemas
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx               # Root router + state
│   │   ├── pages/
│   │   │   ├── UploadPage.tsx    # Step 1+2 startup
│   │   │   ├── DashboardPage.tsx # KPIs + 9 Plotly charts
│   │   │   ├── CustomersPage.tsx # TanStack Table + pagination
│   │   │   ├── PredictPage.tsx   # Manual 26-reading prediction
│   │   │   ├── ReportsPage.tsx   # Download CSV/JSON/TXT
│   │   │   ├── CopilotPage.tsx   # AR/EN XAI assistant
│   │   │   └── SettingsPage.tsx  # Theme, language, threshold
│   │   ├── components/Layout.tsx # Sidebar navigation
│   │   ├── api/client.ts         # Axios API client
│   │   └── types/index.ts        # TypeScript interfaces
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── Dockerfile
├── nginx/nginx.conf               # Reverse proxy config
└── docker-compose.yml
```

---

## 🧠 Model Requirements

The application accepts **only CNN-LSTM models** in `.keras` or `.h5` format.

Your model must have:
- **sequence_input**: shape `(None, 26, 1)` — 26 time-step readings
- **stat_input**: shape `(None, 59)` — 59 statistical features (if dual-input)
- **output**: shape `(None, 1)` — sigmoid probability

Models with BiGRU, BiLSTM, Bidirectional layers, or Ensemble architectures are **rejected**.

---

## 📊 Dataset Format

CSV file with:
- `CONS_NO` — Customer ID (any string)
- 26 reading columns — numeric float values
- `FLAG` — Optional (0=Normal, 1=Theft) used **only for evaluation**, never for prediction

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload/model` | Upload CNN-LSTM model |
| POST | `/api/upload/dataset` | Upload CSV + run predictions |
| POST | `/api/upload/reset-model` | Unload current model |
| GET | `/api/upload/status` | Check model/dataset status |
| GET | `/api/dashboard/stats` | KPI statistics |
| GET | `/api/dashboard/charts` | Chart data (all Plotly charts) |
| GET | `/api/dashboard/customers` | Paginated customer predictions |
| GET | `/api/dashboard/customer/{id}` | Single customer data |
| GET | `/api/dashboard/export/csv` | Download predictions CSV |
| GET | `/api/dashboard/export/json` | Download predictions JSON |
| GET | `/api/dashboard/report/summary` | Download text report |
| POST | `/api/predict/manual` | Predict single customer |
| POST | `/api/predict/update-threshold` | Re-apply threshold |
| POST | `/api/copilot/ask` | XAI explanation (EN/AR) |
| GET | `/api/copilot/explain/{id}` | Explain specific customer |
| GET | `/api/health` | Health check |

---

## 🐳 Docker Deployment on DigitalOcean

```bash
# 1. Create a Droplet (2GB+ RAM recommended for TensorFlow)
# 2. SSH in and install Docker
curl -fsSL https://get.docker.com | sh
apt install docker-compose-plugin -y

# 3. Clone or upload your project
git clone <your-repo> && cd etd-xai-enterprise

# 4. Launch
docker-compose up -d --build

# 5. Access: http://<your-droplet-ip>
```

---

## ✅ Security Guarantees

- CNN-LSTM architecture enforced — other architectures rejected with clear error
- FLAG column never used during prediction
- No mock/fallback predictions — real TF inference only
- File validation on all uploads (format, size, architecture check)

---

## 🎓 Graduation Project

**Title:** ETD-XAI Enterprise — Electricity Theft Detection using Deep Learning  
**Model:** CNN-LSTM (Dual-input: sequence + statistical features)  
**Dataset:** SGCC-style CSV with any number of reading columns (auto-detected)  
**Stack:** FastAPI · TensorFlow · React · TypeScript · Plotly · TanStack Table · Docker

---

## 🖥️ Production Deployment (Ubuntu / DigitalOcean — systemd + Nginx + Gunicorn)

A one-shot script provisions a fresh Ubuntu droplet (installs deps, builds the
frontend, sets up the systemd service and Nginx reverse proxy):

```bash
sudo REPO=https://github.com/me30101152101959-tech/Theft.git BRANCH=main \
  bash deploy/deploy.sh
```

What it configures:

| Component        | Detail                                                            |
|------------------|-------------------------------------------------------------------|
| Backend          | Gunicorn + Uvicorn workers via `backend/gunicorn_conf.py`         |
| Service manager  | `deploy/etd-xai.service` — auto-starts on reboot, `Restart=always`|
| Reverse proxy    | `deploy/nginx-site.conf` — serves `frontend/dist`, proxies `/api/`|
| Persistent DB    | `/var/lib/etd-xai/etd_xai.db` (via `DATABASE_PATH`)               |
| Logs             | `/var/log/etd-xai/{app,error}.log`                                |

Enable HTTPS with Let's Encrypt:

```bash
sudo certbot --nginx -d your-domain.com
```

Service management:

```bash
systemctl status etd-xai      # health
journalctl -u etd-xai -f      # live logs
systemctl restart etd-xai     # restart (model + data persist)
```

### Environment variables

Copy `.env.example` → `.env` and adjust. Key vars: `DATABASE_PATH`, `MODEL_PATH`,
`DATASET_PATH`, `UPLOAD_FOLDER`, `CORS_ORIGINS`, `MAX_UPLOAD_MB`, `SECRET_KEY`,
`LOG_LEVEL`, `VITE_API_URL`. All have safe defaults — nothing is required for local dev.

### Persistence & clone-and-run

- The active model (`backend/uploads/model/cnnlstm_final.keras`) is bundled and
  **auto-loads on startup** — no manual upload needed after a clone or reboot.
- The SQLite DB persists all predictions, customers, history and metadata. In
  Docker it lives on the `etd_db` named volume; under systemd at `DATABASE_PATH`.
- On first boot with an empty DB, a bundled sample dataset is auto-processed so
  the dashboard is populated immediately (set `BOOTSTRAP_DATASET_PATH` to override).

### Backup & restore

```bash
sudo bash deploy/backup.sh                 # → /var/backups/etd-xai/etd-xai_backup_*.tar.gz
sudo bash deploy/restore.sh /var/backups/etd-xai/etd-xai_backup_YYYYMMDD_HHMMSS.tar.gz
```

Backups capture a consistent SQLite snapshot (`.backup`) plus uploaded models/datasets.

## 🩺 Troubleshooting

| Symptom                              | Fix                                                                 |
|--------------------------------------|---------------------------------------------------------------------|
| "No valid CNN-LSTM model is loaded"  | Ensure `MODEL_PATH` points to a real `.keras`; check `journalctl`.  |
| Dashboard empty after deploy         | Upload a dataset, or set `BOOTSTRAP_DATASET_PATH`.                   |
| 502 from Nginx                       | Backend down — `systemctl status etd-xai`, check `/var/log/etd-xai`.|
| Frontend can't reach API             | Set `VITE_API_URL` to the public URL and rebuild (`npm run build`). |
| Upload too large                     | Raise `client_max_body_size` (Nginx) / `MAX_UPLOAD_MB`.             |
