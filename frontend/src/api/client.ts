import axios from 'axios';

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({ baseURL: BASE });

// ── Health / startup check ─────────────────────────────────────────────────
export const healthCheck = () => api.get('/api/health');
export const systemStatus = () => api.get('/api/system/status');

// ── Model ─────────────────────────────────────────────────────────────────
/** Upload cnnlstm_final.keras → keras.models.load_model() → SQLite registry */
export const uploadModel = (file: File) => {
  const fd = new FormData();
  fd.append('model_file', file);
  return api.post('/api/load-model', fd);   // unified endpoint
};

/** Legacy alias kept for any existing callers */
export const uploadModelLegacy = (file: File) => {
  const fd = new FormData();
  fd.append('model_file', file);
  return api.post('/api/upload/model', fd);
};

export const resetModel = () => api.post('/api/upload/reset-model');
export const getModelInfo = () => api.get('/api/dashboard/model-info');

// ── Dataset ────────────────────────────────────────────────────────────────
/** Upload CSV → model.predict() → store ALL results in SQLite */
export const uploadDataset = (file: File, threshold = 0.5) => {
  const fd = new FormData();
  fd.append('dataset_file', file);
  fd.append('threshold', String(threshold));
  return api.post('/api/upload', fd);        // unified endpoint
};

export const getUploadStatus = () => api.get('/api/upload/status');

// ── Dashboard  (all data from SQLite) ─────────────────────────────────────
/** Single call — KPIs + charts + metrics + model info from SQLite */
export const getDashboard = () => api.get('/api/dashboard');

/** Legacy split calls (still SQLite-backed) */
export const getDashboardStats = () => api.get('/api/dashboard/stats');
export const getChartData      = () => api.get('/api/dashboard/charts');

// ── Customers  (SELECT from predictions table) ────────────────────────────
export const getCustomers = (params: {
  page?:          number;
  page_size?:     number;
  search?:        string;
  status_filter?: string;
  sort_by?:       string;
  sort_dir?:      string;
}) => api.get('/api/customers', { params });  // unified endpoint

export const getCustomerById = (id: string) =>
  api.get(`/api/dashboard/customer/${encodeURIComponent(id)}`);

// ── Predict ───────────────────────────────────────────────────────────────
/** Single customer predict → model.predict() → SQLite manual_predictions */
export const predictManual = (payload: {
  customer_id: string;
  readings:    number[];
  threshold?:  number;
}) => api.post('/api/predict', payload);      // unified endpoint

/** Legacy alias */
export const predictManualLegacy = (payload: {
  customer_id: string;
  readings:    number[];
  threshold?:  number;
}) => api.post('/api/predict/manual', payload);

export const updateThreshold = (threshold: number) =>
  api.post('/api/predict/update-threshold', { threshold });

/** Batch predict from CSV (returns results, does not persist) */
export const predictBatch = (file: File, threshold = 0.5) => {
  const fd = new FormData();
  fd.append('csv_file', file);
  fd.append('threshold', String(threshold));
  return api.post('/api/predict-batch', fd);
};

// ── Export ────────────────────────────────────────────────────────────────
export const exportCSV    = () => api.get('/api/dashboard/export/csv',        { responseType: 'blob' });
export const exportJSON   = () => api.get('/api/dashboard/export/json',       { responseType: 'blob' });
export const exportReport = () => api.get('/api/dashboard/report/summary',    { responseType: 'blob' });

// ── Copilot ───────────────────────────────────────────────────────────────
export const askCopilot = (payload: {
  question:    string;
  language:    string;
  customer_id?: string;
}) => api.post('/api/copilot/ask', payload);

export const explainCustomer = (id: string, lang = 'en') =>
  api.get(`/api/copilot/explain/${encodeURIComponent(id)}`, { params: { lang } });
