import axios from 'axios';

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: BASE,
  timeout: 30_000,
});

// Long-timeout instance for batch operations (100k rows can take minutes)
const apiBatch = axios.create({
  baseURL: BASE,
  timeout: 600_000, // 10 minutes
});

// ── Upload ──────────────────────────────────────────────────────────────────
export const uploadModel = (file: File) => {
  const fd = new FormData();
  fd.append('model_file', file);
  return api.post('/api/upload/model', fd);
};

export const uploadDataset = (file: File, threshold = 0.5) => {
  const fd = new FormData();
  fd.append('dataset_file', file);
  fd.append('threshold', String(threshold));
  return apiBatch.post('/api/upload/dataset', fd);
};

export const resetModel   = () => api.post('/api/upload/reset-model');
export const getUploadStatus = () => api.get('/api/upload/status');

// ── Dashboard ──────────────────────────────────────────────────────────────
export const getDashboardStats = () => api.get('/api/dashboard/stats');
export const getChartData      = () => api.get('/api/dashboard/charts');
export const getModelInfo      = () => api.get('/api/dashboard/model-info');

// ── Customers ──────────────────────────────────────────────────────────────
export const getCustomers = (params: {
  page?: number;
  page_size?: number;
  search?: string;
  status_filter?: string;
  sort_by?: string;
  sort_dir?: string;
}) => api.get('/api/dashboard/customers', { params });

export const getCustomerById = (id: string) =>
  api.get(`/api/dashboard/customer/${encodeURIComponent(id)}`);

// ── Predict — single ────────────────────────────────────────────────────────
export const predictManual = (payload: {
  customer_id: string;
  readings: number[];
  threshold?: number;
  strategy?: string;
}) => api.post('/api/predict/manual', payload);

// ── Predict — strategies & validation ────────────────────────────────────────
export const getStrategies = () => api.get('/api/predict/strategies');

export const validateDataset = (file: File, strategy = 'last_n') => {
  const fd = new FormData();
  fd.append('csv_file', file);
  fd.append('strategy', strategy);
  return apiBatch.post('/api/predict/validate-dataset', fd);
};

// ── Predict — batch ─────────────────────────────────────────────────────────
export const predictBatchPreview = (file: File, threshold = 0.5, strategy = 'last_n') => {
  const fd = new FormData();
  fd.append('csv_file', file);
  fd.append('threshold', String(threshold));
  fd.append('strategy', strategy);
  return apiBatch.post('/api/predict/batch-preview', fd);
};

export const predictBatchStore = (file: File, threshold = 0.5, strategy = 'last_n', label = 'Batch Upload') => {
  const fd = new FormData();
  fd.append('csv_file', file);
  fd.append('threshold', String(threshold));
  fd.append('strategy', strategy);
  fd.append('label', label);
  return apiBatch.post('/api/predict/batch-store', fd);
};

// ── System status (model verification) ───────────────────────────────────────
export const getSystemStatus = () => api.get('/api/system/status');

// ── Model status (exclusive-engine verification + last prediction trail) ──────
export const getModelStatus = () => api.get('/api/model/status');

// ── Predict — threshold update ───────────────────────────────────────────────
export const updateThreshold = (threshold: number) =>
  api.post('/api/predict/update-threshold', { threshold });

// ── Predict — SHAP / explainability ─────────────────────────────────────────
export const getShap = (customerId: string) =>
  api.get(`/api/predict/shap/${encodeURIComponent(customerId)}`);

// ── Predict — history ────────────────────────────────────────────────────────
export const getPredictHistory = (limit = 100, source?: string) =>
  api.get('/api/predict/history', { params: { limit, source } });

// ── Predict — template ───────────────────────────────────────────────────────
export const downloadTemplate = () =>
  api.get('/api/predict/template', { responseType: 'blob' });

// ── Export ─────────────────────────────────────────────────────────────────
export const exportCSV    = () => api.get('/api/dashboard/export/csv',      { responseType: 'blob' });
export const exportJSON   = () => api.get('/api/dashboard/export/json',     { responseType: 'blob' });
export const exportReport = () => api.get('/api/dashboard/report/summary',  { responseType: 'blob' });

// ── Copilot ───────────────────────────────────────────────────────────────
export const askCopilot = (payload: {
  question: string;
  language: string;
  customer_id?: string;
}) => api.post('/api/copilot/ask', payload);

export const explainCustomer = (id: string, lang = 'en') =>
  api.get(`/api/copilot/explain/${encodeURIComponent(id)}`, { params: { lang } });

// ── Health ─────────────────────────────────────────────────────────────────
export const healthCheck = () => api.get('/api/health');
