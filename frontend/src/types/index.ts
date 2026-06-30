// ─── Customer ────────────────────────────────────────────────────────────────
export interface Customer {
  id: string;
  readings: number[];
  probability: number;
  prediction: 0 | 1;
  confidence: number;
  risk_score: number;
  status: 'Normal' | 'Theft';
  flag?: number | null;
}

// ─── Dashboard Stats ──────────────────────────────────────────────────────────
export interface DashboardStats {
  total_customers: number;
  processed_customers: number;
  predicted_theft: number;
  predicted_normal: number;
  avg_confidence: number;
  avg_risk_score: number;
  theft_rate: number;
  has_flag: boolean;
  dataset_name: string;
  upload_time: string;
  // evaluation (optional)
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  roc_auc?: number;
}

// ─── Model Info ───────────────────────────────────────────────────────────────
export interface ModelInfo {
  loaded: boolean;
  model_name?: string;
  upload_time?: string;
  input_shape?: string;
  output_shape?: string;
  total_params?: number;
  total_params_fmt?: string;
  is_dual_input?: boolean;
  stat_input_size?: number;
  summary?: string;
  architecture?: string;
}

// ─── Chart Data ───────────────────────────────────────────────────────────────
export interface ChartData {
  risk_distribution?: { values: number[]; labels: string[] };
  pie?: { labels: string[]; values: number[] };
  top10_high?: { ids: string[]; risks: number[]; probs: number[] };
  top10_low?: { ids: string[]; risks: number[] };
  scatter?: { risk: number[]; confidence: number[]; status: string[]; ids: string[] };
  confusion?: number[][];
  roc?: { fpr: number[]; tpr: number[] };
  pr?: { precision: number[]; recall: number[] };
}

// ─── Pagination ───────────────────────────────────────────────────────────────
export interface PaginatedResult {
  data: Customer[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// ─── Upload State ─────────────────────────────────────────────────────────────
export type AppStep = 'upload_model' | 'upload_dataset' | 'ready';

export interface AppState {
  step: AppStep;
  modelInfo: ModelInfo | null;
  datasetSummary: DatasetSummary | null;
}

export interface DatasetSummary {
  total: number;
  theft: number;
  normal: number;
  avg_confidence?: number;
  avg_risk?: number;
  has_flag?: boolean;
  dataset_name: string;
  upload_time?: string;
  model_used?: string;
}

// ─── Manual Prediction ───────────────────────────────────────────────────────
export interface PredictionResult {
  probability: number;
  prediction: 0 | 1;
  confidence: number;
  risk_score: number;
  status: 'Normal' | 'Theft';
  label: string;
  customer_id: string;
  readings: number[];
  model_name: string;
  threshold_used: number;
}

// ─── Copilot ─────────────────────────────────────────────────────────────────
export interface CopilotMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}
