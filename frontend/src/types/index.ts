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
  predicted_at?: string;
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
  threshold?: number;
  // evaluation (optional — only when FLAG present)
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  roc_auc?: number;
  data_source?: string;
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
  seq_len_expected?: number | null;   // null => variable-length
  is_variable_length?: boolean;
  seq_channels?: number;
  tf_version?: string;
  keras_version?: string;
  summary?: string;
  architecture?: string;
}

// ─── Preprocessing strategy ───────────────────────────────────────────────────
export interface Strategy {
  value: string;
  label: string;
}

// ─── Dataset validation (server-side) ─────────────────────────────────────────
export interface DatasetValidation {
  success: boolean;
  dataset_name: string;
  n_customers: number;
  n_reading_cols: number;
  detected_seq_len: number;
  missing_values: number;
  duplicate_customers: number;
  id_column: string | null;
  ground_truth_column: string | null;
  feature_columns: string[];
  status: string;
  compatibility: {
    compatible: boolean;
    expected_len: number | null;
    uploaded_len: number;
    variable_length: boolean;
    preprocessing_needed: boolean;
    reason: string;
    suggested_strategy?: string;
    selected_strategy?: string;
  };
  preview: Record<string, any>[];
  preview_columns: string[];
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
  avg_confidence: number;
  avg_risk: number;
  has_flag: boolean;
  dataset_name: string;
  upload_time: string;
  model_used: string;
}

// ─── Manual Prediction ───────────────────────────────────────────────────────
export type RiskLevel = 'Low' | 'Medium' | 'High';

export interface PredictionResult {
  probability: number;
  prediction: 0 | 1;
  confidence: number;
  risk_score: number;
  risk_level: RiskLevel;
  status: 'Normal' | 'Theft';
  label: string;
  customer_id: string;
  readings: number[];
  model_name: string;
  threshold_used: number;
  predicted_at?: string;
  sqlite_row_id?: number;
  predict_proof?: string;
}

// ─── Batch Prediction ─────────────────────────────────────────────────────────
export interface BatchPredRow {
  customer_id: string;
  probability: number;
  prediction: 0 | 1;
  confidence: number;
  risk_score: number;
  risk_level: RiskLevel;
  status: 'Normal' | 'Theft';
  flag: number | null;
  readings: number[];
  predicted_at: string;
}

export interface BatchPredMetrics {
  accuracy?: number;
  precision_val?: number;
  recall_val?: number;
  f1_score?: number;
  roc_auc?: number;
}

export interface BatchPredResult {
  success: boolean;
  stored: boolean;
  upload_id?: number;
  total: number;
  theft: number;
  normal: number;
  avg_confidence: number;
  avg_risk: number;
  elapsed_seconds: number;
  has_flag: boolean;
  metrics: BatchPredMetrics;
  predictions: BatchPredRow[];
  predict_proof: string;
  model_name: string;
}

// ─── Prediction History ───────────────────────────────────────────────────────
export interface PredHistoryRow {
  id: number;
  customer_id: string | null;
  probability: number;
  prediction: 0 | 1;
  confidence: number;
  risk_score: number;
  status: 'Normal' | 'Theft';
  readings: number[];
  predicted_at: string;
  threshold: number;
  model_name: string | null;
  source: string;
}

// ─── SHAP / Explainability ────────────────────────────────────────────────────
export interface ShapFeature {
  feature: string;
  value: number;
  importance: number;
  rank: number;
}

export interface ShapResult {
  customer_id: string;
  probability: number;
  status: 'Normal' | 'Theft';
  method: string;
  feature_importance: ShapFeature[];
  top5_features: string[];
  readings_labels: string[];
}

// ─── CSV Preview ──────────────────────────────────────────────────────────────
export interface CsvPreview {
  rows: Record<string, string>[];
  columns: string[];
  totalRows: number;
  missingValues: number;
  hasConsNo: boolean;
  hasFlag: boolean;
  readingColCount: number;
  isValid: boolean;
  errors: string[];
}

// ─── Copilot ─────────────────────────────────────────────────────────────────
export interface CopilotMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}
