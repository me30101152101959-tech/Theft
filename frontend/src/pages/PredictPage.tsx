import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Zap, AlertTriangle, CheckCircle2, Loader2, Upload, Download,
  Search, ChevronUp, ChevronDown, FileText, BarChart2, Clock,
  RefreshCw, Database, Shield, X, Info, ChevronLeft, ChevronRight, Settings2,
} from 'lucide-react';
import Plot from 'react-plotly.js';
import toast from 'react-hot-toast';

import {
  predictManual, predictBatchPreview, predictBatchStore,
  getShap, getPredictHistory, downloadTemplate,
  getStrategies, validateDataset,
} from '../api/client';
import type {
  ModelInfo, PredictionResult, BatchPredRow, BatchPredResult,
  ShapResult, PredHistoryRow, RiskLevel, Strategy, DatasetValidation,
} from '../types';

// ─────────────────────────────────────────────────────────────────────────────
// Shared UI
// ─────────────────────────────────────────────────────────────────────────────
const riskColors: Record<RiskLevel, string> = {
  Low:    'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  Medium: 'bg-yellow-500/15  text-yellow-400  border-yellow-500/30',
  High:   'bg-red-500/15     text-red-400     border-red-500/30',
};

const RiskBadge = ({ level }: { level: RiskLevel }) => (
  <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold border ${riskColors[level]}`}>
    {level}
  </span>
);

const StatusBadge = ({ status }: { status: string }) =>
  status === 'Theft' ? (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-red-500/15 text-red-400 border border-red-500/30 rounded-full text-xs font-bold">
      <AlertTriangle className="w-3 h-3" /> Theft
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 rounded-full text-xs font-bold">
      <CheckCircle2 className="w-3 h-3" /> Normal
    </span>
  );

const ProbBar = ({ value, status }: { value: number; status: string }) => (
  <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
    <div className={`h-full rounded-full transition-all duration-700 ${status === 'Theft' ? 'bg-red-500' : 'bg-emerald-500'}`}
      style={{ width: `${value * 100}%` }} />
  </div>
);

const StatCard = ({ label, value, sub }: { label: string; value: string; sub?: string }) => (
  <div className="bg-slate-800/60 rounded-xl p-3 space-y-0.5">
    <p className="text-slate-400 text-xs">{label}</p>
    <p className="text-white font-bold text-base">{value}</p>
    {sub && <p className="text-slate-500 text-[10px]">{sub}</p>}
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// SHAP chart
// ─────────────────────────────────────────────────────────────────────────────
const ShapChart = ({ shap }: { shap: ShapResult }) => {
  const top10 = shap.feature_importance.slice(0, 10);
  const colors = top10.map(f => shap.top5_features.includes(f.feature) ? '#f59e0b' : '#3b82f6');
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-white font-bold text-sm">Feature Importance</h3>
          <p className="text-slate-500 text-xs mt-0.5">{shap.method}</p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="w-3 h-3 rounded-sm bg-yellow-500 inline-block" />
          <span className="text-slate-400">Top 5</span>
        </div>
      </div>
      <Plot
        data={[{
          type: 'bar', orientation: 'h',
          x: top10.map(f => f.importance * 100),
          y: top10.map(f => f.feature),
          marker: { color: colors },
          text: top10.map(f => `${(f.importance * 100).toFixed(1)}%`),
          textposition: 'auto',
        } as any]}
        layout={{
          paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
          font: { color: '#94a3b8', size: 10 },
          margin: { t: 5, b: 30, l: 55, r: 40 }, height: 240,
          xaxis: { gridcolor: '#1e293b', color: '#64748b', title: { text: 'Importance (%)' } },
          yaxis: { gridcolor: '#1e293b', color: '#64748b', autorange: 'reversed' },
          showlegend: false,
        }}
        config={{ displayModeBar: false }} style={{ width: '100%' }}
      />
      <p className="text-slate-500 text-xs mt-1">
        Top contributors: <span className="text-yellow-400 font-mono">{shap.top5_features.join(', ')}</span>
      </p>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Prediction result card
// ─────────────────────────────────────────────────────────────────────────────
const PredictionResultCard = ({ result }: { result: PredictionResult }) => {
  const isTheft = result.status === 'Theft';
  const riskLevel = (result.risk_level ?? (result.risk_score >= 75 ? 'High' : result.risk_score >= 40 ? 'Medium' : 'Low')) as RiskLevel;
  return (
    <div className={`border rounded-2xl p-5 space-y-4 ${isTheft ? 'bg-red-500/8 border-red-500/25' : 'bg-emerald-500/8 border-emerald-500/25'}`}>
      <div className="flex items-start gap-3">
        <div className={`p-2.5 rounded-xl shrink-0 ${isTheft ? 'bg-red-500/20' : 'bg-emerald-500/20'}`}>
          {isTheft ? <AlertTriangle className="w-6 h-6 text-red-400" /> : <CheckCircle2 className="w-6 h-6 text-emerald-400" />}
        </div>
        <div className="min-w-0 flex-1">
          <p className={`text-lg font-black ${isTheft ? 'text-red-400' : 'text-emerald-400'}`}>{result.label}</p>
          <p className="text-slate-400 text-xs truncate">Customer: <span className="font-mono text-slate-300">{result.customer_id}</span></p>
        </div>
        <StatusBadge status={result.status} />
      </div>

      <div className="grid grid-cols-2 gap-2">
        <StatCard label="Probability" value={`${(result.probability * 100).toFixed(2)}%`} sub={`Raw: ${result.probability.toFixed(6)}`} />
        <StatCard label="Confidence" value={`${(result.confidence * 100).toFixed(2)}%`} />
        <StatCard label="Risk Score" value={`${result.risk_score.toFixed(1)}/100`} />
        <div className="bg-slate-800/60 rounded-xl p-3 space-y-1">
          <p className="text-slate-400 text-xs">Risk Level</p>
          <RiskBadge level={riskLevel} />
        </div>
      </div>

      <div>
        <div className="flex justify-between text-xs text-slate-500 mb-1">
          <span>Risk Meter</span><span>{result.risk_score.toFixed(1)}/100</span>
        </div>
        <div className="h-2.5 bg-slate-700 rounded-full overflow-hidden">
          <div className={`h-full rounded-full transition-all duration-700 ${result.risk_score >= 75 ? 'bg-red-500' : result.risk_score >= 40 ? 'bg-yellow-500' : 'bg-emerald-500'}`}
            style={{ width: `${result.risk_score}%` }} />
        </div>
      </div>

      {result.predicted_at && (
        <div className="flex items-center gap-1.5 text-xs text-slate-600">
          <Clock className="w-3 h-3" />{new Date(result.predicted_at + 'Z').toLocaleString()}
        </div>
      )}

      <div className="bg-slate-900/60 rounded-lg px-3 py-2 font-mono text-[10px] text-slate-500 break-all">
        {result.predict_proof}
      </div>
      <p className="text-slate-600 text-[10px]">
        Model: <span className="text-blue-400">{result.model_name}</span>
        {result.sqlite_row_id && <> · SQLite row #{result.sqlite_row_id}</>}
      </p>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Single Customer tab — dynamic length
// ─────────────────────────────────────────────────────────────────────────────
const SingleTab: React.FC<{ modelInfo: ModelInfo | null; threshold: number; strategy: string }> = ({ modelInfo, threshold, strategy }) => {
  const modelLen = modelInfo?.seq_len_expected ?? null;
  const variable = modelInfo?.is_variable_length ?? (modelLen === null);
  // For fixed models use the model length; for variable models default to 26 (user-editable)
  const [seqLen, setSeqLen]       = useState<number>(modelLen ?? 26);
  const [customerId, setCustomerId] = useState('');
  const [readings, setReadings]   = useState<string[]>(Array(modelLen ?? 26).fill(''));
  const [errors, setErrors]       = useState<Record<number, string>>({});
  const [result, setResult]       = useState<PredictionResult | null>(null);
  const [loading, setLoading]     = useState(false);
  const [shap, setShap]           = useState<ShapResult | null>(null);
  const [shapLoading, setShapLoading] = useState(false);

  // Re-sync the form when the model's expected length changes
  useEffect(() => {
    if (modelLen != null) {
      setSeqLen(modelLen);
      setReadings(Array(modelLen).fill(''));
    }
  }, [modelLen]);

  const resizeForm = (n: number) => {
    const clamped = Math.max(2, Math.min(500, n || 2));
    setSeqLen(clamped);
    setReadings(prev => {
      const next = Array(clamped).fill('');
      for (let i = 0; i < Math.min(clamped, prev.length); i++) next[i] = prev[i];
      return next;
    });
  };

  const nums = readings.map(Number);
  const hasAnyValue = readings.some(r => r.trim() !== '');

  const validate = (): boolean => {
    const errs: Record<number, string> = {};
    if (!customerId.trim()) { toast.error('Customer ID is required'); return false; }
    readings.forEach((v, i) => {
      if (v.trim() === '') errs[i] = 'Required';
      else if (isNaN(Number(v))) errs[i] = 'NaN';
    });
    setErrors(errs);
    if (Object.keys(errs).length > 0) { toast.error(`Fix ${Object.keys(errs).length} invalid reading(s)`); return false; }
    return true;
  };

  const handlePredict = async () => {
    if (!validate()) return;
    setLoading(true); setResult(null); setShap(null);
    try {
      const res = await predictManual({ customer_id: customerId, readings: nums, threshold, strategy });
      setResult(res.data.result);
      toast.success(`Prediction: ${res.data.result.status}`);
      fetchShap(customerId);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Prediction failed');
    } finally { setLoading(false); }
  };

  const fetchShap = async (cid: string) => {
    setShapLoading(true);
    try { const res = await getShap(cid); setShap(res.data); } catch { /* optional */ } finally { setShapLoading(false); }
  };

  const fillRandom = (anomalous: boolean) => {
    const vals = Array.from({ length: seqLen }, (_, i) => {
      const base = 1200 + Math.sin(i / 3) * 200;
      if (anomalous && i % 4 === 0) return '0';
      return String(Math.round(base + (Math.random() - 0.5) * 150));
    });
    setReadings(vals);
    setCustomerId(anomalous ? 'DEMO_THEFT_001' : 'DEMO_NORMAL_001');
    setErrors({}); setResult(null); setShap(null);
  };

  const clear = () => { setCustomerId(''); setReadings(Array(seqLen).fill('')); setErrors({}); setResult(null); setShap(null); };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
      <div className="xl:col-span-3 space-y-5">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-5">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div>
              <h2 className="text-white font-bold">Customer Data Entry</h2>
              <p className="text-slate-500 text-xs mt-0.5">
                {variable
                  ? `Model accepts variable length — enter any number of readings`
                  : `Model expects ${modelLen} readings`}
              </p>
            </div>
            <div className="flex gap-2 flex-wrap justify-end">
              <button onClick={() => fillRandom(false)} className="px-3 py-1.5 bg-emerald-500/15 text-emerald-400 border border-emerald-500/20 rounded-lg text-xs font-semibold hover:bg-emerald-500/25">Normal Demo</button>
              <button onClick={() => fillRandom(true)} className="px-3 py-1.5 bg-red-500/15 text-red-400 border border-red-500/20 rounded-lg text-xs font-semibold hover:bg-red-500/25">Theft Demo</button>
              <button onClick={clear} className="px-3 py-1.5 bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold hover:bg-slate-600">Clear</button>
            </div>
          </div>

          <div>
            <label className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-1.5 block">
              Customer ID <span className="text-red-400">*</span>
            </label>
            <input value={customerId} onChange={e => setCustomerId(e.target.value)}
              placeholder="e.g. A0E791400CF1C48C43DC26A68227854A"
              className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 text-white rounded-xl text-sm font-mono focus:outline-none focus:border-blue-500 placeholder:text-slate-600" />
          </div>

          {/* Variable-length: let user set count */}
          {variable && (
            <div className="flex items-center gap-3">
              <label className="text-slate-400 text-sm whitespace-nowrap">Number of readings:</label>
              <input type="number" min={2} max={500} value={seqLen}
                onChange={e => resizeForm(Number(e.target.value))}
                className="w-24 px-3 py-1.5 bg-slate-800 border border-slate-700 text-white rounded-lg text-sm font-mono focus:outline-none focus:border-blue-500" />
            </div>
          )}

          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
                {seqLen} Consumption Readings (kWh) <span className="text-red-400">*</span>
              </label>
              <span className="text-slate-600 text-xs">{readings.filter(r => r.trim() !== '').length}/{seqLen} filled</span>
            </div>
            <div className="grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-8 gap-2 max-h-72 overflow-y-auto pr-1">
              {readings.map((v, i) => (
                <div key={i} className="space-y-0.5">
                  <p className="text-slate-600 text-[9px] text-center">T{i + 1}</p>
                  <input type="number" value={v}
                    onChange={e => { const nx = [...readings]; nx[i] = e.target.value; setReadings(nx); setErrors(p => { const n = { ...p }; delete n[i]; return n; }); }}
                    placeholder="kWh"
                    className={`w-full px-1.5 py-2 text-xs text-center font-mono rounded-lg focus:outline-none transition-colors
                      ${errors[i] ? 'bg-red-500/10 border border-red-500/50 text-red-400' : 'bg-slate-800 border border-slate-700 text-white focus:border-blue-500'}`} />
                </div>
              ))}
            </div>
          </div>

          <button onClick={handlePredict} disabled={loading}
            className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white font-bold flex items-center justify-center gap-2 transition-colors shadow-lg shadow-blue-500/20">
            {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Running model.predict()…</> : <><Zap className="w-4 h-4" />Predict Now</>}
          </button>

          <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl px-4 py-2.5 font-mono text-[10px] text-slate-500">
            <span className="text-slate-400">Proof: </span>
            model.predict(x) · uploaded_len={seqLen} · model_len={modelLen ?? 'variable'} · strategy={modelLen && seqLen !== modelLen ? strategy : 'none'}
          </div>
        </div>

        {hasAnyValue && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <h3 className="text-white font-bold text-sm mb-3">Live Consumption Pattern</h3>
            <Plot
              data={[{
                type: 'scatter', mode: 'lines+markers',
                x: readings.map((_, i) => `T${i + 1}`), y: nums,
                line: { color: result?.status === 'Theft' ? '#ef4444' : '#3b82f6', width: 2 },
                marker: { size: 4 }, fill: 'tozeroy',
                fillcolor: result?.status === 'Theft' ? 'rgba(239,68,68,0.08)' : 'rgba(59,130,246,0.08)',
              } as any]}
              layout={{
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', size: 10 }, margin: { t: 8, b: 40, l: 48, r: 10 }, height: 180,
                xaxis: { gridcolor: '#1e293b', color: '#64748b' },
                yaxis: { gridcolor: '#1e293b', color: '#64748b', title: { text: 'kWh' } },
                showlegend: false,
              }}
              config={{ displayModeBar: false }} style={{ width: '100%' }} />
          </div>
        )}
      </div>

      <div className="xl:col-span-2 space-y-4">
        {!result && !loading && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center">
            <Shield className="w-10 h-10 text-slate-700 mx-auto mb-3" />
            <p className="text-slate-500 text-sm">Fill in the readings and click <strong className="text-slate-400">Predict Now</strong>.</p>
          </div>
        )}
        {loading && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center">
            <Loader2 className="w-8 h-8 text-blue-400 animate-spin mx-auto mb-3" />
            <p className="text-slate-400 text-sm font-semibold">Running CNN-LSTM Inference…</p>
          </div>
        )}
        {result && <PredictionResultCard result={result} />}
        {shapLoading && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex items-center gap-3">
            <Loader2 className="w-4 h-4 text-yellow-400 animate-spin" />
            <p className="text-slate-400 text-sm">Computing feature importance…</p>
          </div>
        )}
        {shap && !shapLoading && <ShapChart shap={shap} />}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Batch CSV tab — server-side validation + strategy
// ─────────────────────────────────────────────────────────────────────────────
const BatchTab: React.FC<{ modelInfo: ModelInfo | null; threshold: number; strategy: string; setStrategy: (s: string) => void; strategies: Strategy[] }> = ({ modelInfo, threshold: defThresh, strategy, setStrategy, strategies }) => {
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile]           = useState<File | null>(null);
  const [validation, setValidation] = useState<DatasetValidation | null>(null);
  const [validating, setValidating] = useState(false);
  const [batchThresh, setBatchThresh] = useState(defThresh);
  const [loading, setLoading]     = useState(false);
  const [progress, setProgress]   = useState(0);
  const [result, setResult]       = useState<BatchPredResult | null>(null);
  const [search, setSearch]       = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [sortCol, setSortCol]     = useState<keyof BatchPredRow>('risk_score');
  const [sortDir, setSortDir]     = useState<'asc' | 'desc'>('desc');
  const [page, setPage]           = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const PAGE_SIZE = 50;

  useEffect(() => {
    if (!loading) { setProgress(0); return; }
    setProgress(5);
    const t = [
      setTimeout(() => setProgress(25), 800), setTimeout(() => setProgress(45), 2500),
      setTimeout(() => setProgress(65), 6000), setTimeout(() => setProgress(80), 14000),
      setTimeout(() => setProgress(90), 28000),
    ];
    return () => t.forEach(clearTimeout);
  }, [loading]);

  // Re-validate when strategy changes (so compatibility text updates)
  const runValidation = useCallback(async (f: File, strat: string) => {
    setValidating(true); setResult(null);
    try {
      const res = await validateDataset(f, strat);
      setValidation(res.data);
      if (res.data.status !== 'Ready') toast.error(res.data.compatibility?.reason || 'Dataset incompatible');
    } catch (e: any) {
      setValidation(null);
      toast.error(e.response?.data?.detail || 'Validation failed');
    } finally { setValidating(false); }
  }, []);

  const handleFile = (f: File) => {
    if (!f.name.toLowerCase().endsWith('.csv')) { toast.error('Only CSV files accepted'); return; }
    setFile(f); runValidation(f, strategy);
  };

  const onStrategyChange = (s: string) => {
    setStrategy(s);
    if (file) runValidation(file, s);
  };

  const handleDownloadTemplate = async () => {
    try {
      const res = await downloadTemplate();
      const url = URL.createObjectURL(res.data);
      const a = document.createElement('a'); a.href = url; a.download = 'etd_template.csv'; a.click(); URL.revokeObjectURL(url);
    } catch { toast.error('Template download failed'); }
  };

  const handleBatchPredict = async () => {
    if (!file || !validation) return;
    setLoading(true); setResult(null);
    try {
      const res = await predictBatchPreview(file, batchThresh, strategy);
      setResult(res.data); setPage(1); setProgress(100);
      toast.success(`${res.data.total.toLocaleString()} predictions in ${res.data.elapsed_seconds}s`);
    } catch (e: any) { toast.error(e.response?.data?.detail || 'Batch prediction failed'); }
    finally { setLoading(false); }
  };

  const handleSave = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const res = await predictBatchStore(file, batchThresh, strategy, file.name.replace('.csv', ''));
      setProgress(100);
      toast.success(`Saved ${res.data.total.toLocaleString()} customers to Dashboard!`);
      setResult(r => r ? { ...r, ...res.data, stored: true } : null);
    } catch (e: any) { toast.error(e.response?.data?.detail || 'Save failed'); }
    finally { setLoading(false); }
  };

  const exportCsv = () => {
    if (!result) return;
    const header = 'customer_id,status,probability,confidence,risk_score,risk_level,flag';
    const rows = result.predictions.map(r =>
      `${r.customer_id},${r.status},${r.probability},${(r.confidence * 100).toFixed(2)}%,${r.risk_score},${r.risk_level},${r.flag ?? ''}`);
    const blob = new Blob([header + '\n' + rows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = `etd_results_${Date.now()}.csv`; a.click(); URL.revokeObjectURL(url);
  };

  const filtered = (result?.predictions ?? []).filter(r =>
    (!search || r.customer_id.toLowerCase().includes(search.toLowerCase())) &&
    (!statusFilter || r.status === statusFilter));
  const sorted = [...filtered].sort((a, b) => {
    const av = a[sortCol] as any, bv = b[sortCol] as any;
    return sortDir === 'desc' ? (bv > av ? 1 : -1) : (av > bv ? 1 : -1);
  });
  const totalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const pageData = sorted.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const toggleSort = (c: keyof BatchPredRow) => { if (sortCol === c) setSortDir(d => d === 'desc' ? 'asc' : 'desc'); else { setSortCol(c); setSortDir('desc'); } };
  const SortIcon = ({ c }: { c: keyof BatchPredRow }) => sortCol === c ? (sortDir === 'desc' ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />) : <ChevronDown className="w-3 h-3 opacity-30" />;

  const compat = validation?.compatibility;

  return (
    <div className="space-y-5">
      {/* Upload + template */}
      <div className="flex gap-4 items-stretch flex-wrap">
        <div
          className={`flex-1 min-w-64 border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all
            ${isDragging ? 'border-blue-400 bg-blue-500/10' : file ? 'border-emerald-500/40 bg-emerald-500/5' : 'border-slate-700 hover:border-slate-500 bg-slate-900'}`}
          onDragEnter={e => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={e => { e.preventDefault(); setIsDragging(false); }}
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); setIsDragging(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
          onClick={() => fileRef.current?.click()}>
          <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
          {file ? (
            <div className="flex flex-col items-center gap-2">
              <FileText className="w-8 h-8 text-emerald-400" />
              <p className="text-emerald-400 font-semibold text-sm">{file.name}</p>
              <p className="text-slate-500 text-xs">{(file.size / 1024).toFixed(1)} KB · Click to change</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload className="w-8 h-8 text-slate-500" />
              <p className="text-slate-300 font-semibold text-sm">Drop CSV here or click to browse</p>
              <p className="text-slate-600 text-xs">Any number of reading columns — auto-detected</p>
            </div>
          )}
        </div>
        <button onClick={handleDownloadTemplate}
          className="flex flex-col items-center justify-center gap-2 px-5 py-4 bg-slate-900 border border-slate-700 rounded-2xl hover:border-blue-500/50 hover:bg-slate-800 text-slate-300 hover:text-white group">
          <Download className="w-6 h-6 group-hover:text-blue-400" />
          <span className="text-xs font-semibold whitespace-nowrap">Download<br />Template</span>
        </button>
      </div>

      {validating && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex items-center gap-3">
          <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
          <p className="text-slate-400 text-sm">Validating dataset & checking model compatibility…</p>
        </div>
      )}

      {/* Validation report */}
      {validation && !validating && (
        <div className={`rounded-2xl border p-4 space-y-4 ${validation.status === 'Ready' ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-red-500/5 border-red-500/20'}`}>
          <div className="flex items-center gap-3">
            {validation.status === 'Ready' ? <CheckCircle2 className="w-5 h-5 text-emerald-400" /> : <AlertTriangle className="w-5 h-5 text-red-400" />}
            <div>
              <p className={`font-semibold text-sm ${validation.status === 'Ready' ? 'text-emerald-400' : 'text-red-400'}`}>
                {validation.dataset_name} — {validation.status}
              </p>
              {compat && <p className="text-slate-400 text-xs">{compat.reason}</p>}
            </div>
          </div>

          {/* Stat grid */}
          <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 text-center">
            {[
              { l: 'Customers', v: validation.n_customers.toLocaleString() },
              { l: 'Reading Cols', v: String(validation.n_reading_cols) },
              { l: 'Detected Len', v: String(validation.detected_seq_len) },
              { l: 'Model Len', v: compat?.expected_len == null ? 'variable' : String(compat.expected_len) },
              { l: 'Missing', v: String(validation.missing_values) },
              { l: 'Duplicates', v: String(validation.duplicate_customers) },
            ].map(({ l, v }) => (
              <div key={l}><p className="text-white font-bold">{v}</p><p className="text-slate-500 text-xs">{l}</p></div>
            ))}
          </div>

          <div className="flex flex-wrap gap-3 text-xs text-slate-400">
            <span>ID column: <span className="text-slate-200 font-mono">{validation.id_column ?? 'auto'}</span></span>
            <span>Ground truth: <span className={validation.ground_truth_column ? 'text-blue-400 font-mono' : 'text-slate-600'}>{validation.ground_truth_column ?? 'none'}</span></span>
            {compat?.preprocessing_needed && <span className="text-yellow-400">Preprocessing required → {strategy}</span>}
          </div>

          {/* Preview */}
          {validation.preview.length > 0 && (
            <div className="overflow-x-auto">
              <p className="text-slate-500 text-xs mb-2">Preview (first {validation.preview.length} customers):</p>
              <table className="w-full text-xs border-collapse">
                <thead><tr className="border-b border-slate-700">
                  {validation.preview_columns.map(c => <th key={c} className="px-2 py-1.5 text-left text-slate-400 font-semibold">{c}</th>)}
                </tr></thead>
                <tbody>
                  {validation.preview.map((row, i) => (
                    <tr key={i} className="border-b border-slate-800/50">
                      {validation.preview_columns.map(c => <td key={c} className="px-2 py-1.5 text-slate-300 font-mono">{row[c] ?? '—'}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Controls */}
      {validation?.compatibility?.compatible && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
          <div className="grid sm:grid-cols-2 gap-4">
            <div className="flex items-center gap-3">
              <Settings2 className="w-4 h-4 text-slate-500 shrink-0" />
              <label className="text-slate-400 text-sm whitespace-nowrap">Strategy:</label>
              <select value={strategy} onChange={e => onStrategyChange(e.target.value)}
                className="flex-1 bg-slate-800 border border-slate-700 text-slate-200 text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500">
                {strategies.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
              </select>
            </div>
            <div className="flex items-center gap-3">
              <label className="text-slate-400 text-sm whitespace-nowrap">Threshold:</label>
              <input type="range" min="0" max="1" step="0.01" value={batchThresh} onChange={e => setBatchThresh(Number(e.target.value))} className="flex-1 accent-blue-500" />
              <span className="text-blue-400 font-mono font-bold w-10 text-right">{batchThresh.toFixed(2)}</span>
            </div>
          </div>

          {loading && (
            <div>
              <div className="flex justify-between text-xs text-slate-400 mb-1.5">
                <span>Running model.predict() (async)…</span><span>{progress}%</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full transition-all duration-1000" style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}

          <button onClick={handleBatchPredict} disabled={loading || !file}
            className="w-full py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white font-bold flex items-center justify-center gap-2 text-sm">
            {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Predicting…</> : <><Zap className="w-4 h-4" />Run Batch Prediction</>}
          </button>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-center"><p className="text-2xl font-black text-white">{result.total.toLocaleString()}</p><p className="text-slate-400 text-xs">Total</p></div>
            <div className="bg-red-500/8 border border-red-500/20 rounded-xl p-4 text-center"><p className="text-2xl font-black text-red-400">{result.theft.toLocaleString()}</p><p className="text-slate-400 text-xs">Theft</p></div>
            <div className="bg-emerald-500/8 border border-emerald-500/20 rounded-xl p-4 text-center"><p className="text-2xl font-black text-emerald-400">{result.normal.toLocaleString()}</p><p className="text-slate-400 text-xs">Normal</p></div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-center"><p className="text-2xl font-black text-yellow-400">{(result.avg_confidence * 100).toFixed(1)}%</p><p className="text-slate-400 text-xs">Avg Conf</p></div>
          </div>

          <div className="flex items-center justify-between flex-wrap gap-2">
            <p className="text-slate-500 text-xs font-mono">{result.predict_proof}</p>
            <p className="text-slate-500 text-xs">Elapsed <span className="text-white font-mono">{result.elapsed_seconds}s</span> · {result.model_name}</p>
          </div>

          <div className="flex gap-3 flex-wrap">
            <button onClick={exportCsv} className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl text-sm text-slate-300 font-semibold"><Download className="w-4 h-4" />Export CSV</button>
            {!result.stored ? (
              <button onClick={handleSave} disabled={loading} className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 rounded-xl text-sm text-white font-semibold">
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Database className="w-4 h-4" />}Save to Customer Predictions
              </button>
            ) : (
              <div className="flex items-center gap-2 px-4 py-2 bg-emerald-500/15 border border-emerald-500/30 rounded-xl text-sm text-emerald-400 font-semibold"><CheckCircle2 className="w-4 h-4" />Saved to Dashboard</div>
            )}
          </div>

          {/* Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="flex items-center gap-3 p-4 border-b border-slate-800 flex-wrap">
              <div className="flex items-center gap-2 flex-1 min-w-40">
                <Search className="w-4 h-4 text-slate-500" />
                <input value={search} onChange={e => { setSearch(e.target.value); setPage(1); }} placeholder="Search customer ID…"
                  className="flex-1 bg-transparent text-white text-sm focus:outline-none placeholder:text-slate-600" />
                {search && <button onClick={() => setSearch('')}><X className="w-3.5 h-3.5 text-slate-500" /></button>}
              </div>
              <select value={statusFilter} onChange={e => { setStatusFilter(e.target.value); setPage(1); }}
                className="bg-slate-800 border border-slate-700 text-slate-300 text-xs rounded-lg px-3 py-1.5 focus:outline-none">
                <option value="">All Status</option><option value="Theft">Theft</option><option value="Normal">Normal</option>
              </select>
              <span className="text-slate-500 text-xs ml-auto">{filtered.length.toLocaleString()} results</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead><tr className="border-b border-slate-800">
                  {[['customer_id', 'Customer ID'], ['status', 'Status'], ['probability', 'Probability'], ['confidence', 'Confidence'], ['risk_score', 'Risk'], ['risk_level', 'Level']].map(([k, l]) => (
                    <th key={k} onClick={() => toggleSort(k as keyof BatchPredRow)} className="px-4 py-3 text-left text-slate-400 text-xs font-semibold cursor-pointer hover:text-white select-none">
                      <div className="flex items-center gap-1">{l} <SortIcon c={k as keyof BatchPredRow} /></div>
                    </th>
                  ))}
                </tr></thead>
                <tbody>
                  {pageData.map((row, i) => (
                    <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                      <td className="px-4 py-2.5 font-mono text-xs text-slate-300 max-w-32 truncate">{row.customer_id}</td>
                      <td className="px-4 py-2.5"><StatusBadge status={row.status} /></td>
                      <td className="px-4 py-2.5"><div className="space-y-1"><p className="text-white text-xs font-mono">{(row.probability * 100).toFixed(2)}%</p><ProbBar value={row.probability} status={row.status} /></div></td>
                      <td className="px-4 py-2.5 text-slate-300 text-xs font-mono">{(row.confidence * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2.5 text-slate-300 text-xs font-mono">{row.risk_score.toFixed(1)}</td>
                      <td className="px-4 py-2.5"><RiskBadge level={row.risk_level} /></td>
                    </tr>
                  ))}
                  {pageData.length === 0 && <tr><td colSpan={6} className="px-4 py-8 text-center text-slate-500 text-sm">No results.</td></tr>}
                </tbody>
              </table>
            </div>
            {totalPages > 1 && (
              <div className="flex items-center justify-between px-4 py-3 border-t border-slate-800">
                <span className="text-slate-500 text-xs">Page {page} of {totalPages}</span>
                <div className="flex items-center gap-1">
                  <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} className="p-1.5 rounded-lg hover:bg-slate-700 disabled:opacity-30 text-slate-400"><ChevronLeft className="w-4 h-4" /></button>
                  <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages} className="p-1.5 rounded-lg hover:bg-slate-700 disabled:opacity-30 text-slate-400"><ChevronRight className="w-4 h-4" /></button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Prediction history
// ─────────────────────────────────────────────────────────────────────────────
const PredictionHistory: React.FC = () => {
  const [history, setHistory] = useState<PredHistoryRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const load = useCallback(async () => {
    setLoading(true);
    try { const res = await getPredictHistory(50); setHistory(res.data.history); } catch { /* optional */ } finally { setLoading(false); }
  }, []);
  useEffect(() => { load(); }, [load]);
  if (history.length === 0 && !loading) return null;
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
      <button onClick={() => setExpanded(e => !e)} className="w-full flex items-center justify-between px-5 py-4 hover:bg-slate-800/50">
        <div className="flex items-center gap-3">
          <Clock className="w-4 h-4 text-slate-400" />
          <span className="text-white font-bold">Prediction History</span>
          <span className="bg-slate-700 text-slate-300 text-xs px-2 py-0.5 rounded-full font-semibold">{history.length}</span>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={e => { e.stopPropagation(); load(); }} className="p-1 rounded hover:bg-slate-700"><RefreshCw className={`w-3.5 h-3.5 text-slate-400 ${loading ? 'animate-spin' : ''}`} /></button>
          {expanded ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </div>
      </button>
      {expanded && (
        <div className="border-t border-slate-800 overflow-x-auto">
          <table className="w-full text-xs">
            <thead><tr className="border-b border-slate-800">{['#', 'Customer ID', 'Status', 'Probability', 'Level', 'Source', 'Time'].map(h => <th key={h} className="px-4 py-2.5 text-left text-slate-400 font-semibold">{h}</th>)}</tr></thead>
            <tbody>
              {history.map(row => {
                const lvl: RiskLevel = row.risk_score >= 75 ? 'High' : row.risk_score >= 40 ? 'Medium' : 'Low';
                return (
                  <tr key={row.id} className="border-b border-slate-800/40 hover:bg-slate-800/30">
                    <td className="px-4 py-2.5 text-slate-600">#{row.id}</td>
                    <td className="px-4 py-2.5 font-mono text-slate-300 max-w-32 truncate">{row.customer_id || '—'}</td>
                    <td className="px-4 py-2.5"><StatusBadge status={row.status} /></td>
                    <td className="px-4 py-2.5 font-mono text-slate-300">{(row.probability * 100).toFixed(2)}%</td>
                    <td className="px-4 py-2.5"><RiskBadge level={lvl} /></td>
                    <td className="px-4 py-2.5"><span className="px-1.5 py-0.5 bg-slate-800 border border-slate-700 rounded text-slate-400">{row.source}</span></td>
                    <td className="px-4 py-2.5 text-slate-500 whitespace-nowrap">{new Date(row.predicted_at + 'Z').toLocaleString()}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Main page
// ─────────────────────────────────────────────────────────────────────────────
const PredictPage: React.FC<{ modelInfo: ModelInfo | null; threshold: number }> = ({ modelInfo, threshold }) => {
  const [activeTab, setActiveTab] = useState<'single' | 'batch'>('single');
  const [strategy, setStrategy]   = useState('last_n');
  const [strategies, setStrategies] = useState<Strategy[]>([]);

  useEffect(() => {
    getStrategies().then(r => setStrategies(r.data.strategies)).catch(() => {});
  }, []);

  const modelLen = modelInfo?.seq_len_expected;

  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-black text-white">Prediction Center</h1>
        <p className="text-slate-400 text-sm mt-0.5">
          Dynamic time-series — sequence length & features discovered from the model. Every prediction runs <span className="text-blue-400 font-mono">model.predict()</span>.
        </p>
      </div>

      {/* Model status */}
      <div className="flex items-center gap-2 flex-wrap bg-slate-900 border border-slate-800 rounded-xl px-4 py-2.5 text-xs">
        <div className="flex items-center gap-1.5">
          <div className={`w-2 h-2 rounded-full ${modelInfo?.loaded ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
          <span className="text-slate-400">Model:</span>
          <span className="text-white font-bold font-mono">{modelInfo?.model_name ?? 'Not loaded'}</span>
        </div>
        <span className="text-slate-700">|</span>
        <span className="text-slate-400">Input:</span>
        <span className="text-yellow-400 font-mono">{modelInfo?.input_shape ?? '—'}</span>
        <span className="text-slate-700">|</span>
        <span className="text-slate-400">Seq len:</span>
        <span className="text-yellow-400 font-mono">{modelLen == null ? 'variable' : modelLen}</span>
        {modelInfo?.is_dual_input && (
          <><span className="text-slate-700">|</span><span className="text-purple-400 font-semibold">+{modelInfo.stat_input_size} stat features</span></>
        )}
        {modelInfo?.tf_version && (
          <><span className="text-slate-700">|</span><span className="text-slate-500">TF {modelInfo.tf_version} · Keras {modelInfo.keras_version}</span></>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 p-1 bg-slate-900 border border-slate-800 rounded-xl w-fit">
        <button onClick={() => setActiveTab('single')} className={`flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold transition-all ${activeTab === 'single' ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}>
          <Zap className="w-4 h-4" />Single Customer
        </button>
        <button onClick={() => setActiveTab('batch')} className={`flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold transition-all ${activeTab === 'batch' ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}>
          <BarChart2 className="w-4 h-4" />Batch CSV Upload
        </button>
      </div>

      {activeTab === 'single'
        ? <SingleTab modelInfo={modelInfo} threshold={threshold} strategy={strategy} />
        : <BatchTab modelInfo={modelInfo} threshold={threshold} strategy={strategy} setStrategy={setStrategy} strategies={strategies} />}

      <PredictionHistory />
    </div>
  );
};

export default PredictPage;
