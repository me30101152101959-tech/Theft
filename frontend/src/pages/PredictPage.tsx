import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Zap, AlertTriangle, CheckCircle2, Loader2, Upload, Download,
  Search, ChevronUp, ChevronDown, FileText, BarChart2, Clock,
  RefreshCw, Database, Shield, X, Info, ChevronLeft, ChevronRight,
} from 'lucide-react';
import Plot from 'react-plotly.js';
import toast from 'react-hot-toast';

import {
  predictManual, predictBatchPreview, predictBatchStore,
  getShap, getPredictHistory, downloadTemplate,
} from '../api/client';
import type {
  ModelInfo, PredictionResult, BatchPredRow, BatchPredResult,
  ShapResult, PredHistoryRow, CsvPreview, RiskLevel,
} from '../types';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
const READING_LABELS = [
  'Jan-Y1','Feb-Y1','Mar-Y1','Apr-Y1','May-Y1','Jun-Y1',
  'Jul-Y1','Aug-Y1','Sep-Y1','Oct-Y1','Nov-Y1','Dec-Y1',
  'Jan-Y2','Feb-Y2','Mar-Y2','Apr-Y2','May-Y2','Jun-Y2',
  'Jul-Y2','Aug-Y2','Sep-Y2','Oct-Y2','Nov-Y2','Dec-Y2',
  'W1','W2',
];

const TOOLTIPS: Record<string, string> = {
  'Jan-Y1': 'January Year 1 electricity consumption (kWh)',
  'Feb-Y1': 'February Year 1 electricity consumption (kWh)',
  'Mar-Y1': 'March Year 1 electricity consumption (kWh)',
  'Apr-Y1': 'April Year 1 electricity consumption (kWh)',
  'May-Y1': 'May Year 1 electricity consumption (kWh)',
  'Jun-Y1': 'June Year 1 electricity consumption (kWh)',
  'Jul-Y1': 'July Year 1 electricity consumption (kWh)',
  'Aug-Y1': 'August Year 1 electricity consumption (kWh)',
  'Sep-Y1': 'September Year 1 electricity consumption (kWh)',
  'Oct-Y1': 'October Year 1 electricity consumption (kWh)',
  'Nov-Y1': 'November Year 1 electricity consumption (kWh)',
  'Dec-Y1': 'December Year 1 electricity consumption (kWh)',
  'Jan-Y2': 'January Year 2 electricity consumption (kWh)',
  'Feb-Y2': 'February Year 2 electricity consumption (kWh)',
  'Mar-Y2': 'March Year 2 electricity consumption (kWh)',
  'Apr-Y2': 'April Year 2 electricity consumption (kWh)',
  'May-Y2': 'May Year 2 electricity consumption (kWh)',
  'Jun-Y2': 'June Year 2 electricity consumption (kWh)',
  'Jul-Y2': 'July Year 2 electricity consumption (kWh)',
  'Aug-Y2': 'August Year 2 electricity consumption (kWh)',
  'Sep-Y2': 'September Year 2 electricity consumption (kWh)',
  'Oct-Y2': 'October Year 2 electricity consumption (kWh)',
  'Nov-Y2': 'November Year 2 electricity consumption (kWh)',
  'Dec-Y2': 'December Year 2 electricity consumption (kWh)',
  'W1':     'Weekly reading 1 — supplementary billing period (kWh)',
  'W2':     'Weekly reading 2 — supplementary billing period (kWh)',
};

const DEMO_THEFT  = ['2400','2500','0','100','2200','0','2800','50','2100','2300','0','1100','1000','3200','2200','2400','0','2400','1800','2100','0','2400','2500','1200','0','2300'];
const DEMO_NORMAL = ['1200','1250','1180','1300','1220','1190','1280','1240','1210','1260','1230','1150','1170','1310','1195','1243','1215','1195','1278','1089','1300','1240','1265','1177','1190','1272'];

// ─────────────────────────────────────────────────────────────────────────────
// Shared UI components
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
    <div
      className={`h-full rounded-full transition-all duration-700 ${status === 'Theft' ? 'bg-red-500' : 'bg-emerald-500'}`}
      style={{ width: `${value * 100}%` }}
    />
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
// SHAP Chart
// ─────────────────────────────────────────────────────────────────────────────
const ShapChart = ({ shap }: { shap: ShapResult }) => {
  const top10 = shap.feature_importance.slice(0, 10);
  const colors = top10.map(f =>
    shap.top5_features.includes(f.feature) ? '#f59e0b' : '#3b82f6'
  );
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
          <span className="w-3 h-3 rounded-sm bg-blue-500 inline-block ml-2" />
          <span className="text-slate-400">Other</span>
        </div>
      </div>
      <Plot
        data={[{
          type:        'bar',
          orientation: 'h',
          x:    top10.map(f => f.importance * 100),
          y:    top10.map(f => f.feature),
          marker: { color: colors },
          text: top10.map(f => `${(f.importance * 100).toFixed(1)}%`),
          textposition: 'auto',
          hovertemplate: '<b>%{y}</b><br>Importance: %{x:.2f}%<br>Value: %{customdata:.1f} kWh<extra></extra>',
          customdata: top10.map(f => f.value),
        }]}
        layout={{
          paper_bgcolor: 'transparent',
          plot_bgcolor:  'transparent',
          font:   { color: '#94a3b8', size: 10 },
          margin: { t: 5, b: 30, l: 65, r: 40 },
          height: 240,
          xaxis:  { gridcolor: '#1e293b', color: '#64748b', title: { text: 'Importance (%)' } },
          yaxis:  { gridcolor: '#1e293b', color: '#64748b', autorange: 'reversed' },
          showlegend: false,
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
      <p className="text-slate-500 text-xs mt-1">
        Top contributors: <span className="text-yellow-400 font-mono">{shap.top5_features.join(', ')}</span>
      </p>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Single Customer Prediction Tab
// ─────────────────────────────────────────────────────────────────────────────
interface SingleTabProps { modelInfo: ModelInfo | null; threshold: number; }

const SingleTab: React.FC<SingleTabProps> = ({ modelInfo, threshold }) => {
  const [customerId, setCustomerId]   = useState('');
  const [readings,   setReadings]     = useState<string[]>(Array(26).fill(''));
  const [errors,     setErrors]       = useState<Record<number, string>>({});
  const [result,     setResult]       = useState<PredictionResult | null>(null);
  const [loading,    setLoading]      = useState(false);
  const [shap,       setShap]         = useState<ShapResult | null>(null);
  const [shapLoading,setShapLoading]  = useState(false);

  const nums = readings.map(Number);
  const hasAnyValue = readings.some(r => r.trim() !== '');

  const validate = (): boolean => {
    const errs: Record<number, string> = {};
    if (!customerId.trim()) { toast.error('Customer ID is required'); return false; }
    readings.forEach((v, i) => {
      if (v.trim() === '') { errs[i] = 'Required'; }
      else if (isNaN(Number(v))) { errs[i] = 'Must be a number'; }
    });
    setErrors(errs);
    if (Object.keys(errs).length > 0) {
      toast.error(`Fix ${Object.keys(errs).length} invalid reading(s)`);
      return false;
    }
    return true;
  };

  const handlePredict = async () => {
    if (!validate()) return;
    setLoading(true);
    setResult(null);
    setShap(null);
    try {
      const res = await predictManual({ customer_id: customerId, readings: nums, threshold });
      const r   = res.data.result as PredictionResult;
      setResult(r);
      toast.success(`Prediction: ${r.status}`);
      // Auto-fetch SHAP
      fetchShap(customerId);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const fetchShap = async (cid: string) => {
    setShapLoading(true);
    try {
      const res = await getShap(cid);
      setShap(res.data);
    } catch {
      // SHAP is optional — silently ignore
    } finally {
      setShapLoading(false);
    }
  };

  const fillDemo = (type: 'theft' | 'normal') => {
    setReadings(type === 'theft' ? DEMO_THEFT : DEMO_NORMAL);
    setCustomerId(type === 'theft' ? 'DEMO_THEFT_001' : 'DEMO_NORMAL_001');
    setErrors({});
    setResult(null);
    setShap(null);
  };

  const clear = () => {
    setCustomerId('');
    setReadings(Array(26).fill(''));
    setErrors({});
    setResult(null);
    setShap(null);
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
      {/* ── Left: Input form ── */}
      <div className="xl:col-span-3 space-y-5">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-5">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-white font-bold">Customer Data Entry</h2>
              <p className="text-slate-500 text-xs mt-0.5">Enter all 26 monthly + weekly readings in kWh</p>
            </div>
            <div className="flex gap-2 flex-wrap justify-end">
              <button onClick={() => fillDemo('normal')}
                className="px-3 py-1.5 bg-emerald-500/15 text-emerald-400 border border-emerald-500/20 rounded-lg text-xs font-semibold hover:bg-emerald-500/25 transition-colors">
                Normal Demo
              </button>
              <button onClick={() => fillDemo('theft')}
                className="px-3 py-1.5 bg-red-500/15 text-red-400 border border-red-500/20 rounded-lg text-xs font-semibold hover:bg-red-500/25 transition-colors">
                Theft Demo
              </button>
              <button onClick={clear}
                className="px-3 py-1.5 bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold hover:bg-slate-600 transition-colors">
                Clear
              </button>
            </div>
          </div>

          {/* Customer ID */}
          <div>
            <label className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-1.5 block">
              Customer ID <span className="text-red-400">*</span>
            </label>
            <input
              value={customerId}
              onChange={e => setCustomerId(e.target.value)}
              placeholder="e.g. A0E791400CF1C48C43DC26A68227854A"
              className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 text-white rounded-xl text-sm font-mono focus:outline-none focus:border-blue-500 placeholder:text-slate-600 transition-colors"
            />
          </div>

          {/* 26 Readings Grid */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
                26 Electricity Consumption Readings (kWh) <span className="text-red-400">*</span>
              </label>
              <span className="text-slate-600 text-xs">
                {readings.filter(r => r.trim() !== '').length}/26 filled
              </span>
            </div>

            {/* Year 1 */}
            <p className="text-slate-600 text-[10px] font-semibold uppercase mb-1">Year 1 — Monthly</p>
            <div className="grid grid-cols-4 sm:grid-cols-6 gap-2 mb-3">
              {READING_LABELS.slice(0, 12).map((label, i) => (
                <ReadingInput key={i} index={i} label={label} value={readings[i]}
                  error={errors[i]}
                  onChange={(v) => { const next=[...readings]; next[i]=v; setReadings(next); setErrors(p=>{ const n={...p}; delete n[i]; return n; }); }}
                  tooltip={TOOLTIPS[label]} />
              ))}
            </div>

            {/* Year 2 */}
            <p className="text-slate-600 text-[10px] font-semibold uppercase mb-1">Year 2 — Monthly</p>
            <div className="grid grid-cols-4 sm:grid-cols-6 gap-2 mb-3">
              {READING_LABELS.slice(12, 24).map((label, i) => (
                <ReadingInput key={i+12} index={i+12} label={label} value={readings[i+12]}
                  error={errors[i+12]}
                  onChange={(v) => { const next=[...readings]; next[i+12]=v; setReadings(next); setErrors(p=>{ const n={...p}; delete n[i+12]; return n; }); }}
                  tooltip={TOOLTIPS[label]} />
              ))}
            </div>

            {/* Weekly */}
            <p className="text-slate-600 text-[10px] font-semibold uppercase mb-1">Weekly</p>
            <div className="grid grid-cols-2 gap-2">
              {READING_LABELS.slice(24).map((label, i) => (
                <ReadingInput key={i+24} index={i+24} label={label} value={readings[i+24]}
                  error={errors[i+24]}
                  onChange={(v) => { const next=[...readings]; next[i+24]=v; setReadings(next); setErrors(p=>{ const n={...p}; delete n[i+24]; return n; }); }}
                  tooltip={TOOLTIPS[label]} wide />
              ))}
            </div>
          </div>

          <button
            onClick={handlePredict}
            disabled={loading}
            className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-500 active:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white font-bold flex items-center justify-center gap-2 transition-colors shadow-lg shadow-blue-500/20"
          >
            {loading
              ? <><Loader2 className="w-4 h-4 animate-spin" />Running CNN-LSTM Prediction…</>
              : <><Zap className="w-4 h-4" />Predict Now</>}
          </button>

          {/* Model verification proof */}
          <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl px-4 py-2.5 font-mono text-[10px] text-slate-500">
            <span className="text-slate-400">Proof: </span>
            model.predict(x) · x.shape=(1,26,1) · model=
            <span className="text-blue-400">{modelInfo?.model_name ?? 'not loaded'}</span>
          </div>
        </div>

        {/* Consumption chart — live as user types */}
        {hasAnyValue && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <h3 className="text-white font-bold text-sm mb-3">Live Consumption Pattern</h3>
            <Plot
              data={[{
                type: 'scatter',
                mode: 'lines+markers',
                x:    READING_LABELS,
                y:    nums,
                line: { color: result?.status === 'Theft' ? '#ef4444' : '#3b82f6', width: 2 },
                marker: { size: 4, color: result?.status === 'Theft' ? '#ef4444' : '#3b82f6' },
                fill: 'tozeroy',
                fillcolor: result?.status === 'Theft' ? 'rgba(239,68,68,0.08)' : 'rgba(59,130,246,0.08)',
                hovertemplate: '<b>%{x}</b><br>%{y:.0f} kWh<extra></extra>',
              }]}
              layout={{
                paper_bgcolor: 'transparent',
                plot_bgcolor:  'transparent',
                font:   { color: '#94a3b8', size: 10 },
                margin: { t: 8, b: 40, l: 48, r: 10 },
                height: 180,
                xaxis:  { gridcolor: '#1e293b', color: '#64748b', tickangle: -45 },
                yaxis:  { gridcolor: '#1e293b', color: '#64748b', title: { text: 'kWh' } },
                showlegend: false,
              }}
              config={{ displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>
        )}
      </div>

      {/* ── Right: Results ── */}
      <div className="xl:col-span-2 space-y-4">
        {!result && !loading && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center">
            <Shield className="w-10 h-10 text-slate-700 mx-auto mb-3" />
            <p className="text-slate-500 text-sm">Fill in the readings and click <strong className="text-slate-400">Predict Now</strong> to run the CNN-LSTM model.</p>
          </div>
        )}

        {loading && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center">
            <Loader2 className="w-8 h-8 text-blue-400 animate-spin mx-auto mb-3" />
            <p className="text-slate-400 text-sm font-semibold">Running CNN-LSTM Inference…</p>
            <p className="text-slate-600 text-xs mt-1">model.predict(x) where x.shape=(1,26,1)</p>
          </div>
        )}

        {result && <PredictionResultCard result={result} />}

        {shapLoading && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex items-center gap-3">
            <Loader2 className="w-4 h-4 text-yellow-400 animate-spin shrink-0" />
            <p className="text-slate-400 text-sm">Computing feature importance…</p>
          </div>
        )}

        {shap && !shapLoading && <ShapChart shap={shap} />}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Reading Input sub-component (with tooltip)
// ─────────────────────────────────────────────────────────────────────────────
interface ReadingInputProps {
  index:    number;
  label:    string;
  value:    string;
  error?:   string;
  tooltip?: string;
  wide?:    boolean;
  onChange: (v: string) => void;
}

const ReadingInput: React.FC<ReadingInputProps> = ({ index, label, value, error, tooltip, wide, onChange }) => {
  const [showTip, setShowTip] = useState(false);
  return (
    <div className={`space-y-0.5 relative ${wide ? 'col-span-1' : ''}`}>
      <div className="flex items-center justify-between px-0.5">
        <p className="text-slate-500 text-[9px] font-medium truncate">{label}</p>
        {tooltip && (
          <div className="relative" onMouseEnter={() => setShowTip(true)} onMouseLeave={() => setShowTip(false)}>
            <Info className="w-2.5 h-2.5 text-slate-600 cursor-help" />
            {showTip && (
              <div className="absolute bottom-full right-0 mb-1 z-50 w-44 bg-slate-700 border border-slate-600 rounded-lg px-2.5 py-2 text-[10px] text-slate-300 shadow-xl pointer-events-none">
                {tooltip}
              </div>
            )}
          </div>
        )}
      </div>
      <input
        type="number"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder="kWh"
        className={`w-full px-1.5 py-2 text-xs text-center font-mono rounded-lg focus:outline-none transition-colors
          ${error
            ? 'bg-red-500/10 border border-red-500/50 text-red-400 focus:border-red-400'
            : 'bg-slate-800 border border-slate-700 text-white focus:border-blue-500 focus:bg-slate-700'
          }`}
      />
      {error && <p className="text-red-400 text-[8px] text-center">{error}</p>}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Prediction Result Card
// ─────────────────────────────────────────────────────────────────────────────
const PredictionResultCard = ({ result }: { result: PredictionResult }) => {
  const isTheft   = result.status === 'Theft';
  const riskLevel = result.risk_level ?? (result.risk_score >= 75 ? 'High' : result.risk_score >= 40 ? 'Medium' : 'Low');

  return (
    <div className={`border rounded-2xl p-5 space-y-4 ${isTheft ? 'bg-red-500/8 border-red-500/25' : 'bg-emerald-500/8 border-emerald-500/25'}`}>
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className={`p-2.5 rounded-xl shrink-0 ${isTheft ? 'bg-red-500/20' : 'bg-emerald-500/20'}`}>
          {isTheft
            ? <AlertTriangle className="w-6 h-6 text-red-400" />
            : <CheckCircle2  className="w-6 h-6 text-emerald-400" />}
        </div>
        <div className="min-w-0 flex-1">
          <p className={`text-lg font-black ${isTheft ? 'text-red-400' : 'text-emerald-400'}`}>
            {result.label}
          </p>
          <p className="text-slate-400 text-xs truncate">Customer: <span className="font-mono text-slate-300">{result.customer_id}</span></p>
        </div>
        <StatusBadge status={result.status} />
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-2">
        <StatCard label="Probability"
          value={`${(result.probability * 100).toFixed(2)}%`}
          sub={`Raw: ${result.probability.toFixed(6)}`} />
        <StatCard label="Confidence"
          value={`${(result.confidence * 100).toFixed(2)}%`} />
        <StatCard label="Risk Score"
          value={`${result.risk_score.toFixed(1)}/100`} />
        <div className="bg-slate-800/60 rounded-xl p-3 space-y-1">
          <p className="text-slate-400 text-xs">Risk Level</p>
          <RiskBadge level={riskLevel as RiskLevel} />
        </div>
      </div>

      {/* Risk meter */}
      <div>
        <div className="flex justify-between text-xs text-slate-500 mb-1">
          <span>Risk Meter</span>
          <span>{result.risk_score.toFixed(1)}/100</span>
        </div>
        <div className="h-2.5 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              result.risk_score >= 75 ? 'bg-red-500' :
              result.risk_score >= 40 ? 'bg-yellow-500' : 'bg-emerald-500'
            }`}
            style={{ width: `${result.risk_score}%` }}
          />
        </div>
        <div className="flex justify-between text-[9px] text-slate-600 mt-0.5">
          <span>Low (0–40)</span><span>Medium (40–75)</span><span>High (75+)</span>
        </div>
      </div>

      {/* Threshold */}
      <div className="flex justify-between text-xs">
        <span className="text-slate-500">Threshold</span>
        <span className="text-yellow-400 font-mono">{result.threshold_used}</span>
      </div>

      {/* Prediction time */}
      {result.predicted_at && (
        <div className="flex items-center gap-1.5 text-xs text-slate-600">
          <Clock className="w-3 h-3" />
          {new Date(result.predicted_at + 'Z').toLocaleString()}
        </div>
      )}

      {/* Proof */}
      <div className="bg-slate-900/60 rounded-lg px-3 py-2 font-mono text-[10px] text-slate-500 break-all">
        {result.predict_proof ?? `model.predict(x) x.shape=(1,26,1) → ${result.probability.toFixed(6)}`}
      </div>

      <p className="text-slate-600 text-[10px]">
        Model: <span className="text-blue-400">{result.model_name}</span>
        {result.sqlite_row_id && <> · SQLite row #{result.sqlite_row_id}</>}
      </p>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// CSV Batch Prediction Tab
// ─────────────────────────────────────────────────────────────────────────────
interface BatchTabProps { modelInfo: ModelInfo | null; threshold: number; }

const BatchTab: React.FC<BatchTabProps> = ({ modelInfo, threshold: defaultThreshold }) => {
  const fileRef = useRef<HTMLInputElement>(null);
  const [file,        setFile]        = useState<File | null>(null);
  const [preview,     setPreview]     = useState<CsvPreview | null>(null);
  const [batchThresh, setBatchThresh] = useState(defaultThreshold);
  const [loading,     setLoading]     = useState(false);
  const [progress,    setProgress]    = useState(0);
  const [result,      setResult]      = useState<BatchPredResult | null>(null);
  const [search,      setSearch]      = useState('');
  const [statusFilter,setStatusFilter]= useState('');
  const [sortCol,     setSortCol]     = useState<keyof BatchPredRow>('risk_score');
  const [sortDir,     setSortDir]     = useState<'asc'|'desc'>('desc');
  const [page,        setPage]        = useState(1);
  const [isDragging,  setIsDragging]  = useState(false);
  const PAGE_SIZE = 50;

  // Animate progress bar while loading
  useEffect(() => {
    if (!loading) { setProgress(0); return; }
    setProgress(5);
    const intervals = [
      setTimeout(() => setProgress(20), 800),
      setTimeout(() => setProgress(40), 2500),
      setTimeout(() => setProgress(60), 6000),
      setTimeout(() => setProgress(75), 12000),
      setTimeout(() => setProgress(85), 20000),
      setTimeout(() => setProgress(92), 35000),
    ];
    return () => intervals.forEach(clearTimeout);
  }, [loading]);

  const parseCsvPreview = (content: string): CsvPreview => {
    const lines = content.split('\n').filter(l => l.trim());
    if (lines.length < 2) return { rows:[], columns:[], totalRows:0, missingValues:0, hasConsNo:false, hasFlag:false, readingColCount:0, isValid:false, errors:['File has no data rows'] };

    const columns = lines[0].split(',').map(c => c.trim().replace(/^"|"$/g,''));
    const upperCols = columns.map(c => c.toUpperCase());
    const hasConsNo = upperCols.includes('CONS_NO');
    const hasFlag   = upperCols.includes('FLAG');
    const skip      = new Set(['CONS_NO','FLAG']);
    const readCols  = columns.filter(c => !skip.has(c.toUpperCase()));

    const errors: string[] = [];
    if (readCols.length < 26) errors.push(`Need 26 reading columns, found ${readCols.length}`);

    const dataLines = lines.slice(1);
    const totalRows = dataLines.length;

    let missingValues = 0;
    const previewRows: Record<string, string>[] = [];
    for (let i = 0; i < Math.min(5, dataLines.length); i++) {
      const vals = dataLines[i].split(',').map(v => v.trim().replace(/^"|"$/g,''));
      const row: Record<string, string> = {};
      columns.forEach((col, ci) => {
        row[col] = vals[ci] ?? '';
        if (!vals[ci] || vals[ci].trim() === '') missingValues++;
      });
      previewRows.push(row);
    }

    return {
      rows:           previewRows,
      columns:        columns.slice(0, 8),
      totalRows,
      missingValues,
      hasConsNo,
      hasFlag,
      readingColCount: readCols.length,
      isValid:         errors.length === 0,
      errors,
    };
  };

  const handleFile = (f: File) => {
    if (!f.name.toLowerCase().endsWith('.csv')) {
      toast.error('Only CSV files are accepted');
      return;
    }
    setFile(f);
    setResult(null);
    const reader = new FileReader();
    reader.onload = e => {
      const text = e.target?.result as string;
      const prev = parseCsvPreview(text);
      setPreview(prev);
      if (!prev.isValid) {
        prev.errors.forEach(err => toast.error(err));
      }
    };
    reader.readAsText(f);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const handleDownloadTemplate = async () => {
    try {
      const res = await downloadTemplate();
      const url = URL.createObjectURL(res.data);
      const a = document.createElement('a');
      a.href = url; a.download = 'etd_prediction_template.csv';
      a.click(); URL.revokeObjectURL(url);
    } catch {
      toast.error('Failed to download template');
    }
  };

  const handleBatchPredict = async () => {
    if (!file) return;
    if (!preview?.isValid) { toast.error('Fix CSV errors first'); return; }
    setLoading(true);
    setResult(null);
    try {
      const res = await predictBatchPreview(file, batchThresh);
      setResult(res.data);
      setPage(1);
      setProgress(100);
      toast.success(`${res.data.total.toLocaleString()} predictions complete in ${res.data.elapsed_seconds}s`);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Batch prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveToDashboard = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const res = await predictBatchStore(file, batchThresh, file.name.replace('.csv',''));
      setProgress(100);
      toast.success(`Saved ${res.data.total.toLocaleString()} customers to Dashboard!`);
      setResult(r => r ? { ...r, ...res.data, stored: true } : null);
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Failed to save to dashboard');
    } finally {
      setLoading(false);
    }
  };

  const exportCsv = () => {
    if (!result) return;
    const header = 'customer_id,status,probability,confidence,risk_score,risk_level,flag,predicted_at';
    const rows   = result.predictions.map(r =>
      `${r.customer_id},${r.status},${r.probability},${(r.confidence*100).toFixed(2)}%,${r.risk_score},${r.risk_level},${r.flag??''},${r.predicted_at}`
    );
    const blob = new Blob([header + '\n' + rows.join('\n')], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = `etd_batch_results_${Date.now()}.csv`;
    a.click(); URL.revokeObjectURL(url);
  };

  // Filter + sort + paginate
  const filtered = (result?.predictions ?? []).filter(r => {
    const matchSearch = !search || r.customer_id.toLowerCase().includes(search.toLowerCase());
    const matchStatus = !statusFilter || r.status === statusFilter;
    return matchSearch && matchStatus;
  });

  const sorted = [...filtered].sort((a, b) => {
    const av = a[sortCol] as any;
    const bv = b[sortCol] as any;
    return sortDir === 'desc' ? (bv > av ? 1 : -1) : (av > bv ? 1 : -1);
  });

  const totalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const pageData   = sorted.slice((page-1)*PAGE_SIZE, page*PAGE_SIZE);

  const toggleSort = (col: keyof BatchPredRow) => {
    if (sortCol === col) setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    else { setSortCol(col); setSortDir('desc'); }
  };

  const SortIcon = ({ col }: { col: keyof BatchPredRow }) =>
    sortCol === col
      ? sortDir === 'desc' ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />
      : <ChevronDown className="w-3 h-3 opacity-30" />;

  return (
    <div className="space-y-5">
      {/* Upload zone + template */}
      <div className="flex gap-4 items-stretch flex-wrap">
        <div
          className={`flex-1 min-w-64 border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all
            ${isDragging ? 'border-blue-400 bg-blue-500/10' : file ? 'border-emerald-500/40 bg-emerald-500/5' : 'border-slate-700 hover:border-slate-500 bg-slate-900'}`}
          onDragEnter={e => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={e => { e.preventDefault(); setIsDragging(false); }}
          onDragOver={e => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
        >
          <input ref={fileRef} type="file" accept=".csv" className="hidden"
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
          {file ? (
            <div className="flex flex-col items-center gap-2">
              <FileText className="w-8 h-8 text-emerald-400" />
              <p className="text-emerald-400 font-semibold text-sm">{file.name}</p>
              <p className="text-slate-500 text-xs">{(file.size / 1024).toFixed(1)} KB · Click to change</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload className="w-8 h-8 text-slate-500" />
              <p className="text-slate-300 font-semibold text-sm">Drop CSV file here or click to browse</p>
              <p className="text-slate-600 text-xs">Requires CONS_NO + 26 reading columns · FLAG optional</p>
            </div>
          )}
        </div>

        <button onClick={handleDownloadTemplate}
          className="flex flex-col items-center justify-center gap-2 px-5 py-4 bg-slate-900 border border-slate-700 rounded-2xl hover:border-blue-500/50 hover:bg-slate-800 transition-colors text-slate-300 hover:text-white group">
          <Download className="w-6 h-6 group-hover:text-blue-400 transition-colors" />
          <span className="text-xs font-semibold whitespace-nowrap">Download<br/>Template</span>
        </button>
      </div>

      {/* Validation summary */}
      {preview && (
        <div className={`rounded-2xl border p-4 ${preview.isValid ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-red-500/5 border-red-500/20'}`}>
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div className="flex items-center gap-3">
              {preview.isValid
                ? <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
                : <AlertTriangle className="w-5 h-5 text-red-400 shrink-0" />}
              <div>
                <p className={`font-semibold text-sm ${preview.isValid ? 'text-emerald-400' : 'text-red-400'}`}>
                  {preview.isValid ? 'File validated — ready to predict' : 'File has errors'}
                </p>
                {preview.errors.map((e, i) => <p key={i} className="text-red-300 text-xs">{e}</p>)}
              </div>
            </div>
            <div className="flex gap-4 text-xs">
              <div className="text-center">
                <p className="text-white font-bold text-lg">{preview.totalRows.toLocaleString()}</p>
                <p className="text-slate-500">Rows</p>
              </div>
              <div className="text-center">
                <p className="text-white font-bold text-lg">{preview.readingColCount}</p>
                <p className="text-slate-500">Reading cols</p>
              </div>
              <div className="text-center">
                <p className={`font-bold text-lg ${preview.missingValues > 0 ? 'text-yellow-400' : 'text-emerald-400'}`}>
                  {preview.missingValues}
                </p>
                <p className="text-slate-500">Missing</p>
              </div>
              <div className="text-center">
                <p className={`font-bold text-lg ${preview.hasFlag ? 'text-blue-400' : 'text-slate-500'}`}>
                  {preview.hasFlag ? 'Yes' : 'No'}
                </p>
                <p className="text-slate-500">FLAG col</p>
              </div>
            </div>
          </div>

          {/* Preview table */}
          {preview.rows.length > 0 && (
            <div className="mt-4 overflow-x-auto">
              <p className="text-slate-500 text-xs mb-2">Preview (first {preview.rows.length} rows):</p>
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-700">
                    {preview.columns.map(col => (
                      <th key={col} className="px-2 py-1.5 text-left text-slate-400 font-semibold">{col}</th>
                    ))}
                    {preview.columns.length < 8 && <th className="px-2 py-1.5 text-slate-600">…</th>}
                  </tr>
                </thead>
                <tbody>
                  {preview.rows.map((row, i) => (
                    <tr key={i} className="border-b border-slate-800/50">
                      {preview.columns.map(col => (
                        <td key={col} className="px-2 py-1.5 text-slate-300 font-mono">{row[col] || '—'}</td>
                      ))}
                      {preview.columns.length < 8 && <td className="px-2 py-1.5 text-slate-600">…</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Controls */}
      {preview?.isValid && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-3 flex-1 min-w-48">
              <label className="text-slate-400 text-sm whitespace-nowrap">Threshold:</label>
              <input type="range" min="0" max="1" step="0.01" value={batchThresh}
                onChange={e => setBatchThresh(Number(e.target.value))}
                className="flex-1 accent-blue-500" />
              <span className="text-blue-400 font-mono font-bold w-10 text-right">{batchThresh.toFixed(2)}</span>
            </div>
          </div>

          {/* Progress bar */}
          {loading && (
            <div>
              <div className="flex justify-between text-xs text-slate-400 mb-1.5">
                <span>Running CNN-LSTM batch inference…</span>
                <span>{progress}%</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full transition-all duration-1000"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-slate-600 text-xs mt-1">
                model.predict(x, batch_size=256) · x.shape=({preview.totalRows},26,1)
              </p>
            </div>
          )}

          <div className="flex gap-3 flex-wrap">
            <button
              onClick={handleBatchPredict}
              disabled={loading || !file}
              className="flex-1 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-bold flex items-center justify-center gap-2 transition-colors text-sm"
            >
              {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Predicting…</> : <><Zap className="w-4 h-4" />Run Batch Prediction</>}
            </button>
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-4">
          {/* Summary stats */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-center">
              <p className="text-2xl font-black text-white">{result.total.toLocaleString()}</p>
              <p className="text-slate-400 text-xs mt-0.5">Total Customers</p>
            </div>
            <div className="bg-red-500/8 border border-red-500/20 rounded-xl p-4 text-center">
              <p className="text-2xl font-black text-red-400">{result.theft.toLocaleString()}</p>
              <p className="text-slate-400 text-xs mt-0.5">Theft Detected</p>
            </div>
            <div className="bg-emerald-500/8 border border-emerald-500/20 rounded-xl p-4 text-center">
              <p className="text-2xl font-black text-emerald-400">{result.normal.toLocaleString()}</p>
              <p className="text-slate-400 text-xs mt-0.5">Normal</p>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-center">
              <p className="text-2xl font-black text-yellow-400">{(result.avg_confidence * 100).toFixed(1)}%</p>
              <p className="text-slate-400 text-xs mt-0.5">Avg Confidence</p>
            </div>
          </div>

          {/* Metrics (if FLAG present) */}
          {result.has_flag && result.metrics && Object.keys(result.metrics).length > 0 && (
            <div className="bg-slate-900 border border-blue-500/20 rounded-2xl p-4">
              <p className="text-blue-400 text-xs font-semibold uppercase tracking-wider mb-3">Evaluation Metrics (vs Ground Truth FLAG)</p>
              <div className="grid grid-cols-5 gap-2">
                {[
                  { label: 'Accuracy',  value: result.metrics.accuracy },
                  { label: 'Precision', value: result.metrics.precision_val },
                  { label: 'Recall',    value: result.metrics.recall_val },
                  { label: 'F1 Score',  value: result.metrics.f1_score },
                  { label: 'ROC-AUC',   value: result.metrics.roc_auc },
                ].map(({ label, value }) => (
                  <div key={label} className="text-center">
                    <p className="text-white font-bold">{value != null ? `${(value*100).toFixed(1)}%` : '—'}</p>
                    <p className="text-slate-500 text-xs">{label}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Elapsed + proof */}
          <div className="flex items-center justify-between flex-wrap gap-2">
            <p className="text-slate-500 text-xs font-mono">{result.predict_proof}</p>
            <p className="text-slate-500 text-xs">
              Elapsed: <span className="text-white font-mono">{result.elapsed_seconds}s</span>
              {' · '}Model: <span className="text-blue-400 font-mono">{result.model_name}</span>
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex gap-3 flex-wrap">
            <button onClick={exportCsv}
              className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl text-sm text-slate-300 font-semibold transition-colors">
              <Download className="w-4 h-4" /> Export CSV
            </button>
            {!result.stored && (
              <button onClick={handleSaveToDashboard} disabled={loading}
                className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 rounded-xl text-sm text-white font-semibold transition-colors">
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Database className="w-4 h-4" />}
                Save to Customer Predictions
              </button>
            )}
            {result.stored && (
              <div className="flex items-center gap-2 px-4 py-2 bg-emerald-500/15 border border-emerald-500/30 rounded-xl text-sm text-emerald-400 font-semibold">
                <CheckCircle2 className="w-4 h-4" /> Saved to Dashboard
              </div>
            )}
          </div>

          {/* Results Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
            {/* Table controls */}
            <div className="flex items-center gap-3 p-4 border-b border-slate-800 flex-wrap">
              <div className="flex items-center gap-2 flex-1 min-w-40">
                <Search className="w-4 h-4 text-slate-500 shrink-0" />
                <input
                  value={search}
                  onChange={e => { setSearch(e.target.value); setPage(1); }}
                  placeholder="Search customer ID…"
                  className="flex-1 bg-transparent text-white text-sm focus:outline-none placeholder:text-slate-600"
                />
                {search && <button onClick={() => setSearch('')}><X className="w-3.5 h-3.5 text-slate-500 hover:text-white" /></button>}
              </div>
              <select
                value={statusFilter}
                onChange={e => { setStatusFilter(e.target.value); setPage(1); }}
                className="bg-slate-800 border border-slate-700 text-slate-300 text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500"
              >
                <option value="">All Status</option>
                <option value="Theft">Theft</option>
                <option value="Normal">Normal</option>
              </select>
              <span className="text-slate-500 text-xs ml-auto whitespace-nowrap">
                {filtered.length.toLocaleString()} result{filtered.length !== 1 ? 's' : ''}
              </span>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-800">
                    {[
                      { key: 'customer_id',  label: 'Customer ID' },
                      { key: 'status',       label: 'Status' },
                      { key: 'probability',  label: 'Probability' },
                      { key: 'confidence',   label: 'Confidence' },
                      { key: 'risk_score',   label: 'Risk Score' },
                      { key: 'risk_level',   label: 'Risk Level' },
                    ].map(({ key, label }) => (
                      <th key={key}
                        onClick={() => toggleSort(key as keyof BatchPredRow)}
                        className="px-4 py-3 text-left text-slate-400 text-xs font-semibold cursor-pointer hover:text-white transition-colors whitespace-nowrap select-none">
                        <div className="flex items-center gap-1">
                          {label} <SortIcon col={key as keyof BatchPredRow} />
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {pageData.map((row, i) => (
                    <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                      <td className="px-4 py-2.5 font-mono text-xs text-slate-300 max-w-32 truncate">{row.customer_id}</td>
                      <td className="px-4 py-2.5"><StatusBadge status={row.status} /></td>
                      <td className="px-4 py-2.5">
                        <div className="space-y-1">
                          <p className="text-white text-xs font-mono">{(row.probability * 100).toFixed(2)}%</p>
                          <ProbBar value={row.probability} status={row.status} />
                        </div>
                      </td>
                      <td className="px-4 py-2.5 text-slate-300 text-xs font-mono">{(row.confidence * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2.5 text-slate-300 text-xs font-mono">{row.risk_score.toFixed(1)}</td>
                      <td className="px-4 py-2.5"><RiskBadge level={row.risk_level} /></td>
                    </tr>
                  ))}
                  {pageData.length === 0 && (
                    <tr><td colSpan={6} className="px-4 py-8 text-center text-slate-500 text-sm">No results match your filter.</td></tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between px-4 py-3 border-t border-slate-800">
                <span className="text-slate-500 text-xs">
                  Page {page} of {totalPages} · {sorted.length.toLocaleString()} rows
                </span>
                <div className="flex items-center gap-1">
                  <button onClick={() => setPage(p => Math.max(1, p-1))} disabled={page === 1}
                    className="p-1.5 rounded-lg hover:bg-slate-700 disabled:opacity-30 transition-colors text-slate-400">
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                    const p = Math.max(1, Math.min(page - 2 + i, totalPages - 4 + i));
                    return (
                      <button key={p} onClick={() => setPage(p)}
                        className={`w-7 h-7 rounded-lg text-xs font-semibold transition-colors ${page === p ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-slate-700'}`}>
                        {p}
                      </button>
                    );
                  })}
                  <button onClick={() => setPage(p => Math.min(totalPages, p+1))} disabled={page === totalPages}
                    className="p-1.5 rounded-lg hover:bg-slate-700 disabled:opacity-30 transition-colors text-slate-400">
                    <ChevronRight className="w-4 h-4" />
                  </button>
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
// Prediction History
// ─────────────────────────────────────────────────────────────────────────────
const PredictionHistory: React.FC = () => {
  const [history,  setHistory]  = useState<PredHistoryRow[]>([]);
  const [loading,  setLoading]  = useState(false);
  const [expanded, setExpanded] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getPredictHistory(50);
      setHistory(res.data.history);
    } catch {
      // history is optional
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  if (history.length === 0 && !loading) return null;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
      <button
        onClick={() => setExpanded(e => !e)}
        className="w-full flex items-center justify-between px-5 py-4 hover:bg-slate-800/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Clock className="w-4 h-4 text-slate-400" />
          <span className="text-white font-bold">Prediction History</span>
          <span className="bg-slate-700 text-slate-300 text-xs px-2 py-0.5 rounded-full font-semibold">
            {history.length}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={e => { e.stopPropagation(); load(); }}
            className="p-1 rounded hover:bg-slate-700 transition-colors">
            <RefreshCw className={`w-3.5 h-3.5 text-slate-400 ${loading ? 'animate-spin' : ''}`} />
          </button>
          {expanded ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </div>
      </button>

      {expanded && (
        <div className="border-t border-slate-800 overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-800">
                {['#', 'Customer ID', 'Status', 'Probability', 'Risk Level', 'Source', 'Time'].map(h => (
                  <th key={h} className="px-4 py-2.5 text-left text-slate-400 font-semibold whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.map((row, i) => {
                const riskLevel: RiskLevel = row.risk_score >= 75 ? 'High' : row.risk_score >= 40 ? 'Medium' : 'Low';
                return (
                  <tr key={row.id} className="border-b border-slate-800/40 hover:bg-slate-800/30 transition-colors">
                    <td className="px-4 py-2.5 text-slate-600">#{row.id}</td>
                    <td className="px-4 py-2.5 font-mono text-slate-300 max-w-32 truncate">{row.customer_id || '—'}</td>
                    <td className="px-4 py-2.5"><StatusBadge status={row.status} /></td>
                    <td className="px-4 py-2.5 font-mono text-slate-300">{(row.probability * 100).toFixed(2)}%</td>
                    <td className="px-4 py-2.5"><RiskBadge level={riskLevel} /></td>
                    <td className="px-4 py-2.5">
                      <span className="px-1.5 py-0.5 bg-slate-800 border border-slate-700 rounded text-slate-400">{row.source}</span>
                    </td>
                    <td className="px-4 py-2.5 text-slate-500 whitespace-nowrap">
                      {new Date(row.predicted_at + 'Z').toLocaleString()}
                    </td>
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
// Main PredictPage
// ─────────────────────────────────────────────────────────────────────────────
interface Props { modelInfo: ModelInfo | null; threshold: number; }

const PredictPage: React.FC<Props> = ({ modelInfo, threshold }) => {
  const [activeTab, setActiveTab] = useState<'single' | 'batch'>('single');

  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-black text-white">Prediction Center</h1>
        <p className="text-slate-400 text-sm mt-0.5">
          Every prediction runs through <span className="text-blue-400 font-mono">model.predict()</span> on the loaded CNN-LSTM model — no mocks, no placeholders.
        </p>
      </div>

      {/* Model status badge */}
      <div className="flex items-center gap-2 flex-wrap bg-slate-900 border border-slate-800 rounded-xl px-4 py-2.5 text-xs">
        <div className="flex items-center gap-1.5">
          <div className={`w-2 h-2 rounded-full ${modelInfo?.loaded ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
          <span className="text-slate-400">Model:</span>
          <span className="text-white font-bold font-mono">{modelInfo?.model_name ?? 'Not loaded'}</span>
        </div>
        <span className="text-slate-700">|</span>
        <span className="text-slate-400">Architecture:</span>
        <span className="text-blue-400 font-bold">CNN-LSTM</span>
        <span className="text-slate-700">|</span>
        <span className="text-slate-400">Input:</span>
        <span className="text-yellow-400 font-mono">(N, 26, 1)</span>
        <span className="text-slate-700">|</span>
        <span className="text-slate-400">Threshold:</span>
        <span className="text-yellow-400 font-mono">{threshold}</span>
        {modelInfo?.is_dual_input && (
          <>
            <span className="text-slate-700">|</span>
            <span className="text-purple-400 font-semibold">Dual-Input (+{modelInfo.stat_input_size} stat features)</span>
          </>
        )}
      </div>

      {/* Tab switcher */}
      <div className="flex gap-1 p-1 bg-slate-900 border border-slate-800 rounded-xl w-fit">
        <button
          onClick={() => setActiveTab('single')}
          className={`flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold transition-all ${
            activeTab === 'single'
              ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
              : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
        >
          <Zap className="w-4 h-4" />
          Single Customer
        </button>
        <button
          onClick={() => setActiveTab('batch')}
          className={`flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold transition-all ${
            activeTab === 'batch'
              ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
              : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
        >
          <BarChart2 className="w-4 h-4" />
          Batch CSV Upload
        </button>
      </div>

      {/* Tab content */}
      {activeTab === 'single'
        ? <SingleTab modelInfo={modelInfo} threshold={threshold} />
        : <BatchTab  modelInfo={modelInfo} threshold={threshold} />}

      {/* Prediction history — shown below both tabs */}
      <PredictionHistory />
    </div>
  );
};

export default PredictPage;
