import React, { useState } from 'react';
import { Zap, AlertTriangle, CheckCircle2, Loader2 } from 'lucide-react';
import { predictManual } from '../api/client';
import type { PredictionResult, ModelInfo } from '../types';
import Plot from 'react-plotly.js';
import toast from 'react-hot-toast';

interface Props { modelInfo: ModelInfo | null; threshold: number; }

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
  'Jan+', 'Feb+', 'Mar+', 'Apr+', 'May+', 'Jun+',
  'Jul+', 'Aug+', 'Sep+', 'Oct+', 'Nov+', 'Dec+',
  'W1', 'W2'];

const PredictPage: React.FC<Props> = ({ modelInfo, threshold }) => {
  const [customerId, setCustomerId] = useState('');
  const [readings, setReadings] = useState<string[]>(Array(26).fill(''));
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleReadingChange = (i: number, val: string) => {
    const next = [...readings];
    next[i] = val;
    setReadings(next);
  };

  const fillDemo = (type: 'theft' | 'normal') => {
    if (type === 'theft') {
      // Anomalous pattern: sudden drops, zeros, negatives
      setReadings(['2400', '2500', '0', '100', '2200', '0', '2800', '50',
        '2100', '2300', '0', '1100', '1000', '3200', '2200', '2400',
        '0', '2400', '1800', '2100', '0', '2400', '2500', '1200', '0', '2300']);
      setCustomerId('DEMO_THEFT_001');
    } else {
      // Normal pattern: consistent readings
      setReadings(['1200', '1250', '1180', '1300', '1220', '1190', '1280',
        '1240', '1210', '1260', '1230', '1150', '1170', '1310', '1195',
        '1243', '1215', '1195', '1278', '1089', '1300', '1240', '1265',
        '1177', '1190', '1272']);
      setCustomerId('DEMO_NORMAL_001');
    }
  };

  const handlePredict = async () => {
    if (!customerId.trim()) { toast.error('Enter a Customer ID'); return; }
    const nums = readings.map(Number);
    if (nums.some(isNaN)) { toast.error('All 26 readings must be valid numbers'); return; }

    setLoading(true);
    try {
      const res = await predictManual({ customer_id: customerId, readings: nums, threshold });
      setResult(res.data.result);
      toast.success('Prediction complete!');
    } catch (e: any) {
      toast.error(e.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setCustomerId('');
    setReadings(Array(26).fill(''));
    setResult(null);
  };

  const readingValues = readings.map(Number);

  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-black text-white">Manual Prediction</h1>
        <p className="text-slate-400 text-sm">Enter one customer's data and get an instant CNN-LSTM prediction.</p>
      </div>

      {/* Model badge */}
      <div className="flex items-center gap-3 text-xs bg-slate-900 border border-slate-800 rounded-xl px-4 py-2.5">
        <span className="text-slate-400">Model:</span>
        <span className="text-white font-bold font-mono">{modelInfo?.model_name}</span>
        <span className="text-slate-600">|</span>
        <span className="text-slate-400">Architecture:</span>
        <span className="text-blue-400 font-bold">CNN-LSTM</span>
        <span className="text-slate-600">|</span>
        <span className="text-slate-400">Threshold:</span>
        <span className="text-yellow-400 font-mono font-bold">{threshold}</span>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* Input Panel */}
        <div className="xl:col-span-3 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-5">
          <div className="flex items-center justify-between">
            <h2 className="text-white font-bold">Customer Data Entry</h2>
            <div className="flex gap-2">
              <button onClick={() => fillDemo('normal')} className="px-3 py-1.5 bg-emerald-500/15 text-emerald-400 border border-emerald-500/20 rounded-lg text-xs font-semibold hover:bg-emerald-500/25 transition-colors">
                Normal Demo
              </button>
              <button onClick={() => fillDemo('theft')} className="px-3 py-1.5 bg-red-500/15 text-red-400 border border-red-500/20 rounded-lg text-xs font-semibold hover:bg-red-500/25 transition-colors">
                Theft Demo
              </button>
              <button onClick={clear} className="px-3 py-1.5 bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold hover:bg-slate-600 transition-colors">
                Clear
              </button>
            </div>
          </div>

          {/* Customer ID */}
          <div>
            <label className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-1.5 block">Customer ID</label>
            <input
              value={customerId}
              onChange={e => setCustomerId(e.target.value)}
              placeholder="e.g. A0E791400CF1C48C43DC26A68227854A"
              className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 text-white rounded-xl text-sm font-mono focus:outline-none focus:border-blue-500 placeholder:text-slate-600"
            />
          </div>

          {/* 26 Readings Grid */}
          <div>
            <label className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2 block">26 Electricity Consumption Readings</label>
            <div className="grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-8 gap-2">
              {readings.map((v, i) => (
                <div key={i} className="space-y-0.5">
                  <p className="text-slate-600 text-[9px] text-center">{i + 1}</p>
                  <input
                    type="number"
                    value={v}
                    onChange={e => handleReadingChange(i, e.target.value)}
                    placeholder="0"
                    className="w-full px-1.5 py-2 bg-slate-800 border border-slate-700 text-white rounded-lg text-xs text-center font-mono focus:outline-none focus:border-blue-500 focus:bg-slate-700"
                  />
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={handlePredict}
            disabled={loading}
            className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white font-bold flex items-center justify-center gap-2 transition-colors"
          >
            {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Running CNN-LSTM Prediction...</> : <><Zap className="w-4 h-4" />Predict Now</>}
          </button>
        </div>

        {/* Result Panel */}
        <div className="xl:col-span-2 space-y-4">
          {/* Consumption chart */}
          {readings.some(r => r !== '') && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
              <h3 className="text-white font-bold text-sm mb-3">Consumption Pattern</h3>
              <Plot
                data={[{
                  type: 'scatter',
                  mode: 'lines+markers',
                  x: MONTHS,
                  y: readingValues,
                  line: { color: result?.status === 'Theft' ? '#ef4444' : '#3b82f6', width: 2 },
                  marker: { size: 5 },
                  fill: 'tozeroy',
                  fillcolor: result?.status === 'Theft' ? 'rgba(239,68,68,0.1)' : 'rgba(59,130,246,0.1)',
                }]}
                layout={{
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#94a3b8', size: 10 },
                  margin: { t: 10, b: 30, l: 45, r: 10 },
                  height: 180,
                  xaxis: { gridcolor: '#1e293b', color: '#64748b' },
                  yaxis: { gridcolor: '#1e293b', color: '#64748b', title: { text: 'kWh' } },
                  showlegend: false,
                }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Prediction result */}
          {result && (
            <div className={`border rounded-2xl p-6 space-y-4
              ${result.status === 'Theft'
                ? 'bg-red-500/10 border-red-500/30'
                : 'bg-emerald-500/10 border-emerald-500/30'}`}>
              <div className="flex items-center gap-3">
                {result.status === 'Theft'
                  ? <AlertTriangle className="w-8 h-8 text-red-400" />
                  : <CheckCircle2 className="w-8 h-8 text-emerald-400" />}
                <div>
                  <p className={`text-xl font-black ${result.status === 'Theft' ? 'text-red-400' : 'text-emerald-400'}`}>
                    {result.label}
                  </p>
                  <p className="text-slate-400 text-xs">Customer: {result.customer_id}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: 'Probability', value: `${(result.probability * 100).toFixed(2)}%` },
                  { label: 'Confidence', value: `${(result.confidence * 100).toFixed(2)}%` },
                  { label: 'Risk Score', value: `${result.risk_score.toFixed(1)}/100` },
                  { label: 'Threshold', value: result.threshold_used.toFixed(2) },
                ].map(({ label, value }) => (
                  <div key={label} className="bg-slate-900/50 rounded-xl p-3">
                    <p className="text-slate-400 text-xs">{label}</p>
                    <p className="text-white font-bold text-lg">{value}</p>
                  </div>
                ))}
              </div>

              <div className="bg-slate-900/50 rounded-xl p-3">
                <p className="text-slate-400 text-xs mb-1">Risk Meter</p>
                <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${result.risk_score > 75 ? 'bg-red-500' : result.risk_score > 50 ? 'bg-orange-500' : result.risk_score > 25 ? 'bg-yellow-500' : 'bg-emerald-500'}`}
                    style={{ width: `${result.risk_score}%` }}
                  />
                </div>
              </div>

              <p className="text-slate-500 text-xs">
                Prediction by: <span className="text-blue-400 font-mono">{result.model_name}</span>
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictPage;
