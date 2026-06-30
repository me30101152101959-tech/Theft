import React, { useEffect, useState } from 'react';
import {
  Users, AlertTriangle, CheckCircle2, TrendingUp,
  Activity, Target, BarChart3, Cpu, RefreshCw, Database,
} from 'lucide-react';
import Plot from 'react-plotly.js';
import toast from 'react-hot-toast';

import { getDashboard } from '../api/client';
import type { ModelInfo } from '../types';

interface Props { modelInfo: ModelInfo | null; }

const KPI = ({ label, value, sub, icon: Icon, color }: {
  label: string; value: string; sub?: string; icon: any; color: string;
}) => (
  <div className={`bg-slate-900 border border-slate-800 rounded-2xl p-5 hover:border-${color}-500/40 transition-colors`}>
    <div className="flex items-start justify-between mb-3">
      <div className={`p-2.5 bg-${color}-500/15 rounded-xl`}>
        <Icon className={`w-5 h-5 text-${color}-400`} />
      </div>
    </div>
    <p className="text-2xl font-black text-white">{value}</p>
    <p className="text-slate-400 text-sm font-medium mt-0.5">{label}</p>
    {sub && <p className="text-slate-500 text-xs mt-1">{sub}</p>}
  </div>
);

const LAYOUT = {
  paper_bgcolor: 'transparent',
  plot_bgcolor:  'transparent',
  font:   { color: '#94a3b8', size: 11 },
  margin: { t: 30, b: 40, l: 50, r: 20 },
  showlegend: true,
  legend: { font: { color: '#94a3b8' } },
  xaxis:  { gridcolor: '#1e293b', color: '#64748b' },
  yaxis:  { gridcolor: '#1e293b', color: '#64748b' },
};

const DashboardPage: React.FC<Props> = ({ modelInfo }) => {
  const [data,    setData]    = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [notReady, setNotReady] = useState(false);

  const load = async () => {
    setLoading(true);
    setNotReady(false);
    try {
      const res = await getDashboard();
      if (!res.data.ready) {
        setNotReady(true);
        setData(null);
      } else {
        setData(res.data);
      }
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? 'Could not reach backend.';
      toast.error(`Dashboard error: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-full min-h-64">
      <div className="text-center">
        <RefreshCw className="w-8 h-8 text-blue-400 animate-spin mx-auto mb-3" />
        <p className="text-slate-400">Loading dashboard from SQLite…</p>
      </div>
    </div>
  );

  if (notReady) return (
    <div className="flex flex-col items-center justify-center h-full min-h-64 gap-4">
      <Database className="w-12 h-12 text-slate-600" />
      <p className="text-slate-300 text-lg font-semibold">No data in database yet</p>
      <p className="text-slate-500 text-sm text-center max-w-sm">
        Upload <code className="text-blue-400">cnnlstm_final.keras</code> then your dataset CSV.
        Predictions will be stored in SQLite and shown here immediately.
      </p>
      <button onClick={load}
        className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-sm font-medium">
        <RefreshCw className="w-4 h-4" /> Retry
      </button>
    </div>
  );

  if (!data) return null;

  const charts = data.charts ?? {};

  return (
    <div className="p-6 space-y-6 max-w-screen-2xl mx-auto">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-black text-white">ETD-XAI Dashboard</h1>
          <p className="text-slate-400 text-sm">
            Dataset: <span className="text-slate-300">{data.dataset_name}</span>
            {data.upload_time && <> · {new Date(data.upload_time).toLocaleString()}</>}
          </p>
        </div>
        <button onClick={load}
          className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-sm font-medium transition-colors">
          <RefreshCw className="w-4 h-4" /> Refresh
        </button>
      </div>

      {/* Data source proof banner */}
      <div className="bg-emerald-500/10 border border-emerald-500/25 rounded-2xl p-3 flex items-center gap-3 flex-wrap text-xs font-mono">
        <Database className="w-4 h-4 text-emerald-400 shrink-0" />
        <span className="text-emerald-300 font-bold">Data source:</span>
        <span className="text-slate-400">{data.data_source}</span>
        <span className="text-slate-600">|</span>
        <span className="text-slate-400">{data.predict_proof}</span>
      </div>

      {/* Model proof banner */}
      <div className="bg-blue-500/10 border border-blue-500/25 rounded-2xl p-4 flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-blue-400" />
          <span className="text-blue-300 font-bold text-sm">CNN-LSTM Model:</span>
        </div>
        <div className="flex gap-5 text-xs font-mono flex-wrap">
          <span><span className="text-slate-500">File:</span> <span className="text-white font-bold">{data.model_name || modelInfo?.model_name}</span></span>
          <span><span className="text-slate-500">Params:</span> <span className="text-yellow-400">{data.model_params || modelInfo?.total_params_fmt}</span></span>
          <span><span className="text-slate-500">Input:</span> <span className="text-white">{data.input_shape || modelInfo?.input_shape}</span></span>
          <span><span className="text-slate-500">Output:</span> <span className="text-white">{data.output_shape || modelInfo?.output_shape}</span></span>
          <span><span className="text-slate-500">Dual-input:</span> <span className="text-blue-400">{String(data.is_dual_input)}</span></span>
        </div>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
        <KPI label="Total Customers"  value={(data.total_customers ?? 0).toLocaleString()} icon={Users}         color="blue" />
        <KPI label="Processed"        value={(data.processed_customers ?? 0).toLocaleString()} icon={Activity}    color="indigo" />
        <KPI label="Predicted Theft"  value={(data.predicted_theft ?? 0).toLocaleString()}
              sub={`${((data.theft_rate ?? 0) * 100).toFixed(1)}% of total`} icon={AlertTriangle} color="red" />
        <KPI label="Predicted Normal" value={(data.predicted_normal ?? 0).toLocaleString()} icon={CheckCircle2}  color="emerald" />
        <KPI label="Avg Confidence"   value={`${((data.avg_confidence ?? 0) * 100).toFixed(1)}%`} icon={Target} color="yellow" />
        <KPI label="Avg Risk Score"   value={(data.avg_risk_score ?? 0).toFixed(1)} sub="out of 100" icon={TrendingUp} color="orange" />
      </div>

      {/* Evaluation KPIs (only when FLAG ground truth was present) */}
      {data.has_flag && data.accuracy != null && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {[
            { label: 'Accuracy',  value: `${(data.accuracy * 100).toFixed(2)}%`,          color: 'green' },
            { label: 'Precision', value: `${((data.precision ?? 0) * 100).toFixed(2)}%`,  color: 'blue' },
            { label: 'Recall',    value: `${((data.recall    ?? 0) * 100).toFixed(2)}%`,  color: 'purple' },
            { label: 'F1 Score',  value: `${((data.f1_score  ?? 0) * 100).toFixed(2)}%`,  color: 'yellow' },
            { label: 'ROC-AUC',   value: `${((data.roc_auc   ?? 0) * 100).toFixed(2)}%`,  color: 'pink' },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
              <p className={`text-2xl font-black text-${color}-400`}>{value}</p>
              <p className="text-slate-400 text-sm mt-1">{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">

        {charts.pie && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <h3 className="text-white font-bold mb-3 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-blue-400" />Normal vs Theft
            </h3>
            <Plot
              data={[{
                type: 'pie',
                labels: charts.pie.labels,
                values: charts.pie.values,
                marker: { colors: ['#22c55e', '#ef4444'] },
                textinfo: 'label+percent',
                textfont: { color: '#fff' as any, size: 13 },
                hole: 0.4,
              }]}
              layout={{ ...LAYOUT, height: 260 }}
              config={{ displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>
        )}

        {charts.risk_distribution && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <h3 className="text-white font-bold mb-3">Risk Score Distribution</h3>
            <Plot
              data={([
                {
                  type: 'histogram',
                  x: charts.risk_distribution.values.filter((_: number, i: number) =>
                    charts.risk_distribution.labels[i] === 'Theft'),
                  name: 'Theft', marker: { color: '#ef4444' }, opacity: 0.8, nbinsx: 30,
                },
                {
                  type: 'histogram',
                  x: charts.risk_distribution.values.filter((_: number, i: number) =>
                    charts.risk_distribution.labels[i] === 'Normal'),
                  name: 'Normal', marker: { color: '#22c55e' }, opacity: 0.8, nbinsx: 30,
                },
              ] as any)}
              layout={{ ...LAYOUT, height: 260, barmode: 'overlay',
                xaxis: { ...LAYOUT.xaxis, title: { text: 'Risk Score' } },
                yaxis: { ...LAYOUT.yaxis, title: { text: 'Count' } } }}
              config={{ displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>
        )}

        {charts.top10_high && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <h3 className="text-white font-bold mb-3 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-red-400" />Top 10 Highest Risk
            </h3>
            <Plot
              data={[{
                type: 'bar',
                x: charts.top10_high.risks,
                y: charts.top10_high.ids,
                orientation: 'h',
                marker: { color: charts.top10_high.risks.map((r: number) =>
                  r > 75 ? '#ef4444' : r > 50 ? '#f97316' : '#eab308') },
              }]}
              layout={{ ...LAYOUT, height: 260,
                xaxis: { ...LAYOUT.xaxis, title: { text: 'Risk Score' }, range: [0, 100] } }}
              config={{ displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>
        )}

        {charts.scatter && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <h3 className="text-white font-bold mb-3">Risk vs Confidence</h3>
            <Plot
              data={[
                {
                  type: 'scatter', mode: 'markers',
                  x: charts.scatter.confidence.filter((_: number, i: number) => charts.scatter.status[i] === 'Theft'),
                  y: charts.scatter.risk.filter((_: number, i: number) => charts.scatter.status[i] === 'Theft'),
                  name: 'Theft', marker: { color: '#ef4444', size: 4, opacity: 0.6 },
                },
                {
                  type: 'scatter', mode: 'markers',
                  x: charts.scatter.confidence.filter((_: number, i: number) => charts.scatter.status[i] === 'Normal'),
                  y: charts.scatter.risk.filter((_: number, i: number) => charts.scatter.status[i] === 'Normal'),
                  name: 'Normal', marker: { color: '#22c55e', size: 4, opacity: 0.4 },
                },
              ]}
              layout={{ ...LAYOUT, height: 260,
                xaxis: { ...LAYOUT.xaxis, title: { text: 'Confidence' } },
                yaxis: { ...LAYOUT.yaxis, title: { text: 'Risk Score' } } }}
              config={{ displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>
        )}

        {charts.top10_low && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <h3 className="text-white font-bold mb-3 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />Top 10 Lowest Risk
            </h3>
            <Plot
              data={[{
                type: 'bar',
                x: charts.top10_low.risks,
                y: charts.top10_low.ids,
                orientation: 'h',
                marker: { color: '#22c55e' },
              }]}
              layout={{ ...LAYOUT, height: 260,
                xaxis: { ...LAYOUT.xaxis, title: { text: 'Risk Score' }, range: [0, 50] } }}
              config={{ displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>
        )}

        {charts.roc?.fpr?.length > 0 && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <h3 className="text-white font-bold mb-3">ROC Curve</h3>
            <Plot
              data={[
                { type: 'scatter', mode: 'lines', x: charts.roc.fpr, y: charts.roc.tpr,
                  name: 'CNN-LSTM', line: { color: '#3b82f6', width: 2 } },
                { type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1],
                  name: 'Random', line: { color: '#475569', dash: 'dash' } },
              ]}
              layout={{ ...LAYOUT, height: 260,
                xaxis: { ...LAYOUT.xaxis, title: { text: 'FPR' } },
                yaxis: { ...LAYOUT.yaxis, title: { text: 'TPR' } } }}
              config={{ displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>
        )}

        {charts.confusion?.length === 2 && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <h3 className="text-white font-bold mb-3">Confusion Matrix</h3>
            <Plot
              data={[{
                type: 'heatmap',
                z: charts.confusion,
                x: ['Predicted Normal', 'Predicted Theft'],
                y: ['Actual Normal',    'Actual Theft'],
                colorscale: [[0, '#0f172a'], [1, '#3b82f6']],
                showscale: false,
                text: charts.confusion.map((row: number[]) => row.map(String)),
                texttemplate: '%{text}',
                textfont: { size: 18, color: '#fff' as any },
              }]}
              layout={{ ...LAYOUT, height: 260 }}
              config={{ displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>
        )}

        {charts.pr?.precision?.length > 0 && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <h3 className="text-white font-bold mb-3">Precision-Recall Curve</h3>
            <Plot
              data={[{
                type: 'scatter', mode: 'lines',
                x: charts.pr.recall, y: charts.pr.precision,
                name: 'CNN-LSTM', line: { color: '#a855f7', width: 2 },
              }]}
              layout={{ ...LAYOUT, height: 260,
                xaxis: { ...LAYOUT.xaxis, title: { text: 'Recall' } },
                yaxis: { ...LAYOUT.yaxis, title: { text: 'Precision' } } }}
              config={{ displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>
        )}

      </div>
    </div>
  );
};

export default DashboardPage;
