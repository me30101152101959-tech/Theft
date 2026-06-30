import React, { useEffect, useState } from 'react';
import {
  Users, AlertTriangle, CheckCircle2, TrendingUp,
  Activity, Target, BarChart3, Cpu, RefreshCw
} from 'lucide-react';
import { getDashboardStats, getChartData } from '../api/client';
import type { DashboardStats, ChartData, ModelInfo } from '../types';
import Plot from 'react-plotly.js';
import toast from 'react-hot-toast';

interface Props { modelInfo: ModelInfo | null; }

const KPI = ({ label, value, sub, icon: Icon, color }: any) => (
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

const PLOTLY_LAYOUT = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { color: '#94a3b8', size: 11 },
  margin: { t: 30, b: 40, l: 50, r: 20 },
  showlegend: true,
  legend: { font: { color: '#94a3b8' } },
  xaxis: { gridcolor: '#1e293b', color: '#64748b' },
  yaxis: { gridcolor: '#1e293b', color: '#64748b' },
};

const DashboardPage: React.FC<Props> = ({ modelInfo }) => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [charts, setCharts] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const [s, c] = await Promise.all([getDashboardStats(), getChartData()]);
      setStats(s.data);
      setCharts(c.data);
    } catch {
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <RefreshCw className="w-8 h-8 text-blue-400 animate-spin mx-auto mb-3" />
        <p className="text-slate-400">Loading dashboard...</p>
      </div>
    </div>
  );

  if (!stats) return null;

  return (
    <div className="p-6 space-y-6 max-w-screen-2xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-black text-white">ETD-XAI Dashboard</h1>
          <p className="text-slate-400 text-sm">Dataset: {stats.dataset_name} · {new Date(stats.upload_time).toLocaleString()}</p>
        </div>
        <button onClick={load} className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-sm font-medium transition-colors">
          <RefreshCw className="w-4 h-4" /> Refresh
        </button>
      </div>

      {/* ── Model proof banner ── */}
      <div className="bg-blue-500/10 border border-blue-500/25 rounded-2xl p-4 flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-blue-400" />
          <span className="text-blue-300 font-bold text-sm">Powered by:</span>
        </div>
        <div className="flex gap-6 text-xs font-mono flex-wrap">
          <span><span className="text-slate-500">Model:</span> <span className="text-white font-bold">{modelInfo?.model_name}</span></span>
          <span><span className="text-slate-500">Architecture:</span> <span className="text-blue-400">CNN-LSTM</span></span>
          <span><span className="text-slate-500">Params:</span> <span className="text-yellow-400">{modelInfo?.total_params_fmt}</span></span>
          <span><span className="text-slate-500">Input:</span> <span className="text-white">{modelInfo?.input_shape}</span></span>
          <span><span className="text-slate-500">Output:</span> <span className="text-white">{modelInfo?.output_shape}</span></span>
          <span><span className="text-slate-500">Uploaded:</span> <span className="text-white">{modelInfo?.upload_time ? new Date(modelInfo.upload_time).toLocaleTimeString() : 'N/A'}</span></span>
        </div>
      </div>

      {/* ── KPI Grid ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-4">
        <KPI label="Total Customers" value={stats.total_customers.toLocaleString()} icon={Users} color="blue" />
        <KPI label="Processed" value={stats.processed_customers.toLocaleString()} icon={Activity} color="indigo" />
        <KPI label="Predicted Theft" value={stats.predicted_theft.toLocaleString()}
          sub={`${(stats.theft_rate * 100).toFixed(1)}% of total`} icon={AlertTriangle} color="red" />
        <KPI label="Predicted Normal" value={stats.predicted_normal.toLocaleString()} icon={CheckCircle2} color="emerald" />
        <KPI label="Avg Confidence" value={`${(stats.avg_confidence * 100).toFixed(1)}%`} icon={Target} color="yellow" />
        <KPI label="Avg Risk Score" value={stats.avg_risk_score.toFixed(1)} sub="out of 100" icon={TrendingUp} color="orange" />
      </div>

      {/* ── Evaluation KPIs (if FLAG present) ── */}
      {stats.has_flag && stats.accuracy !== undefined && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {[
            { label: 'Accuracy', value: `${(stats.accuracy * 100).toFixed(2)}%`, color: 'green' },
            { label: 'Precision', value: `${((stats.precision ?? 0) * 100).toFixed(2)}%`, color: 'blue' },
            { label: 'Recall', value: `${((stats.recall ?? 0) * 100).toFixed(2)}%`, color: 'purple' },
            { label: 'F1 Score', value: `${((stats.f1_score ?? 0) * 100).toFixed(2)}%`, color: 'yellow' },
            { label: 'ROC-AUC', value: `${((stats.roc_auc ?? 0) * 100).toFixed(2)}%`, color: 'pink' },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
              <p className={`text-2xl font-black text-${color}-400`}>{value}</p>
              <p className="text-slate-400 text-sm mt-1">{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* ── Charts Grid ── */}
      {charts && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">

          {/* Pie: Normal vs Theft */}
          {charts.pie && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
              <h3 className="text-white font-bold mb-3 flex items-center gap-2"><BarChart3 className="w-4 h-4 text-blue-400" />Normal vs Theft</h3>
              <Plot
                data={[{
                  type: 'pie',
                  labels: charts.pie.labels,
                  values: charts.pie.values,
                  marker: { colors: ['#22c55e', '#ef4444'] },
                  textinfo: 'label+percent',
                  textfont: { color: '#fff', size: 13 },
                  hole: 0.4,
                }]}
                layout={{ ...PLOTLY_LAYOUT, height: 260 }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Risk Distribution Histogram */}
          {charts.risk_distribution && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
              <h3 className="text-white font-bold mb-3">Risk Score Distribution</h3>
              <Plot
                data={[
                  {
                    type: 'histogram',
                    x: charts.risk_distribution.values.filter((_, i) => charts.risk_distribution!.labels[i] === 'Theft'),
                    name: 'Theft',
                    marker: { color: '#ef4444' },
                    opacity: 0.8,
                    nbinsx: 30,
                  } as any,
                  {
                    type: 'histogram',
                    x: charts.risk_distribution.values.filter((_, i) => charts.risk_distribution!.labels[i] === 'Normal'),
                    name: 'Normal',
                    marker: { color: '#22c55e' },
                    opacity: 0.8,
                    nbinsx: 30,
                  } as any,
                ]}
                layout={{ ...PLOTLY_LAYOUT, height: 260, barmode: 'overlay', xaxis: { ...PLOTLY_LAYOUT.xaxis, title: { text: 'Risk Score' } }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: { text: 'Count' } } }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Top 10 Highest Risk */}
          {charts.top10_high && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
              <h3 className="text-white font-bold mb-3 flex items-center gap-2"><AlertTriangle className="w-4 h-4 text-red-400" />Top 10 Highest Risk</h3>
              <Plot
                data={[{
                  type: 'bar',
                  x: charts.top10_high.risks,
                  y: charts.top10_high.ids,
                  orientation: 'h',
                  marker: { color: charts.top10_high.risks.map(r => r > 75 ? '#ef4444' : r > 50 ? '#f97316' : '#eab308') },
                }]}
                layout={{ ...PLOTLY_LAYOUT, height: 260, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: { text: 'Risk Score' }, range: [0, 100] } }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Risk vs Confidence Scatter */}
          {charts.scatter && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
              <h3 className="text-white font-bold mb-3">Risk vs Confidence</h3>
              <Plot
                data={[
                  {
                    type: 'scatter', mode: 'markers',
                    x: charts.scatter.confidence.filter((_, i) => charts.scatter!.status[i] === 'Theft'),
                    y: charts.scatter.risk.filter((_, i) => charts.scatter!.status[i] === 'Theft'),
                    name: 'Theft', marker: { color: '#ef4444', size: 4, opacity: 0.6 },
                  },
                  {
                    type: 'scatter', mode: 'markers',
                    x: charts.scatter.confidence.filter((_, i) => charts.scatter!.status[i] === 'Normal'),
                    y: charts.scatter.risk.filter((_, i) => charts.scatter!.status[i] === 'Normal'),
                    name: 'Normal', marker: { color: '#22c55e', size: 4, opacity: 0.4 },
                  },
                ]}
                layout={{ ...PLOTLY_LAYOUT, height: 260, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: { text: 'Confidence' } }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: { text: 'Risk Score' } } }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Top 10 Lowest Risk */}
          {charts.top10_low && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
              <h3 className="text-white font-bold mb-3 flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-emerald-400" />Top 10 Lowest Risk</h3>
              <Plot
                data={[{
                  type: 'bar', x: charts.top10_low.risks, y: charts.top10_low.ids,
                  orientation: 'h',
                  marker: { color: '#22c55e' },
                }]}
                layout={{ ...PLOTLY_LAYOUT, height: 260, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: { text: 'Risk Score' }, range: [0, 50] } }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* ROC Curve */}
          {charts.roc?.fpr && charts.roc.fpr.length > 0 && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
              <h3 className="text-white font-bold mb-3">ROC Curve</h3>
              <Plot
                data={[
                  { type: 'scatter', mode: 'lines', x: charts.roc.fpr, y: charts.roc.tpr, name: 'CNN-LSTM', line: { color: '#3b82f6', width: 2 } },
                  { type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], name: 'Random', line: { color: '#475569', dash: 'dash' } },
                ]}
                layout={{ ...PLOTLY_LAYOUT, height: 260, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: { text: 'FPR' } }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: { text: 'TPR' } } }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Confusion Matrix */}
          {charts.confusion && charts.confusion.length === 2 && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
              <h3 className="text-white font-bold mb-3">Confusion Matrix</h3>
              <Plot
                data={[{
                  type: 'heatmap',
                  z: charts.confusion,
                  x: ['Predicted Normal', 'Predicted Theft'],
                  y: ['Actual Normal', 'Actual Theft'],
                  colorscale: [[0, '#0f172a'], [1, '#3b82f6']],
                  showscale: false,
                  text: charts.confusion.map(row => row.map(String)) as any,
                  texttemplate: '%{text}',
                  textfont: { size: 18, color: '#fff' as any },
                }]}
                layout={{ ...PLOTLY_LAYOUT, height: 260 }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Precision-Recall Curve */}
          {charts.pr?.precision && charts.pr.precision.length > 0 && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
              <h3 className="text-white font-bold mb-3">Precision-Recall Curve</h3>
              <Plot
                data={[{
                  type: 'scatter', mode: 'lines',
                  x: charts.pr.recall, y: charts.pr.precision,
                  name: 'CNN-LSTM', line: { color: '#a855f7', width: 2 },
                }]}
                layout={{ ...PLOTLY_LAYOUT, height: 260, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: { text: 'Recall' } }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: { text: 'Precision' } } }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DashboardPage;
