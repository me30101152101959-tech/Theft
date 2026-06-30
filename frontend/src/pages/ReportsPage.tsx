import React, { useState } from 'react';
import { FileText, Download, BarChart3, FileSpreadsheet, Loader2 } from 'lucide-react';
import { exportCSV, exportJSON, exportReport, getDashboardStats } from '../api/client';
import type { DashboardStats, ModelInfo } from '../types';
import toast from 'react-hot-toast';

interface Props { modelInfo: ModelInfo | null; }

const ReportsPage: React.FC<Props> = ({ modelInfo }) => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState<string | null>(null);

  React.useEffect(() => {
    getDashboardStats().then(r => setStats(r.data)).catch(() => {});
  }, []);

  const download = async (type: 'csv' | 'json' | 'report') => {
    setLoading(type);
    try {
      const fn = type === 'csv' ? exportCSV : type === 'json' ? exportJSON : exportReport;
      const res = await fn();
      const ext = type === 'report' ? 'txt' : type;
      const name = `etd_${type === 'report' ? 'summary_report' : 'predictions'}.${ext}`;
      const url = URL.createObjectURL(new Blob([res.data]));
      const a = document.createElement('a'); a.href = url; a.download = name; a.click();
      URL.revokeObjectURL(url);
      toast.success(`${name} downloaded`);
    } catch { toast.error('Download failed'); }
    finally { setLoading(null); }
  };

  const ReportCard = ({ title, desc, ext, icon: Icon, color, type }: any) => (
    <div className={`bg-slate-900 border border-slate-800 rounded-2xl p-6 hover:border-${color}-500/30 transition-colors`}>
      <div className={`p-3 bg-${color}-500/15 rounded-xl w-fit mb-4`}>
        <Icon className={`w-6 h-6 text-${color}-400`} />
      </div>
      <h3 className="text-white font-bold text-lg mb-1">{title}</h3>
      <p className="text-slate-400 text-sm mb-4">{desc}</p>
      <button
        onClick={() => download(type)}
        disabled={loading === type}
        className={`w-full py-2.5 rounded-xl bg-${color}-600/20 hover:bg-${color}-600/40 text-${color}-400 border border-${color}-500/30 font-semibold text-sm flex items-center justify-center gap-2 transition-all disabled:opacity-50`}
      >
        {loading === type ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
        Download {ext}
      </button>
    </div>
  );

  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-black text-white">Reports & Exports</h1>
        <p className="text-slate-400 text-sm">Download prediction data and evaluation reports.</p>
      </div>

      {/* Stat summary */}
      {stats && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
          <h2 className="text-white font-bold mb-4">Current Dataset Summary</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            {[
              { label: 'Dataset', value: stats.dataset_name },
              { label: 'Total Customers', value: stats.total_customers?.toLocaleString() },
              { label: 'Predicted Theft', value: stats.predicted_theft?.toLocaleString() },
              { label: 'Predicted Normal', value: stats.predicted_normal?.toLocaleString() },
              { label: 'Model', value: modelInfo?.model_name || 'CNN-LSTM' },
              { label: 'Architecture', value: 'CNN-LSTM' },
              ...(stats.has_flag ? [
                { label: 'Accuracy', value: `${((stats.accuracy || 0) * 100).toFixed(2)}%` },
                { label: 'F1 Score', value: `${((stats.f1_score || 0) * 100).toFixed(2)}%` },
              ] : []),
            ].map(({ label, value }) => (
              <div key={label} className="bg-slate-800/50 rounded-xl p-3">
                <p className="text-slate-500 text-xs mb-1">{label}</p>
                <p className="text-white font-bold text-sm truncate">{value}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Report cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <ReportCard
          title="Predictions CSV"
          desc="All customers with prediction results, probability, confidence, and risk scores. Import into Excel or any analytics tool."
          ext="CSV"
          icon={FileSpreadsheet}
          color="emerald"
          type="csv"
        />
        <ReportCard
          title="Predictions JSON"
          desc="Complete prediction dataset in JSON format. Includes all fields for API integration or further analysis."
          ext="JSON"
          icon={FileText}
          color="blue"
          type="json"
        />
        <ReportCard
          title="Executive Summary"
          desc="Text report with model info, prediction statistics, theft rates, and evaluation metrics (if labels available)."
          ext="TXT"
          icon={BarChart3}
          color="purple"
          type="report"
        />
      </div>

      {/* Info */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 text-sm text-slate-400">
        <h3 className="text-white font-bold mb-2">Export Contents</h3>
        <ul className="space-y-1.5 list-disc list-inside">
          <li>Customer ID (CONS_NO)</li>
          <li>CNN-LSTM Prediction (0=Normal, 1=Theft)</li>
          <li>Theft Probability (0.0–1.0)</li>
          <li>Confidence Score (0.0–1.0)</li>
          <li>Risk Score (0–100)</li>
          <li>Status (Normal / Theft)</li>
          <li>Ground Truth Flag (if available in dataset)</li>
        </ul>
      </div>
    </div>
  );
};

export default ReportsPage;
