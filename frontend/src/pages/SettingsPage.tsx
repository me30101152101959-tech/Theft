import React, { useState } from 'react';
import { Settings2, Moon, Sun, Globe2, SlidersHorizontal, Shield } from 'lucide-react';
import { updateThreshold } from '../api/client';
import type { ModelInfo } from '../types';
import toast from 'react-hot-toast';

interface Props {
  modelInfo: ModelInfo | null;
  threshold: number;
  onThresholdChange: (t: number) => void;
  isDark: boolean;
  onToggleDark: () => void;
  lang: string;
  onLangChange: (l: string) => void;
}

const SettingsPage: React.FC<Props> = ({
  modelInfo, threshold, onThresholdChange, isDark, onToggleDark, lang, onLangChange
}) => {
  const [localThreshold, setLocalThreshold] = useState(threshold);
  const [saving, setSaving] = useState(false);

  const applyThreshold = async () => {
    setSaving(true);
    try {
      await updateThreshold(localThreshold);
      onThresholdChange(localThreshold);
      toast.success(`Threshold updated to ${localThreshold.toFixed(2)}`);
    } catch { toast.error('Failed to update threshold'); }
    finally { setSaving(false); }
  };

  const Card = ({ title, children }: { title: string; children: React.ReactNode }) => (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
      <h2 className="text-white font-bold text-lg">{title}</h2>
      {children}
    </div>
  );

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-black text-white flex items-center gap-2"><Settings2 className="w-6 h-6 text-slate-400" />Settings</h1>
        <p className="text-slate-400 text-sm">Configure theme, language, and prediction parameters.</p>
      </div>

      {/* Theme */}
      <Card title="🎨 Theme">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-white font-medium">Application Theme</p>
            <p className="text-slate-400 text-sm">Toggle between dark and light mode.</p>
          </div>
          <button onClick={onToggleDark}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-xl border font-semibold text-sm transition-all
              ${isDark ? 'bg-slate-800 border-slate-700 text-yellow-400 hover:border-yellow-500/40' : 'bg-yellow-50 border-yellow-200 text-yellow-700 hover:bg-yellow-100'}`}>
            {isDark ? <><Sun className="w-4 h-4" />Light Mode</> : <><Moon className="w-4 h-4" />Dark Mode</>}
          </button>
        </div>
      </Card>

      {/* Language */}
      <Card title="🌐 Language">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-white font-medium">Interface Language</p>
            <p className="text-slate-400 text-sm">Also affects AI Copilot responses.</p>
          </div>
          <div className="flex items-center gap-2 bg-slate-800 border border-slate-700 rounded-xl p-1">
            <Globe2 className="w-4 h-4 text-slate-400 ml-2" />
            {[['en', 'English'], ['ar', 'العربية']].map(([code, label]) => (
              <button key={code} onClick={() => onLangChange(code)}
                className={`px-4 py-2 rounded-lg text-sm font-bold transition-all ${lang === code ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}>
                {label}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {/* Prediction Threshold */}
      <Card title="⚡ Prediction Threshold">
        <p className="text-slate-400 text-sm">
          Customers with theft probability ≥ threshold are classified as <span className="text-red-400 font-bold">Theft</span>.
          Lower threshold → more theft detections. Higher → fewer false positives.
        </p>
        <div className="space-y-3">
          <div className="flex items-center gap-4">
            <span className="text-slate-500 text-xs w-20">Low risk (0.0)</span>
            <input
              type="range" min="0" max="1" step="0.01"
              value={localThreshold}
              onChange={e => setLocalThreshold(Number(e.target.value))}
              className="flex-1 accent-blue-500"
            />
            <span className="text-slate-500 text-xs w-24 text-right">High risk (1.0)</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400 text-sm">Current threshold: <span className="text-blue-400 font-mono font-bold text-lg">{localThreshold.toFixed(2)}</span></span>
            <div className="flex gap-2">
              {[0.3, 0.4, 0.5, 0.6, 0.7].map(t => (
                <button key={t} onClick={() => setLocalThreshold(t)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-mono font-bold transition-colors ${Math.abs(localThreshold - t) < 0.005 ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}>
                  {t}
                </button>
              ))}
            </div>
          </div>
          <button onClick={applyThreshold} disabled={saving}
            className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-colors">
            <SlidersHorizontal className="w-4 h-4" />
            {saving ? 'Applying...' : 'Apply Threshold to All Predictions'}
          </button>
        </div>
      </Card>

      {/* Model Info */}
      {modelInfo?.loaded && (
        <Card title="🤖 Loaded Model">
          <div className="space-y-2 font-mono text-xs">
            {[
              ['Name', modelInfo.model_name],
              ['Architecture', 'CNN-LSTM (Enforced)'],
              ['Input Shape', modelInfo.input_shape],
              ['Output Shape', modelInfo.output_shape],
              ['Parameters', modelInfo.total_params_fmt],
              ['Dual Input', modelInfo.is_dual_input ? `Yes (seq + ${modelInfo.stat_input_size} stat features)` : 'No'],
              ['Loaded At', new Date(modelInfo.upload_time!).toLocaleString()],
            ].map(([k, v]) => (
              <div key={k as string} className="flex items-center justify-between py-1.5 border-b border-slate-800">
                <span className="text-slate-500">{k}</span>
                <span className="text-white font-bold">{v}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Security */}
      <Card title="🔒 Security">
        <div className="space-y-2 text-sm text-slate-400">
          {[
            'Only CNN-LSTM architecture accepted (.keras or .h5)',
            'BiGRU, BiLSTM, Ensemble, and Transfer Learning models are rejected',
            'FLAG column is never used during prediction — evaluation only',
            'All predictions are generated by the uploaded model exclusively',
            'No fallback or mock predictions are ever used',
          ].map(s => (
            <div key={s} className="flex items-start gap-2">
              <Shield className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
              <span>{s}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};

export default SettingsPage;
