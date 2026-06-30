import React, { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { RefreshCw } from 'lucide-react';

import Layout from './components/Layout';
import UploadPage from './pages/UploadPage';
import DashboardPage from './pages/DashboardPage';
import CustomersPage from './pages/CustomersPage';
import PredictPage from './pages/PredictPage';
import ReportsPage from './pages/ReportsPage';
import CopilotPage from './pages/CopilotPage';
import SettingsPage from './pages/SettingsPage';

import type { ModelInfo, DatasetSummary, AppStep } from './types';
import { healthCheck, resetModel } from './api/client';

const qc = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 30_000 } },
});

const TOAST_STYLE = {
  style: { background: '#1e293b', color: '#f1f5f9', border: '1px solid #334155' },
};

const App: React.FC = () => {
  const [step,      setStep]      = useState<AppStep | 'checking'>('checking');
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [summary,   setSummary]   = useState<DatasetSummary | null>(null);
  const [isDark,    setIsDark]    = useState(true);
  const [lang,      setLang]      = useState('en');
  const [threshold, setThreshold] = useState(0.5);

  // ── On mount: check if backend already has model + SQLite data ─────────
  useEffect(() => {
    (async () => {
      try {
        const { data } = await healthCheck();
        if (data.model_loaded && data.dataset_loaded) {
          // Both ready — skip upload wizard, go straight to dashboard
          const info = data.model_info;
          setModelInfo({
            model_name:      info.model_name      ?? '',
            architecture:    info.architecture    ?? 'CNN-LSTM',
            input_shape:     info.input_shape     ?? '',
            output_shape:    info.output_shape    ?? '',
            total_params:    info.total_params    ?? 0,
            total_params_fmt:info.total_params_fmt ?? '',
            upload_time:     info.upload_time     ?? '',
            is_dual_input:   info.is_dual_input   ?? false,
            loaded:          true,
          });
          setSummary({
            total:        data.prediction_count ?? 0,
            theft:        0,
            normal:       0,
            dataset_name: data.dataset_name ?? '',
            has_flag:     false,
          });
          setStep('ready');
        } else if (data.model_loaded) {
          // Model loaded but no dataset — skip model upload step
          const info = data.model_info;
          setModelInfo({
            model_name:      info.model_name      ?? '',
            architecture:    info.architecture    ?? 'CNN-LSTM',
            input_shape:     info.input_shape     ?? '',
            output_shape:    info.output_shape    ?? '',
            total_params:    info.total_params    ?? 0,
            total_params_fmt:info.total_params_fmt ?? '',
            upload_time:     info.upload_time     ?? '',
            is_dual_input:   info.is_dual_input   ?? false,
            loaded:          true,
          });
          setStep('upload_dataset');
        } else {
          setStep('upload_model');
        }
      } catch {
        // Backend unreachable — show upload flow
        setStep('upload_model');
      }
    })();
  }, []);

  const handleReady = (m: ModelInfo, s: DatasetSummary) => {
    setModelInfo(m);
    setSummary(s);
    setStep('ready');
  };

  const handleChangeModel = async () => {
    await resetModel().catch(() => {});
    setModelInfo(null);
    setSummary(null);
    setStep('upload_model');
  };

  const handleChangeDataset = () => {
    setSummary(null);
    setStep('upload_dataset');
  };

  // ── Loading screen while checking backend ─────────────────────────────
  if (step === 'checking') {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-10 h-10 text-blue-400 animate-spin mx-auto mb-4" />
          <p className="text-slate-300 text-lg font-semibold">Connecting to backend…</p>
          <p className="text-slate-500 text-sm mt-1">Checking SQLite for existing data</p>
        </div>
      </div>
    );
  }

  // ── Upload wizard ──────────────────────────────────────────────────────
  if (step !== 'ready') {
    return (
      <QueryClientProvider client={qc}>
        <div className={isDark ? 'dark' : ''}>
          <UploadPage initialStep={step} onReady={handleReady} />
          <Toaster position="bottom-right" toastOptions={TOAST_STYLE} />
        </div>
      </QueryClientProvider>
    );
  }

  // ── Main app ───────────────────────────────────────────────────────────
  return (
    <QueryClientProvider client={qc}>
      <div className={isDark ? 'dark' : ''}>
        <BrowserRouter>
          <Layout
            modelInfo={modelInfo}
            summary={summary}
            isDark={isDark}
            onToggleDark={() => setIsDark(d => !d)}
            onChangeModel={handleChangeModel}
            onChangeDataset={handleChangeDataset}
          >
            <Routes>
              <Route path="/"          element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<DashboardPage modelInfo={modelInfo} />} />
              <Route path="/customers" element={<CustomersPage />} />
              <Route path="/predict"   element={<PredictPage modelInfo={modelInfo} threshold={threshold} />} />
              <Route path="/reports"   element={<ReportsPage modelInfo={modelInfo} />} />
              <Route path="/copilot"   element={<CopilotPage />} />
              <Route path="/settings"  element={
                <SettingsPage
                  modelInfo={modelInfo}
                  threshold={threshold}
                  onThresholdChange={setThreshold}
                  isDark={isDark}
                  onToggleDark={() => setIsDark(d => !d)}
                  lang={lang}
                  onLangChange={setLang}
                />
              } />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </Layout>
        </BrowserRouter>
        <Toaster position="bottom-right" toastOptions={TOAST_STYLE} />
      </div>
    </QueryClientProvider>
  );
};

export default App;
