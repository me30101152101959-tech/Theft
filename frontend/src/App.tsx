import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';

import Layout from './components/Layout';
import UploadPage from './pages/UploadPage';
import DashboardPage from './pages/DashboardPage';
import CustomersPage from './pages/CustomersPage';
import PredictPage from './pages/PredictPage';
import ReportsPage from './pages/ReportsPage';
import CopilotPage from './pages/CopilotPage';
import SettingsPage from './pages/SettingsPage';

import type { ModelInfo, DatasetSummary, AppStep } from './types';
import { uploadDataset, resetModel } from './api/client';

const qc = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 30_000 } }
});

const App: React.FC = () => {
  const [step, setStep] = useState<AppStep>('upload_model');
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [isDark, setIsDark] = useState(true);
  const [lang, setLang] = useState('en');
  const [threshold, setThreshold] = useState(0.5);

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

  // While still uploading, show the upload page
  if (step !== 'ready') {
    return (
      <QueryClientProvider client={qc}>
        <div className={isDark ? 'dark' : ''}>
          <UploadPage onReady={handleReady} />
          <Toaster position="bottom-right" toastOptions={{ style: { background: '#1e293b', color: '#f1f5f9', border: '1px solid #334155' } }} />
        </div>
      </QueryClientProvider>
    );
  }

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
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<DashboardPage modelInfo={modelInfo} />} />
              <Route path="/customers" element={<CustomersPage />} />
              <Route path="/predict" element={<PredictPage modelInfo={modelInfo} threshold={threshold} />} />
              <Route path="/reports" element={<ReportsPage modelInfo={modelInfo} />} />
              <Route path="/copilot" element={<CopilotPage />} />
              <Route path="/settings" element={
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
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: { background: '#1e293b', color: '#f1f5f9', border: '1px solid #334155' }
          }}
        />
      </div>
    </QueryClientProvider>
  );
};

export default App;
