import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
  Shield, LayoutDashboard, Users, Zap, FileText,
  BrainCircuit, Settings, ChevronLeft, ChevronRight,
  Cpu, Upload, Moon, Sun
} from 'lucide-react';
import type { ModelInfo, DatasetSummary } from '../types';

interface Props {
  children: React.ReactNode;
  modelInfo: ModelInfo | null;
  summary: DatasetSummary | null;
  isDark: boolean;
  onToggleDark: () => void;
  onChangeModel: () => void;
  onChangeDataset: () => void;
}

const NAV = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/customers', icon: Users, label: 'Customers' },
  { to: '/predict', icon: Zap, label: 'Manual Predict' },
  { to: '/reports', icon: FileText, label: 'Reports' },
  { to: '/copilot', icon: BrainCircuit, label: 'AI Copilot' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

const Layout: React.FC<Props> = ({
  children, modelInfo, summary, isDark, onToggleDark,
  onChangeModel, onChangeDataset
}) => {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className={`flex h-screen ${isDark ? 'dark' : ''} overflow-hidden`}>
      {/* Sidebar */}
      <aside className={`flex flex-col bg-slate-900 border-r border-slate-800 transition-all duration-300 ${collapsed ? 'w-16' : 'w-64'}`}>
        {/* Logo */}
        <div className="flex items-center gap-3 p-4 border-b border-slate-800 h-16">
          <div className="flex-shrink-0 w-8 h-8 bg-blue-600 rounded-xl flex items-center justify-center shadow-md shadow-blue-500/30">
            <Shield className="w-4 h-4 text-white" />
          </div>
          {!collapsed && (
            <div className="overflow-hidden">
              <p className="text-white font-black text-sm leading-none">ETD-XAI</p>
              <p className="text-blue-400 text-[10px] font-semibold">Enterprise v1.0</p>
            </div>
          )}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="ml-auto text-slate-500 hover:text-white transition-colors"
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* Model badge */}
        {!collapsed && modelInfo?.loaded && (
          <div className="mx-3 mt-3 p-3 bg-blue-500/10 border border-blue-500/20 rounded-xl">
            <div className="flex items-center gap-2 mb-1">
              <Cpu className="w-3.5 h-3.5 text-blue-400" />
              <span className="text-blue-400 text-[10px] font-bold uppercase tracking-wider">Active Model</span>
            </div>
            <p className="text-white text-xs font-mono truncate">{modelInfo.model_name}</p>
            <p className="text-slate-400 text-[10px]">CNN-LSTM · {modelInfo.total_params_fmt} params</p>
          </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all
                ${isActive
                  ? 'bg-blue-600 text-white shadow-md shadow-blue-500/20'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                } ${collapsed ? 'justify-center' : ''}`
              }
              title={collapsed ? label : undefined}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {!collapsed && <span>{label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Bottom actions */}
        {!collapsed && (
          <div className="p-3 border-t border-slate-800 space-y-2">
            <button
              onClick={onChangeModel}
              className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-400 hover:text-blue-400 hover:bg-slate-800 rounded-lg transition-colors"
            >
              <Upload className="w-3.5 h-3.5" />
              Load New Model
            </button>
            <button
              onClick={onChangeDataset}
              className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-400 hover:text-purple-400 hover:bg-slate-800 rounded-lg transition-colors"
            >
              <Upload className="w-3.5 h-3.5" />
              Load New Dataset
            </button>
            <button
              onClick={onToggleDark}
              className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
            >
              {isDark ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
              {isDark ? 'Light Mode' : 'Dark Mode'}
            </button>
          </div>
        )}
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-slate-950 dark:bg-slate-950">
        {children}
      </main>
    </div>
  );
};

export default Layout;
