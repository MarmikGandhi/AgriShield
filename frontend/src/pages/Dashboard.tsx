// src/pages/Dashboard.tsx

import React, { useEffect, useState } from 'react';
import { useLanguage } from '../context/LanguageContext';
import { useAuth } from '../context/AuthContext';
import { dashboardApi, weatherApi, marketApi } from '../services/api';
import { DashboardSummary, WeatherRiskData, MandiPriceItem, ScanRecord } from '../types';
import { WeatherRiskCard } from '../components/WeatherRiskCard';
import { MandiPricesTicker } from '../components/MandiPricesTicker';
import {
  Scan,
  ShieldCheck,
  AlertTriangle,
  AlertCircle,
  ArrowRight,
  Clock,
  ChevronRight,
  Sparkles,
  RefreshCw,
  Cpu,
  Bot,
  HelpCircle,
  CheckCircle2
} from 'lucide-react';

interface DashboardProps {
  onNavigate: (page: string, params?: any) => void;
  onSelectScan: (scan: ScanRecord) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onNavigate, onSelectScan }) => {
  const { t } = useLanguage();
  const { user } = useAuth();
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [weatherRisk, setWeatherRisk] = useState<WeatherRiskData | null>(null);
  const [mandiPrices, setMandiPrices] = useState<MandiPriceItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const [sumRes, weatherRes, pricesRes] = await Promise.all([
        dashboardApi.getSummary().catch(() => null),
        weatherApi.getRisk('Anand').catch(() => null),
        marketApi.getPrices('Anand').catch(() => []),
      ]);

      if (sumRes) setSummary(sumRes);
      if (weatherRes) setWeatherRisk(weatherRes);
      if (pricesRes) setMandiPrices(pricesRes);
    } catch (e) {
      console.error('Failed to load dashboard data:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  return (
    <div className="space-y-6 sm:space-y-8 animate-in fade-in duration-200">
      
      {/* 1. Welcome Section */}
      <div className="bg-white rounded-2xl p-6 sm:p-7 border border-slate-200/90 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="max-w-2xl space-y-2">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200/70 text-emerald-800 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
            <span>{t.dashboard.regionBadge}</span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
            {t.dashboard.welcome}, {user?.full_name || 'Farmer'}
          </h1>
          <p className="text-slate-600 text-xs sm:text-sm leading-relaxed">
            AI foliar diagnosis engine is standing by. Upload leaf photos for instant disease detection, check weather risks, and monitor local APMC Mandi rates.
          </p>
        </div>

        <div className="flex sm:flex-col items-center sm:items-end gap-3 shrink-0">
          <button
            onClick={() => onNavigate('scanner')}
            className="w-full sm:w-auto px-5 py-2.5 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white font-semibold text-xs sm:text-sm transition shadow-xs flex items-center justify-center gap-2"
          >
            <Scan className="w-4 h-4 text-emerald-200" />
            <span>{t.dashboard.startScanBtn}</span>
            <ArrowRight className="w-4 h-4" />
          </button>
          <span className="text-[11px] text-slate-500 hidden sm:block">
            Fast diagnostic pipeline ready
          </span>
        </div>
      </div>

      {/* 2. Summary KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        
        <div className="bg-white rounded-xl p-4 sm:p-5 border border-slate-200/90 shadow-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">{t.dashboard.totalScans}</span>
            <div className="w-8 h-8 rounded-lg bg-slate-100 text-slate-700 flex items-center justify-center">
              <Scan className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-bold text-slate-900 mb-0.5">{summary?.total_scans ?? 0}</div>
          <p className="text-[11px] text-slate-500">Total processed field samples</p>
        </div>

        <div className="bg-white rounded-xl p-4 sm:p-5 border border-slate-200/90 shadow-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">{t.dashboard.healthyFoliage}</span>
            <div className="w-8 h-8 rounded-lg bg-emerald-50 text-emerald-700 flex items-center justify-center">
              <ShieldCheck className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-bold text-emerald-700 mb-0.5">{summary?.healthy_scans ?? 0}</div>
          <p className="text-[11px] text-slate-500">Asymptomatic healthy foliage</p>
        </div>

        <div className="bg-white rounded-xl p-4 sm:p-5 border border-slate-200/90 shadow-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">{t.dashboard.diseasedCrops}</span>
            <div className="w-8 h-8 rounded-lg bg-amber-50 text-amber-700 flex items-center justify-center">
              <AlertTriangle className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-bold text-amber-700 mb-0.5">{summary?.diseased_scans ?? 0}</div>
          <p className="text-[11px] text-slate-500">Identified crop stress cases</p>
        </div>

        <div className="bg-white rounded-xl p-4 sm:p-5 border border-slate-200/90 shadow-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">{t.dashboard.criticalAlerts}</span>
            <div className="w-8 h-8 rounded-lg bg-rose-50 text-rose-700 flex items-center justify-center">
              <AlertCircle className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-bold text-rose-700 mb-0.5">{summary?.critical_scans ?? 0}</div>
          <p className="text-[11px] text-slate-500">High-severity advisory dispatches</p>
        </div>

      </div>

      {/* 3. Operational Information Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column (Col-span 2): Weather Risk & Mandi Prices */}
        <div className="lg:col-span-2 space-y-6">
          {weatherRisk ? (
            <WeatherRiskCard data={weatherRisk} />
          ) : (
            <div className="bg-white rounded-xl p-6 border border-slate-200 text-center">
              <RefreshCw className="w-5 h-5 text-slate-400 mx-auto mb-2 animate-spin" />
              <p className="text-xs text-slate-500 font-medium">Checking live ICAR Anand weather risk...</p>
            </div>
          )}

          {/* Mandi Prices */}
          {mandiPrices.length > 0 && (
            <MandiPricesTicker prices={mandiPrices} />
          )}
        </div>

        {/* Right Column (Col-span 1): Agricultural Advisory & Quick Actions */}
        <div className="space-y-6">
          
          {/* AI Pipeline Readiness Card */}
          <div className="bg-white rounded-xl p-5 border border-slate-200/90 shadow-xs space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-lg bg-emerald-50 text-emerald-700 flex items-center justify-center">
                  <Cpu className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="font-bold text-xs text-slate-900">AI Diagnostic Pipeline</h3>
                  <p className="text-[10px] text-slate-500">Model Verification Status</p>
                </div>
              </div>
              <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-800 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full">
                <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                Ready
              </span>
            </div>

            <div className="space-y-2.5 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-600">Dual-CNN Ensemble</span>
                <span className="font-mono text-[11px] font-semibold text-slate-800">EffB0 + MobV3 (38 cls)</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-600">Foliar Segmentation</span>
                <span className="font-mono text-[11px] font-semibold text-slate-800">PlantSeg U-Net</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-600">Quality Guard</span>
                <span className="font-mono text-[11px] font-semibold text-slate-800">OpenCV Pre-filter</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-600">Visual Attribution</span>
                <span className="font-mono text-[11px] font-semibold text-slate-800">Grad-CAM (EffB0)</span>
              </div>
            </div>

            <button
              onClick={() => onNavigate('datasets')}
              className="w-full text-center text-xs font-semibold text-emerald-800 hover:text-emerald-900 pt-2 border-t border-slate-100 block transition"
            >
              Inspect Model Weights & Audit Metrics →
            </button>
          </div>

          {/* Quick Support Card */}
          <div className="bg-slate-50 rounded-xl p-5 border border-slate-200/90 space-y-3">
            <h3 className="font-bold text-xs text-slate-900 uppercase tracking-wider">Farmer Assistance</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Have questions regarding fertilizer dosage, pest control schedules, or an uncertain crop diagnosis?
            </p>
            
            <div className="space-y-2 pt-1">
              <button
                onClick={() => onNavigate('kisanBot')}
                className="w-full flex items-center justify-between p-2.5 bg-white hover:bg-emerald-50/50 border border-slate-200 rounded-lg text-xs font-medium text-slate-800 transition"
              >
                <div className="flex items-center gap-2">
                  <Bot className="w-4 h-4 text-emerald-700" />
                  <span>Ask Kisan AI Assistant</span>
                </div>
                <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
              </button>

              <button
                onClick={() => onNavigate('askExpert')}
                className="w-full flex items-center justify-between p-2.5 bg-white hover:bg-indigo-50/50 border border-slate-200 rounded-lg text-xs font-medium text-slate-800 transition"
              >
                <div className="flex items-center gap-2">
                  <HelpCircle className="w-4 h-4 text-indigo-700" />
                  <span>Consult University Agronomist</span>
                </div>
                <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
              </button>
            </div>
          </div>

        </div>

      </div>

      {/* 4. Recent Field Activity Section */}
      <div className="bg-white rounded-xl p-5 sm:p-6 border border-slate-200/90 shadow-xs space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <div>
            <h3 className="font-bold text-sm text-slate-900">{t.dashboard.recentScansTitle}</h3>
            <p className="text-xs text-slate-500">Review recent foliar scans, Grad-CAM heatmaps, and treatment advice</p>
          </div>
          <button
            onClick={() => onNavigate('history')}
            className="text-xs font-semibold text-emerald-700 hover:text-emerald-800 flex items-center gap-1 transition"
          >
            <span>View All Records</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {summary?.recent_scans && summary.recent_scans.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
            {summary.recent_scans.slice(0, 6).map((scan) => (
              <div
                key={scan.id}
                onClick={() => {
                  onSelectScan(scan);
                  onNavigate('scanner');
                }}
                className="bg-white hover:bg-slate-50/80 rounded-lg p-3 border border-slate-200 hover:border-slate-300 transition cursor-pointer flex gap-3 group"
              >
                <div className="w-14 h-14 rounded-md bg-slate-100 overflow-hidden shrink-0 border border-slate-200/80">
                  <img
                    src={scan.image_url}
                    alt={scan.crop}
                    className="w-full h-full object-cover group-hover:scale-105 transition"
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-0.5">
                    <span className="font-semibold text-xs text-slate-900 truncate">{scan.crop}</span>
                    <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                      scan.is_healthy
                        ? 'bg-emerald-50 text-emerald-800 border border-emerald-200/60'
                        : 'bg-rose-50 text-rose-800 border border-rose-200/60'
                    }`}>
                      {scan.is_healthy ? 'Healthy' : scan.severity_level}
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 truncate">{scan.disease_condition}</p>
                  <p className="text-[10px] text-slate-400 mt-1 flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {new Date(scan.created_at).toLocaleDateString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-8 text-center bg-slate-50/60 rounded-xl border border-dashed border-slate-200">
            <Scan className="w-7 h-7 text-slate-400 mx-auto mb-2" />
            <p className="text-xs font-medium text-slate-700">{t.dashboard.noScansYet}</p>
            <button
              onClick={() => onNavigate('scanner')}
              className="mt-3 px-4 py-2 bg-emerald-700 hover:bg-emerald-800 text-white rounded-lg text-xs font-semibold transition shadow-xs"
            >
              Scan Your First Leaf
            </button>
          </div>
        )}
      </div>

    </div>
  );
};
