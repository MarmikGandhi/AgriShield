// src/components/QualityGuardModal.tsx

import React from 'react';
import { QualityGuardResult } from '../types';
import { AlertTriangle, Focus, Sun, Leaf, RefreshCw, XCircle } from 'lucide-react';
import { useLanguage } from '../context/LanguageContext';

interface QualityGuardModalProps {
  result: QualityGuardResult;
  onRetry: () => void;
  onClose: () => void;
}

export const QualityGuardModal: React.FC<QualityGuardModalProps> = ({ result, onRetry, onClose }) => {
  const { t } = useLanguage();

  const getReasonIcon = () => {
    switch (result.reason) {
      case 'BLURRY':
        return <Focus className="w-7 h-7 text-amber-600" />;
      case 'DARK':
      case 'OVEREXPOSED':
        return <Sun className="w-7 h-7 text-amber-600" />;
      case 'NON_PLANT':
        return <Leaf className="w-7 h-7 text-rose-600" />;
      default:
        return <AlertTriangle className="w-7 h-7 text-rose-600" />;
    }
  };

  const getCorrectiveAdvice = () => {
    switch (result.reason) {
      case 'BLURRY':
        return t.qualityGuard.blurIssue;
      case 'DARK':
      case 'OVEREXPOSED':
        return t.qualityGuard.darkIssue;
      case 'NON_PLANT':
        return t.qualityGuard.foliageIssue;
      default:
        return result.message || 'Please upload a clean, focused photograph of a crop leaf.';
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs">
      <div className="bg-white rounded-xl shadow-xl border border-slate-200/90 max-w-md w-full p-6 text-center">
        
        {/* Icon & Title */}
        <div className="w-12 h-12 rounded-xl bg-amber-50 border border-amber-200/90 flex items-center justify-center mx-auto mb-3">
          {getReasonIcon()}
        </div>

        <h3 className="text-base font-bold text-slate-900 mb-0.5">
          {t.qualityGuard.rejectedTitle}
        </h3>
        <p className="text-xs text-slate-500 mb-4">
          {t.qualityGuard.rejectedSubtitle}
        </p>

        {/* Diagnosis & Actionable Guidance */}
        <div className="bg-amber-50/80 border border-amber-200/80 rounded-lg p-3 text-left mb-4">
          <div className="flex items-start gap-2.5">
            <XCircle className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
            <div>
              <p className="text-xs font-semibold text-amber-900 mb-0.5">
                Rejection Code: {result.reason}
              </p>
              <p className="text-xs text-amber-800 leading-relaxed">
                {getCorrectiveAdvice()}
              </p>
            </div>
          </div>
        </div>

        {/* Metrics Grid */}
        {result.metrics && Object.keys(result.metrics).length > 0 && (
          <div className="grid grid-cols-2 gap-2 text-left bg-slate-50 p-3 rounded-lg border border-slate-200/80 mb-5 text-[11px]">
            {result.metrics.laplacian_variance !== undefined && (
              <div>
                <span className="text-slate-400">Sharpness (Laplacian):</span>
                <p className="font-semibold text-slate-700">
                  {result.metrics.laplacian_variance.toFixed(1)} <span className="text-[10px] text-slate-400">(Min: 65.0)</span>
                </p>
              </div>
            )}
            {result.metrics.mean_brightness !== undefined && (
              <div>
                <span className="text-slate-400">Brightness:</span>
                <p className="font-semibold text-slate-700">
                  {result.metrics.mean_brightness.toFixed(1)} <span className="text-[10px] text-slate-400">(Norm: 35-230)</span>
                </p>
              </div>
            )}
            {result.metrics.foliage_percentage !== undefined && (
              <div className="col-span-2">
                <span className="text-slate-400">Vegetation Foliage Ratio:</span>
                <p className="font-semibold text-slate-700">
                  {result.metrics.foliage_percentage.toFixed(1)}% <span className="text-[10px] text-slate-400">(Min: 12.0%)</span>
                </p>
              </div>
            )}
          </div>
        )}

        {/* Buttons */}
        <div className="flex items-center gap-2.5">
          <button
            onClick={onRetry}
            className="flex-1 py-2 px-3 bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-semibold rounded-lg transition shadow-xs flex items-center justify-center gap-1.5 cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>{t.qualityGuard.tryAgain}</span>
          </button>
          <button
            onClick={onClose}
            className="py-2 px-4 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold rounded-lg transition cursor-pointer"
          >
            Dismiss
          </button>
        </div>

      </div>
    </div>
  );
};
