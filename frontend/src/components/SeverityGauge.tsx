// src/components/SeverityGauge.tsx

import React from 'react';
import { AlertCircle, CheckCircle2, AlertTriangle, Flame, ShieldAlert, HelpCircle } from 'lucide-react';

interface SeverityGaugeProps {
  severityLevel: string;
  affectedPercentage: number;
  isHealthy: boolean;
  severityReliability?: string;
  isApproximate?: boolean;
  discrepancyDetected?: boolean;
  rawAffectedPercentage?: number;
}

export const SeverityGauge: React.FC<SeverityGaugeProps> = ({
  severityLevel,
  affectedPercentage,
  isHealthy,
  severityReliability = 'RELIABLE',
  isApproximate = false,
  discrepancyDetected = false,
  rawAffectedPercentage,
}) => {
  const getSeverityConfig = () => {
    if (discrepancyDetected) {
      return {
        label: 'Discrepancy (Review Required)',
        badgeColor: 'bg-amber-50 text-amber-900 border-amber-300',
        barColor: 'bg-amber-500',
        textColor: 'text-amber-800',
        icon: ShieldAlert,
        urgency: 'Cross-Model Conflict — Inspection Required',
      };
    }

    if (severityReliability === 'UNRELIABLE' || severityLevel.toLowerCase().includes('unavailable')) {
      return {
        label: 'Boundary Unreliable',
        badgeColor: 'bg-slate-100 text-slate-700 border-slate-200',
        barColor: 'bg-slate-300',
        textColor: 'text-slate-600',
        icon: HelpCircle,
        urgency: 'Cannot calculate reliably — Retake photo',
      };
    }

    if (isHealthy || severityLevel.toLowerCase().includes('none')) {
      return {
        label: 'None (Healthy)',
        badgeColor: 'bg-emerald-50 text-emerald-800 border-emerald-200/80',
        barColor: 'bg-emerald-500',
        textColor: 'text-emerald-700',
        icon: CheckCircle2,
        urgency: 'Normal Field Care & Nutrition',
      };
    }

    switch (severityLevel.toLowerCase()) {
      case 'low':
        return {
          label: 'Low Severity',
          badgeColor: 'bg-emerald-50 text-emerald-800 border-emerald-200/80',
          barColor: 'bg-emerald-500',
          textColor: 'text-emerald-700',
          icon: AlertCircle,
          urgency: 'Early Stage — Preventative Intervention',
        };
      case 'medium':
        return {
          label: 'Medium Severity',
          badgeColor: 'bg-amber-50 text-amber-800 border-amber-200/80',
          barColor: 'bg-amber-500',
          textColor: 'text-amber-700',
          icon: AlertTriangle,
          urgency: 'Moderate Damage — Targeted Curative Spray',
        };
      case 'high':
        return {
          label: 'High Severity',
          badgeColor: 'bg-rose-50 text-rose-800 border-rose-200/80',
          barColor: 'bg-rose-500',
          textColor: 'text-rose-700',
          icon: AlertTriangle,
          urgency: 'Significant Outbreak — Rapid Foliar Spray Needed',
        };
      case 'critical':
      default:
        return {
          label: 'Critical Damage',
          badgeColor: 'bg-rose-50 text-rose-800 border-rose-200/80',
          barColor: 'bg-rose-600',
          textColor: 'text-rose-700',
          icon: Flame,
          urgency: 'Severe Canopy Loss — Agronomist Review Needed',
        };
    }
  };

  const config = getSeverityConfig();
  const Icon = config.icon;
  const isUnreliable = severityReliability === 'UNRELIABLE';

  return (
    <div className="bg-white border border-slate-200/90 rounded-xl p-5 shadow-xs space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Severity Classification</span>
        <div className="flex items-center gap-1.5">
          {isApproximate && !isUnreliable && (
            <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
              Approximate
            </span>
          )}
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${config.badgeColor}`}>
            <Icon className="w-3.5 h-3.5" />
            {config.label}
          </span>
        </div>
      </div>

      {/* Discrepancy Notice */}
      {discrepancyDetected && (
        <div className="bg-amber-50/80 border border-amber-300 rounded-lg p-3 text-xs text-amber-900 flex items-start gap-2">
          <ShieldAlert className="w-4 h-4 text-amber-700 mt-0.5 shrink-0" />
          <div>
            <strong>Classification / Segmentation Disagreement:</strong> Classifier predicted Healthy, but U-Net localized {rawAffectedPercentage ?? affectedPercentage}% lesion clusters. Human agronomist inspection recommended.
          </div>
        </div>
      )}

      {/* Numerical percentage & progress bar */}
      <div>
        <div className="flex items-baseline justify-between mb-1.5">
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-slate-900 tracking-tight">
              {isUnreliable ? 'Unavailable' : `${affectedPercentage.toFixed(1)}%`}
            </span>
            {isApproximate && !isUnreliable && (
              <span className="text-xs font-medium text-amber-600">± approx.</span>
            )}
          </div>
          <span className="text-xs font-medium text-slate-500">
            {isUnreliable ? 'Boundary unverified (<4% canopy)' : 'Foliar Canopy Area Affected'}
          </span>
        </div>

        {/* Progress bar */}
        <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden flex relative">
          <div
            className={`h-full transition-all duration-500 rounded-full ${config.barColor}`}
            style={{ width: `${isUnreliable ? 0 : Math.min(100, Math.max(isHealthy ? 0 : 2, affectedPercentage))}%` }}
          ></div>
        </div>

        {/* Severity Scale ticks */}
        <div className="flex justify-between text-[10px] text-slate-400 mt-1 font-mono">
          <span>0%</span>
          <span>10% Low</span>
          <span>25% Med</span>
          <span>50% High</span>
          <span>100% Crit</span>
        </div>
      </div>

      {/* Agronomic action callout */}
      <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-xs">
        <span className="text-slate-500 font-medium">Decision Support:</span>
        <span className={`font-semibold ${config.textColor}`}>
          {config.urgency}
        </span>
      </div>
    </div>
  );
};
