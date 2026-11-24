// src/components/WeatherRiskCard.tsx

import React from 'react';
import { WeatherRiskData } from '../types';
import { CloudRain, Thermometer, Droplets, Wind, ShieldAlert, CheckCircle, Info } from 'lucide-react';

interface WeatherRiskCardProps {
  data: WeatherRiskData;
}

export const WeatherRiskCard: React.FC<WeatherRiskCardProps> = ({ data }) => {
  const { current_weather, overall_risk_level, disease_risks } = data;

  const getRiskBadge = (level: string) => {
    switch (level.toLowerCase()) {
      case 'high':
        return {
          badge: 'bg-rose-50 text-rose-700 border-rose-200',
          dot: 'bg-rose-600',
        };
      case 'moderate':
        return {
          badge: 'bg-amber-50 text-amber-700 border-amber-200',
          dot: 'bg-amber-500',
        };
      case 'low':
      default:
        return {
          badge: 'bg-emerald-50 text-emerald-700 border-emerald-200',
          dot: 'bg-emerald-600',
        };
    }
  };

  const riskStyle = getRiskBadge(overall_risk_level);

  return (
    <div className="rounded-xl border border-slate-200/90 bg-white p-5 shadow-xs space-y-4">
      {/* Top Header */}
      <div className="flex items-start justify-between gap-3 pb-3 border-b border-slate-100">
        <div>
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${riskStyle.dot}`}></span>
            <h4 className="font-bold text-sm text-slate-900">
              Weather-Based Foliar Disease Risk
            </h4>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Station: <strong className="text-slate-700">{current_weather.location}, Gujarat</strong> • ICAR Agronomic Model
          </p>
        </div>

        <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${riskStyle.badge}`}>
          {overall_risk_level} Risk
        </span>
      </div>

      {/* Weather Parameters Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 p-3 rounded-lg bg-slate-50 border border-slate-200/80 text-xs">
        <div className="flex items-center gap-2">
          <Thermometer className="w-4 h-4 text-rose-500 shrink-0" />
          <div>
            <span className="text-[10px] text-slate-500 block">Temperature</span>
            <span className="font-semibold text-slate-900">{current_weather.temperature_c}°C</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Droplets className="w-4 h-4 text-sky-500 shrink-0" />
          <div>
            <span className="text-[10px] text-slate-500 block">Relative Humidity</span>
            <span className="font-semibold text-slate-900">{current_weather.humidity_pct}%</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <CloudRain className="w-4 h-4 text-indigo-500 shrink-0" />
          <div>
            <span className="text-[10px] text-slate-500 block">Precipitation</span>
            <span className="font-semibold text-slate-900">{current_weather.rainfall_mm} mm</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Wind className="w-4 h-4 text-teal-600 shrink-0" />
          <div>
            <span className="text-[10px] text-slate-500 block">Wind Speed</span>
            <span className="font-semibold text-slate-900">{current_weather.wind_speed_kmh} km/h</span>
          </div>
        </div>
      </div>

      {/* Specific Disease Threats */}
      <div className="space-y-2">
        <p className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
          <ShieldAlert className="w-3.5 h-3.5 text-slate-500" />
          Crop Vulnerability Forecast
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
          {disease_risks.slice(0, 4).map((risk, idx) => (
            <div
              key={idx}
              className="bg-slate-50/70 p-3 rounded-lg border border-slate-200/80 flex flex-col justify-between"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-semibold text-xs text-slate-900">{risk.crop} — {risk.disease}</span>
                <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${
                  risk.risk_level === 'High' ? 'bg-rose-50 text-rose-700 border-rose-200' :
                  risk.risk_level === 'Moderate' ? 'bg-amber-50 text-amber-700 border-amber-200' : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                }`}>
                  {risk.risk_level}
                </span>
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed mb-2">
                {risk.management_advice}
              </p>
              <div className="flex flex-wrap gap-1">
                {risk.conducive_factors?.map((factor, fIdx) => (
                  <span
                    key={fIdx}
                    className="text-[10px] bg-white text-slate-600 px-2 py-0.5 rounded border border-slate-200"
                  >
                    {factor}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
