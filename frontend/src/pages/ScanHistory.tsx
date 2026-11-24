// src/pages/ScanHistory.tsx

import React, { useEffect, useState } from 'react';
import { useLanguage } from '../context/LanguageContext';
import { scanApi } from '../services/api';
import { ScanRecord } from '../types';
import {
  History,
  Search,
  Filter,
  Eye,
  LayoutGrid,
  List,
  RefreshCw,
  Clock
} from 'lucide-react';

interface ScanHistoryProps {
  onSelectScan: (scan: ScanRecord) => void;
  onNavigate: (page: string) => void;
}

export const ScanHistory: React.FC<ScanHistoryProps> = ({ onSelectScan, onNavigate }) => {
  const { t, language } = useLanguage();
  const [scans, setScans] = useState<ScanRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('grid');

  const fetchScans = async () => {
    setLoading(true);
    try {
      const data = await scanApi.listScans({
        severity: severityFilter !== 'all' ? severityFilter : undefined,
        crop: searchQuery || undefined,
      });
      setScans(data);
    } catch (err) {
      console.error('Failed to load scan history:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchScans();
  }, [severityFilter]);

  const filteredScans = scans.filter((s) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return s.crop.toLowerCase().includes(q) || s.disease_condition.toLowerCase().includes(q);
  });

  const getCropName = (scan: ScanRecord) => {
    if (language === 'hi' && scan.crop_hi) return scan.crop_hi;
    if (language === 'gu' && scan.crop_gu) return scan.crop_gu;
    return scan.crop;
  };

  const getConditionName = (scan: ScanRecord) => {
    if (language === 'hi' && scan.disease_condition_hi) return scan.disease_condition_hi;
    if (language === 'gu' && scan.disease_condition_gu) return scan.disease_condition_gu;
    return scan.disease_condition;
  };

  const getSeverityBadge = (scan: ScanRecord) => {
    if (scan.is_healthy) {
      return {
        label: 'Healthy',
        className: 'bg-emerald-50 text-emerald-800 border-emerald-200/80',
      };
    }
    const lvl = (scan.severity_level || '').toLowerCase();
    if (lvl.includes('low')) {
      return {
        label: 'Low',
        className: 'bg-emerald-50 text-emerald-800 border-emerald-200/80',
      };
    }
    if (lvl.includes('medium')) {
      return {
        label: 'Moderate',
        className: 'bg-amber-50 text-amber-800 border-amber-200/80',
      };
    }
    if (lvl.includes('high') || lvl.includes('critical')) {
      return {
        label: scan.severity_level,
        className: 'bg-rose-50 text-rose-800 border-rose-200/80',
      };
    }
    return {
      label: 'Unverified',
      className: 'bg-slate-100 text-slate-700 border-slate-200',
    };
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <div className="w-8 h-8 rounded-lg bg-emerald-700 text-white flex items-center justify-center">
              <History className="w-4 h-4" />
            </div>
            <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
              {t.history.title}
            </h1>
          </div>
          <p className="text-xs sm:text-sm text-slate-500">
            Archived pathology scans with Grad-CAM visual heatmaps & U-Net lesion overlays
          </p>
        </div>

        {/* View mode toggle */}
        <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg text-xs font-medium self-start sm:self-auto border border-slate-200">
          <button
            onClick={() => setViewMode('grid')}
            className={`p-1.5 rounded-md transition ${
              viewMode === 'grid' ? 'bg-white text-slate-900 shadow-2xs font-semibold' : 'text-slate-500 hover:text-slate-800'
            }`}
            title="Grid View"
          >
            <LayoutGrid className="w-4 h-4" />
          </button>
          <button
            onClick={() => setViewMode('table')}
            className={`p-1.5 rounded-md transition ${
              viewMode === 'table' ? 'bg-white text-slate-900 shadow-2xs font-semibold' : 'text-slate-500 hover:text-slate-800'
            }`}
            title="Table View"
          >
            <List className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-center gap-2.5">
        <div className="relative flex-1 w-full">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder={t.history.searchPlaceholder}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 rounded-lg border border-slate-200 bg-white text-xs text-slate-900 focus:outline-none focus:ring-1 focus:ring-emerald-600 focus:border-emerald-600 shadow-2xs"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Filter className="w-4 h-4 text-slate-400 shrink-0 hidden sm:block" />
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="w-full sm:w-44 py-2 px-3 rounded-lg border border-slate-200 bg-white text-xs font-medium text-slate-700 focus:outline-none focus:ring-1 focus:ring-emerald-600 shadow-2xs"
          >
            <option value="all">{t.history.filterAllSeverities}</option>
            <option value="Low">Low Severity</option>
            <option value="Medium">Medium Severity</option>
            <option value="High">High Severity</option>
            <option value="Critical">Critical Severity</option>
          </select>

          <button
            onClick={fetchScans}
            className="p-2 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-slate-600 transition shadow-2xs"
            title="Refresh Scan Records"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Scans Listing */}
      {loading ? (
        <div className="p-12 text-center bg-white rounded-xl border border-slate-200">
          <RefreshCw className="w-6 h-6 text-emerald-700 mx-auto mb-2 animate-spin" />
          <p className="text-xs text-slate-500 font-medium">Retrieving diagnostic records from database...</p>
        </div>
      ) : filteredScans.length === 0 ? (
        <div className="p-12 text-center bg-white rounded-xl border border-dashed border-slate-200 space-y-3">
          <History className="w-8 h-8 text-slate-300 mx-auto" />
          <div>
            <h3 className="text-sm font-semibold text-slate-800">{t.history.noRecords}</h3>
            <p className="text-xs text-slate-500 mt-0.5">Try adjusting your filters or upload a leaf photo to diagnose.</p>
          </div>
          <button
            onClick={() => onNavigate('scanner')}
            className="px-4 py-2 bg-emerald-700 hover:bg-emerald-800 text-white rounded-lg text-xs font-semibold transition shadow-xs"
          >
            Scan a Crop Leaf
          </button>
        </div>
      ) : viewMode === 'grid' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredScans.map((scan) => {
            const badge = getSeverityBadge(scan);
            return (
              <div
                key={scan.id}
                onClick={() => {
                  onSelectScan(scan);
                  onNavigate('scanner');
                }}
                className="bg-white rounded-xl border border-slate-200 hover:border-slate-300 overflow-hidden shadow-xs hover:shadow-sm transition cursor-pointer flex flex-col group"
              >
                <div className="relative aspect-[4/3] bg-slate-900 overflow-hidden">
                  <img
                    src={scan.image_url}
                    alt={scan.crop}
                    className="w-full h-full object-cover group-hover:scale-105 transition duration-300"
                  />
                  <div className="absolute top-2.5 right-2.5">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${badge.className}`}>
                      {badge.label}
                    </span>
                  </div>
                </div>

                <div className="p-4 flex-1 flex flex-col justify-between space-y-2">
                  <div>
                    <div className="flex items-center justify-between text-xs font-bold text-slate-900 mb-0.5">
                      <span className="truncate">{getCropName(scan)}</span>
                      <span className="text-emerald-700 font-semibold shrink-0">{scan.confidence.toFixed(1)}%</span>
                    </div>
                    <p className="text-xs text-slate-600 truncate font-medium">
                      {getConditionName(scan)}
                    </p>
                  </div>

                  <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {new Date(scan.created_at).toLocaleDateString()}
                    </span>
                    <span className="text-emerald-700 font-medium group-hover:translate-x-0.5 transition flex items-center gap-1">
                      View Report
                      <Eye className="w-3 h-3" />
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-xs">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-600">
              <thead className="bg-slate-50 border-b border-slate-200 font-semibold text-slate-700">
                <tr>
                  <th className="py-3 px-4">{t.history.tableDate}</th>
                  <th className="py-3 px-4">{t.history.tableCrop}</th>
                  <th className="py-3 px-4">{t.history.tableCondition}</th>
                  <th className="py-3 px-4">{t.history.tableConfidence}</th>
                  <th className="py-3 px-4">{t.history.tableSeverity}</th>
                  <th className="py-3 px-4 text-right">{t.history.tableActions}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredScans.map((scan) => {
                  const badge = getSeverityBadge(scan);
                  return (
                    <tr
                      key={scan.id}
                      onClick={() => {
                        onSelectScan(scan);
                        onNavigate('scanner');
                      }}
                      className="hover:bg-slate-50/80 cursor-pointer transition"
                    >
                      <td className="py-3 px-4 whitespace-nowrap text-slate-500">
                        {new Date(scan.created_at).toLocaleDateString()}
                      </td>
                      <td className="py-3 px-4 font-semibold text-slate-900">
                        {getCropName(scan)}
                      </td>
                      <td className="py-3 px-4 font-medium text-slate-700">
                        {getConditionName(scan)}
                      </td>
                      <td className="py-3 px-4 font-semibold text-emerald-700">
                        {scan.confidence.toFixed(1)}%
                      </td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${badge.className}`}>
                          {badge.label}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button className="px-2.5 py-1 rounded-md bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold transition">
                          View
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

    </div>
  );
};
