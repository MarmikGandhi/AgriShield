// src/pages/AlertsFeed.tsx

import React, { useEffect, useState } from 'react';
import { useLanguage } from '../context/LanguageContext';
import { notificationApi } from '../services/api';
import { NotificationItem } from '../types';
import {
  Bell,
  AlertTriangle,
  AlertCircle,
  CheckCircle2,
  Mail,
  MessageSquare,
  Smartphone,
  RefreshCw,
  Check,
  Clock
} from 'lucide-react';

export const AlertsFeed: React.FC = () => {
  const { t, language } = useLanguage();
  const [alerts, setAlerts] = useState<NotificationItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [filter, setFilter] = useState<'all' | 'critical'>('all');

  const fetchAlerts = async () => {
    setLoading(true);
    try {
      const data = await notificationApi.list();
      setAlerts(data);
    } catch (e) {
      console.error('Failed to load notifications:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
  }, []);

  const handleMarkRead = async (id: number) => {
    try {
      await notificationApi.markRead(id);
      setAlerts((prev) =>
        prev.map((a) => (a.id === id ? { ...a, is_read: true } : a))
      );
    } catch (e) {
      console.error('Mark read error:', e);
    }
  };

  const filteredAlerts = alerts.filter((a) => {
    if (filter === 'critical') return a.severity === 'critical';
    return true;
  });

  const getAlertTitle = (item: NotificationItem) => {
    if (language === 'hi' && item.title_hi) return item.title_hi;
    if (language === 'gu' && item.title_gu) return item.title_gu;
    return item.title;
  };

  const getAlertMessage = (item: NotificationItem) => {
    if (language === 'hi' && item.message_hi) return item.message_hi;
    if (language === 'gu' && item.message_gu) return item.message_gu;
    return item.message;
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <div className="w-8 h-8 rounded-lg bg-emerald-700 text-white flex items-center justify-center">
              <Bell className="w-4 h-4" />
            </div>
            <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
              {t.alerts.title}
            </h1>
          </div>
          <p className="text-xs sm:text-sm text-slate-500">
            {t.alerts.subtitle}
          </p>
        </div>

        {/* Filter controls */}
        <div className="flex items-center gap-2">
          <div className="bg-slate-100 p-1 rounded-lg flex items-center text-xs font-medium border border-slate-200">
            <button
              onClick={() => setFilter('all')}
              className={`px-3 py-1 rounded-md transition ${
                filter === 'all' ? 'bg-white text-slate-900 shadow-2xs font-semibold' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {t.alerts.allAlerts} ({alerts.length})
            </button>
            <button
              onClick={() => setFilter('critical')}
              className={`px-3 py-1 rounded-md transition ${
                filter === 'critical' ? 'bg-white text-rose-700 shadow-2xs font-semibold' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {t.alerts.criticalOnly} ({alerts.filter((a) => a.severity === 'critical').length})
            </button>
          </div>

          <button
            onClick={fetchAlerts}
            className="p-2 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-slate-600 transition shadow-2xs"
            title="Refresh Alerts"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Alerts Feed */}
      {loading ? (
        <div className="p-12 text-center bg-white rounded-xl border border-slate-200">
          <RefreshCw className="w-6 h-6 text-emerald-700 mx-auto mb-2 animate-spin" />
          <p className="text-xs text-slate-500 font-medium">Retrieving alert notifications...</p>
        </div>
      ) : filteredAlerts.length === 0 ? (
        <div className="p-12 text-center bg-white rounded-xl border border-dashed border-slate-200 space-y-2">
          <CheckCircle2 className="w-8 h-8 text-emerald-600 mx-auto" />
          <h3 className="text-sm font-semibold text-slate-800">No Active Outbreak Alerts</h3>
          <p className="text-xs text-slate-500">{t.alerts.noAlerts}</p>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredAlerts.map((item) => (
            <div
              key={item.id}
              className={`p-4 sm:p-5 rounded-xl border transition-all ${
                item.is_read
                  ? 'bg-white border-slate-200 opacity-80'
                  : item.severity === 'critical'
                  ? 'bg-rose-50/50 border-rose-200 shadow-xs'
                  : 'bg-amber-50/50 border-amber-200 shadow-xs'
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                
                <div className="flex items-start gap-3">
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-0.5 ${
                    item.severity === 'critical'
                      ? 'bg-rose-100 text-rose-800 border border-rose-200'
                      : item.severity === 'warning'
                      ? 'bg-amber-100 text-amber-800 border border-amber-200'
                      : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                  }`}>
                    {item.severity === 'critical' ? (
                      <AlertCircle className="w-4 h-4 text-rose-700" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-amber-700" />
                    )}
                  </div>

                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <h4 className="text-xs sm:text-sm font-bold text-slate-900">
                        {getAlertTitle(item)}
                      </h4>
                      {!item.is_read && (
                        <span className="w-2 h-2 rounded-full bg-emerald-600"></span>
                      )}
                    </div>
                    <p className="text-xs text-slate-700 leading-relaxed">
                      {getAlertMessage(item)}
                    </p>

                    {/* Delivery audit badges */}
                    <div className="flex flex-wrap items-center gap-2 pt-2 text-[10px] text-slate-500 font-medium">
                      <span className="flex items-center gap-1 bg-white px-2 py-0.5 rounded border border-slate-200">
                        <Smartphone className="w-3 h-3 text-emerald-700" />
                        In-App: Delivered
                      </span>
                      <span className="flex items-center gap-1 bg-white px-2 py-0.5 rounded border border-slate-200">
                        <MessageSquare className="w-3 h-3 text-sky-700" />
                        SMS: Dispatched
                      </span>
                      <span className="flex items-center gap-1 bg-white px-2 py-0.5 rounded border border-slate-200">
                        <Mail className="w-3 h-3 text-slate-700" />
                        Email: Sent
                      </span>
                      <span className="text-slate-400 flex items-center gap-1 ml-auto">
                        <Clock className="w-3 h-3" />
                        {new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                  </div>
                </div>

                {!item.is_read && (
                  <button
                    onClick={() => handleMarkRead(item.id)}
                    className="p-1.5 rounded-md bg-white hover:bg-slate-50 text-slate-500 hover:text-slate-800 transition shrink-0 border border-slate-200 shadow-2xs"
                    title="Mark as Read"
                  >
                    <Check className="w-3.5 h-3.5" />
                  </button>
                )}

              </div>
            </div>
          ))}
        </div>
      )}

    </div>
  );
};
