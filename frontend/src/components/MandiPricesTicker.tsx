// src/components/MandiPricesTicker.tsx

import React from 'react';
import { MandiPriceItem } from '../types';
import { TrendingUp, MapPin, Calendar, Tag } from 'lucide-react';

interface MandiPricesTickerProps {
  prices: MandiPriceItem[];
}

export const MandiPricesTicker: React.FC<MandiPricesTickerProps> = ({ prices }) => {
  return (
    <div className="bg-white border border-slate-200/90 rounded-xl p-5 shadow-xs space-y-3">
      <div className="flex items-center justify-between pb-2 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-lg bg-emerald-50 text-emerald-700 flex items-center justify-center">
            <TrendingUp className="w-3.5 h-3.5" />
          </div>
          <div>
            <h4 className="font-bold text-sm text-slate-900">Gujarat APMC Mandi Prices</h4>
            <p className="text-[11px] text-slate-500">Official Agmarknet / APMC Market Rates</p>
          </div>
        </div>
        <span className="text-[10px] font-semibold text-emerald-700 bg-emerald-50 px-2.5 py-0.5 rounded-full border border-emerald-200">
          Live Rates
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
        {prices.map((item, idx) => (
          <div
            key={idx}
            className="p-3 rounded-lg border border-slate-200/80 hover:border-slate-300 bg-slate-50/50 hover:bg-slate-50 transition"
          >
            <div className="flex items-start justify-between mb-1">
              <div>
                <span className="font-semibold text-xs text-slate-900">{item.commodity}</span>
                <span className="text-[10px] text-slate-500 block">{item.variety}</span>
              </div>
              <div className="text-right">
                <span className="text-sm font-bold text-emerald-700">₹{item.modal_price.toLocaleString()}</span>
                <span className="text-[9px] text-slate-400 block">per Quintal</span>
              </div>
            </div>

            <div className="flex items-center justify-between text-[10px] text-slate-500 pt-2 border-t border-slate-200/60 mt-1">
              <span className="flex items-center gap-1 truncate max-w-[60%]">
                <MapPin className="w-3 h-3 text-slate-400 shrink-0" />
                <span className="truncate">{item.market} ({item.district})</span>
              </span>
              <span className="text-slate-600 font-medium">₹{item.min_price} - ₹{item.max_price}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
