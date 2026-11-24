// src/components/Footer.tsx

import React from 'react';
import { Sprout, ShieldCheck } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-slate-900 text-slate-400 text-xs border-t border-slate-800 py-10 mt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          
          {/* Col 1: System */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-white">
              <div className="w-6 h-6 rounded-lg bg-emerald-700 flex items-center justify-center">
                <Sprout className="w-3.5 h-3.5 text-white" />
              </div>
              <span className="font-bold text-sm tracking-tight">AgriShield AI</span>
            </div>
            <p className="text-slate-400 text-xs leading-relaxed">
              Early foliar stress detection, real-time lesion segmentation, explainable Grad-CAM, and multi-channel farmer alerts.
            </p>
            <div className="flex items-center gap-1.5 text-emerald-400 text-[11px] font-medium">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>Production AI Verified Pipeline</span>
            </div>
          </div>

          {/* Col 2: Agronomic Guidance */}
          <div className="space-y-2">
            <h4 className="text-white font-semibold text-xs uppercase tracking-wider">Agronomic Guidance</h4>
            <ul className="space-y-1.5 text-slate-400 text-xs">
              <li>• Anand Agricultural University (AAU)</li>
              <li>• Navsari Agricultural University (NAU)</li>
              <li>• Indian Council of Agricultural Research (ICAR)</li>
              <li>• Krishi Vigyan Kendra (KVK) Network</li>
              <li>• Gujarat State Seed & Crop Protection Board</li>
            </ul>
          </div>

          {/* Col 3: AI Benchmarks */}
          <div className="space-y-2">
            <h4 className="text-white font-semibold text-xs uppercase tracking-wider">AI Benchmarks</h4>
            <ul className="space-y-1.5 text-slate-400 text-xs">
              <li>• PlantVillage (54,305 controlled images)</li>
              <li>• PlantDoc (2,573 field pathology images)</li>
              <li>• PlantSeg (7,774 pixel-level lesion masks)</li>
              <li>• PlantWild (18,542 in-the-wild evaluations)</li>
              <li>• Dual-CNN Ensemble (EfficientNet + MobileNet)</li>
            </ul>
          </div>

          {/* Col 4: Project Team */}
          <div className="space-y-2">
            <h4 className="text-white font-semibold text-xs uppercase tracking-wider">Project Engineering</h4>
            <p className="text-xs text-slate-300">
              Department of Information Technology & Computer Engineering
            </p>
            <div className="text-[11px] space-y-1 text-slate-400">
              <p><strong className="text-slate-300">Marmik Gandhi:</strong> Team Leader / Frontend / DB</p>
              <p><strong className="text-slate-300">Mehal Jain:</strong> Backend / Analytics / Baseline</p>
              <p><strong className="text-slate-300">Brij Patel:</strong> Backend / FastAPI / Quality Guard</p>
              <p><strong className="text-slate-300">Aayush Barot:</strong> Frontend / UX / Localization</p>
            </div>
          </div>
        </div>

        <div className="pt-6 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between text-[11px] text-slate-500 gap-3">
          <p>© 2026 AgriShield AI. Built for Gujarat & Indian Farmers in English, हिन्दी, and ગુજરાતી.</p>
          <div className="flex items-center gap-1 text-slate-400">
            <span>Powered by PyTorch, OpenCV, FastAPI & React</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
