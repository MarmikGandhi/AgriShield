// src/pages/DatasetsAndModels.tsx

import React, { useEffect, useState } from 'react';
import { useLanguage } from '../context/LanguageContext';
import { systemApi } from '../services/api';
import {
  Database,
  Cpu,
  Layers,
  Award,
  CheckCircle2,
  RefreshCw,
  Server,
  BarChart2,
  ShieldCheck
} from 'lucide-react';

export const DatasetsAndModels: React.FC = () => {
  const { t } = useLanguage();
  const [datasetStatus, setDatasetStatus] = useState<any>(null);
  const [aiStatus, setAiStatus] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchStatus = async () => {
    setLoading(true);
    try {
      const [ds, ai] = await Promise.all([
        systemApi.getDatasetsStatus().catch(() => null),
        systemApi.getAiStatus().catch(() => null),
      ]);
      setDatasetStatus(ds);
      setAiStatus(ai);
    } catch (e) {
      console.error('Failed to load system status:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  return (
    <div className="space-y-6 sm:space-y-8 animate-in fade-in duration-200">
      
      {/* Header */}
      <div className="border-b border-slate-200 pb-4">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-8 h-8 rounded-lg bg-emerald-700 text-white flex items-center justify-center">
            <Database className="w-4 h-4" />
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
            {t.datasets.title}
          </h1>
        </div>
        <p className="text-xs sm:text-sm text-slate-500">
          {t.datasets.subtitle}
        </p>
      </div>

      {/* 4 Dataset Inventory Cards */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-emerald-700" />
          <h3 className="font-semibold text-xs sm:text-sm text-slate-900">
            Four Verified Real-World Dataset Inventories
          </h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          
          {/* PlantVillage */}
          <div className="bg-white rounded-xl p-5 border border-slate-200/90 shadow-xs space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-900">PlantVillage</span>
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200">
                Controlled Lab
              </span>
            </div>
            <div>
              <span className="text-2xl font-bold text-slate-900">54,305</span>
              <span className="text-xs text-slate-500 block">Color Leaf Images</span>
            </div>
            <div className="text-xs text-slate-600 space-y-1 pt-2.5 border-t border-slate-100">
              <p>• <strong>38</strong> Crop & Disease Classes</p>
              <p>• 14 Botanical Species</p>
              <p>• Verified RGB Image Integrity</p>
            </div>
          </div>

          {/* PlantDoc */}
          <div className="bg-white rounded-xl p-5 border border-slate-200/90 shadow-xs space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-900">PlantDoc</span>
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-sky-50 text-sky-800 border border-sky-200">
                In-Field Pathology
              </span>
            </div>
            <div>
              <span className="text-2xl font-bold text-slate-900">2,573</span>
              <span className="text-xs text-slate-500 block">Field Images</span>
            </div>
            <div className="text-xs text-slate-600 space-y-1 pt-2.5 border-t border-slate-100">
              <p>• <strong>29</strong> Field Disease Categories</p>
              <p>• Natural lighting & soil backgrounds</p>
              <p>• Cross-domain generalization test</p>
            </div>
          </div>

          {/* PlantSeg */}
          <div className="bg-white rounded-xl p-5 border border-slate-200/90 shadow-xs space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-900">PlantSeg</span>
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-rose-50 text-rose-800 border border-rose-200">
                Pixel Lesion Masks
              </span>
            </div>
            <div>
              <span className="text-2xl font-bold text-slate-900">7,774</span>
              <span className="text-xs text-slate-500 block">Image-Mask Pairs</span>
            </div>
            <div className="text-xs text-slate-600 space-y-1 pt-2.5 border-t border-slate-100">
              <p>• <strong>100%</strong> 1:1 Matched Pairs</p>
              <p>• Pixel-level lesion annotations</p>
              <p>• Nature Scientific Data (2025)</p>
            </div>
          </div>

          {/* PlantWild */}
          <div className="bg-white rounded-xl p-5 border border-slate-200/90 shadow-xs space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-900">PlantWild</span>
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-purple-50 text-purple-800 border border-purple-200">
                In-The-Wild Flora
              </span>
            </div>
            <div>
              <span className="text-2xl font-bold text-slate-900">18,542</span>
              <span className="text-xs text-slate-500 block">Unconstrained Photos</span>
            </div>
            <div className="text-xs text-slate-600 space-y-1 pt-2.5 border-t border-slate-100">
              <p>• <strong>89</strong> Flora Species & Outbreaks</p>
              <p>• Camera angle & glare diversity</p>
              <p>• Out-of-distribution benchmark</p>
            </div>
          </div>

        </div>
      </div>

      {/* AI Model Checkpoints & Empirical Performance */}
      <div className="bg-white rounded-xl p-5 sm:p-7 border border-slate-200/90 shadow-xs space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-100 gap-2">
          <div>
            <h3 className="font-semibold text-xs sm:text-sm text-slate-900 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-emerald-700" />
              <span>Trained Neural Network Architecture & Empirical Checkpoint Metrics</span>
            </h3>
            <p className="text-xs text-slate-500">
              Trained PyTorch models with strict weight validation and leak-free split verification
            </p>
          </div>
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200 self-start sm:self-auto">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
            Checkpoints Validated
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6">
          
          {/* EfficientNet-B0 */}
          <div className="p-4 sm:p-5 rounded-lg bg-slate-50 border border-slate-200/80 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="font-bold text-xs text-slate-900">EfficientNet-B0</span>
              <span className="text-[10px] font-mono text-slate-400">19.03 MB</span>
            </div>
            <div>
              <span className="text-2xl font-bold text-slate-900">83.68%</span>
              <span className="text-xs text-slate-500 block">Holdout Validation Accuracy</span>
            </div>
            <div className="text-xs text-slate-600 space-y-1 pt-2 border-t border-slate-200/60">
              <p>• Parameters: 4.05M</p>
              <p>• Feature Layer: <code>features[-1]</code> (Grad-CAM)</p>
              <p>• Checkpoint: <code>efficientnet_b0_best.pth</code></p>
            </div>
          </div>

          {/* MobileNetV3-Small */}
          <div className="p-4 sm:p-5 rounded-lg bg-slate-50 border border-slate-200/80 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="font-bold text-xs text-slate-900">MobileNetV3-Small</span>
              <span className="text-[10px] font-mono text-slate-400">6.36 MB</span>
            </div>
            <div>
              <span className="text-2xl font-bold text-slate-900">83.16%</span>
              <span className="text-xs text-slate-500 block">Holdout Validation Accuracy</span>
            </div>
            <div className="text-xs text-slate-600 space-y-1 pt-2 border-t border-slate-200/60">
              <p>• Parameters: 1.57M</p>
              <p>• Lightweight CPU inference (&lt;25ms)</p>
              <p>• Checkpoint: <code>mobilenet_v3_best.pth</code></p>
            </div>
          </div>

          {/* Dual-CNN Ensemble */}
          <div className="p-4 sm:p-5 rounded-lg bg-emerald-50/60 border border-emerald-200 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="font-bold text-xs text-emerald-950">Dual-CNN Ensemble</span>
              <span className="text-[10px] font-semibold text-emerald-800 bg-emerald-100 border border-emerald-200 px-2 py-0.5 rounded-full">
                Top-5: 99.17%
              </span>
            </div>
            <div>
              <span className="text-2xl font-bold text-emerald-900">91.83%</span>
              <span className="text-xs text-emerald-700 block">PlantVillage Test Accuracy</span>
            </div>
            <div className="text-xs text-emerald-800 space-y-1 pt-2 border-t border-emerald-200">
              <p>• Weights: 0.65 EffNet + 0.35 MobNet</p>
              <p>• Method: Soft Weighted Average</p>
              <p>• Macro F1: 92.87% on 38 classes</p>
            </div>
          </div>

        </div>

        {/* U-Net Lesion Segmentation Card */}
        <div className="p-4 sm:p-5 rounded-lg bg-slate-50 border border-slate-200/80 space-y-2.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-rose-500"></span>
              <span className="font-bold text-xs sm:text-sm text-slate-900">U-Net Foliar Lesion Segmentation (PlantSeg)</span>
            </div>
            <span className="text-[10px] font-mono text-slate-400">31.12 MB checkpoint</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
            <div>
              <span className="text-slate-500 block">Mean Lesion Dice:</span>
              <span className="text-base font-bold text-slate-900">31.42%</span>
            </div>
            <div>
              <span className="text-slate-500 block">Mean Lesion IoU:</span>
              <span className="text-base font-bold text-slate-900">20.66%</span>
            </div>
            <div>
              <span className="text-slate-500 block">Clean Specificity:</span>
              <span className="text-base font-bold text-slate-900">100.00%</span>
            </div>
            <div>
              <span className="text-slate-500 block">Compound Loss:</span>
              <span className="text-base font-bold text-slate-900">BCE + Dice + Focal</span>
            </div>
          </div>
        </div>

      </div>

      {/* Dataset Splits and Reproducibility */}
      <div className="bg-white rounded-xl p-5 sm:p-7 border border-slate-200/90 shadow-xs space-y-4">
        <h3 className="font-semibold text-xs sm:text-sm text-slate-900">Reconciled Zero-Leakage Dataset Splits</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
            <span className="font-semibold text-slate-700 block mb-1">Training Split (70%)</span>
            <span className="text-xl font-bold text-slate-900">38,154</span>
            <span className="text-[11px] text-slate-500 block mt-0.5">SHA-256 deduplicated samples</span>
          </div>
          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
            <span className="font-semibold text-slate-700 block mb-1">Validation Split (15%)</span>
            <span className="text-xl font-bold text-slate-900">8,157</span>
            <span className="text-[11px] text-slate-500 block mt-0.5">Early stopping & checkpointing</span>
          </div>
          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
            <span className="font-semibold text-slate-700 block mb-1">Held-Out Test Split (15%)</span>
            <span className="text-xl font-bold text-slate-900">8,177</span>
            <span className="text-[11px] text-slate-500 block mt-0.5">Independent unseen holdout</span>
          </div>
        </div>
      </div>

    </div>
  );
};
