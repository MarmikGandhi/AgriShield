// src/components/GradCamViewer.tsx

import React, { useState } from 'react';
import { Eye, Layers, Info, Sliders } from 'lucide-react';

interface GradCamViewerProps {
  originalImageUrl: string;
  gradcamImageUrl?: string;
  cropName: string;
  conditionName: string;
}

export const GradCamViewer: React.FC<GradCamViewerProps> = ({
  originalImageUrl,
  gradcamImageUrl,
  cropName,
  conditionName,
}) => {
  const [mode, setMode] = useState<'side-by-side' | 'blend'>('side-by-side');
  const [blendOpacity, setBlendOpacity] = useState<number>(65);

  if (!gradcamImageUrl) {
    return (
      <div className="bg-slate-50 border border-slate-200/90 rounded-xl p-6 text-center">
        <Info className="w-7 h-7 text-slate-400 mx-auto mb-2" />
        <p className="text-xs font-semibold text-slate-700">Grad-CAM Heatmap Unavailable</p>
        <p className="text-[11px] text-slate-500 mt-0.5">Grad-CAM heatmaps are synthesized for diseased foliage classifications.</p>
      </div>
    );
  }

  return (
    <div className="bg-white border border-slate-200/90 rounded-xl p-5 shadow-xs space-y-4">
      {/* Header & Mode controls */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-amber-500"></span>
            <h4 className="font-bold text-sm text-slate-900">Grad-CAM Class Activation Map</h4>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Gradient-weighted visual attention of deep features for: <strong className="text-slate-800">{conditionName}</strong>
          </p>
        </div>

        <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg text-xs font-medium">
          <button
            onClick={() => setMode('side-by-side')}
            className={`px-3 py-1 rounded-md transition cursor-pointer ${
              mode === 'side-by-side' ? 'bg-white text-slate-900 shadow-2xs font-semibold' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Side-by-Side
          </button>
          <button
            onClick={() => setMode('blend')}
            className={`px-3 py-1 rounded-md transition cursor-pointer ${
              mode === 'blend' ? 'bg-white text-slate-900 shadow-2xs font-semibold' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Interactive Overlay
          </button>
        </div>
      </div>

      {/* Viewer Canvas */}
      {mode === 'side-by-side' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
              <span>Original Leaf Photo</span>
              <span className="text-[10px] text-slate-400 font-normal">Input Image</span>
            </div>
            <div className="aspect-square bg-slate-900 rounded-xl overflow-hidden border border-slate-200 flex items-center justify-center relative group">
              <img
                src={originalImageUrl}
                alt="Original leaf"
                className="w-full h-full object-contain"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs font-semibold text-amber-700">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                Grad-CAM Activation Heatmap
              </span>
              <span className="text-[10px] text-slate-400 font-normal">Last Conv Layer</span>
            </div>
            <div className="aspect-square bg-slate-900 rounded-xl overflow-hidden border border-slate-200 flex items-center justify-center relative">
              <img
                src={gradcamImageUrl}
                alt="Grad-CAM overlay"
                className="w-full h-full object-contain"
              />
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="relative aspect-[4/3] max-h-96 mx-auto bg-slate-900 rounded-xl overflow-hidden border border-slate-200 flex items-center justify-center">
            {/* Background original */}
            <img
              src={originalImageUrl}
              alt="Original leaf"
              className="absolute inset-0 w-full h-full object-contain"
            />
            {/* Foreground GradCAM with opacity */}
            <img
              src={gradcamImageUrl}
              alt="Grad-CAM"
              style={{ opacity: blendOpacity / 100 }}
              className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity"
            />
          </div>

          {/* Slider control */}
          <div className="flex items-center gap-3 bg-slate-50 p-2.5 rounded-lg border border-slate-200 text-xs">
            <Sliders className="w-4 h-4 text-slate-500 shrink-0" />
            <span className="font-semibold text-slate-700">Heatmap Intensity:</span>
            <input
              type="range"
              min="0"
              max="100"
              value={blendOpacity}
              onChange={(e) => setBlendOpacity(Number(e.target.value))}
              className="flex-1 accent-emerald-700 cursor-pointer"
            />
            <span className="font-bold text-slate-800 w-9 text-right">{blendOpacity}%</span>
          </div>
        </div>
      )}

      {/* Explanatory Legend & Model Source Disclaimer */}
      <div className="bg-slate-50 rounded-lg p-3 border border-slate-200/80 space-y-2 text-[11px] text-slate-600">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-700">Color Spectrum:</span>
            <div className="flex items-center gap-1 font-mono text-[10px]">
              <span className="px-1.5 py-0.5 rounded bg-blue-600 text-white font-bold">Blue</span>
              <span>(Low attention)</span>
              <span className="mx-1">→</span>
              <span className="px-1.5 py-0.5 rounded bg-amber-500 text-slate-900 font-bold">Yellow</span>
              <span>(Moderate)</span>
              <span className="mx-1">→</span>
              <span className="px-1.5 py-0.5 rounded bg-rose-600 text-white font-bold">Red</span>
              <span>(Max attention)</span>
            </div>
          </div>
          <span className="text-[10px] font-semibold text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
            Source: EfficientNet-B0 features[-1]
          </span>
        </div>
        <p className="text-[10px] text-slate-500 border-t border-slate-200/60 pt-1.5 leading-relaxed">
          <strong className="text-slate-600">Attribution Notice:</strong> This Grad-CAM visual attention map is computed specifically from the EfficientNet-B0 CNN branch. It visualizes the spatial filters of EfficientNet-B0 and does not reflect MobileNetV3 activations.
        </p>
      </div>
    </div>
  );
};
