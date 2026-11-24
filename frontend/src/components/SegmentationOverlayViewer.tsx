// src/components/SegmentationOverlayViewer.tsx

import React, { useState } from 'react';
import { BoundingBox } from '../types';
import { Grid, Eye, Square, AlertCircle, CheckCircle } from 'lucide-react';

interface SegmentationOverlayViewerProps {
  segmentationUrl?: string;
  originalUrl: string;
  affectedPercentage: number;
  lesionCount: number;
  boundingBoxes?: BoundingBox[];
  severityLevel: string;
}

export const SegmentationOverlayViewer: React.FC<SegmentationOverlayViewerProps> = ({
  segmentationUrl,
  originalUrl,
  affectedPercentage,
  lesionCount,
  boundingBoxes = [],
  severityLevel,
}) => {
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [selectedBox, setSelectedBox] = useState<number | null>(null);

  if (!segmentationUrl) {
    return (
      <div className="bg-emerald-50 border border-emerald-200/90 rounded-xl p-6 text-center">
        <CheckCircle className="w-7 h-7 text-emerald-700 mx-auto mb-2" />
        <h4 className="text-xs font-bold text-emerald-900">Zero Foliar Lesions Detected</h4>
        <p className="text-[11px] text-emerald-700 mt-1">
          U-Net pixel segmentation confirmed no necrotic or chlorotic lesion clusters on the foliage.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white border border-slate-200/90 rounded-xl p-5 shadow-xs space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-rose-500"></span>
            <h4 className="font-bold text-sm text-slate-900">U-Net Foliar Lesion Segmentation (PlantSeg)</h4>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Pixel-level localization: <span className="font-semibold text-rose-600">{affectedPercentage}% canopy damaged</span> across <span className="font-semibold text-slate-800">{lesionCount} lesion clusters</span>
          </p>
        </div>

        {/* Toggle Bounding Boxes */}
        {boundingBoxes.length > 0 && (
          <button
            onClick={() => setShowBoxes(!showBoxes)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition border cursor-pointer ${
              showBoxes
                ? 'bg-rose-50 border-rose-200 text-rose-700'
                : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
            }`}
          >
            <Square className="w-3.5 h-3.5" />
            <span>{showBoxes ? 'Hide Bounding Boxes' : 'Show Bounding Boxes'}</span>
          </button>
        )}
      </div>

      {/* Main Image Viewer */}
      <div className="relative aspect-[4/3] max-h-96 mx-auto bg-slate-900 rounded-xl overflow-hidden border border-slate-200 flex items-center justify-center">
        <img
          src={segmentationUrl}
          alt="U-Net lesion segmentation overlay"
          className="w-full h-full object-contain"
        />

        {showBoxes && boundingBoxes.length > 0 && (
          <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
            <div className="absolute top-3 left-3 bg-slate-900/80 backdrop-blur-xs text-white text-[10px] font-mono px-2 py-1 rounded border border-slate-700">
              {boundingBoxes.length} Lesion Spots Highlighted
            </div>
          </div>
        )}
      </div>

      {/* Lesion Clusters Table & Metrics */}
      <div className="bg-slate-50 rounded-lg p-3 border border-slate-200/80">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
            <Grid className="w-3.5 h-3.5 text-emerald-700" />
            Largest Detected Lesion Clusters
          </span>
          <span className="text-[10px] text-slate-400">Contours sorted by pixel area</span>
        </div>

        {boundingBoxes.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
            {boundingBoxes.slice(0, 8).map((box, idx) => (
              <div
                key={idx}
                className="bg-white p-2 rounded-lg border border-slate-200/90 shadow-2xs hover:border-rose-300 transition"
              >
                <div className="flex items-center justify-between text-[10px] font-semibold text-slate-600 mb-0.5">
                  <span>Spot #{idx + 1}</span>
                  <span className="text-rose-600 font-mono">{box.area_pixels} px²</span>
                </div>
                <p className="text-[10px] text-slate-400 font-mono">
                  {box.width}×{box.height} @ ({box.x},{box.y})
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-slate-500 italic">No discrete bounding boxes (diffuse foliage discoloration).</p>
        )}
      </div>
    </div>
  );
};
