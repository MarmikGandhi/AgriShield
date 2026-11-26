// src/pages/CropScanner.tsx

import React, { useState, useRef } from 'react';
import { useLanguage } from '../context/LanguageContext';
import { scanApi, API_BASE } from '../services/api';
import { ScanRecord, QualityGuardResult } from '../types';
import { QualityGuardModal } from '../components/QualityGuardModal';
import { GradCamViewer } from '../components/GradCamViewer';
import { SegmentationOverlayViewer } from '../components/SegmentationOverlayViewer';
import { SeverityGauge } from '../components/SeverityGauge';
import {
  UploadCloud,
  Camera,
  Sparkles,
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Info,
  ShieldCheck,
  Printer,
  ChevronRight,
  HelpCircle,
  FileText
} from 'lucide-react';

interface CropScannerProps {
  initialScan?: ScanRecord | null;
  onNavigate?: (page: string) => void;
}

export const CropScanner: React.FC<CropScannerProps> = ({ initialScan, onNavigate }) => {
  const { t, language } = useLanguage();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(initialScan ? initialScan.image_url : null);
  const [notes, setNotes] = useState<string>('');
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [analysisStage, setAnalysisStage] = useState<string>('');
  const [activeScan, setActiveScan] = useState<ScanRecord | null>(initialScan || null);
  const [qualityError, setQualityError] = useState<QualityGuardResult | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setActiveScan(null);
      setQualityError(null);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setActiveScan(null);
      setQualityError(null);
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;

    setAnalyzing(true);
    setQualityError(null);

    // Staged progress updates
    setAnalysisStage(t.scanner.qualityCheckStage);
    const stageTimer1 = setTimeout(() => setAnalysisStage(t.scanner.classificationStage), 800);
    const stageTimer2 = setTimeout(() => setAnalysisStage(t.scanner.gradcamStage), 1600);
    const stageTimer3 = setTimeout(() => setAnalysisStage(t.scanner.segmentationStage), 2400);

    const formData = new FormData();
    formData.append('file', selectedFile);
    if (notes) formData.append('notes', notes);

    try {
      const result = await scanApi.uploadScan(formData);
      setActiveScan(result);
    } catch (err: any) {
      console.error('Scan error:', err);
      if (err.response?.status === 422 && err.response?.data?.detail) {
        const detail = err.response.data.detail;
        setQualityError({
          is_valid: false,
          status: 'REJECTED',
          reason: detail.reason || 'IMAGE_UNSUITABLE',
          message: detail.message || detail.reason,
          metrics: detail.metrics || {},
        });
      } else if (err.response?.status === 503) {
        const detail = err.response.data?.detail;
        alert(`AI Model Unavailable: ${detail?.message || detail?.error || 'Required AI models are not ready or checkpoints are missing.'}`);
      } else if (!err.response) {
        const targetHost = API_BASE || window.location.origin;
        console.error('Target backend:', targetHost);
        alert(
          `Cannot connect to backend server (${targetHost}).\n\n` +
          `Common causes:\n` +
          `1. Cold Start: Render free tier sleeps after 15m. Please wait 45s for it to wake up and try again.\n` +
          `2. Vercel Config: VITE_API_BASE_URL environment variable is missing in Vercel settings.\n` +
          `3. CORS / Network: Browser blocked request to ${targetHost}.`
        );
      } else {
        const detail = err.response?.data?.detail;
        alert(typeof detail === 'string' ? detail : (detail?.message || `Server error (${err.response.status}).`));
      }
    } finally {
      clearTimeout(stageTimer1);
      clearTimeout(stageTimer2);
      clearTimeout(stageTimer3);
      setAnalyzing(false);
      setAnalysisStage('');
    }
  };

  const resetScanner = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setActiveScan(null);
    setQualityError(null);
    setNotes('');
  };

  const printReport = () => {
    window.print();
  };

  // Localized crop and disease names
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

  // Status indicator styling
  const getSeverityBadge = (scan: ScanRecord) => {
    if (scan.is_healthy) {
      return {
        text: 'Healthy Foliage',
        bg: 'bg-emerald-50 text-emerald-800 border-emerald-200/80',
        dot: 'bg-emerald-600',
      };
    }
    const lvl = (scan.severity_level || '').toLowerCase();
    if (lvl.includes('low')) {
      return {
        text: 'Low Severity',
        bg: 'bg-emerald-50 text-emerald-800 border-emerald-200/80',
        dot: 'bg-emerald-600',
      };
    }
    if (lvl.includes('medium')) {
      return {
        text: 'Moderate Severity',
        bg: 'bg-amber-50 text-amber-800 border-amber-200/80',
        dot: 'bg-amber-500',
      };
    }
    if (lvl.includes('high') || lvl.includes('critical')) {
      return {
        text: `${scan.severity_level} Severity`,
        bg: 'bg-rose-50 text-rose-800 border-rose-200/80',
        dot: 'bg-rose-600',
      };
    }
    return {
      text: 'Unverified Severity',
      bg: 'bg-slate-100 text-slate-700 border-slate-200',
      dot: 'bg-slate-400',
    };
  };

  return (
    <div className="space-y-6 sm:space-y-8 animate-in fade-in duration-200">
      
      {/* 1. Header */}
      <div className="border-b border-slate-200 pb-4">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-8 h-8 rounded-lg bg-emerald-700 text-white flex items-center justify-center">
            <Sparkles className="w-4 h-4" />
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
            {t.scanner.title}
          </h1>
        </div>
        <p className="text-xs sm:text-sm text-slate-500 max-w-3xl">
          {t.scanner.subtitle}
        </p>
      </div>

      {/* Quality Guard Rejection Modal */}
      {qualityError && (
        <QualityGuardModal
          result={qualityError}
          onRetry={() => {
            setQualityError(null);
            fileInputRef.current?.click();
          }}
          onClose={() => setQualityError(null)}
        />
      )}

      {/* 2. Upload Zone (Rendered when no active scan result is displayed) */}
      {!activeScan && (
        <div className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200/90 shadow-xs space-y-6">
          
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => !previewUrl && fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-8 sm:p-12 text-center transition cursor-pointer relative ${
              previewUrl
                ? 'border-emerald-500 bg-slate-900'
                : 'border-slate-300 hover:border-emerald-600 bg-slate-50/50 hover:bg-emerald-50/20'
            }`}
          >
            {previewUrl ? (
              <div className="relative aspect-[4/3] max-h-80 mx-auto flex items-center justify-center">
                <img
                  src={previewUrl}
                  alt="Selected leaf preview"
                  className="w-full h-full object-contain rounded-lg"
                />
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    resetScanner();
                  }}
                  className="absolute top-2 right-2 px-3 py-1.5 rounded-md bg-slate-900/80 hover:bg-slate-900 text-white text-xs font-semibold shadow-xs transition"
                >
                  Change Photo
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="w-14 h-14 rounded-xl bg-emerald-50 text-emerald-700 border border-emerald-200/80 flex items-center justify-center mx-auto shadow-2xs">
                  <UploadCloud className="w-7 h-7" />
                </div>
                <div>
                  <h3 className="font-semibold text-sm sm:text-base text-slate-800">
                    {t.scanner.dropzoneTitle}
                  </h3>
                  <p className="text-xs text-slate-500 max-w-md mx-auto mt-1">
                    {t.scanner.dropzoneSubtitle}
                  </p>
                </div>
                <div className="flex items-center justify-center gap-2 pt-1">
                  <span className="text-[11px] font-medium text-slate-400 bg-white px-2 py-0.5 rounded border border-slate-200">JPG</span>
                  <span className="text-[11px] font-medium text-slate-400 bg-white px-2 py-0.5 rounded border border-slate-200">PNG</span>
                  <span className="text-[11px] font-medium text-slate-400 bg-white px-2 py-0.5 rounded border border-slate-200">WebP</span>
                </div>
              </div>
            )}
          </div>

          {/* Hidden inputs */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*"
            className="hidden"
          />
          <input
            type="file"
            ref={cameraInputRef}
            onChange={handleFileChange}
            accept="image/*"
            capture="environment"
            className="hidden"
          />

          {/* Action Bar */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-1">
            
            <div className="flex items-center gap-2 w-full sm:w-auto">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="flex-1 sm:flex-none px-4 py-2 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-semibold transition flex items-center justify-center gap-2"
              >
                <UploadCloud className="w-4 h-4 text-emerald-700" />
                <span>Choose File</span>
              </button>

              <button
                type="button"
                onClick={() => cameraInputRef.current?.click()}
                className="flex-1 sm:flex-none px-4 py-2 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-semibold transition flex items-center justify-center gap-2"
              >
                <Camera className="w-4 h-4 text-emerald-700" />
                <span>{t.scanner.useCamera}</span>
              </button>
            </div>

            {/* Run Analysis CTA */}
            <button
              onClick={runAnalysis}
              disabled={!selectedFile || analyzing}
              className={`w-full sm:w-auto px-6 py-2.5 rounded-lg font-semibold text-xs sm:text-sm transition flex items-center justify-center gap-2 shadow-xs ${
                !selectedFile || analyzing
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed shadow-none'
                  : 'bg-emerald-700 hover:bg-emerald-800 text-white'
              }`}
            >
              {analyzing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-white" />
                  <span>{t.scanner.analyzing}</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 text-emerald-200" />
                  <span>{t.scanner.analyzeBtn}</span>
                </>
              )}
            </button>
          </div>

          {/* Progress Bar during Inference */}
          {analyzing && (
            <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 space-y-2 animate-in fade-in">
              <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
                <span>{analysisStage}</span>
                <span className="text-emerald-700">Executing Neural Models...</span>
              </div>
              <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                <div className="h-full bg-emerald-600 rounded-full w-3/4 animate-pulse"></div>
              </div>
              <p className="text-[11px] text-slate-500">
                Dual-CNN Ensemble (EfficientNet-B0 + MobileNetV3) & PlantSeg U-Net segmentation
              </p>
            </div>
          )}

          {/* Simple photography guidance */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-4 border-t border-slate-100 text-xs text-slate-600">
            <div className="flex items-start gap-2">
              <span className="w-5 h-5 rounded-full bg-slate-100 text-slate-700 font-bold text-[11px] flex items-center justify-center shrink-0">1</span>
              <span>Position camera 10–20 cm above the single affected leaf.</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="w-5 h-5 rounded-full bg-slate-100 text-slate-700 font-bold text-[11px] flex items-center justify-center shrink-0">2</span>
              <span>Tap screen to ensure clear focus before snapping.</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="w-5 h-5 rounded-full bg-slate-100 text-slate-700 font-bold text-[11px] flex items-center justify-center shrink-0">3</span>
              <span>Capture in natural daylight. Avoid extreme flash reflection.</span>
            </div>
          </div>

        </div>
      )}

      {/* 3. Diagnostic Results View */}
      {activeScan && (
        <div className="space-y-6">
          
          {/* Top Result Card */}
          <div className="bg-white rounded-2xl p-6 sm:p-7 border border-slate-200/90 shadow-xs space-y-4">
            
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-100">
              
              <div className="space-y-1.5">
                <div className="flex flex-wrap items-center gap-2">
                  {(() => {
                    const badge = getSeverityBadge(activeScan);
                    return (
                      <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${badge.bg}`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${badge.dot}`}></span>
                        {badge.text}
                      </span>
                    );
                  })()}

                  {activeScan.is_uncertain_classification && (
                    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-50 text-amber-800 border border-amber-300">
                      <AlertTriangle className="w-3 h-3 text-amber-600" />
                      Uncertain Prediction (&lt;40% confidence)
                    </span>
                  )}

                  {activeScan.discrepancy_detected && (
                    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-50 text-amber-800 border border-amber-300">
                      <AlertCircle className="w-3 h-3 text-amber-600" />
                      Cross-Model Discrepancy
                    </span>
                  )}

                  <span className="text-xs text-slate-400">Scan ID: #{activeScan.id}</span>
                </div>

                <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
                  {getCropName(activeScan)} — {getConditionName(activeScan)}
                </h2>

                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-600 pt-0.5">
                  <span>Causal Agent: <strong className="text-slate-900 font-semibold">{activeScan.causal_agent || 'N/A'}</strong></span>
                  <span>•</span>
                  <span>Ensemble Confidence: <strong className="text-slate-900 font-semibold">{activeScan.confidence.toFixed(1)}%</strong></span>
                  <span>•</span>
                  <span>Canopy Damage: <strong className="text-slate-900 font-semibold">{activeScan.affected_percentage.toFixed(1)}%</strong></span>
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-2 shrink-0">
                <button
                  onClick={printReport}
                  className="px-3.5 py-2 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-semibold transition flex items-center gap-1.5 shadow-2xs"
                >
                  <Printer className="w-3.5 h-3.5" />
                  <span>{t.scanner.downloadReport}</span>
                </button>
                <button
                  onClick={resetScanner}
                  className="px-4 py-2 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-semibold transition flex items-center gap-1.5 shadow-xs"
                >
                  <RefreshCw className="w-3.5 h-3.5 text-emerald-200" />
                  <span>{t.scanner.scanAnother}</span>
                </button>
              </div>

            </div>

            {/* Warning Notices if needed */}
            {(activeScan.is_uncertain_classification || activeScan.discrepancy_detected || activeScan.severity_reliability === 'UNRELIABLE') && (
              <div className="p-3.5 rounded-xl bg-amber-50/70 border border-amber-200 text-xs text-amber-900 space-y-1">
                <div className="flex items-center gap-1.5 font-bold text-amber-800">
                  <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                  <span>Important Diagnostic Notice for Farmers & Extension Staff:</span>
                </div>
                {activeScan.is_uncertain_classification && (
                  <p className="text-[11px] leading-relaxed">
                    • The AI ensemble confidence is below 40%. The symptoms may resemble multiple foliar conditions. Inspect the alternative diagnoses below or consult an agronomist before spraying.
                  </p>
                )}
                {activeScan.discrepancy_detected && (
                  <p className="text-[11px] leading-relaxed">
                    • The classifier identified the crop as healthy, but U-Net localized active foliar lesions exceeding the noise floor. Ground inspection recommended.
                  </p>
                )}
                {activeScan.severity_reliability === 'UNRELIABLE' && (
                  <p className="text-[11px] leading-relaxed">
                    • Leaf boundaries could not be reliably segmented (canopy &lt;4% or background clutter). Retake a closer, centered photo of the affected leaf.
                  </p>
                )}
              </div>
            )}

          </div>

          {/* Severity Gauge & Quality Indicators */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Severity Gauge */}
            <div className="md:col-span-2">
              <SeverityGauge
                severityLevel={activeScan.severity_level}
                affectedPercentage={activeScan.affected_percentage}
                isHealthy={activeScan.is_healthy}
                severityReliability={activeScan.severity_reliability}
                isApproximate={activeScan.is_approximate}
                discrepancyDetected={activeScan.discrepancy_detected}
                rawAffectedPercentage={activeScan.raw_affected_percentage}
              />
            </div>

            {/* Quality Guard Status */}
            <div className="bg-white rounded-xl p-5 border border-slate-200/90 shadow-xs flex flex-col justify-between">
              <div>
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">Image Quality Verification</span>
                <div className="flex items-center gap-2 text-emerald-800 font-semibold text-sm mb-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  <span>Input Image Integrity PASSED</span>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                  OpenCV quality guard verified sharpness, illumination, and foliar vegetation spectrum prior to neural network inference.
                </p>
              </div>

              {activeScan.quality_metrics && (
                <div className="mt-3 pt-3 border-t border-slate-100 grid grid-cols-2 gap-2 text-[11px] text-slate-500 font-mono">
                  <div>Sharpness: {activeScan.quality_metrics.laplacian_variance?.toFixed(1) || 'OK'}</div>
                  <div>Brightness: {activeScan.quality_metrics.mean_brightness?.toFixed(1) || 'OK'}</div>
                </div>
              )}
            </div>

          </div>

          {/* Explainable AI Visualizations: Grad-CAM and U-Net Segmentation */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            <GradCamViewer
              originalImageUrl={activeScan.image_url}
              gradcamImageUrl={activeScan.gradcam_url}
              cropName={activeScan.crop}
              conditionName={activeScan.disease_condition}
            />

            <SegmentationOverlayViewer
              segmentationUrl={activeScan.segmentation_url}
              originalUrl={activeScan.image_url}
              affectedPercentage={activeScan.affected_percentage}
              lesionCount={activeScan.lesion_count}
              boundingBoxes={activeScan.bounding_boxes}
              severityLevel={activeScan.severity_level}
            />

          </div>

          {/* Treatment & Cultural Prevention Guidance */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Immediate Curative Action */}
            <div className="bg-white rounded-xl p-5 sm:p-6 border border-slate-200/90 shadow-xs space-y-3">
              <div className="flex items-center gap-2 text-rose-800 font-semibold text-sm">
                <div className="w-6 h-6 rounded-md bg-rose-50 text-rose-700 flex items-center justify-center">
                  <AlertCircle className="w-3.5 h-3.5" />
                </div>
                <span>{t.scanner.managementTitle}</span>
              </div>
              <p className="text-xs text-slate-700 leading-relaxed">
                {activeScan.management_advice}
              </p>
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-200/80 text-[11px] text-slate-600">
                <strong className="text-slate-800">ICAR / AAU Advisory:</strong> Apply recommended spray in early morning or late afternoon when foliar stomata are receptive. Ensure uniform coverage.
              </div>
            </div>

            {/* Long-term Cultural Prevention */}
            <div className="bg-white rounded-xl p-5 sm:p-6 border border-slate-200/90 shadow-xs space-y-3">
              <div className="flex items-center gap-2 text-emerald-800 font-semibold text-sm">
                <div className="w-6 h-6 rounded-md bg-emerald-50 text-emerald-700 flex items-center justify-center">
                  <ShieldCheck className="w-3.5 h-3.5" />
                </div>
                <span>{t.scanner.preventionTitle}</span>
              </div>
              <p className="text-xs text-slate-700 leading-relaxed">
                {activeScan.prevention_advice}
              </p>
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-200/80 text-[11px] text-slate-600">
                <strong className="text-slate-800">Field Hygiene:</strong> Safely collect and destroy heavily infected foliar residues to prevent spores overwintering in soil.
              </div>
            </div>

          </div>

          {/* Top-5 Alternative Diagnoses */}
          {activeScan.alternatives && activeScan.alternatives.length > 0 && (
            <div className="bg-white rounded-xl p-5 sm:p-6 border border-slate-200/90 shadow-xs space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                <h4 className="font-semibold text-xs sm:text-sm text-slate-900">{t.scanner.alternativeDiagnoses}</h4>
                <span className="text-[11px] text-slate-400 font-medium">Ranked Neural Hypotheses</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {activeScan.alternatives.map((alt, idx) => (
                  <div key={idx} className="bg-slate-50/80 p-3 rounded-lg border border-slate-200/80 flex items-center justify-between text-xs">
                    <div>
                      <span className="font-semibold text-slate-800 block truncate max-w-[170px]">{alt.crop} — {alt.condition}</span>
                      <span className="text-[10px] text-slate-500">{alt.is_healthy ? 'Healthy Foliage' : 'Pathogenic Foliage'}</span>
                    </div>
                    <span className="font-bold text-slate-700 bg-white px-2 py-0.5 rounded border border-slate-200 text-xs shadow-2xs">
                      {alt.confidence.toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Specialist Consultation Card */}
          <div className="bg-slate-50 border border-slate-200/90 rounded-xl p-5 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-emerald-700 text-white flex items-center justify-center shrink-0">
                <HelpCircle className="w-5 h-5" />
              </div>
              <div>
                <h4 className="font-bold text-xs text-slate-900">Have questions about this diagnosis?</h4>
                <p className="text-[11px] text-slate-600">
                  Submit this scan directly to Anand Agricultural University agronomists for verification.
                </p>
              </div>
            </div>
            <button
              onClick={() => onNavigate && onNavigate('askExpert')}
              className="px-4 py-2 bg-emerald-700 hover:bg-emerald-800 text-white rounded-lg text-xs font-semibold transition shadow-xs shrink-0"
            >
              Consult University Specialist
            </button>
          </div>

        </div>
      )}

    </div>
  );
};
