// src/pages/AgronomistPortal.tsx

import React, { useState, useEffect } from 'react';
import { useLanguage } from '../context/LanguageContext';
import { useAuth } from '../context/AuthContext';
import { agronomistApi } from '../services/api';
import { ExpertQuestionItem } from '../types';
import {
  GraduationCap,
  CheckCircle,
  Clock,
  Send,
  Sparkles,
  RefreshCw,
  Building,
  Award,
  AlertCircle
} from 'lucide-react';

export const AgronomistPortal: React.FC = () => {
  const { t } = useLanguage();
  const { user, isAgronomist, quickLogin } = useAuth();
  const [cases, setCases] = useState<ExpertQuestionItem[]>([]);
  const [selectedCase, setSelectedCase] = useState<ExpertQuestionItem | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [submitting, setSubmitting] = useState<boolean>(false);

  // Response inputs
  const [responseText, setResponseText] = useState<string>('');
  const [treatment, setTreatment] = useState<string>('');
  const [prevention, setPrevention] = useState<string>('');

  const fetchPendingCases = async () => {
    setLoading(true);
    try {
      const data = await agronomistApi.getPendingCases();
      setCases(data);
      if (data.length > 0 && !selectedCase) {
        setSelectedCase(data[0]);
      }
    } catch (e) {
      console.error('Failed to load agronomist cases:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPendingCases();
  }, []);

  const handleSubmitAdvisory = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedCase || !responseText) return;

    setSubmitting(true);
    try {
      await agronomistApi.submitAdvisory(selectedCase.id, {
        response_text: responseText,
        recommended_treatment: treatment || undefined,
        preventive_measures: prevention || undefined,
      });

      alert('Official advisory successfully approved and dispatched to farmer!');
      setResponseText('');
      setTreatment('');
      setPrevention('');
      setSelectedCase(null);
      await fetchPendingCases();
    } catch (err: any) {
      console.error('Submit advisory error:', err);
      alert(err.response?.data?.detail || 'Failed to submit advisory.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6 sm:space-y-8 animate-in fade-in duration-200">
      
      {/* Header Banner */}
      <div className="bg-white rounded-2xl p-6 sm:p-7 border border-slate-200/90 shadow-xs space-y-2">
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-indigo-50 border border-indigo-200 text-indigo-800 text-xs font-semibold">
          <Award className="w-3.5 h-3.5 text-indigo-600" />
          <span>Anand Agricultural University (AAU) Specialist Workstation</span>
        </div>
        <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
          {t.agronomist.title}
        </h1>
        <p className="text-slate-600 text-xs sm:text-sm max-w-3xl leading-relaxed">
          Review unresolved farmer inquiries, inspect foliar pathology symptoms and AI scans, and prescribe verified chemical and cultural interventions.
        </p>

        {!isAgronomist && (
          <div className="mt-3 p-3 rounded-lg bg-amber-50 border border-amber-200 text-amber-900 text-xs flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <span>You are viewing in farmer preview mode. Switch to official agronomist mode to approve cases:</span>
            <button
              onClick={() => quickLogin('agronomist')}
              className="px-3 py-1.5 rounded-md bg-indigo-700 hover:bg-indigo-800 text-white font-semibold transition shrink-0 self-start sm:self-auto"
            >
              Switch to Agronomist
            </button>
          </div>
        )}
      </div>

      {/* Main Split Interface */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 sm:gap-8">
        
        {/* Cases Triage Queue */}
        <div className="bg-white rounded-xl p-5 border border-slate-200/90 shadow-xs space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-slate-100">
            <h3 className="font-semibold text-xs sm:text-sm text-slate-900 flex items-center gap-2">
              <Clock className="w-4 h-4 text-indigo-700" />
              <span>Pending Cases ({cases.length})</span>
            </h3>
            <button
              onClick={fetchPendingCases}
              className="p-1.5 rounded-md border border-slate-200 bg-white hover:bg-slate-50 text-slate-600 transition shadow-2xs"
              title="Refresh queue"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {loading ? (
            <div className="p-8 text-center text-xs text-slate-400">Loading cases...</div>
          ) : cases.length === 0 ? (
            <div className="p-8 text-center bg-slate-50 rounded-xl border border-dashed border-slate-200 space-y-2">
              <CheckCircle className="w-7 h-7 text-emerald-600 mx-auto" />
              <p className="text-xs font-semibold text-slate-800">All Inquiries Resolved</p>
              <p className="text-[11px] text-slate-500">Zero pending farmer questions in the triage queue.</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-[550px] overflow-y-auto pr-1">
              {cases.map((c) => (
                <div
                  key={c.id}
                  onClick={() => setSelectedCase(c)}
                  className={`p-3.5 rounded-xl border transition cursor-pointer text-xs ${
                    selectedCase?.id === c.id
                      ? 'bg-indigo-50/70 border-indigo-300 shadow-2xs'
                      : 'bg-white border-slate-200 hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center justify-between font-semibold text-slate-900 mb-1">
                    <span className="truncate">{c.crop}: {c.title}</span>
                    <span className="text-[10px] text-indigo-700 bg-indigo-50 border border-indigo-200 px-2 py-0.2 rounded-md shrink-0">
                      Case #{c.id}
                    </span>
                  </div>
                  <p className="text-slate-500 line-clamp-2 text-[11px] mb-2">{c.description}</p>
                  <div className="flex items-center justify-between text-[10px] text-slate-400 pt-1 border-t border-slate-100">
                    <span>Farmer: {c.farmer_name}</span>
                    <span>{new Date(c.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Case Inspector & Official Advisory Submission Form */}
        <div className="lg:col-span-2 space-y-6">
          {selectedCase ? (
            <div className="bg-white rounded-xl p-5 sm:p-7 border border-slate-200/90 shadow-xs space-y-6">
              
              {/* Case Details */}
              <div className="space-y-3 pb-5 border-b border-slate-100">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-indigo-800 bg-indigo-50 px-2.5 py-0.5 rounded border border-indigo-200">
                    Crop: {selectedCase.crop}
                  </span>
                  <span className="text-xs text-slate-400">
                    Submitted: {new Date(selectedCase.created_at).toLocaleString()}
                  </span>
                </div>

                <h3 className="text-base sm:text-lg font-bold text-slate-900">{selectedCase.title}</h3>
                
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-200/80 text-xs text-slate-700 space-y-2">
                  <p><strong className="text-slate-900">Farmer Observations:</strong> {selectedCase.description}</p>
                  {selectedCase.symptoms && (
                    <p><strong className="text-slate-900">Reported Symptoms:</strong> {selectedCase.symptoms}</p>
                  )}
                  <p className="text-[11px] text-slate-400 pt-1 border-t border-slate-200/60">
                    Farmer Contact: {selectedCase.farmer_name}
                  </p>
                </div>

                {selectedCase.attached_image_url && (
                  <div>
                    <span className="text-xs font-semibold text-slate-700 block mb-1">Attached Symptom Photograph:</span>
                    <div className="max-w-xs aspect-[4/3] rounded-lg overflow-hidden bg-slate-900 border border-slate-200">
                      <img src={selectedCase.attached_image_url} alt="Attached leaf" className="w-full h-full object-contain" />
                    </div>
                  </div>
                )}
              </div>

              {/* Advisory Response Form */}
              <form onSubmit={handleSubmitAdvisory} className="space-y-4">
                <h4 className="font-semibold text-sm text-slate-900">Official Agronomic Diagnosis & Advisory</h4>

                <div>
                  <label className="text-xs font-semibold text-slate-700 block mb-1">
                    Pathology Diagnosis & Advisory Notes *
                  </label>
                  <textarea
                    rows={4}
                    value={responseText}
                    onChange={(e) => setResponseText(e.target.value)}
                    placeholder="Provide detailed diagnostic assessment of the foliar stress, causal fungal/bacterial pathogen, and severity..."
                    required
                    className="w-full p-3 rounded-lg border border-slate-200 text-xs text-slate-900 focus:outline-none focus:ring-1 focus:ring-indigo-600 focus:border-indigo-600 resize-none shadow-2xs"
                  />
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs font-semibold text-slate-700 block mb-1">
                      Recommended Chemical / Bio-Fungicide
                    </label>
                    <input
                      type="text"
                      value={treatment}
                      onChange={(e) => setTreatment(e.target.value)}
                      placeholder="e.g. Copper Oxychloride 50% WP @ 2.5 g/L"
                      className="w-full px-3 py-2 rounded-lg border border-slate-200 text-xs text-slate-900 focus:outline-none focus:ring-1 focus:ring-indigo-600 shadow-2xs"
                    />
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-700 block mb-1">
                      Preventive Agronomic Measures
                    </label>
                    <input
                      type="text"
                      value={prevention}
                      onChange={(e) => setPrevention(e.target.value)}
                      placeholder="e.g. Regulate furrow irrigation, avoid sprinkler wetting"
                      className="w-full px-3 py-2 rounded-lg border border-slate-200 text-xs text-slate-900 focus:outline-none focus:ring-1 focus:ring-indigo-600 shadow-2xs"
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={submitting || !responseText}
                  className={`px-6 py-2.5 rounded-lg font-semibold text-xs text-white transition shadow-xs flex items-center justify-center gap-2 ${
                    submitting || !responseText
                      ? 'bg-slate-300 text-slate-500 cursor-not-allowed shadow-none'
                      : 'bg-indigo-700 hover:bg-indigo-800'
                  }`}
                >
                  {submitting ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Approving Advisory...</span>
                    </>
                  ) : (
                    <>
                      <Send className="w-3.5 h-3.5" />
                      <span>Approve & Dispatch Advisory</span>
                    </>
                  )}
                </button>
              </form>

            </div>
          ) : (
            <div className="p-12 text-center bg-white rounded-xl border border-dashed border-slate-200">
              <Clock className="w-8 h-8 text-slate-300 mx-auto mb-2" />
              <h3 className="text-sm font-semibold text-slate-800">Select an Inquiry</h3>
              <p className="text-xs text-slate-500 mt-1">Select a pending case from the queue to inspect symptoms and prescribe treatment.</p>
            </div>
          )}
        </div>

      </div>

    </div>
  );
};
