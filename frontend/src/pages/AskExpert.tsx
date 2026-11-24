// src/pages/AskExpert.tsx

import React, { useState, useEffect } from 'react';
import { useLanguage } from '../context/LanguageContext';
import { useAuth } from '../context/AuthContext';
import { expertApi } from '../services/api';
import { ExpertQuestionItem } from '../types';
import {
  HelpCircle,
  Send,
  Upload,
  Clock,
  GraduationCap,
  RefreshCw,
  MessageSquare,
  CheckCircle2
} from 'lucide-react';

export const AskExpert: React.FC = () => {
  const { t } = useLanguage();
  const { user } = useAuth();
  const [questions, setQuestions] = useState<ExpertQuestionItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [submitting, setSubmitting] = useState<boolean>(false);

  // Form states
  const [crop, setCrop] = useState<string>('');
  const [title, setTitle] = useState<string>('');
  const [description, setDescription] = useState<string>('');
  const [symptoms, setSymptoms] = useState<string>('');
  const [photo, setPhoto] = useState<File | null>(null);

  const fetchQuestions = async () => {
    setLoading(true);
    try {
      const data = await expertApi.listQuestions();
      setQuestions(data);
    } catch (e) {
      console.error('Failed to load expert questions:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQuestions();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!crop || !title || !description) {
      alert('Please fill out all required fields.');
      return;
    }

    setSubmitting(true);
    const formData = new FormData();
    formData.append('crop', crop);
    formData.append('title', title);
    formData.append('description', description);
    if (symptoms) formData.append('symptoms', symptoms);
    if (photo) formData.append('image', photo);

    try {
      await expertApi.createQuestion(formData);
      setCrop('');
      setTitle('');
      setDescription('');
      setSymptoms('');
      setPhoto(null);
      await fetchQuestions();
    } catch (err: any) {
      console.error('Submit query error:', err);
      alert(err.response?.data?.detail || 'Failed to submit inquiry.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6 sm:space-y-8 animate-in fade-in duration-200">
      
      {/* Header */}
      <div className="border-b border-slate-200 pb-4">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-8 h-8 rounded-lg bg-emerald-700 text-white flex items-center justify-center">
            <HelpCircle className="w-4 h-4" />
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
            {t.askExpert.title}
          </h1>
        </div>
        <p className="text-xs sm:text-sm text-slate-500">
          {t.askExpert.subtitle}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 sm:gap-8">
        
        {/* Inquiry Form */}
        <div className="bg-white rounded-xl p-5 sm:p-6 border border-slate-200/90 shadow-xs space-y-4">
          <div className="flex items-center gap-2 pb-3 border-b border-slate-100">
            <GraduationCap className="w-4 h-4 text-emerald-700" />
            <h3 className="font-semibold text-xs sm:text-sm text-slate-900">{t.askExpert.newQuestionTitle}</h3>
          </div>

          <form onSubmit={handleSubmit} className="space-y-3.5">
            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">
                {t.askExpert.cropLabel} *
              </label>
              <input
                type="text"
                value={crop}
                onChange={(e) => setCrop(e.target.value)}
                placeholder="e.g. Tomato, Cotton, Cumin, Potato"
                required
                className="w-full px-3 py-2 rounded-lg border border-slate-200 text-xs text-slate-900 focus:outline-none focus:ring-1 focus:ring-emerald-600 focus:border-emerald-600 shadow-2xs"
              />
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">
                {t.askExpert.titleLabel} *
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Brief summary of the issue"
                required
                className="w-full px-3 py-2 rounded-lg border border-slate-200 text-xs text-slate-900 focus:outline-none focus:ring-1 focus:ring-emerald-600 focus:border-emerald-600 shadow-2xs"
              />
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">
                {t.askExpert.descLabel} *
              </label>
              <textarea
                rows={3}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe leaf spots, curling, discoloration, duration noticed..."
                required
                className="w-full px-3 py-2 rounded-lg border border-slate-200 text-xs text-slate-900 focus:outline-none focus:ring-1 focus:ring-emerald-600 focus:border-emerald-600 resize-none shadow-2xs"
              />
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">
                Leaf Photo (Optional)
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files && setPhoto(e.target.files[0])}
                className="w-full text-xs text-slate-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-slate-100 file:text-slate-700 hover:file:bg-slate-200 cursor-pointer"
              />
            </div>

            <button
              type="submit"
              disabled={submitting}
              className={`w-full py-2.5 rounded-lg font-semibold text-xs text-white transition shadow-xs flex items-center justify-center gap-2 ${
                submitting
                  ? 'bg-emerald-400 cursor-not-allowed'
                  : 'bg-emerald-700 hover:bg-emerald-800'
              }`}
            >
              {submitting ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Submitting to Specialist...</span>
                </>
              ) : (
                <>
                  <Send className="w-3.5 h-3.5" />
                  <span>{t.askExpert.submitBtn}</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Existing Questions Feed */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-100">
            <h3 className="font-semibold text-xs sm:text-sm text-slate-900">{t.askExpert.myQuestionsTitle}</h3>
            <button
              onClick={fetchQuestions}
              className="p-1.5 rounded-md border border-slate-200 bg-white hover:bg-slate-50 text-slate-600 transition shadow-2xs"
              title="Refresh questions"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {loading ? (
            <div className="p-8 text-center bg-white rounded-xl border border-slate-200">
              <RefreshCw className="w-6 h-6 text-emerald-700 mx-auto mb-2 animate-spin" />
              <p className="text-xs text-slate-500 font-medium">Loading agronomic inquiry threads...</p>
            </div>
          ) : questions.length === 0 ? (
            <div className="p-8 text-center bg-white rounded-xl border border-dashed border-slate-200 space-y-2">
              <MessageSquare className="w-7 h-7 text-slate-300 mx-auto" />
              <p className="text-xs font-semibold text-slate-700">No Inquiries Submitted Yet</p>
              <p className="text-[11px] text-slate-500">
                Have an unidentifiable crop symptom? Submit your question on the left to consult university agronomists.
              </p>
            </div>
          ) : (
            <div className="space-y-3.5">
              {questions.map((q) => (
                <div
                  key={q.id}
                  className="bg-white rounded-xl p-4 sm:p-5 border border-slate-200/90 shadow-xs space-y-3"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-semibold text-xs text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                          {q.crop}
                        </span>
                        <span className="text-[10px] text-slate-400 flex items-center gap-1 font-medium">
                          <Clock className="w-3 h-3" />
                          {new Date(q.created_at).toLocaleDateString()}
                        </span>
                      </div>
                      <h4 className="text-sm font-bold text-slate-900">{q.title}</h4>
                      <p className="text-xs text-slate-600 mt-1 leading-relaxed">{q.description}</p>
                    </div>

                    <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-semibold shrink-0 border ${
                      q.status === 'answered'
                        ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                        : 'bg-amber-50 text-amber-800 border-amber-200'
                    }`}>
                      {q.status === 'answered' ? t.askExpert.answered : t.askExpert.pending}
                    </span>
                  </div>

                  {/* Attached photo if present */}
                  {q.attached_image_url && (
                    <div className="w-16 h-16 rounded-lg overflow-hidden bg-slate-100 border border-slate-200">
                      <img src={q.attached_image_url} alt="Attached leaf" className="w-full h-full object-cover" />
                    </div>
                  )}

                  {/* Official Agronomist Response if answered */}
                  {q.responses && q.responses.length > 0 && (
                    <div className="bg-emerald-50/50 border border-emerald-200/80 rounded-lg p-3.5 space-y-1.5 text-xs">
                      <div className="flex items-center gap-2 text-emerald-900 font-semibold">
                        <GraduationCap className="w-4 h-4 text-emerald-700" />
                        <span>Official Agronomist Advisory: {q.responses[0].agronomist_name}</span>
                      </div>
                      <p className="text-slate-700 leading-relaxed">
                        {q.responses[0].response_text}
                      </p>
                      {q.responses[0].recommended_treatment && (
                        <div className="text-[11px] bg-white p-2.5 rounded border border-emerald-200 mt-1">
                          <strong className="text-emerald-900">Recommended Treatment:</strong> {q.responses[0].recommended_treatment}
                        </div>
                      )}
                    </div>
                  )}

                </div>
              ))}
            </div>
          )}
        </div>

      </div>

    </div>
  );
};
