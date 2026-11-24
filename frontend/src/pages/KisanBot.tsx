// src/pages/KisanBot.tsx

import React, { useState, useRef, useEffect } from 'react';
import { useLanguage } from '../context/LanguageContext';
import { chatApi } from '../services/api';
import { ChatMessage } from '../types';
import {
  Bot,
  Send,
  User,
  Sparkles,
  BookOpen,
  HelpCircle,
  RefreshCw,
  CheckCircle2
} from 'lucide-react';

export const KisanBot: React.FC = () => {
  const { t, language } = useLanguage();
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: language === 'gu'
        ? "નમસ્તે ખેડૂત મિત્ર! હું કિસાન એઆઈ સહાયક છું. આણંદ કૃષિ યુનિવર્સિટી અને ICAR દ્વારા પ્રમાણિત ખેતી, પાક રોગ, જંતુ નિયંત્રણ અને ખાતર સંબંધિત કોઈપણ પ્રશ્ન પૂછો."
        : language === 'hi'
        ? "नमस्ते किसान भाई! मैं किसान एआई सहायक हूँ। आनंद कृषि विश्वविद्यालय और आईसीएआर द्वारा सत्यापित फसल रोग, कीट प्रबंधन और खाद संबंधित कोई भी प्रश्न पूछें।"
        : "Namaste Farmer! I am your Kisan AI Agronomic Assistant. Ask me anything about crop diseases, pest management, or fertilizer schedules verified by ICAR & Anand Agricultural University (AAU).",
      sources: [
        "ICAR Crop Protection Handbooks (2025)",
        "Anand Agricultural University (AAU) Extension Manual",
        "Navsari Agricultural University (NAU) Recommendations",
      ],
    },
  ]);

  const [input, setInput] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async (messageText?: string) => {
    const textToSend = messageText || input;
    if (!textToSend.trim() || loading) return;

    const userMessage: ChatMessage = { role: 'user', content: textToSend };
    setMessages((prev) => [...prev, userMessage]);
    if (!messageText) setInput('');
    setLoading(true);

    try {
      const data = await chatApi.sendMessage({
        message: textToSend,
        conversation_history: messages,
        language,
      });

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: data.reply,
        sources: data.sources || [],
        recommended_actions: data.recommended_actions || [],
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Chat error:', err);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: "I'm having trouble connecting to the agronomic knowledge server. Please verify your connection or consult your local KVK extension specialist.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const sampleQuestions = [
    t.kisanBot.sampleQ1,
    t.kisanBot.sampleQ2,
    t.kisanBot.sampleQ3,
  ];

  return (
    <div className="max-w-4xl mx-auto space-y-4 sm:space-y-5 animate-in fade-in duration-200">
      
      {/* Header */}
      <div className="border-b border-slate-200 pb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-emerald-700 text-white flex items-center justify-center shadow-xs">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
              {t.kisanBot.title}
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200">
                ICAR / AAU Advisory RAG
              </span>
            </h1>
            <p className="text-xs text-slate-500">{t.kisanBot.subtitle}</p>
          </div>
        </div>
      </div>

      {/* Suggested Questions */}
      <div className="flex flex-wrap gap-2">
        {sampleQuestions.map((q, idx) => (
          <button
            key={idx}
            onClick={() => handleSend(q)}
            className="text-xs bg-white hover:bg-emerald-50/60 hover:border-emerald-300 text-slate-700 px-3 py-1.5 rounded-lg border border-slate-200 transition shadow-2xs font-medium text-left"
          >
            💡 {q}
          </button>
        ))}
      </div>

      {/* Chat Window */}
      <div className="bg-white rounded-2xl border border-slate-200/90 shadow-xs flex flex-col h-[520px] overflow-hidden">
        
        {/* Messages scroll area */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4 bg-slate-50/40">
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={`flex items-start gap-2.5 ${
                m.role === 'user' ? 'flex-row-reverse' : 'flex-row'
              }`}
            >
              <div className={`w-7 h-7 rounded-md flex items-center justify-center shrink-0 text-white text-xs font-bold ${
                m.role === 'user' ? 'bg-slate-800' : 'bg-emerald-700'
              }`}>
                {m.role === 'user' ? <User className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5" />}
              </div>

              <div className={`max-w-[85%] sm:max-w-[75%] rounded-xl p-3.5 sm:p-4 text-xs leading-relaxed ${
                m.role === 'user'
                  ? 'bg-emerald-700 text-white rounded-tr-xs shadow-xs'
                  : 'bg-white border border-slate-200 text-slate-800 rounded-tl-xs shadow-2xs'
              }`}>
                <p className="whitespace-pre-line">{m.content}</p>

                {/* Sources & Citations */}
                {m.sources && m.sources.length > 0 && (
                  <div className="mt-3 pt-2.5 border-t border-slate-100 text-[11px] text-slate-500">
                    <span className="font-semibold text-slate-700 block mb-1 flex items-center gap-1">
                      <BookOpen className="w-3 h-3 text-emerald-700" />
                      {t.kisanBot.sourcesTitle}
                    </span>
                    <ul className="space-y-0.5">
                      {m.sources.map((s, sIdx) => (
                        <li key={sIdx} className="text-slate-600">• {s}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex items-start gap-2.5">
              <div className="w-7 h-7 rounded-md bg-emerald-700 flex items-center justify-center text-white shrink-0">
                <Bot className="w-3.5 h-3.5" />
              </div>
              <div className="bg-white border border-slate-200 rounded-xl rounded-tl-xs p-3 text-xs text-slate-500 flex items-center gap-2 shadow-2xs">
                <RefreshCw className="w-3.5 h-3.5 text-emerald-700 animate-spin" />
                <span>{t.kisanBot.thinking}</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="p-3 sm:p-3.5 border-t border-slate-200 bg-white">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="flex items-center gap-2"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={t.kisanBot.inputPlaceholder}
              className="flex-1 bg-slate-50 border border-slate-200 rounded-lg px-3.5 py-2.5 text-xs text-slate-900 focus:outline-none focus:ring-1 focus:ring-emerald-600 focus:border-emerald-600 focus:bg-white transition"
            />
            <button
              type="submit"
              disabled={!input.trim() || loading}
              className={`px-4 py-2.5 rounded-lg font-semibold text-xs transition flex items-center gap-1.5 shadow-xs shrink-0 ${
                !input.trim() || loading
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  : 'bg-emerald-700 hover:bg-emerald-800 text-white'
              }`}
            >
              <span>{t.kisanBot.send}</span>
              <Send className="w-3.5 h-3.5" />
            </button>
          </form>
        </div>

      </div>

    </div>
  );
};
