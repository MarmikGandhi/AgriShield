// src/pages/Login.tsx

import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useLanguage } from '../context/LanguageContext';
import { authApi } from '../services/api';
import {
  Sprout,
  Lock,
  Mail,
  User,
  MapPin,
  ArrowRight,
  ShieldCheck,
  CheckCircle2
} from 'lucide-react';

interface LoginProps {
  onSuccess: () => void;
}

export const Login: React.FC<LoginProps> = ({ onSuccess }) => {
  const { t } = useLanguage();
  const { login, quickLogin } = useAuth();
  const [isRegister, setIsRegister] = useState<boolean>(false);

  // Form states
  const [email, setEmail] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [fullName, setFullName] = useState<string>('');
  const [role, setRole] = useState<'farmer' | 'agronomist'>('farmer');
  const [village, setVillage] = useState<string>('Borsad');
  const [district, setDistrict] = useState<string>('Anand');
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string>('');

  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg('');

    try {
      if (isRegister) {
        const res = await authApi.register({
          email,
          password,
          full_name: fullName,
          role,
          village,
          district,
        });
        login(res.access_token, res.user);
      } else {
        const res = await authApi.login({ email, password });
        login(res.access_token, res.user);
      }
      onSuccess();
    } catch (err: any) {
      console.error('Auth error:', err);
      setErrorMsg(err.response?.data?.detail || 'Authentication failed. Please check credentials.');
    } finally {
      setLoading(false);
    }
  };

  const handleDemoClick = async (demoRole: 'farmer' | 'agronomist') => {
    setLoading(true);
    await quickLogin(demoRole);
    setLoading(false);
    onSuccess();
  };

  return (
    <div className="max-w-md mx-auto my-10 p-6 sm:p-8 bg-white rounded-xl border border-slate-200/90 shadow-xs space-y-6">
      
      {/* Brand Header */}
      <div className="text-center space-y-2">
        <div className="w-10 h-10 rounded-xl bg-emerald-700 text-white flex items-center justify-center mx-auto shadow-xs">
          <Sprout className="w-5 h-5" />
        </div>
        <h2 className="text-xl font-bold text-slate-900 tracking-tight">
          {isRegister ? 'Create AgriShield Account' : 'Sign in to AgriShield AI'}
        </h2>
        <p className="text-xs text-slate-500">
          Precision foliar pathology for Gujarat & Indian agriculture
        </p>
      </div>

      {/* One-Click Quick Access Box */}
      <div className="p-3.5 rounded-xl bg-slate-50/80 border border-slate-200/80 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Quick Demo Access</span>
          <span className="text-[10px] text-slate-400">1-Click Sign In</span>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => handleDemoClick('farmer')}
            className="py-2 px-3 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-semibold transition shadow-xs flex items-center justify-center gap-1.5 cursor-pointer"
          >
            <Sprout className="w-3.5 h-3.5" />
            <span>Farmer Demo</span>
          </button>
          <button
            type="button"
            onClick={() => handleDemoClick('agronomist')}
            className="py-2 px-3 rounded-lg bg-slate-800 hover:bg-slate-900 text-white text-xs font-semibold transition shadow-xs flex items-center justify-center gap-1.5 cursor-pointer"
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Agronomist Demo</span>
          </button>
        </div>
        <p className="text-[10px] text-slate-400 text-center">
          Pre-configured with Anand District test profiles (AAU Benchmark)
        </p>
      </div>

      {errorMsg && (
        <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs font-medium">
          {errorMsg}
        </div>
      )}

      {/* Login Form */}
      <form onSubmit={handleLoginSubmit} className="space-y-4">
        {isRegister && (
          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-1">Full Name</label>
            <div className="relative">
              <User className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Rameshbhai Patel"
                required
                className="w-full pl-9 pr-3.5 py-2 rounded-lg border border-slate-200 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-1 focus:ring-emerald-600 focus:border-emerald-600 transition"
              />
            </div>
          </div>
        )}

        <div>
          <label className="text-xs font-semibold text-slate-700 block mb-1">Email Address</label>
          <div className="relative">
            <Mail className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="farmer@agri.in"
              required
              className="w-full pl-9 pr-3.5 py-2 rounded-lg border border-slate-200 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-1 focus:ring-emerald-600 focus:border-emerald-600 transition"
            />
          </div>
        </div>

        <div>
          <label className="text-xs font-semibold text-slate-700 block mb-1">Password</label>
          <div className="relative">
            <Lock className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              className="w-full pl-9 pr-3.5 py-2 rounded-lg border border-slate-200 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-1 focus:ring-emerald-600 focus:border-emerald-600 transition"
            />
          </div>
        </div>

        {isRegister && (
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">District</label>
              <input
                type="text"
                value={district}
                onChange={(e) => setDistrict(e.target.value)}
                placeholder="Anand"
                className="w-full px-3 py-2 rounded-lg border border-slate-200 text-xs text-slate-900 focus:outline-none focus:ring-1 focus:ring-emerald-600 focus:border-emerald-600 transition"
              />
            </div>
            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">Role</label>
              <select
                value={role}
                onChange={(e: any) => setRole(e.target.value)}
                className="w-full px-3 py-2 rounded-lg border border-slate-200 text-xs font-semibold text-slate-700 focus:outline-none focus:ring-1 focus:ring-emerald-600 focus:border-emerald-600 transition"
              >
                <option value="farmer">Farmer</option>
                <option value="agronomist">Agronomist</option>
              </select>
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full py-2.5 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white font-semibold text-xs transition shadow-xs flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
        >
          <span>{loading ? 'Authenticating...' : isRegister ? 'Register & Sign In' : 'Sign In'}</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </form>

      {/* Switch between sign in and register */}
      <div className="text-center text-xs text-slate-500 pt-3 border-t border-slate-100">
        {isRegister ? (
          <p>
            Already have an account?{' '}
            <button
              onClick={() => setIsRegister(false)}
              className="font-semibold text-emerald-700 hover:underline cursor-pointer"
            >
              Sign In here
            </button>
          </p>
        ) : (
          <p>
            Need a new account?{' '}
            <button
              onClick={() => setIsRegister(true)}
              className="font-semibold text-emerald-700 hover:underline cursor-pointer"
            >
              Register here
            </button>
          </p>
        )}
      </div>

    </div>
  );
};
