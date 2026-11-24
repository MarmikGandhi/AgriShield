// src/components/Navbar.tsx

import React, { useState } from 'react';
import { useLanguage } from '../context/LanguageContext';
import { useAuth } from '../context/AuthContext';
import { Language } from '../types';
import {
  Sprout,
  Scan,
  LayoutDashboard,
  History,
  Bell,
  Bot,
  HelpCircle,
  GraduationCap,
  Database,
  Globe,
  LogOut,
  User,
  Menu,
  X,
  ChevronDown
} from 'lucide-react';

interface NavbarProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  unreadCount?: number;
}

export const Navbar: React.FC<NavbarProps> = ({ currentPage, onNavigate, unreadCount = 0 }) => {
  const { language, setLanguage, t } = useLanguage();
  const { user, isAuthenticated, logout, quickLogin } = useAuth();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [langDropdownOpen, setLangDropdownOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  const navItems = [
    { id: 'dashboard', label: t.nav.dashboard, icon: LayoutDashboard },
    { id: 'scanner', label: t.nav.scanner, icon: Scan, isScanAction: true },
    { id: 'history', label: t.nav.history, icon: History },
    { id: 'alerts', label: t.nav.alerts, icon: Bell, badge: unreadCount > 0 ? unreadCount : undefined },
    { id: 'kisanBot', label: t.nav.kisanBot, icon: Bot },
    { id: 'askExpert', label: t.nav.askExpert, icon: HelpCircle },
    { id: 'agronomist', label: t.nav.agronomist, icon: GraduationCap, agronomistOnly: true },
    { id: 'datasets', label: t.nav.datasets, icon: Database },
  ];

  const handleNav = (id: string) => {
    onNavigate(id);
    setMobileMenuOpen(false);
  };

  const languageLabels: Record<Language, string> = {
    en: 'English (EN)',
    hi: 'हिन्दी (HI)',
    gu: 'ગુજરાતી (GU)',
  };

  return (
    <header className="sticky top-0 z-40 bg-white border-b border-slate-200/80 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo & Brand Identity */}
          <div
            className="flex items-center gap-3 cursor-pointer group"
            onClick={() => handleNav('dashboard')}
          >
            <div className="w-9 h-9 rounded-lg bg-emerald-700 flex items-center justify-center text-white shadow-xs group-hover:bg-emerald-800 transition">
              <Sprout className="w-5 h-5 stroke-[2.2]" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-lg tracking-tight text-slate-900">AgriShield</span>
                <span className="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200/60">
                  AI
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium -mt-0.5 hidden sm:block">
                Early Crop Stress & Alert System
              </p>
            </div>
          </div>

          {/* Desktop Nav Items */}
          <nav className="hidden lg:flex items-center gap-1">
            {navItems.map((item) => {
              if (item.agronomistOnly && user?.role !== 'agronomist') return null;
              const Icon = item.icon;
              const isActive = currentPage === item.id;

              if (item.isScanAction) {
                return (
                  <button
                    key={item.id}
                    onClick={() => handleNav(item.id)}
                    className={`ml-1 mr-1 flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition shadow-xs ${
                      isActive
                        ? 'bg-emerald-800 text-white ring-2 ring-emerald-600/30'
                        : 'bg-emerald-700 hover:bg-emerald-800 text-white'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5 text-emerald-200" />
                    <span>{item.label}</span>
                  </button>
                );
              }

              return (
                <button
                  key={item.id}
                  onClick={() => handleNav(item.id)}
                  className={`relative flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                    isActive
                      ? 'bg-emerald-50 text-emerald-800 font-semibold border border-emerald-200/70 shadow-2xs'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-emerald-700' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                  {item.badge !== undefined && (
                    <span className="ml-0.5 px-1.5 py-0.2 text-[10px] font-bold rounded-full bg-rose-500 text-white">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Right actions: Language & Profile */}
          <div className="flex items-center gap-2 sm:gap-2.5">
            
            {/* Language Selector */}
            <div className="relative">
              <button
                onClick={() => setLangDropdownOpen(!langDropdownOpen)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition shadow-2xs"
                title="Change Language"
              >
                <Globe className="w-3.5 h-3.5 text-emerald-600" />
                <span className="uppercase font-semibold">{language}</span>
                <ChevronDown className="w-3 h-3 text-slate-400" />
              </button>

              {langDropdownOpen && (
                <div className="absolute right-0 mt-1.5 w-44 bg-white rounded-xl shadow-md border border-slate-200/80 py-1 z-50 animate-in fade-in zoom-in-95">
                  {(['en', 'hi', 'gu'] as Language[]).map((lang) => (
                    <button
                      key={lang}
                      onClick={() => {
                        setLanguage(lang);
                        setLangDropdownOpen(false);
                      }}
                      className={`w-full text-left px-3 py-2 text-xs font-medium flex items-center justify-between transition ${
                        language === lang ? 'bg-emerald-50 text-emerald-800 font-semibold' : 'text-slate-700 hover:bg-slate-50'
                      }`}
                    >
                      <span>{languageLabels[lang]}</span>
                      {language === lang && <span className="w-1.5 h-1.5 rounded-full bg-emerald-600"></span>}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* User Profile / Quick Switch */}
            {isAuthenticated ? (
              <div className="relative">
                <button
                  onClick={() => setUserMenuOpen(!userMenuOpen)}
                  className="flex items-center gap-2 pl-1.5 pr-2 py-1 rounded-lg border border-slate-200 hover:bg-slate-50 transition bg-white shadow-2xs"
                >
                  <div className={`w-6 h-6 rounded-md flex items-center justify-center text-white text-[11px] font-bold ${
                    user?.role === 'agronomist' ? 'bg-indigo-700' : 'bg-emerald-700'
                  }`}>
                    {user?.full_name ? user.full_name[0] : 'U'}
                  </div>
                  <div className="text-left hidden md:block">
                    <p className="text-xs font-semibold text-slate-800 leading-tight">{user?.full_name}</p>
                    <p className="text-[10px] text-slate-500 capitalize">{user?.role}</p>
                  </div>
                  <ChevronDown className="w-3 h-3 text-slate-400" />
                </button>

                {userMenuOpen && (
                  <div className="absolute right-0 mt-1.5 w-60 bg-white rounded-xl shadow-lg border border-slate-200/80 p-2 z-50">
                    <div className="px-3 py-2 border-b border-slate-100">
                      <p className="text-xs font-bold text-slate-900">{user?.full_name}</p>
                      <p className="text-xs text-slate-500 truncate">{user?.email}</p>
                      <span className={`inline-block mt-1 px-2 py-0.5 text-[10px] font-semibold rounded-md uppercase ${
                        user?.role === 'agronomist' ? 'bg-indigo-50 text-indigo-700 border border-indigo-200' : 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                      }`}>
                        {user?.role}
                      </span>
                    </div>

                    {/* Quick Demo Switchers */}
                    <div className="py-1.5 border-b border-slate-100">
                      <p className="text-[10px] font-semibold text-slate-400 px-3 uppercase tracking-wider mb-1">Demo Profile</p>
                      <button
                        onClick={() => {
                          quickLogin('farmer');
                          setUserMenuOpen(false);
                        }}
                        className={`w-full text-left px-3 py-1.5 text-xs rounded-lg transition flex items-center justify-between ${
                          user?.role === 'farmer' ? 'bg-emerald-50 text-emerald-800 font-semibold' : 'text-slate-600 hover:bg-slate-50'
                        }`}
                      >
                        <span>Farmer (Rameshbhai)</span>
                        {user?.role === 'farmer' && <span className="text-[10px] bg-emerald-200 text-emerald-800 px-1.5 py-0.2 rounded font-bold">Active</span>}
                      </button>
                      <button
                        onClick={() => {
                          quickLogin('agronomist');
                          setUserMenuOpen(false);
                        }}
                        className={`w-full text-left px-3 py-1.5 text-xs rounded-lg transition flex items-center justify-between ${
                          user?.role === 'agronomist' ? 'bg-indigo-50 text-indigo-800 font-semibold' : 'text-slate-600 hover:bg-slate-50'
                        }`}
                      >
                        <span>Agronomist (Dr. Hitesh)</span>
                        {user?.role === 'agronomist' && <span className="text-[10px] bg-indigo-200 text-indigo-800 px-1.5 py-0.2 rounded font-bold">Active</span>}
                      </button>
                    </div>

                    <button
                      onClick={() => {
                        logout();
                        setUserMenuOpen(false);
                      }}
                      className="w-full text-left px-3 py-1.5 text-xs font-medium text-rose-600 hover:bg-rose-50 rounded-lg flex items-center gap-2 transition"
                    >
                      <LogOut className="w-3.5 h-3.5" />
                      <span>{t.nav.logout}</span>
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center gap-1.5">
                <button
                  onClick={() => quickLogin('farmer')}
                  className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-emerald-700 text-white hover:bg-emerald-800 transition shadow-xs"
                >
                  Farmer Demo
                </button>
                <button
                  onClick={() => quickLogin('agronomist')}
                  className="hidden sm:inline-flex px-3 py-1.5 text-xs font-semibold rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-50 transition"
                >
                  Expert Demo
                </button>
              </div>
            )}

            {/* Mobile menu toggle */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden p-2 rounded-lg text-slate-600 hover:bg-slate-100 transition"
              aria-label="Toggle Navigation Menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="lg:hidden border-t border-slate-200 bg-white px-4 pt-2 pb-5 space-y-1 shadow-md">
          {navItems.map((item) => {
            if (item.agronomistOnly && user?.role !== 'agronomist') return null;
            const Icon = item.icon;
            const isActive = currentPage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleNav(item.id)}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition ${
                  isActive
                    ? 'bg-emerald-50 text-emerald-800 font-semibold border border-emerald-200/70'
                    : item.isScanAction
                    ? 'bg-emerald-700 text-white font-semibold'
                    : 'text-slate-700 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-700' : item.isScanAction ? 'text-white' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </div>
                {item.badge !== undefined && (
                  <span className="px-2 py-0.5 text-[10px] font-bold rounded-full bg-rose-500 text-white">
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
    </header>
  );
};
