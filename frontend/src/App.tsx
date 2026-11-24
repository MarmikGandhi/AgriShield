// src/App.tsx

import React, { useState, useEffect } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { LanguageProvider, useLanguage } from './context/LanguageContext';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';

// Pages
import { Dashboard } from './pages/Dashboard';
import { CropScanner } from './pages/CropScanner';
import { ScanHistory } from './pages/ScanHistory';
import { AlertsFeed } from './pages/AlertsFeed';
import { KisanBot } from './pages/KisanBot';
import { AskExpert } from './pages/AskExpert';
import { AgronomistPortal } from './pages/AgronomistPortal';
import { DatasetsAndModels } from './pages/DatasetsAndModels';
import { Login } from './pages/Login';

import { ScanRecord } from './types';
import { notificationApi } from './services/api';

const AppContent: React.FC = () => {
  const { isAuthenticated } = useAuth();
  const [currentPage, setCurrentPage] = useState<string>('dashboard');
  const [selectedScan, setSelectedScan] = useState<ScanRecord | null>(null);
  const [unreadAlertsCount, setUnreadAlertsCount] = useState<number>(0);

  const fetchUnreadAlerts = async () => {
    try {
      const notifs = await notificationApi.list();
      const unread = notifs.filter((n) => !n.is_read).length;
      setUnreadAlertsCount(unread);
    } catch (e) {
      // offline / not logged in
    }
  };

  useEffect(() => {
    fetchUnreadAlerts();
    const interval = setInterval(fetchUnreadAlerts, 20000);
    return () => clearInterval(interval);
  }, [isAuthenticated]);

  const handleNavigate = (page: string) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleSelectScan = (scan: ScanRecord) => {
    setSelectedScan(scan);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      
      {/* Top Navbar */}
      <Navbar
        currentPage={currentPage}
        onNavigate={handleNavigate}
        unreadCount={unreadAlertsCount}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {currentPage === 'dashboard' && (
          <Dashboard
            onNavigate={handleNavigate}
            onSelectScan={handleSelectScan}
          />
        )}

        {currentPage === 'scanner' && (
          <CropScanner
            initialScan={selectedScan}
            onNavigate={handleNavigate}
          />
        )}

        {currentPage === 'history' && (
          <ScanHistory
            onSelectScan={handleSelectScan}
            onNavigate={handleNavigate}
          />
        )}

        {currentPage === 'alerts' && (
          <AlertsFeed />
        )}

        {currentPage === 'kisanBot' && (
          <KisanBot />
        )}

        {currentPage === 'askExpert' && (
          <AskExpert />
        )}

        {currentPage === 'agronomist' && (
          <AgronomistPortal />
        )}

        {currentPage === 'datasets' && (
          <DatasetsAndModels />
        )}

        {currentPage === 'login' && (
          <Login onSuccess={() => handleNavigate('dashboard')} />
        )}
      </main>

      {/* Footer */}
      <Footer />

    </div>
  );
};

export function App() {
  return (
    <LanguageProvider>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </LanguageProvider>
  );
}

export default App;
