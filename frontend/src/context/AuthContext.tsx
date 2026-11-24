// src/context/AuthContext.tsx

import React, { createContext, useContext, useState, useEffect } from 'react';
import { User, UserRole } from '../types';
import axios from 'axios';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isAgronomist: boolean;
  login: (token: string, user: User) => void;
  logout: () => void;
  quickLogin: (role: 'farmer' | 'agronomist') => Promise<void>;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  token: null,
  isAuthenticated: false,
  isAgronomist: false,
  login: () => {},
  logout: () => {},
  quickLogin: async () => {},
  loading: true,
});

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const savedToken = localStorage.getItem('agrishield_token');
    const savedUser = localStorage.getItem('agrishield_user');
    if (savedToken && savedUser) {
      try {
        setToken(savedToken);
        setUser(JSON.parse(savedUser));
      } catch (e) {
        localStorage.removeItem('agrishield_token');
        localStorage.removeItem('agrishield_user');
      }
    }
    setLoading(false);
  }, []);

  const login = (newToken: string, newUser: User) => {
    setToken(newToken);
    setUser(newUser);
    localStorage.setItem('agrishield_token', newToken);
    localStorage.setItem('agrishield_user', JSON.stringify(newUser));
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('agrishield_token');
    localStorage.removeItem('agrishield_user');
  };

  const quickLogin = async (role: 'farmer' | 'agronomist') => {
    const email = role === 'farmer' ? 'farmer@agri.in' : 'expert@agri.in';
    const password = role === 'farmer' ? 'farmer123' : 'expert123';

    try {
      const response = await axios.post('/api/v1/auth/login', { email, password });
      login(response.data.access_token, response.data.user);
    } catch (err) {
      console.error('Quick login error:', err);
      // Fallback demo mock if offline
      const mockUser: User = {
        id: role === 'farmer' ? 1 : 2,
        email,
        full_name: role === 'farmer' ? 'Rameshbhai Patel' : 'Dr. Hitesh Sharma',
        role,
        language_preference: role === 'farmer' ? 'gu' : 'en',
        created_at: new Date().toISOString()
      };
      login('demo_token', mockUser);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user,
        isAgronomist: user?.role === 'agronomist',
        login,
        logout,
        quickLogin,
        loading,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
