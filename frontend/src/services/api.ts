// src/services/api.ts

import axios from 'axios';
import {
  AuthResponse,
  DashboardSummary,
  ScanRecord,
  NotificationItem,
  WeatherRiskData,
  MandiPriceItem,
  ChatMessage,
  ExpertQuestionItem,
} from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL ? (import.meta.env.VITE_API_BASE_URL as string).replace(/\/$/, '') : '';

export const getAssetUrl = (url?: string | null): string => {
  if (!url) return '';
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  return `${API_BASE}${url.startsWith('/') ? '' : '/'}${url}`;
};

const api = axios.create({
  baseURL: `${API_BASE}/api/v1`,
});

// Request interceptor to attach JWT token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('agrishield_token');
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      // Optional auto-logout
    }
    return Promise.reject(error);
  }
);

export const authApi = {
  login: async (credentials: { email: string; password: string }): Promise<AuthResponse> => {
    const res = await api.post('/auth/login', credentials);
    return res.data;
  },
  register: async (data: any): Promise<AuthResponse> => {
    const res = await api.post('/auth/register', data);
    return res.data;
  },
  me: async () => {
    const res = await api.get('/auth/me');
    return res.data;
  },
};

export const scanApi = {
  uploadScan: async (formData: FormData): Promise<ScanRecord> => {
    const res = await api.post('/scans', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },
  listScans: async (params?: { crop?: string; disease?: string; severity?: string; limit?: number }): Promise<ScanRecord[]> => {
    const res = await api.get('/scans', { params });
    return res.data;
  },
  getScanDetail: async (id: number): Promise<ScanRecord> => {
    const res = await api.get(`/scans/${id}`);
    return res.data;
  },
  getScanReport: async (id: number) => {
    const res = await api.get(`/scans/${id}/report`);
    return res.data;
  },
};

export const dashboardApi = {
  getSummary: async (): Promise<DashboardSummary> => {
    const res = await api.get('/dashboard/summary');
    return res.data;
  },
};

export const notificationApi = {
  list: async (): Promise<NotificationItem[]> => {
    const res = await api.get('/notifications');
    return res.data;
  },
  markRead: async (id: number) => {
    const res = await api.put(`/notifications/${id}/read`);
    return res.data;
  },
};

export const weatherApi = {
  getRisk: async (location: string = 'Anand'): Promise<WeatherRiskData> => {
    const res = await api.get('/weather/risk', { params: { location } });
    return res.data;
  },
};

export const marketApi = {
  getPrices: async (district: string = 'Anand'): Promise<MandiPriceItem[]> => {
    const res = await api.get('/market/prices', { params: { district } });
    return res.data;
  },
};

export const chatApi = {
  sendMessage: async (data: { message: string; conversation_history?: ChatMessage[]; crop?: string; language?: string }) => {
    const res = await api.post('/chat', data);
    return res.data;
  },
};

export const expertApi = {
  listQuestions: async (): Promise<ExpertQuestionItem[]> => {
    const res = await api.get('/expert/questions');
    return res.data;
  },
  createQuestion: async (formData: FormData): Promise<ExpertQuestionItem> => {
    const res = await api.post('/expert/questions', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },
  getQuestion: async (id: number): Promise<ExpertQuestionItem> => {
    const res = await api.get(`/expert/questions/${id}`);
    return res.data;
  },
  answerQuestion: async (id: number, data: { response_text: string; recommended_treatment?: string; preventive_measures?: string }) => {
    const res = await api.post(`/expert/questions/${id}/respond`, data);
    return res.data;
  },
};

export const agronomistApi = {
  getPendingCases: async (): Promise<ExpertQuestionItem[]> => {
    const res = await api.get('/agronomist/cases/pending');
    return res.data;
  },
  submitAdvisory: async (id: number, data: { response_text: string; recommended_treatment?: string; preventive_measures?: string }) => {
    const res = await api.post(`/agronomist/cases/${id}/advisory`, data);
    return res.data;
  },
};

export const systemApi = {
  getDatasetsStatus: async () => {
    const res = await api.get('/datasets/status');
    return res.data;
  },
  getAiStatus: async () => {
    const res = await api.get('/ai/status');
    return res.data;
  },
};
