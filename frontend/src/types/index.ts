// src/types/index.ts

export type UserRole = 'farmer' | 'agronomist' | 'admin';
export type Language = 'en' | 'hi' | 'gu';

export interface User {
  id: number;
  email: string;
  full_name: string;
  phone_number?: string;
  role: UserRole;
  language_preference: Language;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface QualityMetrics {
  height?: number;
  width?: number;
  laplacian_variance?: number;
  mean_brightness?: number;
  contrast_std?: number;
  foliage_ratio?: number;
  foliage_percentage?: number;
}

export interface QualityGuardResult {
  is_valid: boolean;
  status: string;
  reason: string;
  message: string;
  metrics: QualityMetrics;
}

export interface PredictionItem {
  class_id: number;
  crop: string;
  crop_hi?: string;
  crop_gu?: string;
  condition: string;
  condition_hi?: string;
  condition_gu?: string;
  is_healthy: boolean;
  confidence: number;
  probability: number;
  causal_agent?: string;
  default_severity?: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  area_pixels: number;
}

export interface SeverityResult {
  affected_percentage: number;
  severity_level: 'None' | 'None (Healthy)' | 'Low' | 'Medium' | 'High' | 'Critical';
  diseased_pixels: number;
  leaf_pixels: number;
  lesion_count: number;
  bounding_boxes: BoundingBox[];
  method: string;
  is_canopy_relative: boolean;
  recommendation: string;
  disclaimer: string;
}

export interface ScanRecord {
  id: number;
  user_id: number;
  original_filename: string;
  image_url: string;
  crop: string;
  crop_hi?: string;
  crop_gu?: string;
  disease_condition: string;
  disease_condition_hi?: string;
  disease_condition_gu?: string;
  confidence: number;
  is_healthy: boolean;
  causal_agent?: string;
  severity_level: string;
  affected_percentage: number;
  raw_affected_percentage?: number;
  noise_filtered_percentage?: number;
  severity_reliability?: 'RELIABLE' | 'APPROXIMATE' | 'UNRELIABLE';
  is_approximate?: boolean;
  discrepancy_detected?: boolean;
  is_uncertain_classification?: boolean;
  explainability_source?: string;
  lesion_count: number;
  bounding_boxes?: BoundingBox[];
  gradcam_url?: string;
  segmentation_url?: string;
  management_advice?: string;
  prevention_advice?: string;
  quality_status: string;
  quality_metrics?: QualityMetrics;
  alternatives: PredictionItem[];
  created_at: string;
}

export interface DashboardSummary {
  total_scans: number;
  healthy_scans: number;
  diseased_scans: number;
  critical_scans: number;
  recent_scans: ScanRecord[];
  active_alerts_count: number;
  weather_summary: {
    location: string;
    temperature_c: number;
    humidity_pct: number;
    condition: string;
    overall_risk: string;
  };
  disease_distribution: Record<string, number>;
  crop_distribution: Record<string, number>;
}

export interface NotificationItem {
  id: number;
  title: string;
  title_hi?: string;
  title_gu?: string;
  message: string;
  message_hi?: string;
  message_gu?: string;
  alert_type: string;
  severity: 'info' | 'warning' | 'critical';
  is_read: boolean;
  channel: string;
  delivery_status: string;
  created_at: string;
}

export interface WeatherCurrent {
  location: string;
  temperature_c: number;
  humidity_pct: number;
  rainfall_mm: number;
  wind_speed_kmh: number;
  condition: string;
  is_live: boolean;
  timestamp: string;
}

export interface DiseaseRiskItem {
  crop: string;
  disease: string;
  risk_level: 'Low' | 'Moderate' | 'High';
  conducive_factors: string[];
  management_advice: string;
}

export interface WeatherRiskData {
  location: string;
  current_weather: WeatherCurrent;
  overall_risk_level: 'Low' | 'Moderate' | 'High';
  disease_risks: DiseaseRiskItem[];
  model_type: string;
  disclaimer: string;
}

export interface MandiPriceItem {
  commodity: string;
  market: string;
  district: string;
  state: string;
  variety: string;
  min_price: number;
  max_price: number;
  modal_price: number;
  price_date: string;
  unit: string;
  is_live: boolean;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
  recommended_actions?: string[];
}

export interface ExpertResponseView {
  id: number;
  agronomist_id: number;
  agronomist_name: string;
  response_text: string;
  recommended_treatment?: string;
  preventive_measures?: string;
  created_at: string;
}

export interface ExpertQuestionItem {
  id: number;
  farmer_id: number;
  farmer_name: string;
  crop: string;
  title: string;
  description: string;
  symptoms?: string;
  status: 'pending' | 'answered';
  attached_image_url?: string;
  scan_id?: number;
  responses: ExpertResponseView[];
  created_at: string;
  updated_at: string;
}
