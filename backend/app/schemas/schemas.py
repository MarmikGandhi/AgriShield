"""
schemas.py
==========
Pydantic Request & Response Data Contracts for FastAPI REST APIs.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field

# --- Auth Schemas ---
class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    phone_number: Optional[str] = None
    role: str = "farmer" # 'farmer' | 'agronomist'
    language_preference: str = "en" # 'en' | 'hi' | 'gu'

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)
    # Optional farmer details
    village: Optional[str] = "Anand"
    taluka: Optional[str] = "Anand"
    district: Optional[str] = "Anand"
    farm_size_acres: Optional[float] = 2.5
    # Optional agronomist details
    institution: Optional[str] = None
    specialization: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class TokenData(BaseModel):
    sub: Optional[str] = None
    role: Optional[str] = None

# --- AI Diagnostics & Scan Schemas ---
class QualityMetrics(BaseModel):
    height: Optional[int] = None
    width: Optional[int] = None
    laplacian_variance: Optional[float] = None
    mean_brightness: Optional[float] = None
    contrast_std: Optional[float] = None
    foliage_ratio: Optional[float] = None
    foliage_percentage: Optional[float] = None

class QualityGuardResult(BaseModel):
    is_valid: bool
    status: str
    reason: str
    message: str
    metrics: Dict[str, Any] = {}

class PredictionItem(BaseModel):
    class_id: int
    crop: str
    crop_hi: Optional[str] = ""
    crop_gu: Optional[str] = ""
    condition: str
    condition_hi: Optional[str] = ""
    condition_gu: Optional[str] = ""
    is_healthy: bool
    confidence: float
    probability: float
    causal_agent: Optional[str] = None
    default_severity: Optional[str] = "Medium"

ScanPrediction = PredictionItem

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    area_pixels: int

class SeverityResult(BaseModel):
    affected_percentage: float
    severity_level: str # 'None' | 'Low' | 'Medium' | 'High' | 'Critical'
    diseased_pixels: int
    leaf_pixels: int
    lesion_count: int
    bounding_boxes: List[BoundingBox] = []
    method: str
    is_canopy_relative: bool
    recommendation: str
    disclaimer: str

class ScanResponse(BaseModel):
    id: int
    user_id: int
    original_filename: str
    image_url: str
    crop: str
    crop_hi: Optional[str] = ""
    crop_gu: Optional[str] = ""
    disease_condition: str
    disease_condition_hi: Optional[str] = ""
    disease_condition_gu: Optional[str] = ""
    confidence: float
    is_healthy: bool
    causal_agent: Optional[str] = None
    severity_level: str
    affected_percentage: float
    lesion_count: int
    bounding_boxes: Optional[List[Dict[str, Any]]] = []
    gradcam_url: Optional[str] = None
    segmentation_url: Optional[str] = None
    management_advice: Optional[str] = None
    prevention_advice: Optional[str] = None
    quality_status: str
    quality_metrics: Optional[Dict[str, Any]] = None
    severity_reliability: Optional[str] = "RELIABLE"
    is_approximate: Optional[bool] = False
    discrepancy_detected: Optional[bool] = False
    is_uncertain_classification: Optional[bool] = False
    raw_affected_percentage: Optional[float] = None
    noise_filtered_percentage: Optional[float] = None
    explainability_source: Optional[str] = "EfficientNet-B0 (features[-1])"
    alternatives: List[PredictionItem] = []
    created_at: datetime

    class Config:
        from_attributes = True

# --- Dashboard & Trends Schemas ---
class DashboardSummary(BaseModel):
    total_scans: int
    healthy_scans: int
    diseased_scans: int
    critical_scans: int
    recent_scans: List[ScanResponse] = []
    active_alerts_count: int
    weather_summary: Dict[str, Any]
    disease_distribution: Dict[str, int] = {}
    crop_distribution: Dict[str, int] = {}

# --- Farmer Notifications Schemas ---
class NotificationResponse(BaseModel):
    id: int
    title: str
    title_hi: Optional[str] = None
    title_gu: Optional[str] = None
    message: str
    message_hi: Optional[str] = None
    message_gu: Optional[str] = None
    alert_type: str
    severity: str
    is_read: bool
    channel: str
    delivery_status: str
    created_at: datetime

    class Config:
        from_attributes = True

# --- Expert / Agronomist Portal Schemas ---
class ExpertResponseCreate(BaseModel):
    response_text: str
    recommended_treatment: Optional[str] = None
    preventive_measures: Optional[str] = None

class ExpertResponseView(BaseModel):
    id: int
    agronomist_id: int
    agronomist_name: str
    response_text: str
    recommended_treatment: Optional[str] = None
    preventive_measures: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class ExpertQuestionCreate(BaseModel):
    crop: str
    title: str
    description: str
    symptoms: Optional[str] = None
    scan_id: Optional[int] = None

class ExpertQuestionResponse(BaseModel):
    id: int
    farmer_id: int
    farmer_name: str
    crop: str
    title: str
    description: str
    symptoms: Optional[str] = None
    status: str
    attached_image_url: Optional[str] = None
    scan_id: Optional[int] = None
    responses: List[ExpertResponseView] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# --- Weather & Risk Schemas ---
class WeatherCurrentResponse(BaseModel):
    location: str
    temperature_c: float
    humidity_pct: float
    rainfall_mm: float
    wind_speed_kmh: float
    condition: str
    is_live: bool
    data_source: Optional[str] = "STATIC_AGROCLIMATIC_FALLBACK"
    provider_notes: Optional[str] = None
    timestamp: datetime

class DiseaseRiskItem(BaseModel):
    crop: str
    disease: str
    risk_level: str # 'Low' | 'Moderate' | 'High'
    conducive_factors: List[str]
    management_advice: str

class WeatherRiskResponse(BaseModel):
    location: str
    current_weather: WeatherCurrentResponse
    overall_risk_level: str
    disease_risks: List[DiseaseRiskItem]
    model_type: str = "Rule-based Agronomic Weather Risk Model"
    disclaimer: str

# --- Mandi / Market Price Schemas ---
class MandiPriceItem(BaseModel):
    commodity: str
    market: str
    district: str
    state: str
    variety: str
    min_price: float
    max_price: float
    modal_price: float
    price_date: str
    unit: str = "₹ / Quintal"
    is_live: bool
    data_source: Optional[str] = "APMC_BENCHMARK_CACHE"
    provenance: Optional[str] = None

# --- Kisan Bot Chat Schemas ---
class ChatMessage(BaseModel):
    role: str # 'user' | 'assistant'
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []
    crop: Optional[str] = None
    language: str = "en" # 'en' | 'hi' | 'gu'

class ChatResponse(BaseModel):
    reply: str
    sources: List[str] = []
    recommended_actions: List[str] = []
    language: str
