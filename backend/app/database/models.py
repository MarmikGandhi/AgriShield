"""
models.py
=========
SQLAlchemy Relational Database Schema for the Crop Health and Farmer Alert System.
Supports PostgreSQL and SQLite fallback with full relational integrity, foreign keys,
and indexing.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    phone_number = Column(String(50), nullable=True)
    role = Column(String(50), default="farmer", nullable=False) # 'farmer' | 'agronomist' | 'admin'
    language_preference = Column(String(10), default="en", nullable=False) # 'en' | 'hi' | 'gu'
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    farmer_profile = relationship("FarmerProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    agronomist_profile = relationship("AgronomistProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    scans = relationship("Scan", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    questions = relationship("ExpertQuestion", back_populates="farmer", foreign_keys="ExpertQuestion.farmer_id")
    responses = relationship("ExpertResponse", back_populates="agronomist")

class FarmerProfile(Base):
    __tablename__ = "farmer_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    village = Column(String(255), default="Anand")
    taluka = Column(String(255), default="Anand")
    district = Column(String(255), default="Anand") # e.g. Anand, Junagadh, Rajkot, Mehsana, Surat
    state = Column(String(100), default="Gujarat")
    farm_size_acres = Column(Float, default=2.5)
    primary_crops = Column(String(255), default="Tomato, Potato, Cotton, Groundnut")
    soil_type = Column(String(100), default="Black Loam / Alluvial")

    user = relationship("User", back_populates="farmer_profile")

class AgronomistProfile(Base):
    __tablename__ = "agronomist_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    institution = Column(String(255), default="Anand Agricultural University (AAU)")
    qualification = Column(String(255), default="M.Sc. / Ph.D. Plant Pathology")
    specialization = Column(String(255), default="Foliar Crop Diseases & Organic Pathology")
    license_number = Column(String(100), default="GJ-AGRO-2026-8812")
    is_verified = Column(Boolean, default=True)

    user = relationship("User", back_populates="agronomist_profile")

class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    
    # Classification results
    crop = Column(String(100), nullable=False)
    crop_hi = Column(String(100), nullable=True)
    crop_gu = Column(String(100), nullable=True)
    disease_condition = Column(String(255), nullable=False)
    disease_condition_hi = Column(String(255), nullable=True)
    disease_condition_gu = Column(String(255), nullable=True)
    confidence = Column(Float, nullable=False)
    is_healthy = Column(Boolean, default=False)
    causal_agent = Column(String(255), nullable=True)

    # Severity & Segmentation results
    severity_level = Column(String(50), default="Low") # 'None' | 'Low' | 'Medium' | 'High' | 'Critical'
    affected_percentage = Column(Float, default=0.0)
    lesion_count = Column(Integer, default=0)
    bounding_boxes = Column(JSON, nullable=True)
    gradcam_path = Column(String(500), nullable=True)
    segmentation_path = Column(String(500), nullable=True)

    # Quality Guard status
    quality_status = Column(String(50), default="PASSED")
    quality_metrics = Column(JSON, nullable=True)

    # Advisory
    management_advice = Column(Text, nullable=True)
    prevention_advice = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="scans")
    predictions = relationship("ScanPrediction", back_populates="scan", cascade="all, delete-orphan")

class ScanPrediction(Base):
    __tablename__ = "scan_predictions"

    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(Integer, ForeignKey("scans.id"), nullable=False, index=True)
    class_id = Column(Integer, nullable=False)
    crop = Column(String(100), nullable=False)
    condition = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)

    scan = relationship("Scan", back_populates="predictions")

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    title_hi = Column(String(255), nullable=True)
    title_gu = Column(String(255), nullable=True)
    message = Column(Text, nullable=False)
    message_hi = Column(Text, nullable=True)
    message_gu = Column(Text, nullable=True)
    alert_type = Column(String(50), default="DISEASE_DETECTED")
    severity = Column(String(50), default="warning") # 'info' | 'warning' | 'critical'
    is_read = Column(Boolean, default=False)
    channel = Column(String(50), default="in_app") # 'in_app' | 'email' | 'sms'
    delivery_status = Column(String(50), default="delivered") # 'delivered' | 'skipped_no_credentials' | 'failed'
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="notifications")

class ExpertQuestion(Base):
    __tablename__ = "expert_questions"

    id = Column(Integer, primary_key=True, index=True)
    farmer_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    crop = Column(String(100), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    symptoms = Column(Text, nullable=True)
    status = Column(String(50), default="Pending") # 'Pending' | 'Under Review' | 'Answered' | 'Closed'
    attached_image_path = Column(String(500), nullable=True)
    scan_id = Column(Integer, ForeignKey("scans.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    farmer = relationship("User", back_populates="questions", foreign_keys=[farmer_id])
    responses = relationship("ExpertResponse", back_populates="question", cascade="all, delete-orphan")

class ExpertResponse(Base):
    __tablename__ = "expert_responses"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("expert_questions.id"), nullable=False, index=True)
    agronomist_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    response_text = Column(Text, nullable=False)
    recommended_treatment = Column(Text, nullable=True)
    preventive_measures = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    question = relationship("ExpertQuestion", back_populates="responses")
    agronomist = relationship("User", back_populates="responses")

class WeatherRecord(Base):
    __tablename__ = "weather_records"

    id = Column(Integer, primary_key=True, index=True)
    location = Column(String(100), nullable=False, default="Anand, Gujarat")
    temperature_c = Column(Float, nullable=False)
    humidity_pct = Column(Float, nullable=False)
    rainfall_mm = Column(Float, default=0.0)
    wind_speed_kmh = Column(Float, default=12.0)
    condition_text = Column(String(100), default="Clear Sky")
    risk_level = Column(String(50), default="Low") # 'Low' | 'Moderate' | 'High'
    risk_factors = Column(JSON, nullable=True)
    is_live = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

class MarketPrice(Base):
    __tablename__ = "market_prices"

    id = Column(Integer, primary_key=True, index=True)
    commodity = Column(String(100), nullable=False)
    market = Column(String(100), nullable=False)
    district = Column(String(100), nullable=False)
    state = Column(String(100), default="Gujarat")
    variety = Column(String(100), default="Local / Hybrid")
    min_price = Column(Float, nullable=False)
    max_price = Column(Float, nullable=False)
    modal_price = Column(Float, nullable=False)
    price_date = Column(String(50), nullable=False)
    is_live = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    checkpoint_path = Column(String(500), nullable=False)
    accuracy = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
