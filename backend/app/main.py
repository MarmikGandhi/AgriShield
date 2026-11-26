"""
main.py
=======
FastAPI Main Application Entrypoint for the AI Crop Stress Detection and Farmer Alert System.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.database.session import init_db, SessionLocal
from app.database.models import User, FarmerProfile, AgronomistProfile
from app.core.security import get_password_hash

# Routers
from app.api.v1.auth import router as auth_router
from app.api.v1.scans import router as scans_router
from app.api.v1.dashboard import router as dashboard_router
from app.api.v1.notifications import router as notifications_router
from app.api.v1.weather import router as weather_router
from app.api.v1.market import router as market_router
from app.api.v1.chat import router as chat_router
from app.api.v1.expert import router as expert_router
from app.api.v1.agronomist import router as agronomist_router
from app.api.v1.datasets import router as datasets_router
from app.api.v1.ai import router as ai_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Production-grade AI crop disease diagnosis, segmentation, explainable Grad-CAM, and farmer alert platform."
)

# CORS - support all local & deployed domains with credentials
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Upload mount points
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")

# Include API Routers (both /api/v1 and /api for compatibility)
for prefix in [settings.API_V1_STR, "/api"]:
    app.include_router(auth_router, prefix=prefix)
    app.include_router(scans_router, prefix=prefix)
    app.include_router(dashboard_router, prefix=prefix)
    app.include_router(notifications_router, prefix=prefix)
    app.include_router(weather_router, prefix=prefix)
    app.include_router(market_router, prefix=prefix)
    app.include_router(chat_router, prefix=prefix)
    app.include_router(expert_router, prefix=prefix)
    app.include_router(agronomist_router, prefix=prefix)
    app.include_router(datasets_router, prefix=prefix)
    app.include_router(ai_router, prefix=prefix)

@app.on_event("startup")
def on_startup():
    init_db()
    seed_initial_accounts()

def seed_initial_accounts():
    """Seeds default demo farmer and agronomist accounts if they don't exist."""
    db = SessionLocal()
    try:
        # Seed Farmer
        farmer = db.query(User).filter(User.email == "farmer@agri.in").first()
        if not farmer:
            farmer = User(
                email="farmer@agri.in",
                hashed_password=get_password_hash("farmer123"),
                full_name="Rameshbhai Patel",
                phone_number="+919876543210",
                role="farmer",
                language_preference="gu"
            )
            db.add(farmer)
            db.commit()
            db.refresh(farmer)
            f_profile = FarmerProfile(
                user_id=farmer.id,
                village="Borsad",
                taluka="Anand",
                district="Anand",
                state="Gujarat",
                farm_size_acres=3.5,
                primary_crops="Tomato, Potato, Tobacco, Cotton"
            )
            db.add(f_profile)
            db.commit()

        # Seed Agronomist
        agronomist = db.query(User).filter(User.email == "expert@agri.in").first()
        if not agronomist:
            agronomist = User(
                email="expert@agri.in",
                hashed_password=get_password_hash("expert123"),
                full_name="Dr. Hitesh Sharma",
                phone_number="+919825012345",
                role="agronomist",
                language_preference="en"
            )
            db.add(agronomist)
            db.commit()
            db.refresh(agronomist)
            a_profile = AgronomistProfile(
                user_id=agronomist.id,
                institution="Anand Agricultural University (AAU)",
                specialization="Foliar Pathology & Crop Epidemiology",
                license_number="GJ-AAU-PATH-2026-04"
            )
            db.add(a_profile)
            db.commit()

    finally:
        db.close()

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AgriShield AI",
        "database": "connected",
        "version": "1.0.1",
        "build": "deploy_fix_20261126"
    }
