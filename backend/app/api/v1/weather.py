"""
weather.py
==========
Weather integration and rule-based foliar disease outbreak risk estimation endpoints.
"""

from fastapi import APIRouter, Depends, Query
from app.schemas.schemas import WeatherCurrentResponse, WeatherRiskResponse
from app.services.weather_service import get_current_weather, evaluate_disease_risks

router = APIRouter(prefix="/weather", tags=["Weather & Outbreak Risk"])

@router.get("/current", response_model=WeatherCurrentResponse)
def get_weather_now(location: str = Query("Anand, Gujarat")):
    weather = get_current_weather(location=location)
    return weather

@router.get("/risk", response_model=WeatherRiskResponse)
def get_disease_risk(location: str = Query("Anand, Gujarat")):
    weather = get_current_weather(location=location)
    risk_evaluation = evaluate_disease_risks(weather)
    return risk_evaluation
