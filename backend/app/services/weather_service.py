"""
weather_service.py
==================
Integrates weather data (OpenWeather API or localized Gujarat agro-meteorological observations)
and implements rule-based foliar disease outbreak risk models based on temperature, relative humidity,
and rainfall patterns for Indian agriculture.
"""

import os
import requests
from datetime import datetime, timezone
from typing import Dict, Any, List
from app.core.config import settings

# Agronomic risk rules based on ICAR & agricultural university guidelines
DISEASE_RISK_RULES = [
    {
        "crop": "Tomato / Potato",
        "disease": "Late Blight (Phytophthora infestans)",
        "temp_min": 15.0,
        "temp_max": 24.0,
        "humidity_min": 85.0,
        "conducive_text": "High relative humidity (>85%) and moderate cool temperatures (15-24°C) create optimal conditions for rapid sporangial germination and foliar spread.",
        "management": "Apply preventive copper-based fungicides (Copper Oxychloride 50 WP @ 2.5g/L) or Mancozeb 75 WP before rain events."
    },
    {
        "crop": "Tomato / Potato",
        "disease": "Early Blight (Alternaria solani)",
        "temp_min": 24.0,
        "temp_max": 32.0,
        "humidity_min": 75.0,
        "conducive_text": "Warm temperatures (24-32°C) alternating with wet and humid canopy conditions promote concentric target-spot lesion expansion.",
        "management": "Avoid overhead sprinkler irrigation. Apply Chlorothalonil or Azoxystrobin on lower foliage."
    },
    {
        "crop": "Cucurbits / Cherry / Grapes",
        "disease": "Powdery Mildew (Erysiphaceae)",
        "temp_min": 20.0,
        "temp_max": 30.0,
        "humidity_min": 50.0,
        "conducive_text": "Warm, dry afternoons followed by humid, cool nights favor fungal conidia dispersion without requiring free water.",
        "management": "Spray Wettable Sulphur 80 WP @ 3g/L or Hexaconazole 5% EC @ 1ml/L at early white dusting stage."
    },
    {
        "crop": "Bell Pepper / Tomato",
        "disease": "Bacterial Spot (Xanthomonas spp.)",
        "temp_min": 25.0,
        "temp_max": 35.0,
        "humidity_min": 80.0,
        "conducive_text": "High temperature and humidity combined with wind-driven rain or mechanical splashing spread bacterial exudates rapidly.",
        "management": "Spray Streptocycline (1g per 10L water) mixed with Copper Oxychloride (25g per 10L). Avoid working in fields when foliage is wet."
    },
    {
        "crop": "Corn (Maize) / Apple",
        "disease": "Rust (Puccinia sorghi / Gymnosporangium)",
        "temp_min": 16.0,
        "temp_max": 26.0,
        "humidity_min": 80.0,
        "conducive_text": "Moderate temperatures and persistent dew or leaf wetness exceeding 6 hours trigger urediniospore germination.",
        "management": "Ensure adequate field spacing for sunlight penetration. Apply Propiconazole 25 EC @ 1ml/L if pustules are observed."
    }
]

def get_current_weather(location: str = "Anand, Gujarat") -> Dict[str, Any]:
    """
    Fetches real weather from OpenWeather if API key is configured,
    else returns calibrated local meteorological observation with explicit provenance.
    """
    api_key = settings.OPENWEATHER_API_KEY
    if api_key and api_key.strip():
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&units=metric&appid={api_key}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "location": f"{data.get('name', location)}, Gujarat",
                    "temperature_c": round(data["main"]["temp"], 1),
                    "humidity_pct": round(data["main"]["humidity"], 1),
                    "rainfall_mm": round(data.get("rain", {}).get("1h", 0.0), 1),
                    "wind_speed_kmh": round(data["wind"]["speed"] * 3.6, 1),
                    "condition": data["weather"][0]["description"].title(),
                    "is_live": True,
                    "data_source": "LIVE_OPENWEATHER_API",
                    "provider_notes": "Real-time telemetry streamed via OpenWeatherMap API v2.5",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            else:
                print(f"[!] Weather API returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[!] Weather API error: {e}")

    # Explicitly labeled fallback to realistic agro-climatic baseline for central Gujarat (Charotar region) in November
    return {
        "location": location,
        "temperature_c": 28.5,
        "humidity_pct": 62.0,
        "rainfall_mm": 0.0,
        "wind_speed_kmh": 11.2,
        "condition": "Dry & Sunny (Post-Monsoon Kharif/Rabi)",
        "is_live": False,
        "data_source": "STATIC_AGROCLIMATIC_FALLBACK",
        "provider_notes": "Calibrated agro-climatic baseline for Anand, Gujarat (Charotar Rabi Season). Configure OPENWEATHER_API_KEY in .env for live telemetry.",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

def evaluate_disease_risks(weather: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates crop disease risks against weather parameters using rule-based agronomic logic."""
    temp = weather["temperature_c"]
    humidity = weather["humidity_pct"]

    active_risks = []
    max_risk_level = "Low"

    for rule in DISEASE_RISK_RULES:
        factors = []
        is_temp_match = rule["temp_min"] <= temp <= rule["temp_max"]
        is_hum_match = humidity >= rule["humidity_min"]

        if is_temp_match and is_hum_match:
            risk = "High"
            factors.append(f"Temperature ({temp}°C) is in optimal disease range ({rule['temp_min']}-{rule['temp_max']}°C)")
            factors.append(f"Elevated relative humidity ({humidity}%) surpasses pathogen germination threshold (≥{rule['humidity_min']}%)")
            max_risk_level = "High"
        elif is_temp_match or (humidity >= rule["humidity_min"] - 10):
            risk = "Moderate"
            if is_temp_match:
                factors.append(f"Favorable temperature ({temp}°C)")
            if humidity >= rule["humidity_min"] - 10:
                factors.append(f"Sub-optimal but increasing moisture ({humidity}%)")
            if max_risk_level != "High":
                max_risk_level = "Moderate"
        else:
            risk = "Low"
            factors.append(f"Current weather ({temp}°C, {humidity}% RH) inhibits severe foliar outbreak")

        active_risks.append({
            "crop": rule["crop"],
            "disease": rule["disease"],
            "risk_level": risk,
            "conducive_factors": factors,
            "management_advice": rule["management"]
        })

    return {
        "location": weather["location"],
        "current_weather": weather,
        "overall_risk_level": max_risk_level,
        "disease_risks": active_risks,
        "model_type": "Rule-based Agronomic Disease Conduciveness Model",
        "disclaimer": "Weather-based risk estimation identifies microclimatic conduciveness and should be validated with physical field scouting."
    }
