"""
dashboard.py
============
Farmer dashboard metrics and historical crop health trend aggregation.
"""

from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database.session import get_db
from app.database.models import User, Scan, Notification
from app.schemas.schemas import DashboardSummary
from app.api.deps import get_current_user
from app.services.weather_service import get_current_weather
from app.api.v1.scans import format_scan_response

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/summary", response_model=DashboardSummary)
def get_dashboard_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    scans = db.query(Scan).filter(Scan.user_id == current_user.id).order_by(desc(Scan.created_at)).all()

    total_scans = len(scans)
    healthy_scans = sum(1 for s in scans if s.is_healthy)
    diseased_scans = total_scans - healthy_scans
    critical_scans = sum(1 for s in scans if s.severity_level in ["High", "Critical"])

    # Distributions
    crop_dist = Counter(s.crop for s in scans)
    disease_dist = Counter(s.disease_condition for s in scans if not s.is_healthy)

    # Active unread alerts
    unread_alerts = db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.is_read == False
    ).count()

    # Weather summary
    location = "Anand, Gujarat"
    if current_user.farmer_profile and current_user.farmer_profile.district:
        location = f"{current_user.farmer_profile.district}, Gujarat"
    weather = get_current_weather(location=location)

    recent_scans = [format_scan_response(s) for s in scans[:5]]

    return {
        "total_scans": total_scans,
        "healthy_scans": healthy_scans,
        "diseased_scans": diseased_scans,
        "critical_scans": critical_scans,
        "recent_scans": recent_scans,
        "active_alerts_count": unread_alerts,
        "weather_summary": weather,
        "disease_distribution": dict(disease_dist.most_common(8)),
        "crop_distribution": dict(crop_dist.most_common(8))
    }

@router.get("/trends")
def get_dashboard_trends(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Returns 7-day disease progression trend aggregation."""
    today = datetime.utcnow().date()
    days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]

    timeline = []
    for d in days:
        d_start = datetime.combine(d, datetime.min.time())
        d_end = datetime.combine(d, datetime.max.time())
        day_scans = db.query(Scan).filter(
            Scan.user_id == current_user.id,
            Scan.created_at >= d_start,
            Scan.created_at <= d_end
        ).all()

        timeline.append({
            "date": d.strftime("%b %d"),
            "total": len(day_scans),
            "healthy": sum(1 for s in day_scans if s.is_healthy),
            "diseased": sum(1 for s in day_scans if not s.is_healthy),
            "avg_severity": round(sum(s.affected_percentage for s in day_scans) / len(day_scans), 1) if day_scans else 0.0
        })

    return {"timeline": timeline}
