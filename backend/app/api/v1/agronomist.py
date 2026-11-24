"""
agronomist.py
=============
Dedicated portal endpoints for accredited agronomists and plant pathologists.
Includes question triage, farmer scan inspection, and regional disease trends.
"""

from collections import Counter
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database.session import get_db
from app.database.models import User, Scan, ExpertQuestion, ExpertResponse
from app.api.deps import require_agronomist
from app.api.v1.scans import format_scan_response
from app.api.v1.expert import format_question_response

router = APIRouter(prefix="/agronomist", tags=["Agronomist Portal"])

@router.get("/dashboard")
def get_agronomist_dashboard(
    current_agronomist: User = Depends(require_agronomist),
    db: Session = Depends(get_db)
):
    all_questions = db.query(ExpertQuestion).order_by(desc(ExpertQuestion.created_at)).all()
    all_scans = db.query(Scan).order_by(desc(Scan.created_at)).all()

    pending_q = sum(1 for q in all_questions if q.status == "Pending")
    under_review_q = sum(1 for q in all_questions if q.status == "Under Review")
    answered_q = sum(1 for q in all_questions if q.status == "Answered")

    # Crop and disease breakdown across all farmers
    crop_counts = Counter(s.crop for s in all_scans)
    disease_counts = Counter(s.disease_condition for s in all_scans if not s.is_healthy)
    severity_counts = Counter(s.severity_level for s in all_scans)

    return {
        "agronomist": {
            "name": current_agronomist.full_name,
            "institution": current_agronomist.agronomist_profile.institution if current_agronomist.agronomist_profile else "Agricultural University",
            "specialization": current_agronomist.agronomist_profile.specialization if current_agronomist.agronomist_profile else "Pathology"
        },
        "stats": {
            "pending_queries": pending_q,
            "under_review_queries": under_review_q,
            "answered_queries": answered_q,
            "total_farmer_scans": len(all_scans),
            "critical_outbreaks": severity_counts.get("Critical", 0) + severity_counts.get("High", 0)
        },
        "regional_crop_distribution": dict(crop_counts.most_common(6)),
        "prevalent_diseases": dict(disease_counts.most_common(6)),
        "severity_breakdown": dict(severity_counts),
        "recent_queries": [format_question_response(q) for q in all_questions[:5]]
    }

@router.get("/questions")
def get_agronomist_questions(
    status_filter: str = None,
    current_agronomist: User = Depends(require_agronomist),
    db: Session = Depends(get_db)
):
    query = db.query(ExpertQuestion)
    if status_filter:
        query = query.filter(ExpertQuestion.status == status_filter)
    questions = query.order_by(desc(ExpertQuestion.created_at)).all()
    return [format_question_response(q) for q in questions]

@router.patch("/questions/{question_id}")
def update_question_status(
    question_id: int,
    status_value: str,
    current_agronomist: User = Depends(require_agronomist),
    db: Session = Depends(get_db)
):
    question = db.query(ExpertQuestion).filter(ExpertQuestion.id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found.")

    if status_value not in ["Pending", "Under Review", "Answered", "Closed"]:
        raise HTTPException(status_code=400, detail="Invalid status value.")

    question.status = status_value
    db.commit()
    db.refresh(question)
    return format_question_response(question)
