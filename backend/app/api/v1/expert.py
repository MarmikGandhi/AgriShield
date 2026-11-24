"""
expert.py
=========
Ask Expert module: Farmer question submission, tracking, and agronomist responses.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database.session import get_db
from app.database.models import User, ExpertQuestion, ExpertResponse, Scan
from app.schemas.schemas import ExpertQuestionResponse, ExpertResponseCreate, ExpertResponseView
from app.api.deps import get_current_user, require_agronomist

router = APIRouter(prefix="/expert", tags=["Ask Expert"])

@router.post("/questions", response_model=ExpertQuestionResponse)
async def submit_question(
    crop: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    symptoms: Optional[str] = Form(None),
    scan_id: Optional[int] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    attached_img = None
    if scan_id:
        scan = db.query(Scan).filter(Scan.id == scan_id).first()
        if scan:
            attached_img = scan.image_path

    question = ExpertQuestion(
        farmer_id=current_user.id,
        crop=crop,
        title=title,
        description=description,
        symptoms=symptoms,
        scan_id=scan_id,
        attached_image_path=attached_img,
        status="Pending"
    )
    db.add(question)
    db.commit()
    db.refresh(question)

    return format_question_response(question)

@router.get("/questions", response_model=List[ExpertQuestionResponse])
def list_questions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role in ["agronomist", "admin"]:
        # Agronomists can see all questions
        questions = db.query(ExpertQuestion).order_by(desc(ExpertQuestion.created_at)).all()
    else:
        # Farmers only see their questions
        questions = db.query(ExpertQuestion).filter(ExpertQuestion.farmer_id == current_user.id).order_by(desc(ExpertQuestion.created_at)).all()

    return [format_question_response(q) for q in questions]

@router.post("/questions/{question_id}/response", response_model=ExpertQuestionResponse)
def respond_to_question(
    question_id: int,
    resp_in: ExpertResponseCreate,
    current_agronomist: User = Depends(require_agronomist),
    db: Session = Depends(get_db)
):
    question = db.query(ExpertQuestion).filter(ExpertQuestion.id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found.")

    response = ExpertResponse(
        question_id=question.id,
        agronomist_id=current_agronomist.id,
        response_text=resp_in.response_text,
        recommended_treatment=resp_in.recommended_treatment,
        preventive_measures=resp_in.preventive_measures
    )
    db.add(response)
    question.status = "Answered"
    db.commit()
    db.refresh(question)

    return format_question_response(question)

def format_question_response(q: ExpertQuestion) -> dict:
    resp_list = []
    for r in q.responses:
        resp_list.append({
            "id": r.id,
            "agronomist_id": r.agronomist_id,
            "agronomist_name": r.agronomist.full_name if r.agronomist else "Agronomist",
            "response_text": r.response_text,
            "recommended_treatment": r.recommended_treatment,
            "preventive_measures": r.preventive_measures,
            "created_at": r.created_at
        })

    return {
        "id": q.id,
        "farmer_id": q.farmer_id,
        "farmer_name": q.farmer.full_name if q.farmer else "Farmer",
        "crop": q.crop,
        "title": q.title,
        "description": q.description,
        "symptoms": q.symptoms,
        "status": q.status,
        "attached_image_url": q.attached_image_path,
        "scan_id": q.scan_id,
        "responses": resp_list,
        "created_at": q.created_at,
        "updated_at": q.updated_at
    }
