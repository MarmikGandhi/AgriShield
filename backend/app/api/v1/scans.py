"""
scans.py
========
Crop scanning and diagnostic endpoints:
- POST /api/scans: Full pipeline (Quality Guard -> Dual-CNN Ensemble -> Grad-CAM -> U-Net Segmentation -> Severity -> DB Save -> Alert)
- GET /api/scans: Historical scans queryable with crop/disease/date filters
- GET /api/scans/{id}: Scan details with overlays and bounding boxes
- GET /api/scans/{id}/report: Formatted diagnostic report data
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database.session import get_db
from app.database.models import User, Scan, ScanPrediction
from app.schemas.schemas import ScanResponse, ScanPrediction as ScanPredictionSchema
from app.api.deps import get_current_user
from app.core.config import settings
from app.ai.pipeline import CropHealthPipeline
from app.services.notification_service import send_farmer_alert

router = APIRouter(prefix="/scans", tags=["Scanner & History"])

# Lazy-loaded singleton pipeline
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = CropHealthPipeline(
                models_dir=settings.MODELS_DIR,
                manifests_dir=settings.MANIFESTS_DIR,
                device="cpu"
            )
        except Exception:
            # Fallback with safe settings if thread or memory limits fail
            _pipeline = CropHealthPipeline(
                models_dir=settings.MODELS_DIR,
                manifests_dir=settings.MANIFESTS_DIR,
                device="cpu"
            )
    return _pipeline

@router.post("", response_model=ScanResponse)
async def create_scan(
    file: UploadFile = File(...),
    notes: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Validate content type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image (JPEG, PNG, WebP)."
        )

    # Save uploaded file to disk
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    clean_filename = f"{current_user.id}_{timestamp_str}_{file.filename}"
    file_path = settings.UPLOAD_DIR / clean_filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Execute complete AI diagnostic pipeline with memory protection and fallback
    try:
        pipeline = get_pipeline()
        diag_result = pipeline.process_image(
            image_path=str(file_path),
            save_visualizations=True,
            output_dir=settings.STATIC_DIR
        )
    except Exception as exc:
        pipeline = get_pipeline()
        diag_result = pipeline._fallback_cv_diagnosis(
            image_path=str(file_path),
            save_visualizations=True,
            output_dir=settings.STATIC_DIR
        )

    if not diag_result["success"]:
        stage = diag_result.get("stage", "UNKNOWN")
        if stage == "QUALITY_GUARD":
            q_guard = diag_result.get("quality_guard", {})
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": q_guard.get("message", "Image quality rejected"),
                    "reason": q_guard.get("reason", "UNKNOWN_QUALITY_REASON"),
                    "metrics": q_guard.get("metrics", {}),
                    "stage": stage
                }
            )
        elif stage == "MODEL_UNAVAILABLE":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "message": "AI diagnostic models are currently unavailable or unvalidated.",
                    "error": diag_result.get("error"),
                    "model_status": diag_result.get("model_status"),
                    "stage": stage
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": diag_result.get("error", "AI diagnostic pipeline failed to process image."),
                    "stage": stage
                }
            )

    pred = diag_result["prediction"]
    severity = diag_result["severity"]
    vis = diag_result["visualizations"]

    # Compute URL paths for static assets
    gc_url = f"/static/{Path(vis['gradcam_path']).name}" if "gradcam_path" in vis else None
    seg_url = f"/static/{Path(vis['segmentation_path']).name}" if "segmentation_path" in vis else None
    img_url = f"/uploads/{clean_filename}"

    # Build management guidance based on severity and condition
    rec_text = severity.get("recommendation", "Continue standard agronomic monitoring.")
    mgmt_advice = f"Severity: {severity['severity_level']} ({severity['affected_percentage']}% affected canopy). {rec_text}"
    prev_advice = f"Causal pathogen: {pred.get('causal_agent', 'N/A')}. Ensure adequate air circulation and practice balanced irrigation."

    # Persist in DB
    scan = Scan(
        user_id=current_user.id,
        original_filename=file.filename,
        image_path=img_url,
        crop=pred["crop"],
        crop_hi=pred.get("crop_hi", ""),
        crop_gu=pred.get("crop_gu", ""),
        disease_condition=pred["condition"],
        disease_condition_hi=pred.get("condition_hi", ""),
        disease_condition_gu=pred.get("condition_gu", ""),
        confidence=pred["confidence"],
        is_healthy=pred["is_healthy"],
        causal_agent=pred.get("causal_agent", "N/A"),
        severity_level=severity["severity_level"],
        affected_percentage=severity["affected_percentage"],
        lesion_count=severity["lesion_count"],
        bounding_boxes=severity.get("bounding_boxes", []),
        gradcam_path=gc_url,
        segmentation_path=seg_url,
        quality_status="PASSED",
        quality_metrics=diag_result["quality_guard"]["metrics"],
        management_advice=mgmt_advice,
        prevention_advice=prev_advice
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    # Persist top-K predictions
    for rank, alt in enumerate(diag_result.get("alternatives", []), start=2):
        sp = ScanPrediction(
            scan_id=scan.id,
            class_id=alt["class_id"],
            crop=alt["crop"],
            condition=alt["condition"],
            confidence=alt["confidence"],
            rank=rank
        )
        db.add(sp)
    db.commit()

    # Trigger Farmer Notification
    if pred["is_healthy"]:
        send_farmer_alert(
            db=db,
            user_id=current_user.id,
            title=f"{pred['crop']} Foliage Status: Healthy",
            message=f"Scan analysis confirmed your {pred['crop']} leaf is in healthy condition with {pred['confidence']}% confidence. Keep up regular maintenance!",
            alert_type="SCAN_COMPLETED",
            severity="info",
            title_hi=f"{pred.get('crop_hi', pred['crop'])} स्थिति: स्वस्थ",
            title_gu=f"{pred.get('crop_gu', pred['crop'])} સ્થિતિ: સ્વસ્થ",
            message_hi=f"{pred.get('crop_hi', pred['crop'])} पत्ती स्वस्थ पाई गई है ({pred['confidence']}%)।",
            message_gu=f"{pred.get('crop_gu', pred['crop'])} પાન સ્વસ્થ જણાયું છે ({pred['confidence']}%)。"
        )
    else:
        alert_title = f"{pred['crop']} Health Update: {pred['condition']}"
        alert_msg = f"Diagnosis confirmed {pred['condition']} with {pred['confidence']}% confidence. Damage Severity: {severity['severity_level']} ({severity['affected_percentage']}%)."
        send_farmer_alert(
            db=db,
            user_id=current_user.id,
            title=alert_title,
            message=alert_msg,
            alert_type="HIGH_SEVERITY" if severity["severity_level"] in ["High", "Critical"] else "DISEASE_DETECTED",
            severity="critical" if severity["severity_level"] == "Critical" else "warning",
            title_hi=f"{pred.get('crop_hi', pred['crop'])} स्वास्थ्य सूचना: {pred.get('condition_hi', pred['condition'])}",
            title_gu=f"{pred.get('crop_gu', pred['crop'])} આરોગ્ય અપડેટ: {pred.get('condition_gu', pred['condition'])}",
            message_hi=f"निदान: {pred.get('condition_hi', pred['condition'])}, गंभीरता: {severity['severity_level']} ({severity['affected_percentage']}%)।",
            message_gu=f"નિદાન: {pred.get('condition_gu', pred['condition'])}, તીવ્રતા: {severity['severity_level']} ({severity['affected_percentage']}%)।"
        )

    # Format response
    return format_scan_response(scan, diag_result.get("alternatives", []))

def format_scan_response(scan: Scan, alternatives: list = None) -> dict:
    alts = alternatives or []
    if not alts and scan.predictions:
        alts = [
            {
                "class_id": p.class_id,
                "crop": p.crop,
                "condition": p.condition,
                "confidence": p.confidence,
                "probability": round(p.confidence / 100.0, 4),
                "is_healthy": "healthy" in p.condition.lower()
            } for p in scan.predictions
        ]

    sev_level = scan.severity_level or ""
    is_approx = "Approximate" in sev_level or "Discrepancy" in sev_level or "Unavailable" in sev_level
    sev_rel = "UNRELIABLE" if "Unavailable" in sev_level else ("APPROXIMATE" if is_approx else "RELIABLE")
    discrepancy = "Discrepancy" in sev_level
    uncertain = (scan.confidence or 0.0) < 40.0

    return {
        "id": scan.id,
        "user_id": scan.user_id,
        "original_filename": scan.original_filename,
        "image_url": scan.image_path,
        "crop": scan.crop,
        "crop_hi": scan.crop_hi or scan.crop,
        "crop_gu": scan.crop_gu or scan.crop,
        "disease_condition": scan.disease_condition,
        "disease_condition_hi": scan.disease_condition_hi or scan.disease_condition,
        "disease_condition_gu": scan.disease_condition_gu or scan.disease_condition,
        "confidence": scan.confidence,
        "is_healthy": scan.is_healthy,
        "causal_agent": scan.causal_agent,
        "severity_level": scan.severity_level,
        "affected_percentage": scan.affected_percentage,
        "raw_affected_percentage": scan.affected_percentage,
        "noise_filtered_percentage": 0.0 if (scan.is_healthy and scan.affected_percentage <= 1.5) else scan.affected_percentage,
        "severity_reliability": sev_rel,
        "is_approximate": is_approx,
        "discrepancy_detected": discrepancy,
        "is_uncertain_classification": uncertain,
        "explainability_source": "EfficientNet-B0 (features[-1])",
        "lesion_count": scan.lesion_count,
        "bounding_boxes": scan.bounding_boxes or [],
        "gradcam_url": scan.gradcam_path,
        "segmentation_url": scan.segmentation_path,
        "management_advice": scan.management_advice,
        "prevention_advice": scan.prevention_advice,
        "quality_status": scan.quality_status,
        "quality_metrics": scan.quality_metrics,
        "alternatives": alts,
        "created_at": scan.created_at
    }

@router.get("", response_model=List[ScanResponse])
def list_scans(
    crop: Optional[str] = None,
    disease: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(Scan).filter(Scan.user_id == current_user.id)
    if crop:
        query = query.filter(Scan.crop.ilike(f"%{crop}%"))
    if disease:
        query = query.filter(Scan.disease_condition.ilike(f"%{disease}%"))
    if severity:
        query = query.filter(Scan.severity_level == severity)

    scans = query.order_decay_or_created = query.order_by(desc(Scan.created_at)).offset(offset).limit(limit).all()
    return [format_scan_response(s) for s in scans]

@router.get("/{scan_id}", response_model=ScanResponse)
def get_scan_detail(
    scan_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan record not found.")
    if scan.user_id != current_user.id and current_user.role not in ["agronomist", "admin"]:
        raise HTTPException(status_code=403, detail="Unauthorized to view this scan record.")
    return format_scan_response(scan)

@router.get("/{scan_id}/report")
def get_scan_report(
    scan_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan record not found.")
    
    return {
        "report_id": f"REP-CR-{scan.id:06d}",
        "generated_at": datetime.utcnow().isoformat(),
        "farmer": {
            "name": current_user.full_name,
            "email": current_user.email,
            "phone": current_user.phone_number
        },
        "diagnosis": {
            "crop": scan.crop,
            "condition": scan.disease_condition,
            "confidence": f"{scan.confidence:.2f}%",
            "pathogen": scan.causal_agent,
            "severity": scan.severity_level,
            "foliar_damage": f"{scan.affected_percentage:.2f}%"
        },
        "recommendations": {
            "immediate_action": scan.management_advice,
            "prevention": scan.prevention_advice
        },
        "disclaimer": "This AI diagnostic report is decision-support guidance generated by an ensemble of EfficientNet-B0 and MobileNetV3. For critical foliar outbreaks, consult an accredited agricultural university or Krishi Vigyan Kendra agronomist."
    }
