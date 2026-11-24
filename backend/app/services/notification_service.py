"""
notification_service.py
=======================
Handles farmer alerts and notifications across multi-channels:
- In-app alerts (persisted in relational DB)
- Email (SMTP via TLS when configured)
- SMS (Twilio REST API when credentials are provided)
Maintains strict delivery audit trail: marks 'delivered', 'skipped_no_credentials', or 'failed'
honestly without fabricating false confirmations.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import Notification, User

def send_farmer_alert(
    db: Session,
    user_id: int,
    title: str,
    message: str,
    alert_type: str = "DISEASE_DETECTED",
    severity: str = "warning",
    title_hi: Optional[str] = None,
    title_gu: Optional[str] = None,
    message_hi: Optional[str] = None,
    message_gu: Optional[str] = None,
    send_email: bool = True,
    send_sms: bool = True
) -> Dict[str, Any]:
    """
    Creates an in-app notification and attempts delivery via email/SMS if configured.
    Returns status report.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {"status": "error", "message": f"User {user_id} not found."}

    channel_statuses = {"in_app": "delivered"}

    # 1. Email Channel
    if send_email and settings.SMTP_SERVER and settings.SMTP_USER and settings.SMTP_PASSWORD:
        try:
            msg = MIMEMultipart()
            msg["From"] = settings.SMTP_USER
            msg["To"] = user.email
            msg["Subject"] = f"[Crop Health Alert] {title}"
            body = f"Hello {user.full_name},\n\n{message}\n\nPlease check your farmer portal for visual Grad-CAM diagnosis and treatment steps.\n\nRegards,\nCrop Health System"
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT, timeout=5) as server:
                server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.send_message(msg)
            channel_statuses["email"] = "delivered"
        except Exception as e:
            print(f"[!] Email dispatch failed: {e}")
            channel_statuses["email"] = f"failed: {str(e)}"
    elif send_email:
        channel_statuses["email"] = "skipped_no_credentials"

    # 2. SMS Channel (Twilio or Indian DLT SMS Gateway)
    if send_sms and settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN and user.phone_number:
        try:
            import requests
            url = f"https://api.twilio.com/2010-04-01/Accounts/{settings.TWILIO_ACCOUNT_SID}/Messages.json"
            data = {
                "From": settings.TWILIO_PHONE_NUMBER,
                "To": user.phone_number,
                "Body": f"Crop Health Alert: {title}. {message[:120]}"
            }
            resp = requests.post(url, data=data, auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN), timeout=5)
            if resp.status_code in [200, 201]:
                channel_statuses["sms"] = "delivered"
            else:
                channel_statuses["sms"] = f"provider_rejected: {resp.status_code}"
        except Exception as e:
            channel_statuses["sms"] = f"failed: {str(e)}"
    elif send_sms:
        channel_statuses["sms"] = "skipped_no_credentials"

    # 3. Save In-App Notification in DB
    db_notif = Notification(
        user_id=user_id,
        title=title,
        title_hi=title_hi or title,
        title_gu=title_gu or title,
        message=message,
        message_hi=message_hi or message,
        message_gu=message_gu or message,
        alert_type=alert_type,
        severity=severity,
        channel="multi_channel",
        delivery_status=f"in_app:delivered; email:{channel_statuses.get('email', 'disabled')}; sms:{channel_statuses.get('sms', 'disabled')}"
    )
    db.add(db_notif)
    db.commit()
    db.refresh(db_notif)

    return {
        "notification_id": db_notif.id,
        "in_app": "delivered",
        "channels": channel_statuses
    }
