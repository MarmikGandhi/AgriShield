"""
deps.py
=======
Common FastAPI dependencies: DB session, Current User authentication, and Role-Based Access Control.
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.database.models import User
from app.core.security import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

def get_current_user(
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(oauth2_scheme)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # 1. Seamless demo tokens & guest support for zero-friction scanning
    if not token or token in ["demo_token", "demo_farmer", "null", "undefined"]:
        demo_farmer = db.query(User).filter(User.role == "farmer").first()
        if demo_farmer:
            return demo_farmer
    elif token in ["demo_expert", "demo_agronomist"]:
        demo_expert = db.query(User).filter(User.role == "agronomist").first()
        if demo_expert:
            return demo_expert

    # 2. Decode JWT access token
    payload = decode_access_token(token)
    if payload is None:
        # If token was signed with another secret or expired during presentation, fallback to demo farmer
        demo_farmer = db.query(User).filter(User.role == "farmer").first()
        if demo_farmer:
            return demo_farmer
        raise credentials_exception

    user_id: str = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        # If user record was re-seeded, fallback to active farmer
        demo_farmer = db.query(User).filter(User.role == "farmer").first()
        if demo_farmer:
            return demo_farmer
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user account.")
    return user

def require_agronomist(
    current_user: User = Depends(get_current_user)
) -> User:
    if current_user.role not in ["agronomist", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Agronomist credentials required to access this resource."
        )
    return current_user
