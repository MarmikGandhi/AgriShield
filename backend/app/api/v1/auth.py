"""
auth.py
=======
Authentication endpoints: Registration, Login (JWT Token), and Profile details.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.database.models import User, FarmerProfile, AgronomistProfile
from app.schemas.schemas import UserCreate, UserLogin, UserResponse, Token
from app.core.security import verify_password, get_password_hash, create_access_token
from app.api.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=Token)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    # Check if email exists
    existing = db.query(User).filter(User.email == user_in.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address is already registered."
        )

    # Create user
    user = User(
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        full_name=user_in.full_name,
        phone_number=user_in.phone_number,
        role=user_in.role.lower(),
        language_preference=user_in.language_preference
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Create role profile
    if user.role == "farmer":
        profile = FarmerProfile(
            user_id=user.id,
            village=user_in.village or "Anand",
            taluka=user_in.taluka or "Anand",
            district=user_in.district or "Anand",
            farm_size_acres=user_in.farm_size_acres or 2.5
        )
        db.add(profile)
    elif user.role == "agronomist":
        profile = AgronomistProfile(
            user_id=user.id,
            institution=user_in.institution or "Anand Agricultural University",
            specialization=user_in.specialization or "Plant Pathology"
        )
        db.add(profile)

    db.commit()
    db.refresh(user)

    access_token = create_access_token(subject=user.id, role=user.role)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

@router.post("/login", response_model=Token)
def login(login_in: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == login_in.email).first()
    if not user or not verify_password(login_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user account.")

    access_token = create_access_token(subject=user.id, role=user.role)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
