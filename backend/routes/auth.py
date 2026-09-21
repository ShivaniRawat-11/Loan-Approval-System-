from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import timedelta
from database.auth_db import create_user, authenticate_user
from utils.security import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter()

class SignupRequest(BaseModel):
    email: str
    password: str
    name: str = None

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    success: bool
    name: str = None
    email: str = None
    message: str = ""
    token: str = None

@router.post("/signup", response_model=AuthResponse)
def signup(req: SignupRequest):
    """Create a new user account."""
    if not req.email or not req.password:
        raise HTTPException(status_code=400, detail="Email and password are required.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    
    success, message = create_user(req.email, req.password, req.name)
    if success:
        email_clean = req.email.strip().lower()
        name = req.name or email_clean.split("@")[0].capitalize()
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = create_access_token(
            data={"sub": email_clean}, expires_delta=access_token_expires
        )
        return AuthResponse(success=True, name=name, email=email_clean, message=message, token=token)
    else:
        return AuthResponse(success=False, message=message)

@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """Authenticate a user."""
    if not req.email or not req.password:
        raise HTTPException(status_code=400, detail="Email and password are required.")
    
    success, name, message = authenticate_user(req.email, req.password)
    if success:
        email_clean = req.email.strip().lower()
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = create_access_token(
            data={"sub": email_clean}, expires_delta=access_token_expires
        )
        return AuthResponse(success=True, name=name, email=email_clean, message=message, token=token)
    else:
        return AuthResponse(success=False, message=message)

@router.post("/logout")
def logout():
    """Client-side removes the JWT token."""
    return {"success": True, "message": "Logged out."}
