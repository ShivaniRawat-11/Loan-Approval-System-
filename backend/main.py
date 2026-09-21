import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Normalize API keys
for env_name in ("NVIDIA_API_KEY", "GOOGLE_API_KEY"):
    raw_key = os.getenv(env_name, "")
    if raw_key:
        os.environ[env_name] = raw_key.strip().replace('"', "").replace("'", "")

from database.auth_db import init_db
from routes import auth, chat, documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    init_db()
    yield


app = FastAPI(
    title="Loan Approval API",
    description="Backend API for the AI Loan Approval System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://frontend:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount route groups
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])


@app.get("/api/health")
def health_check():
    """Health check endpoint for Docker."""
    return {"status": "ok", "service": "loan-approval-backend"}
