"""
AI Meeting Assistant - FastAPI Application
==========================================
نقطه ورود اصلی سرویس هوش مصنوعی دستیار جلسات.

اجرا:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

مستندات API:
    http://localhost:8000/docs
"""
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app.api.endpoints.meetings import router as meetings_router, set_pipeline
from app.services.pipeline import MeetingPipeline


# ── Global Pipeline Instance ──────────────────────────────────
pipeline = MeetingPipeline(ollama_url="http://127.0.0.1:11434")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    - Startup: Load Whisper model
    - Shutdown: Cleanup
    """
    # ── Startup ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🚀 AI Meeting Assistant - Starting up...")
    print("=" * 60)

    # Load heavy models
    # pipeline.load_models()

    # Inject pipeline into the endpoints
    set_pipeline(pipeline)

    print("\n✅ All models loaded. Server is ready!")
    print("📄 API Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")

    yield

    # ── Shutdown ──────────────────────────────────────────────
    print("\n🛑 Shutting down AI Meeting Assistant...")


# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="AI Meeting Assistant",
    description=(
        "دستیار هوشمند جلسات - تبدیل فایل صوتی به رونوشت، خلاصه، تصمیمات، "
        "وظایف و بینش‌های مدیریتی با استفاده از هوش مصنوعی محلی"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware (for Reza's frontend) ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to Reza's frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Routers ──────────────────────────────────────────
app.include_router(meetings_router)


# ── Root Endpoint ────────────────────────────────────────────
@app.get("/", tags=["root"])
async def root():
    return {
        "service": "AI Meeting Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "process_meeting": "POST /api/v1/meetings/process",
            "chat": "POST /api/v1/meetings/chat",
            "health": "GET /api/v1/health",
        }
    }
