"""
Meeting API Endpoints - REST interface for Reza's backend.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schema import (
    MeetingProcessResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    LocalChatCompleteRequest,
    LocalChatCompleteResponse,
)

import json
import urllib.request
import urllib.error

router = APIRouter(prefix="/api/v1", tags=["meetings"])

# Pipeline instance will be set by app_main.py at startup
_pipeline = None


def set_pipeline(pipeline):
    """Called by app_main.py to inject the pipeline instance."""
    global _pipeline
    _pipeline = pipeline


def _get_pipeline():
    """Get the pipeline instance, raise error if not initialized."""
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Pipeline not initialized."
        )
    return _pipeline


@router.post(
    "/meetings/process",
    response_model=MeetingProcessResponse,
    summary="پردازش فایل صوتی جلسه",
    description="فایل WebM/MP4/WAV جلسه را آپلود کنید تا رونوشت، خلاصه، تصمیمات، وظایف و بینش‌ها استخراج شود.",
)
async def process_meeting(
    audio: UploadFile = File(..., description="فایل صوتی جلسه (WebM, MP4, WAV, etc.)"),
    meeting_id: str = None,
):
    """
    Main endpoint: Upload audio → Get full meeting analysis as JSON.

    Pipeline:
    1. Save uploaded file
    2. Whisper ASR (speech-to-text)
    3. DeepSeek text correction
    4. Persian normalization
    5. AI analysis (summary, decisions, tasks, insights)
    6. Store in vector DB
    7. Return JSON
    """
    pipeline = _get_pipeline()

    # Validate file type
    allowed_extensions = {".webm", ".mp4", ".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    file_ext = ""
    if audio.filename:
        file_ext = "." + audio.filename.rsplit(".", 1)[-1].lower() if "." in audio.filename else ""

    if file_ext and file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    try:
        # Read file content
        file_content = await audio.read()

        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Save to uploads directory
        filename = audio.filename or "unknown_audio.webm"
        saved_path = pipeline.save_uploaded_file(file_content, filename)

        # Process through the full pipeline
        result = pipeline.process_meeting(
            audio_path=str(saved_path),
            meeting_id=meeting_id,
        )

        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post(
    "/meetings/chat",
    response_model=ChatResponse,
    summary="چت هوشمند درباره جلسات",
    description="سوالی درباره جلسات گذشته بپرسید و پاسخ مبتنی بر RAG دریافت کنید.",
)
async def chat_about_meetings(request: ChatRequest):
    """
    RAG-based Q&A about past meetings.
    Searches vector DB for relevant segments and answers using Qwen2.5.
    """
    pipeline = _get_pipeline()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = pipeline.chat(request.question)
        return ChatResponse(
            answer=result["answer"],
            citations=result.get("citations", []),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="بررسی سلامت سرویس",
    description="وضعیت سرویس، مدل Whisper و اتصال Ollama را بررسی می‌کند.",
)
async def health_check():
    """Check if the service and its dependencies are healthy."""
    pipeline = _get_pipeline()

    # Check Whisper model
    whisper_status = "loaded" if pipeline.asr_service.whisper_model is not None else "not_loaded"

    # Check Ollama connection
    ollama_status = "unknown"
    try:
        url = f"{pipeline.ollama_url}/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                ollama_status = "connected"
    except Exception:
        ollama_status = "disconnected"

    return HealthResponse(
        status="ok",
        whisper_model=whisper_status,
        ollama_status=ollama_status,
    )


@router.post(
    "/chat/complete",
    response_model=LocalChatCompleteResponse,
    summary="چت اختصاصی برای مونو‌ریپوی رضا",
    description="دریافت پرامپت سیستم و کاربر و تولید پاسخ هوش مصنوعی به فرمت ساختاریافته JSON",
)
async def chat_complete(request: LocalChatCompleteRequest):
    pipeline = _get_pipeline()
    try:
        result = pipeline.complete_chat(request.system_prompt, request.user_prompt)
        return LocalChatCompleteResponse(
            answer=result["answer"],
            refused=result["refused"],
            usedSourceLabels=result.get("usedSourceLabels", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

