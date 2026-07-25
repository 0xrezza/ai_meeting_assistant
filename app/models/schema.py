from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class TranscriptSegment(BaseModel):
    speaker: str = Field(..., description="Name or identifier of the speaker")
    start_time: str = Field(..., description="Start timestamp of the segment")
    end_time: str = Field(..., description="End timestamp of the segment")
    text: str = Field(..., description="The spoken text")

class Transcript(BaseModel):
    meeting_id: str
    segments: List[TranscriptSegment]
    
class ActionItem(BaseModel):
    task: str = Field(..., description="The specific task to be done")
    assignee: Optional[str] = Field("", description="Person responsible for the task")
    deadline: Optional[str] = Field(None, description="Deadline if mentioned")
    priority: Optional[str] = Field("Normal", description="Priority level")

class Decision(BaseModel):
    decision: str = Field(..., description="The decision that was made")
    context: Optional[str] = Field(None, description="Context or reason for the decision")

class MeetingSummary(BaseModel):
    meeting_id: str
    summary_text: str = Field(..., description="A one-page summary of the meeting")
    key_points: List[str] = Field(default_factory=list, description="Key points discussed")
    decisions: List[Decision] = Field(default_factory=list, description="Decisions made")
    action_items: List[ActionItem] = Field(default_factory=list, description="Extracted tasks")

class StrategicInsights(BaseModel):
    risks: List[str] = Field(..., description="Risks identified in the meeting")
    opportunities: List[str] = Field(..., description="Opportunities identified in the meeting")
    recommendations: List[str] = Field(..., description="AI recommendations for the CEO")


# ===== API Response/Request Models =====

class MeetingProcessResponse(BaseModel):
    """خروجی نهایی JSON که به رضا فرستاده می‌شود"""
    meeting_id: str = Field(..., description="Unique meeting identifier")
    transcript: List[TranscriptSegment] = Field(default_factory=list, description="Full transcript with timestamps")
    summary: str = Field("", description="One-page meeting summary")
    key_points: List[str] = Field(default_factory=list, description="Key points discussed")
    decisions: List[Decision] = Field(default_factory=list, description="Decisions made")
    action_items: List[ActionItem] = Field(default_factory=list, description="Extracted tasks")
    insights: StrategicInsights = Field(default=None, description="Strategic insights (risks, opportunities, recommendations)")

class ChatRequest(BaseModel):
    """درخواست چت RAG"""
    question: str = Field(..., description="User question about meetings")

class ChatResponse(BaseModel):
    """پاسخ چت RAG"""
    answer: str = Field(..., description="AI generated answer")
    citations: List[dict] = Field(default_factory=list, description="Source citations from transcripts")

class HealthResponse(BaseModel):
    """پاسخ بررسی سلامت سرویس"""
    status: str = "ok"
    whisper_model: str = ""
    ollama_status: str = ""

