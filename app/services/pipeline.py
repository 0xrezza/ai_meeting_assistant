"""
Pipeline Orchestrator - Coordinates all AI processing steps.
WebM → ASR → Normalize → Intelligence → VectorDB → Final JSON
"""
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from app.models.schema import (
    Transcript,
    TranscriptSegment,
    MeetingProcessResponse,
    StrategicInsights,
)
from app.services.asr_service import ASRService
from app.services.meeting_intelligence import MeetingIntelligenceService
from app.services.vector_db import SimpleVectorDB
from app.services.rag_chat import RAGChatService


# Directories
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

TRANSCRIPTS_DIR = Path("transcripts")
TRANSCRIPTS_DIR.mkdir(exist_ok=True)


class MeetingPipeline:
    """
    Main orchestrator that chains all AI services together.
    """

    def __init__(self, ollama_url: str = "http://127.0.0.1:11434"):
        self.ollama_url = ollama_url

        # Initialize services
        self.asr_service = ASRService(ollama_url=ollama_url)
        self.intelligence_service = MeetingIntelligenceService(ollama_url=ollama_url)
        self.vector_db = SimpleVectorDB(
            storage_path=str(TRANSCRIPTS_DIR / "vector_store.json"),
            ollama_url=ollama_url,
        )
        self.rag_service = RAGChatService(
            vector_db=self.vector_db,
            ollama_url=ollama_url,
        )

    def load_models(self):
        """Load heavy models at startup (Whisper)."""
        self.asr_service.load_whisper_model("large-v3")

    def _generate_meeting_id(self) -> str:
        """Generate a unique meeting ID based on timestamp."""
        now = datetime.now()
        return f"meeting_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    def save_uploaded_file(self, file_content: bytes, original_filename: str) -> Path:
        """
        Save uploaded WebM file to the uploads directory.

        Returns
        -------
        Path
            Path to the saved file
        """
        # Create a unique filename to avoid collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = original_filename.replace(" ", "_")
        file_path = UPLOADS_DIR / f"{timestamp}_{safe_name}"

        with open(file_path, "wb") as f:
            f.write(file_content)

        print(f"[Pipeline] Saved uploaded file: {file_path}")
        return file_path

    def process_meeting(
        self,
        audio_path: str,
        meeting_id: Optional[str] = None,
    ) -> MeetingProcessResponse:
        """
        Full meeting processing pipeline:
        1. ASR (Whisper + DeepSeek correction + normalization)
        2. Meeting intelligence (summary, decisions, action items)
        3. Strategic insights (risks, opportunities, recommendations)
        4. Store in vector DB for RAG
        5. Build and return final JSON

        Parameters
        ----------
        audio_path : str
            Path to the audio file (WebM, MP4, WAV, etc.)
        meeting_id : str, optional
            Custom meeting ID. Auto-generated if not provided.

        Returns
        -------
        MeetingProcessResponse
            Complete meeting analysis as JSON-serializable object
        """
        if not meeting_id:
            meeting_id = self._generate_meeting_id()

        print(f"\n{'='*60}")
        print(f"[Pipeline] Starting processing for: {meeting_id}")
        print(f"{'='*60}")

        # ── Step 1: ASR (Speech-to-Text) ──────────────────────────
        print(f"\n[Pipeline] Step 1/4: ASR Transcription...")
        transcript = self.asr_service.transcribe_audio(audio_path, meeting_id)
        print(f"[Pipeline] ✓ ASR complete: {len(transcript.segments)} segments")

        # ── Step 2: Meeting Intelligence ──────────────────────────
        print(f"\n[Pipeline] Step 2/4: Analyzing meeting (summary, decisions, tasks)...")
        meeting_summary = self.intelligence_service.analyze_meeting(transcript)
        print(f"[Pipeline] ✓ Analysis complete:")
        print(f"    - Key points: {len(meeting_summary.key_points)}")
        print(f"    - Decisions: {len(meeting_summary.decisions)}")
        print(f"    - Action items: {len(meeting_summary.action_items)}")

        # ── Step 3: Strategic Insights ────────────────────────────
        print(f"\n[Pipeline] Step 3/4: Extracting strategic insights...")
        try:
            insights = self.intelligence_service.extract_insights(transcript)
            print(f"[Pipeline] ✓ Insights complete:")
            print(f"    - Risks: {len(insights.risks)}")
            print(f"    - Opportunities: {len(insights.opportunities)}")
            print(f"    - Recommendations: {len(insights.recommendations)}")
        except Exception as e:
            print(f"[Pipeline] ⚠ Insights extraction failed: {e}")
            insights = StrategicInsights(risks=[], opportunities=[], recommendations=[])

        # ── Step 4: Store in Vector DB ────────────────────────────
        print(f"\n[Pipeline] Step 4/4: Storing in vector database for RAG...")
        try:
            segments_data = [
                {
                    "speaker": seg.speaker,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.text,
                }
                for seg in transcript.segments
            ]
            self.vector_db.add_segments(meeting_id, segments_data)
            print(f"[Pipeline] ✓ Stored {len(segments_data)} segments in vector DB")
        except Exception as e:
            print(f"[Pipeline] ⚠ Vector DB storage failed: {e}")

        # ── Build Final Response ──────────────────────────────────
        response = MeetingProcessResponse(
            meeting_id=meeting_id,
            transcript=transcript.segments,
            summary=meeting_summary.summary_text,
            key_points=meeting_summary.key_points,
            decisions=meeting_summary.decisions,
            action_items=meeting_summary.action_items,
            insights=insights,
        )

        print(f"\n{'='*60}")
        print(f"[Pipeline] ✅ Processing complete for: {meeting_id}")
        print(f"{'='*60}\n")

        return response

    def chat(self, question: str) -> dict:
        """
        Answer a question about past meetings using RAG.

        Parameters
        ----------
        question : str
            User's question

        Returns
        -------
        dict
            Answer with citations
        """
        print(f"[Pipeline] RAG Chat: {question[:50]}...")
        result = self.rag_service.answer_question(question)
        return result
