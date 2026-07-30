"""
ASR Service - Wrapper around Armin's Whisper transcription code.
Converts WebM/audio files to text using Whisper large-v3 and corrects with DeepSeek/Qwen.
"""
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional
from app.models.schema import TranscriptSegment, Transcript
from app.nlp.normalizer import PersianNormalizer


class ASRService:
    """
    Speech-to-Text service using OpenAI Whisper (local) + text correction via Ollama.
    Based on Armin's main.py but refactored as a clean service class.
    """

    def load_env_variables(self):
        """Load environment variables from the monorepo .env file."""
        import os
        from pathlib import Path
        paths_to_check = [
            Path(".") / ".env",
            Path("..") / ".env",
            Path("..") / "modira-main" / "modira" / ".env",
            Path("..") / "modira" / ".env",
            Path("D:/Aleph company/modira-main/modira/.env"),
        ]
        for p in paths_to_check:
            if p.exists():
                print(f"[ASR] Loading environment variables from: {p.resolve()}")
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            v = v.strip().strip("'").strip('"')
                            os.environ[k.strip()] = v
                break

    def __init__(self, ollama_url: str = "http://127.0.0.1:11434"):
        self.ollama_url = ollama_url
        self.normalizer = PersianNormalizer()
        # Always run env load during initialization
        self.load_env_variables()

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS.mmm format."""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"

    def _correct_text_with_llm(self, text: str) -> str:
        """
        Correct transcription errors using OpenAI GPT-4o-mini API.
        This runs completely serverless/API-based with zero local CPU load.
        """
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[ASR] Warning: OPENAI_API_KEY not found. Skipping text correction.")
            return text

        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert Automatic Speech Recognition (ASR) post-processor specializing in Persian (Farsi) language.\n"
                        "Your goal is to clean up and minimally correct ASR transcription errors to make the transcript readable, clean, and grammatically standard, without changing the meaning or speaker style.\n\n"
                        "Core Principles:\n"
                        "1. SEMANTIC PRESERVATION: Keep the original words and order. Do NOT summarize, paraphrase, rewrite, or replace words with synonyms.\n"
                        "2. ORTHOGRAPHY & HALF-SPACING: Correct half-spacing (نیم‌فاصله) using the Zero-Width Non-Joiner (ZWNJ). For example, prefixes like 'می‌' (e.g., می‌خواهم) and suffixes like 'ها' (e.g., کتاب‌ها)، 'تر' (e.g., بزرگ‌تر)، and 'ام/ات/اش' should be correctly half-spaced.\n"
                        "3. SPELLING CORRECTION: Fix obvious typos or phonetic transcription mistakes (e.g., 'سحبت' to 'صحبت').\n"
                        "4. NUMERALS & DATES: Format numbers, dates, monetary units, and times cleanly (e.g., 'پنج میلیون' to '۵,۰۰۰,۰۰۰').\n"
                        "5. NO ADDED COMMENTS: Return ONLY the corrected Farsi text. Do NOT output markdown code blocks (like ```farsi or ```text), notes, explanations, or preambles."
                    )
                },
                {
                    "role": "user",
                    "content": f"Correct this text: {text}"
                }
            ],
            "temperature": 0.1
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        )
        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[ASR] Warning: GPT-4o-mini text correction failed. Error: {e}")
            return text
    def _upload_file_soniox(self, file_path: Path, api_key: str) -> str:
        import uuid
        import mimetypes
        
        filename = file_path.name
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            mime_type = "application/octet-stream"
            
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        
        part_header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8")
        
        part_footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
        
        with open(file_path, "rb") as f:
            file_content = f.read()
            
        body = part_header + file_content + part_footer
        
        url = "https://api.soniox.com/v1/files"
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body))
            },
            method="POST"
        )
        
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["id"]

    def _create_transcription_soniox(self, file_id: str, api_key: str) -> str:
        url = "https://api.soniox.com/v1/transcriptions"
        payload = {
            "file_id": file_id,
            "model": "stt-async-v5",
            "language_hints": ["fa"]
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            method="POST"
        )
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["id"]

    def _poll_transcription_status_soniox(self, transcription_id: str, api_key: str) -> None:
        import time
        url = f"https://api.soniox.com/v1/transcriptions/{transcription_id}"
        
        max_attempts = 120
        for _ in range(max_attempts):
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}"
                },
                method="GET"
            )
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                status = res_data.get("status")
                if status == "completed":
                    return
                elif status in ("error", "failed"):
                    error_msg = res_data.get("error_message") or "Unknown transcription error"
                    raise ValueError(f"Soniox transcription failed: {error_msg}")
            time.sleep(5)
            
        raise TimeoutError("Soniox transcription timed out.")

    def _get_transcript_soniox(self, transcription_id: str, api_key: str) -> dict:
        url = f"https://api.soniox.com/v1/transcriptions/{transcription_id}/transcript"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            method="GET"
        )
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))

    def _transcribe_audio_soniox(self, audio_path: str, meeting_id: str, api_key: str) -> Transcript:
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Step A: Upload file
        print(f"[ASR-Soniox] Uploading file: {audio_file.name}...")
        file_id = self._upload_file_soniox(audio_file, api_key)
        print(f"[ASR-Soniox] File uploaded successfully. File ID: {file_id}")
        
        # Step B: Start transcription
        print(f"[ASR-Soniox] Starting transcription job...")
        transcription_id = self._create_transcription_soniox(file_id, api_key)
        print(f"[ASR-Soniox] Job created. ID: {transcription_id}")
        
        # Step C: Poll status
        print(f"[ASR-Soniox] Polling transcription status...")
        self._poll_transcription_status_soniox(transcription_id, api_key)
        print(f"[ASR-Soniox] Transcription completed successfully!")
        
        # Step D: Get transcript
        print(f"[ASR-Soniox] Fetching transcript data...")
        result = self._get_transcript_soniox(transcription_id, api_key)
        
        tokens = result.get("tokens", [])
        if not tokens:
            raise ValueError("No speech tokens returned from Soniox API.")
            
        # Step E: Group tokens into segments
        print(f"[ASR-Soniox] Grouping {len(tokens)} tokens into segments...")
        grouped_segs: List[TranscriptSegment] = []
        current_words = []
        start_time = None
        
        for i, token in enumerate(tokens):
            word_text = token.get("text", "")
            if not word_text:
                continue
                
            if start_time is None:
                start_time = token.get("start_ms", 0) / 1000.0
                
            current_words.append(word_text)
            end_time = token.get("end_ms", 0) / 1000.0
            
            # Segment conditions
            has_punctuation = any(p in word_text for p in (".", "؟", "!", "?"))
            long_enough = len(current_words) >= 25
            
            large_gap = False
            if i < len(tokens) - 1:
                next_start = tokens[i+1].get("start_ms", 0) / 1000.0
                if next_start - end_time > 1.5:
                    large_gap = True
                    
            if has_punctuation or long_enough or large_gap or (i == len(tokens) - 1):
                raw_segment_text = " ".join([w.strip() for w in current_words if w.strip()])
                if raw_segment_text:
                    # Run LLM text correction on segment
                    corrected_text = self._correct_text_with_llm(raw_segment_text)
                    # Normalize Persian text
                    normalized_text = self.normalizer.normalize_text(corrected_text)
                    
                    grouped_segs.append(TranscriptSegment(
                        speaker="Speaker",
                        start_time=self._format_timestamp(start_time),
                        end_time=self._format_timestamp(end_time),
                        text=normalized_text,
                    ))
                current_words = []
                start_time = None
                
        if not grouped_segs:
            grouped_segs.append(TranscriptSegment(
                speaker="System",
                start_time="00:00:00",
                end_time="00:00:05",
                text="گفتاری در این فایل صوتی تشخیص داده نشد."
            ))
            
        return Transcript(
            meeting_id=meeting_id,
            segments=grouped_segs,
        )

    def transcribe_audio(self, audio_path: str, meeting_id: str) -> Transcript:
        """
        Full ASR pipeline:
        1. Load environment variables dynamically to capture updated credentials.
        2. Transcribe exclusively with Soniox Cloud API (no local Whisper fallback).
        """
        import os
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Always reload environment variables to check for charged API keys
        self.load_env_variables()

        # Check for Soniox API key
        soniox_api_key = os.environ.get("SONIOX_API_KEY")
        if not soniox_api_key:
            # Fallback to the provided default key if not set in environment
            soniox_api_key = "d43a2f6b67d1e9fab178791cf4381ebb8bcbe0e24b72da5dc7d2446bcc26eb2e"

        print(f"[ASR] Attempting Soniox Cloud API transcription for: {audio_file.name}")
        return self._transcribe_audio_soniox(audio_path, meeting_id, soniox_api_key)

