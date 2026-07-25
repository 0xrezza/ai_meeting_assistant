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

    def __init__(self, ollama_url: str = "http://127.0.0.1:11434"):
        self.ollama_url = ollama_url
        self.correction_model = "qwen2.5:7b"
        self.normalizer = PersianNormalizer()
        self.whisper_model = None  # Lazy loaded

    def load_whisper_model(self, model_size: str = "large-v3"):
        """
        Load the Whisper model. Call this once at application startup.
        """
        import whisper
        print(f"[ASR] Loading Whisper model '{model_size}'... (this may take a moment)")
        self.whisper_model = whisper.load_model(model_size)
        print(f"[ASR] Whisper model '{model_size}' loaded successfully.")

    def _ensure_model_loaded(self):
        """Ensure Whisper model is loaded before use."""
        if self.whisper_model is None:
            self.load_whisper_model()

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS.mmm format."""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"

    def _correct_text_with_llm(self, text: str) -> str:
        """
        Correct transcription errors using DeepSeek-R1 via Ollama.
        Based on Armin's correct_text_with_gpt() function.
        """
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.correction_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an Automatic Speech Recognition (ASR) post-processing model.\n\n"
                        "The input transcript is in Persian (Farsi).\n"
                        "The output MUST also be in Persian (Farsi). Never translate the transcript into another language.\n\n"
                        "Your job is to minimally correct ASR transcripts while preserving the speaker's intended meaning.\n\n"
                        "Your primary objective is semantic preservation, NOT producing well-written Persian.\n\n"
                        "Allowed corrections (only when highly confident):\n\n"
                        "1. Correct obvious spelling mistakes.\n"
                        "2. Correct Persian half-spacing (e.g. میخواهم, کتابها).\n"
                        "3. Correct punctuation.\n"
                        "4. Normalize numbers, dates, and units into a consistent format.\n"
                        "5. Correct obvious ASR recognition mistakes only when the intended word is nearly certain.\n"
                        "6. Normalize domain-specific terminology using the provided glossary.\n\n"
                        "Forbidden operations:\n\n"
                        "- Do NOT paraphrase.\n"
                        "- Do NOT rewrite sentences.\n"
                        "- Do NOT summarize.\n"
                        "- Do NOT reorder words.\n"
                        "- Do NOT improve grammar.\n"
                        "- Do NOT replace words with synonyms.\n"
                        "- Do NOT remove repetitions or hesitations.\n"
                        "- Do NOT add missing words.\n"
                        "- Do NOT remove words.\n"
                        "- Do NOT infer information that is not explicitly present.\n"
                        "- Do NOT make corrections based on what \"sounds better.\"\n"
                        "- Do NOT translate any part of the transcript into another language.\n\n"
                        "If multiple interpretations are possible, keep the original transcript unchanged.\n\n"
                        "If your confidence is below 95%, do not modify that part of the transcript.\n\n"
                        "Return ONLY the corrected Persian transcript. Do not include explanations, comments, notes, confidence scores, or any additional text."
                    )
                },
                {
                    "role": "user",
                    "content": f"Correct this text: {text}"
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8")
                print(f"[ASR] Ollama HTTP Error {e.code}: {body}")
            except Exception:
                print(f"[ASR] Ollama HTTP Error {e.code}")
            return text
        except Exception as e:
            print(f"[ASR] Warning: Text correction failed, using original text. Error: {str(e)[:100]}")
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
        1. Try Soniox Cloud API (if API Key available)
        2. Fallback to local Whisper transcription if Soniox fails
        3. DeepSeek text correction (applied inside segment loops)
        4. Persian normalization
        5. Return structured Transcript object
        """
        import os
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Check for Soniox API key
        soniox_api_key = os.environ.get("SONIOX_API_KEY")
        if not soniox_api_key:
            # Fallback to the provided key
            soniox_api_key = "d43a2f6b67d1e9fab178791cf4381ebb8bcbe0e24b72da5dc7d2446bcc26eb2e"

        if soniox_api_key:
            try:
                print(f"[ASR] Attempting Soniox Cloud API transcription for: {audio_file.name}")
                return self._transcribe_audio_soniox(audio_path, meeting_id, soniox_api_key)
            except Exception as e:
                print(f"[ASR] Warning: Soniox transcription failed. Falling back to local Whisper. Error: {e}")

        # --- Whisper Fallback ---
        self._ensure_model_loaded()

        print(f"[ASR] Transcribing locally with Whisper: {audio_file.name}")

        result = self.whisper_model.transcribe(
            audio_path,
            language="fa",
            fp16=False,
            verbose=False,
        )

        full_text = result["text"].strip()
        if not full_text:
            raise ValueError("No text could be extracted from the audio file.")

        print(f"[ASR] Raw local transcription complete. {len(result['segments'])} segments found.")

        segments: List[TranscriptSegment] = []
        hallucination_substrings = ["موسیقی", "music", "subtitles", "subtitle", "سپاس", "ممنون", "تشکر"]
        
        for i, seg in enumerate(result["segments"]):
            raw_text = seg["text"].strip()
            if not raw_text:
                continue
                
            if seg.get("no_speech_prob", 0.0) > 0.45:
                print(f"[ASR] Skipping silent segment {i+1} (no_speech_prob={seg['no_speech_prob']:.2f} > 0.45)")
                continue
                
            check_text = raw_text.lower().strip()
            is_hallucination = False
            for word in hallucination_substrings:
                if word in check_text:
                    is_hallucination = True
                    break
                    
            if is_hallucination or not any(c.isalnum() for c in check_text):
                print(f"[ASR] Skipping hallucination segment {i+1}: '{raw_text}'")
                continue

            corrected_text = self._correct_text_with_llm(raw_text)
            
            check_corrected = corrected_text.lower().strip()
            is_corrected_hallucination = False
            for word in hallucination_substrings:
                if word in check_corrected:
                    is_corrected_hallucination = True
                    break
                    
            if is_corrected_hallucination:
                continue

            normalized_text = self.normalizer.normalize_text(corrected_text)

            segment = TranscriptSegment(
                speaker=f"Speaker",
                start_time=self._format_timestamp(seg["start"]),
                end_time=self._format_timestamp(seg["end"]),
                text=normalized_text,
            )
            segments.append(segment)

            if (i + 1) % 10 == 0:
                print(f"[ASR] Processed {i + 1}/{len(result['segments'])} segments...")

        if not segments:
            print("[ASR] No valid speech segments found. Adding default notice.")
            segments.append(TranscriptSegment(
                speaker="System",
                start_time="00:00:00",
                end_time="00:00:05",
                text="گفتاری در این فایل صوتی تشخیص داده نشد. لطفا مطمئن شوید میکروفون یا صدای سیستم هنگام ضبط فعال بوده است."
            ))

        print(f"[ASR] Local transcription complete. {len(segments)} segments processed.")

        return Transcript(
            meeting_id=meeting_id,
            segments=segments,
        )

