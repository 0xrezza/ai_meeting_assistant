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
    def transcribe_audio(self, audio_path: str, meeting_id: str) -> Transcript:
        """
        Full ASR pipeline:
        1. Whisper transcription (audio → raw text with timestamps)
        2. DeepSeek text correction
        3. Persian normalization
        4. Return structured Transcript object

        Parameters
        ----------
        audio_path : str
            Path to the audio file (WebM, MP4, WAV, etc.)
        meeting_id : str
            Unique meeting identifier

        Returns
        -------
        Transcript
            Structured transcript with segments
        """
        self._ensure_model_loaded()

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"[ASR] Transcribing: {audio_file.name}")

        # Step 1: Whisper transcription
        result = self.whisper_model.transcribe(
            audio_path,
            language="fa",
            fp16=False,
            verbose=False,
        )

        full_text = result["text"].strip()
        if not full_text:
            raise ValueError("No text could be extracted from the audio file.")

        print(f"[ASR] Raw transcription complete. {len(result['segments'])} segments found.")

        # Step 2: Correct each segment's text with DeepSeek
        segments: List[TranscriptSegment] = []
        hallucination_substrings = ["موسیقی", "music", "subtitles", "subtitle", "سپاس", "ممنون", "تشکر"]
        
        for i, seg in enumerate(result["segments"]):
            raw_text = seg["text"].strip()
            if not raw_text:
                continue
                
            # Filter silence/noise using a stricter Whisper probability score threshold of 0.45
            if seg.get("no_speech_prob", 0.0) > 0.45:
                print(f"[ASR] Skipping silent segment {i+1} (no_speech_prob={seg['no_speech_prob']:.2f} > 0.45)")
                continue
                
            # Clean text and run substring-level hallucination filtering
            check_text = raw_text.lower().strip()
            is_hallucination = False
            for word in hallucination_substrings:
                if word in check_text:
                    is_hallucination = True
                    break
                    
            if is_hallucination or not any(c.isalnum() for c in check_text):
                print(f"[ASR] Skipping hallucination segment {i+1}: '{raw_text}'")
                continue

            # Correct text
            corrected_text = self._correct_text_with_llm(raw_text)
            
            # Additional check on corrected text
            check_corrected = corrected_text.lower().strip()
            is_corrected_hallucination = False
            for word in hallucination_substrings:
                if word in check_corrected:
                    is_corrected_hallucination = True
                    break
                    
            if is_corrected_hallucination:
                continue

            # Step 3: Normalize Persian text
            normalized_text = self.normalizer.normalize_text(corrected_text)

            segment = TranscriptSegment(
                speaker=f"Speaker",  # Whisper does not do speaker diarization
                start_time=self._format_timestamp(seg["start"]),
                end_time=self._format_timestamp(seg["end"]),
                text=normalized_text,
            )
            segments.append(segment)

            if (i + 1) % 10 == 0:
                print(f"[ASR] Processed {i + 1}/{len(result['segments'])} segments...")

        # Fallback if no valid segments found
        if not segments:
            print("[ASR] No valid speech segments found. Adding default notice.")
            segments.append(TranscriptSegment(
                speaker="System",
                start_time="00:00:00",
                end_time="00:00:05",
                text="گفتاری در این فایل صوتی تشخیص داده نشد. لطفا مطمئن شوید میکروفون یا صدای سیستم هنگام ضبط فعال بوده است."
            ))

        print(f"[ASR] Transcription complete. {len(segments)} segments processed.")

        return Transcript(
            meeting_id=meeting_id,
            segments=segments,
        )
