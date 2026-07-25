import json
import urllib.request
import urllib.error
from typing import Dict, Any, List
from app.models.schema import MeetingSummary, ActionItem, Decision, Transcript, StrategicInsights

class MeetingIntelligenceService:
    def __init__(self, ollama_url: str = "http://127.0.0.1:11434"):
        self.ollama_url = ollama_url
        self.model_name = "qwen2.5:7b"

    def _call_ollama(self, system_prompt: str, user_prompt: str, json_format: bool = False) -> str:
        url = f"{self.ollama_url}/api/chat"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
        
        if json_format:
            payload["format"] = "json"
            
        data = json.dumps(payload).encode("utf-8")
        
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["message"]["content"]
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e.reason}. Make sure Ollama is running.")
        except Exception as e:
            raise RuntimeError(f"Error calling Ollama API: {str(e)}")

    def analyze_meeting(self, transcript: Transcript) -> MeetingSummary:
        """
        Analyzes the transcript and extracts summary, decisions, and action items using local Qwen2.5 model.
        """
        # Format the transcript text for the LLM
        formatted_transcript = ""
        for seg in transcript.segments:
            formatted_transcript += f"[{seg.start_time} - {seg.end_time}] {seg.speaker}: {seg.text}\n"

        system_prompt = (
            "شما یک دستیار هوش مصنوعی هوشمند برای تحلیل جلسات اداری و سازمانی هستید. "
            "باید خروجی خود را دقیقاً به صورت یک معتبر JSON با ساختار زیر ارسال کنید. "
            "پاسخ شما نباید شامل هیچ متن اضافی، مقدمه، مؤخره یا مارک‌داون باشد؛ فقط و فقط خود JSON خام را ارسال کنید. "
            "ساختار JSON مورد انتظار:\n"
            "{\n"
            '  "summary_text": "یک خلاصه بسیار دقیق، روان و منسجم در حدود یک صفحه از موضوعات بحث شده در جلسه به زبان فارسی.",\n'
            '  "key_points": ["نکته کلیدی اول", "نکته کلیدی دوم", ...],\n'
            '  "decisions": [\n'
            '    {"decision": "عنوان تصمیم اتخاذ شده", "context": "توضیح کوتاه در مورد دلیل یا زمینه این تصمیم"}\n'
            '  ],\n'
            '  "action_items": [\n'
            '    {"task": "عنوان کار یا وظیفه‌ای که باید انجام شود", "assignee": "نام دقیق شخص مسئول کار", "deadline": "مهلت مشخص شده (یا null)", "priority": "اولویت شامل High یا Normal یا Low"}\n'
            '  ]\n'
            "}"
        )

        user_prompt = (
            f"لطفاً رونوشت جلسه زیر را با دقت بررسی و تحلیل کرده و ساختار درخواستی را استخراج کنید:\n\n"
            f"رونوشت جلسه:\n{formatted_transcript}"
        )

        # Call the local Ollama with JSON mode enabled
        raw_response = self._call_ollama(system_prompt, user_prompt, json_format=True)
        
        try:
            result_json = json.loads(raw_response)
            
            # Map JSON to Pydantic objects
            decisions = [
                Decision(decision=d.get("decision", ""), context=d.get("context"))
                for d in result_json.get("decisions", [])
            ]
            
            action_items = [
                ActionItem(
                    task=a.get("task", ""),
                    assignee=a.get("assignee") or "",
                    deadline=a.get("deadline"),
                    priority=a.get("priority", "Normal")
                )
                for a in result_json.get("action_items", [])
            ]
            
            return MeetingSummary(
                meeting_id=transcript.meeting_id,
                summary_text=result_json.get("summary_text", ""),
                key_points=result_json.get("key_points", []),
                decisions=decisions,
                action_items=action_items
            )
            
        except json.JSONDecodeError:
            raise RuntimeError(f"Ollama returned invalid JSON: {raw_response}")
        except Exception as e:
            print(f"[Intelligence] Validation failed for Ollama response!")
            print(f"[Intelligence] Raw Response: {raw_response}")
            print(f"[Intelligence] Error details: {e}")
            raise ValueError(f"AI response validation error: {str(e)}")

    def extract_insights(self, transcript: Transcript) -> StrategicInsights:
        """
        Extracts strategic insights (risks, opportunities, recommendations) from the transcript.
        """
        formatted_transcript = ""
        for seg in transcript.segments:
            formatted_transcript += f"[{seg.start_time} - {seg.end_time}] {seg.speaker}: {seg.text}\n"

        system_prompt = (
            "شما یک مشاور ارشد مدیریت و هوش مصنوعی هستید. باید رونوشت جلسه را تحلیل کرده و ریسک‌ها، فرصت‌ها "
            "و توصیه‌های راهبردی هوشمند برای مدیرعامل استخراج کنید. "
            "باید خروجی خود را دقیقاً به صورت یک معتبر JSON با ساختار زیر ارسال کنید. "
            "پاسخ شما نباید شامل هیچ متن اضافی، مقدمه، مؤخره یا مارک‌داون باشد؛ فقط و فقط خود JSON خام را ارسال کنید. "
            "ساختار JSON مورد انتظار:\n"
            "{\n"
            '  "risks": ["ریسک یا تهدید شناسایی شده اول به زبان فارسی", ...],\n'
            '  "opportunities": ["فرصت شناسایی شده اول به زبان فارسی", ...],\n'
            '  "recommendations": ["توصیه عملیاتی و راهبردی اول برای مدیرعامل به زبان فارسی", ...]\n'
            "}"
        )

        user_prompt = (
            f"لطفاً رونوشت جلسه زیر را بررسی کرده و ریسک‌ها، فرصت‌ها و توصیه‌ها را استخراج کنید:\n\n"
            f"رونوشت جلسه:\n{formatted_transcript}"
        )

        raw_response = self._call_ollama(system_prompt, user_prompt, json_format=True)
        try:
            result_json = json.loads(raw_response)
            return StrategicInsights(
                risks=result_json.get("risks", []),
                opportunities=result_json.get("opportunities", []),
                recommendations=result_json.get("recommendations", [])
            )
        except json.JSONDecodeError:
            raise RuntimeError(f"Ollama returned invalid JSON for insights: {raw_response}")

