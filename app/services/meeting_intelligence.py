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
            "شما یک تحلیل‌گر ارشد و مشاور توسعه کسب‌وکار هستید که جلسات اداری و تصمیم‌گیری‌های سازمانی را تحلیل می‌کند. "
            "وظیفه شما استخراج خلاصه جامع، نکات کلیدی، تصمیمات مهم و وظایف عملیاتی (Action Items) از رونوشت جلسه است.\n\n"
            "پاسخ شما باید دقیقاً یک شیء معتبر JSON با ساختار مشخص‌شده زیر باشد و هیچ‌گونه توضیح اضافی، مقدمه، یا قالب‌بندی مارک‌داون (مانند ```json) قبل یا بعد از آن ارسال نشود.\n\n"
            "دستورالعمل‌های تولید محتوا:\n"
            "۱. خلاصه اجرایی (summary_text): یک خلاصه منسجم، شیوا، حرفه‌ای و ساختاریافته (حدود ۵۰۰ کلمه) به زبان فارسی رسمی بنویسید که اهداف اصلی جلسه، چالش‌های مطرح‌شده و دستاوردهای کلی را پوشش دهد.\n"
            "۲. نکات کلیدی (key_points): موضوعات مهم بحث شده را به صورت جملات کامل، آموزنده و مجزا استخراج کنید (حداقل ۵ مورد).\n"
            "۳. تصمیمات اتخاذ شده (decisions): تصمیمات سرنوشت‌ساز جلسه را به همراه دلیل یا زمینه آن مشخص کنید.\n"
            "۴. وظایف عملیاتی (action_items): وظایف را بسیار مشخص و قابل‌اندازه‌گیری بنویسید. نام مسئول کار (assignee) را دقیقاً از متن استخراج کنید؛ اگر مسئول مشخص نیست، 'نامشخص' بگذارید. اولویت‌ها (priority) را بر اساس فوریت کار روی High، Normal یا Low تنظیم کنید. مهلت انجام کار (deadline) را به صورت متنی (مثلا 'تا پایان هفته'، یا تاریخ شمسی، یا null در صورت عدم ذکر) مشخص کنید.\n\n"
            "ساختار JSON خروجی:\n"
            "{\n"
            '  "summary_text": "خلاصه اجرایی حرفه‌ای...",\n'
            '  "key_points": ["نکته کلیدی اول با جزییات کافی...", "نکته کلیدی دوم..."],\n'
            '  "decisions": [\n'
            '    {"decision": "عنوان تصمیم مشخص و واضح", "context": "زمینه، دلیل اتخاذ یا پیامد این تصمیم"}\n'
            '  ],\n'
            '  "action_items": [\n'
            '    {"task": "شرح دقیق کار عملیاتی", "assignee": "نام فرد مسئول یا \'نامشخص\'", "deadline": "مهلت انجام کار یا null", "priority": "High/Normal/Low"}\n'
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
            "شما یک مشاور مدیریت ارشد، متخصص تحلیل استراتژیک و توسعه کسب‌وکار هستید. "
            "وظیفه شما بررسی موشکافانه گفتگوهای جلسه و استخراج بینش‌های استراتژیک برای مدیرعامل (CEO) است.\n\n"
            "پاسخ شما باید دقیقاً یک شیء معتبر JSON با ساختار مشخص‌شده زیر باشد و هیچ‌گونه توضیح اضافی، مقدمه، یا قالب‌بندی مارک‌داون قبل یا بعد از آن ارسال نشود.\n\n"
            "دستورالعمل‌ها:\n"
            "۱. ریسک‌ها (risks): تهدیدها، نقاط ضعف، کمبود منابع، ریسک‌های مالی، عملیاتی یا زمانی که در جلسه مطرح شده یا به طور غیرمستقیم وجود دارد را با جزییات و به زبان فارسی بنویسید.\n"
            "۲. فرصت‌ها (opportunities): پتانسیل‌های رشد، بازارهای جدید، بهبود فرآیندها و مزایای رقابتی که شرکت می‌تواند از آن‌ها بهره‌مند شود را به صورت شفاف شناسایی کنید.\n"
            "۳. توصیه‌های راهبردی (recommendations): اقدامات مشخص، عملی و بلندمدتی که مدیرعامل باید برای کاهش ریسک‌ها و بهره‌برداری از فرصت‌ها انجام دهد را فرمول‌بندی کنید.\n\n"
            "ساختار JSON خروجی:\n"
            "{\n"
            '  "risks": ["شرح ریسک با جزییات کافی...", ...],\n'
            '  "opportunities": ["شرح فرصت رشد یا بهبود...", ...],\n'
            '  "recommendations": ["توصیه عملی و راهبردی برای مدیرعامل...", ...]\n'
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

