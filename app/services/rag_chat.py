import json
import urllib.request
import urllib.error
from typing import Dict, Any, List
from app.services.vector_db import SimpleVectorDB

class RAGChatService:
    def __init__(self, vector_db: SimpleVectorDB, ollama_url: str = "http://127.0.0.1:11434"):
        self.vector_db = vector_db
        self.ollama_url = ollama_url
        self.model_name = "qwen2.5:7b"

    def _call_ollama_chat(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
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
            raise RuntimeError(f"Ollama connection error: {e.reason}")
        except Exception as e:
            raise RuntimeError(f"Error calling Ollama: {str(e)}")

    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Answers a user query based on the retrieved segments from the vector store.
        """
        # Retrieve top 4 most relevant segments
        search_results = self.vector_db.search(query, top_k=4)
        
        if not search_results:
            return {
                "answer": "هیچ سندی در دیتابیس برای پاسخگویی یافت نشد.",
                "citations": []
            }

        # Build context
        context_str = ""
        citations = []
        for i, (doc, score) in enumerate(search_results):
            context_str += f"سند [{i+1}]:\nزمان: {doc['start_time']} تا {doc['end_time']}\nگوینده: {doc['speaker']}\nمتن: {doc['text']}\n\n"
            citations.append({
                "source_num": i + 1,
                "speaker": doc["speaker"],
                "start_time": doc["start_time"],
                "end_time": doc["end_time"],
                "text": doc["text"],
                "relevance_score": round(score, 3)
            })

        system_prompt = (
            "شما یک دستیار هوشمند هستید که به سوالات مربوط به جلسات پاسخ می‌دهد. "
            "باید پاسخ خود را دقیقاً و فقط بر اساس مستندات ارائه شده در بخش 'بستر متنی (Context)' تنظیم کنید. "
            "اگر پاسخ سوال در متن موجود نیست، بنویسید 'پاسخ این سوال در رونوشت جلسه یافت نشد.' و از خود اطلاعاتی نسازید. "
            "در طول پاسخ خود، هر زمان به اطلاعاتی از اسناد ارجاع می‌دهید، شماره سند را به صورت [سند X] ذکر کنید."
        )

        user_prompt = (
            f"بستر متنی (Context):\n{context_str}\n"
            f"سوال کاربر: {query}\n\n"
            f"پاسخ به سوال کاربر به زبان فارسی:"
        )

        answer = self._call_ollama_chat(system_prompt, user_prompt)
        
        return {
            "answer": answer,
            "citations": citations
        }
