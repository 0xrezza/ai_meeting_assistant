import json
import os
import urllib.request
import urllib.error
from typing import List, Dict, Any, Tuple

class SimpleVectorDB:
    def __init__(self, storage_path: str = "vector_store.json", ollama_url: str = "http://127.0.0.1:11434"):
        self.storage_path = storage_path
        self.ollama_url = ollama_url
        self.model_name = "nomic-embed-text"
        self.documents = []
        self.load()

    def _get_embedding(self, text: str) -> List[float]:
        url = f"{self.ollama_url}/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text
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
                return res_data["embedding"]
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama connection error: {e.reason}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    def add_segments(self, meeting_id: str, segments: List[Dict[str, Any]]):
        """
        Embeds and adds transcript segments to the vector store.
        """
        for i, seg in enumerate(segments):
            text_to_embed = f"{seg['speaker']}: {seg['text']}"
            print(f"Embedding segment {i+1}/{len(segments)}...")
            try:
                embedding = self._get_embedding(text_to_embed)
                doc = {
                    "meeting_id": meeting_id,
                    "segment_index": i,
                    "speaker": seg["speaker"],
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "text": seg["text"],
                    "embedding": embedding
                }
                self.documents.append(doc)
            except Exception as e:
                print(f"Skipped segment {i+1} due to error: {e}")
        self.save()

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        mag1 = sum(a * a for a in v1) ** 0.5
        mag2 = sum(a * a for a in v2) ** 0.5
        if not mag1 or not mag2:
            return 0.0
        return dot_product / (mag1 * mag2)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches for the most similar segments.
        """
        query_embedding = self._get_embedding(query)
        results = []
        for doc in self.documents:
            sim = self._cosine_similarity(query_embedding, doc["embedding"])
            results.append((doc, sim))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except Exception as e:
                print(f"Failed to load vector store: {e}")
                self.documents = []
        else:
            self.documents = []
