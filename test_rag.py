import sys
# Reconfigure stdout to use UTF-8 to prevent encoding issues on Windows
sys.stdout.reconfigure(encoding='utf-8')

import os
from test_ingestion import extract_text_from_docx, parse_transcript
from app.services.vector_db import SimpleVectorDB
from app.services.rag_chat import RAGChatService

def main():
    docx_path = "transcript sample.docx"
    storage_path = "vector_store.json"
    
    db = SimpleVectorDB(storage_path=storage_path)
    
    # Check if vector DB is empty (i.e. first run)
    if not db.documents:
        print("Vector DB is empty. Reading transcript and generating embeddings...")
        paragraphs = extract_text_from_docx(docx_path)
        if not paragraphs:
            print("Failed to read docx file.")
            return
        transcript = parse_transcript(paragraphs)
        
        # Convert segments to dict format for the DB
        segments_dict = [
            {
                "speaker": seg.speaker,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "text": seg.text
            }
            for seg in transcript.segments
        ]
        
        # Add to DB (this will call Ollama embeddings)
        db.add_segments(meeting_id="test_meeting_001", segments=segments_dict)
        print(f"Successfully embedded and saved {len(db.documents)} segments.")
    else:
        print(f"Loaded {len(db.documents)} pre-embedded segments from {storage_path}.")
        
    chat_service = RAGChatService(vector_db=db)
    
    queries = [
        "آقای کمالی در مورد پنج سال آینده چه پیش‌بینی می‌کند؟",
        "نظر آقای رضایی در مورد جایگزین شدن انسان‌ها چیست؟"
    ]
    
    for q in queries:
        print(f"\n==================================================")
        print(f"USER QUERY: {q}")
        print(f"==================================================")
        try:
            result = chat_service.answer_question(q)
            print("\nAI RESPONSE:")
            print(result["answer"])
            
            print("\nSOURCES CITED:")
            for cite in result["citations"]:
                print(f"- [Doc {cite['source_num']}] [{cite['start_time']} - {cite['end_time']}] {cite['speaker']}: \"{cite['text']}\" (Score: {cite['relevance_score']})")
        except Exception as e:
            print(f"Error occurred during query: {e}")

if __name__ == "__main__":
    main()
