import sys
# Reconfigure stdout to use UTF-8 to prevent encoding issues on Windows
sys.stdout.reconfigure(encoding='utf-8')

import json
from test_ingestion import extract_text_from_docx, parse_transcript
from app.nlp.normalizer import PersianNormalizer
from app.services.meeting_intelligence import MeetingIntelligenceService
from app.services.vector_db import SimpleVectorDB
from app.services.rag_chat import RAGChatService

def main():
    docx_path = "transcript sample.docx"
    
    print("==================================================")
    print("1. تست مرحله اول: بارگذاری و تفکیک متون جلسه (Ingestion)")
    print("==================================================")
    paragraphs = extract_text_from_docx(docx_path)
    if not paragraphs:
        print("خطا در خواندن فایل Word.")
        return
        
    transcript = parse_transcript(paragraphs)
    print(f"تعداد بخش‌های استخراج شده: {len(transcript.segments)}")
    
    print("\n==================================================")
    print("2. تست مرحله دوم: نرمالایزر و پیش‌پردازش متن فارسی")
    print("==================================================")
    normalizer = PersianNormalizer()
    raw_sample = "تقریباً هر هفته خبر جدیدی درباره مدل های پیشرفته منتشر می شود ."
    normalized_sample = normalizer.normalize_text(raw_sample)
    print(f"متن خام: {raw_sample}")
    print(f"متن نرمال شده: {normalized_sample}")
    
    # Analyze meeting using Qwen2.5:7b
    print("\n==================================================")
    print("3. تست مرحله سوم: خلاصه‌سازی هوشمند و استخراج تصمیمات و وظایف")
    print("==================================================")
    intel_service = MeetingIntelligenceService()
    try:
        analysis = intel_service.analyze_meeting(transcript)
        print("\n[ خلاصه جلسه ]:")
        print(analysis.summary_text)
        
        print("\n[ نکات کلیدی ]:")
        for pt in analysis.key_points:
            print(f"- {pt}")
            
        print("\n[ تصمیمات اتخاذ شده ]:")
        for d in analysis.decisions:
            print(f"- {d.decision} (زمینه بحث: {d.context})")
            
        print("\n[ وظایف استخراج شده (Action Items) ]:")
        for a in analysis.action_items:
            print(f"- کار: {a.task} | مسئول: {a.assignee} | اولویت: {a.priority} | مهلت: {a.deadline}")
    except Exception as e:
        print(f"خطا در خلاصه‌سازی: {e}")
        
    # Analyze insights (Risks, Opportunities, Recommendations)
    print("\n==================================================")
    print("4. تست مرحله چهارم: تولید بینش راهبردی و توصیه برای مدیرعامل")
    print("==================================================")
    try:
        insights = intel_service.extract_insights(transcript)
        print("\n[ ریسک‌های شناسایی شده ]:")
        for risk in insights.risks:
            print(f"- {risk}")
            
        print("\n[ فرصت‌های شناسایی شده ]:")
        for opp in insights.opportunities:
            print(f"- {opp}")
            
        print("\n[ توصیه‌های عملیاتی و راهبردی برای مدیرعامل ]:")
        for rec in insights.recommendations:
            print(f"- {rec}")
    except Exception as e:
        print(f"خطا در استخراج بینش‌ها: {e}")
        
    # Semantic Search and RAG QA
    print("\n==================================================")
    print("5. تست مرحله پنجم: دیتابیس برداری و جستجوی معنایی (Vector Search)")
    print("==================================================")
    db = SimpleVectorDB(storage_path="vector_store.json")
    if not db.documents:
        # Load and embed
        segments_dict = [
            {"speaker": s.speaker, "start_time": s.start_time, "end_time": s.end_time, "text": s.text}
            for s in transcript.segments
        ]
        db.add_segments("test_meeting_001", segments_dict)
        
    search_query = "نگرانی بابت شغل"
    print(f"جستجوی عبارت: '{search_query}'")
    search_results = db.search(search_query, top_k=2)
    for doc, score in search_results:
        print(f"- [{doc['start_time']} - {doc['end_time']}] {doc['speaker']}: \"{doc['text']}\" (شباهت: {round(score, 3)})")
        
    print("\n==================================================")
    print("6. تست مرحله ششم: سیستم چت هوشمند (RAG QA) با ارجاع به منبع")
    print("==================================================")
    chat_service = RAGChatService(vector_db=db)
    
    rag_queries = [
        "راهکار مواجهه با حذف مشاغل چیست؟",
        "چه ابزارهایی برای اتوماسیون تکراری ذکر شدند؟"
    ]
    
    for rq in rag_queries:
        print(f"\nسوال کاربر: {rq}")
        try:
            res = chat_service.answer_question(rq)
            print(f"پاسخ هوش مصنوعی:\n{res['answer']}")
            print("منابع مورد استفاده:")
            for cite in res["citations"]:
                print(f"  * [سند {cite['source_num']}] [{cite['start_time']} - {cite['end_time']}] {cite['speaker']}: \"{cite['text']}\"")
        except Exception as e:
            print(f"خطا در چت: {e}")

if __name__ == "__main__":
    main()
