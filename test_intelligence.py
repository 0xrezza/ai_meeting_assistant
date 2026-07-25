import sys
# Reconfigure stdout to use UTF-8 to prevent encoding issues on Windows
sys.stdout.reconfigure(encoding='utf-8')

import json
from test_ingestion import extract_text_from_docx, parse_transcript
from app.services.meeting_intelligence import MeetingIntelligenceService

def main():
    docx_path = "transcript sample.docx"
    print("Reading and parsing transcript...")
    raw_text = extract_text_from_docx(docx_path)
    
    if not raw_text:
        print("Failed to read transcript file.")
        return
        
    transcript = parse_transcript(raw_text)
    print(f"Parsed {len(transcript.segments)} segments.")
    
    if len(transcript.segments) == 0:
        print("No segments parsed. Check transcript parsing logic.")
        return
        
    print("\nCalling Ollama with qwen2.5:7b for meeting intelligence analysis...")
    service = MeetingIntelligenceService()
    
    try:
        summary = service.analyze_meeting(transcript)
        
        print("\n================ MEETING SUMMARY ================")
        print(summary.summary_text)
        print("\n================ KEY POINTS ================")
        for pt in summary.key_points:
            print(f"- {pt}")
            
        print("\n================ DECISIONS ================")
        for d in summary.decisions:
            print(f"- Decision: {d.decision}")
            if d.context:
                print(f"  Context: {d.context}")
                
        print("\n================ ACTION ITEMS ================")
        for a in summary.action_items:
            print(f"- Task: {a.task}")
            print(f"  Assignee: {a.assignee}")
            print(f"  Deadline: {a.deadline}")
            print(f"  Priority: {a.priority}")
            
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
