import sys
# Reconfigure stdout to use UTF-8 to prevent encoding issues on Windows
sys.stdout.reconfigure(encoding='utf-8')

import docx
import re
from app.models.schema import TranscriptSegment, Transcript
from app.nlp.normalizer import PersianNormalizer

def extract_text_from_docx(docx_path: str) -> list:
    try:
        doc = docx.Document(docx_path)
        # Return list of non-empty paragraphs
        return [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    except Exception as e:
        print(f"Error reading docx: {e}")
        return []

def parse_transcript(paragraphs: list) -> Transcript:
    normalizer = PersianNormalizer()
    segments = []
    
    current_time = ""
    current_speaker = ""
    current_text = ""
    
    for line in paragraphs:
        # Match header line containing time and optional speaker, e.g. [00:00 - 00:03] آقای رضایی:
        header_match = re.match(r'^\[(\d{2}:\d{2}\s*-\s*\d{2}:\d{2})\]\s*(.*?):?$', line)
        
        if header_match:
            # Save previous segment if exists
            if current_speaker and current_text.strip():
                times = current_time.split('-')
                segments.append(TranscriptSegment(
                    speaker=current_speaker,
                    start_time=times[0].strip(),
                    end_time=times[1].strip() if len(times)>1 else "",
                    text=normalizer.normalize_text(current_text)
                ))
                current_text = ""
            
            current_time = header_match.group(1)
            speaker = header_match.group(2).strip()
            if speaker.endswith(':'):
                speaker = speaker[:-1].strip()
            current_speaker = normalizer.normalize_text(speaker)
        else:
            # Accumulate text for the current speaker
            current_text += line + " "

    # Add the last segment
    if current_speaker and current_text.strip():
        times = current_time.split('-')
        segments.append(TranscriptSegment(
            speaker=current_speaker,
            start_time=times[0].strip(),
            end_time=times[1].strip() if len(times)>1 else "",
            text=normalizer.normalize_text(current_text)
        ))

    return Transcript(meeting_id="test_meeting_001", segments=segments)

if __name__ == "__main__":
    docx_path = "transcript sample.docx"
    paragraphs = extract_text_from_docx(docx_path)
    
    if paragraphs:
        transcript = parse_transcript(paragraphs)
        print("Transcript parsed successfully!")
        print(f"Total Segments: {len(transcript.segments)}\n")
        for seg in transcript.segments[:3]: # print first 3
            print(f"[{seg.start_time} - {seg.end_time}] {seg.speaker}: {seg.text}")
