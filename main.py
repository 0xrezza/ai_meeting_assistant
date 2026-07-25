import whisper
from pathlib import Path
from docx import Document
from jiwer import wer, cer
from hazm import Normalizer
import ollama
import json
from datetime import datetime
import chromadb
from chromadb.config import Settings


# Load once when application starts
MODEL = whisper.load_model("large-v3")
NORMALIZER = Normalizer()

# Directory for storing transcripts
TRANSCRIPTS_DIR = Path("transcripts")
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

# Initialize ChromaDB for RAG
CHROMA_CLIENT = chromadb.PersistentClient(path=str(TRANSCRIPTS_DIR / "chroma_db"))
COLLECTION = CHROMA_CLIENT.get_or_create_collection(name="voice_transcripts")


def read_docx_text(docx_path: str) -> str:
    """
    Extract text from a DOCX file.
    """
    doc = Document(docx_path)

    text = "\n".join(
        paragraph.text.strip()
        for paragraph in doc.paragraphs
        if paragraph.text.strip()
    )

    return text


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to format:
    00:00.000
    """

    minutes = int(seconds // 60)
    seconds = seconds % 60

    return f"{minutes:02d}:{seconds:06.3f}"


def normalize_text(text: str) -> str:
    """
    Normalize Persian text before evaluation.
    """

    text = NORMALIZER.normalize(text)

    # Remove extra spaces/newlines
    text = " ".join(text.split())

    return text


def correct_text_with_gpt(text: str) -> str:
    """
    Correct text using Qwen2.5 via Ollama to fix word errors.
    Strictly returns only corrected text without any additions.
    """
    
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
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
        options={
            "temperature": 0.1,
        }
    )
    
    corrected_text = response['message']['content'].strip()
    return corrected_text


def is_transcribed(audio_path: str) -> bool:
    """
    Check if a voice recording has already been transcribed.
    Checks for existing transcript JSON file based on audio filename.
    """
    audio_filename = Path(audio_path).stem
    transcript_file = TRANSCRIPTS_DIR / f"{audio_filename}.json"
    return transcript_file.exists()


def save_transcript_with_metadata(
    audio_path: str,
    corrected_text: str,
    title: str | None = None,
    additional_metadata: dict | None = None
) -> str:
    """
    Save transcript with metadata to JSON file and add to ChromaDB.
    
    Parameters
    ----------
    audio_path : str
        Path to the audio file
    corrected_text : str
        The corrected transcript text
    title : str | None
        Title of the recording (defaults to filename)
    additional_metadata : dict | None
        Additional metadata to store
    
    Returns
    -------
    str
        Path to the saved transcript file
    """
    audio_filename = Path(audio_path).stem
    transcript_file = TRANSCRIPTS_DIR / f"{audio_filename}.json"
    
    # Get file creation time
    audio_file = Path(audio_path)
    creation_time = datetime.fromtimestamp(audio_file.stat().st_ctime).isoformat()
    
    # Prepare metadata
    metadata = {
        "audio_filename": audio_filename,
        "audio_path": str(audio_path),
        "title": title or audio_filename,
        "creation_time": creation_time,
        "transcription_time": datetime.now().isoformat(),
        "corrected_text": corrected_text,
    }
    
    if additional_metadata:
        metadata.update(additional_metadata)
    
    # Save to JSON file
    with open(transcript_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Add to ChromaDB for RAG
    COLLECTION.add(
        documents=[corrected_text],
        metadatas=[{
            "title": metadata["title"],
            "audio_filename": audio_filename,
            "creation_time": creation_time,
        }],
        ids=[audio_filename]
    )
    
    return str(transcript_file)


def load_all_transcripts() -> list[dict]:
    """
    Load all transcript metadata from the transcripts directory.
    """
    transcripts = []
    for json_file in TRANSCRIPTS_DIR.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            transcripts.append(json.load(f))
    return transcripts


def batch_transcribe_directory(
    directory: str,
    audio_extensions: list[str] = None,
    reference_dir: str | None = None,
):
    """
    Automatically transcribe all audio files in a directory that haven't been transcribed yet.
    
    Parameters
    ----------
    directory : str
        Path to directory containing audio files
    audio_extensions : list[str]
        List of audio file extensions to process (default: common audio formats)
    reference_dir : str | None
        Optional directory containing reference DOCX files
    
    Returns
    -------
    dict
        Summary of transcription results
    """
    if audio_extensions is None:
        audio_extensions = ['.mp3', '.mp4', '.wav', '.m4a', '.flac', '.ogg']
    
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    audio_files = [
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]
    
    results = {
        "total_files": len(audio_files),
        "transcribed": 0,
        "skipped": 0,
        "failed": 0,
        "details": []
    }
    
    for audio_file in audio_files:
        try:
            # Check if already transcribed
            if is_transcribed(str(audio_file)):
                results["skipped"] += 1
                results["details"].append({
                    "file": str(audio_file),
                    "status": "skipped",
                    "reason": "already transcribed"
                })
                continue
            
            # Look for reference file
            reference_path = None
            if reference_dir:
                ref_dir = Path(reference_dir)
                reference_path = ref_dir / f"{audio_file.stem}.docx"
                if not reference_path.exists():
                    reference_path = None
            
            # Transcribe
            result = transcribe_persian(
                audio_path=str(audio_file),
                reference_docx_path=str(reference_path) if reference_path else None,
                title=audio_file.stem,
                save_transcript=True,
            )
            
            results["transcribed"] += 1
            results["details"].append({
                "file": str(audio_file),
                "status": "success",
                "transcript_file": result["transcript_file"],
            })
            
            print(f"✓ Transcribed: {audio_file.name}")
            
        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "file": str(audio_file),
                "status": "failed",
                "error": str(e),
            })
            print(f"✗ Failed: {audio_file.name} - {e}")
    
    return results


def search_transcripts(query: str, n_results: int = 3) -> list[dict]:
    """
    Search transcripts using RAG to find relevant voice recordings.
    
    Parameters
    ----------
    query : str
        The question or search query
    n_results : int
        Number of results to return
    
    Returns
    -------
    list[dict]
        List of relevant transcripts with metadata
    """
    results = COLLECTION.query(
        query_texts=[query],
        n_results=n_results,
    )
    
    formatted_results = []
    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        document = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        
        # Load full transcript metadata
        transcript_file = TRANSCRIPTS_DIR / f"{doc_id}.json"
        full_metadata = {}
        if transcript_file.exists():
            with open(transcript_file, 'r', encoding='utf-8') as f:
                full_metadata = json.load(f)
        
        formatted_results.append({
            "audio_filename": doc_id,
            "title": metadata.get('title', doc_id),
            "relevant_text": document,
            "distance": distance,
            "full_metadata": full_metadata,
        })
    
    return formatted_results


def ask_about_recordings(question: str, use_rag: bool = True) -> dict:
    """
    Ask a question about voice recordings. Uses RAG to find relevant recordings,
    then uses DeepSeek to answer the question based on the relevant context.
    
    Parameters
    ----------
    question : str
        The question to ask
    use_rag : bool
        Whether to use RAG to find relevant recordings (default: True)
    
    Returns
    -------
    dict
        Answer with source information
    """
    # Find relevant transcripts using RAG
    relevant_transcripts = search_transcripts(question, n_results=3)
    
    if not relevant_transcripts:
        return {
            "answer": "No relevant transcripts found.",
            "sources": [],
        }
    
    # Build context from relevant transcripts
    context_parts = []
    for transcript in relevant_transcripts:
        context_parts.append(
            f"Recording: {transcript['title']} ({transcript['audio_filename']})\n"
            f"Content: {transcript['relevant_text']}\n"
        )
    
    context = "\n---\n".join(context_parts)
    
    # Use Qwen2.5 to answer the question
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on voice recording transcripts. Use the provided context to answer questions accurately. If the context doesn't contain the answer, say so. Always mention which recording(s) your answer is based on."
            },
            {
                "role": "user",
                "content": f"Context from voice recordings:\n{context}\n\nQuestion: {question}"
            }
        ],
        options={
            "temperature": 0.3,
        }
    )
    
    answer = response['message']['content'].strip()
    
    return {
        "answer": answer,
        "sources": [
            {
                "title": t['title'],
                "audio_filename": t['audio_filename'],
                "distance": t['distance'],
            }
            for t in relevant_transcripts
        ],
        "relevant_transcripts": relevant_transcripts,
    }


def calculate_accuracy(reference_text: str, predicted_text: str) -> dict:
    """
    Calculate transcription accuracy.
    """

    reference_text = normalize_text(reference_text)
    predicted_text = normalize_text(predicted_text)

    word_error_rate = wer(reference_text, predicted_text)
    char_error_rate = cer(reference_text, predicted_text)

    return {
        "wer": round(word_error_rate, 4),
        "cer": round(char_error_rate, 4),
        "word_accuracy": round((1 - word_error_rate) * 100, 2),
        "char_accuracy": round((1 - char_error_rate) * 100, 2),
    }


def transcribe_persian(
    audio_path: str,
    reference_docx_path: str | None = None,
    title: str | None = None,
    save_transcript: bool = True,
):
    """
    Transcribe Persian audio/video and optionally evaluate it.

    Parameters
    ----------
    audio_path : str
        Path to mp4, mp3, wav, m4a, etc.

    reference_docx_path : str | None
        Optional path to DOCX transcript.

    title : str | None
        Title of the recording for metadata.

    save_transcript : bool
        Whether to save transcript with metadata for RAG system.

    Returns
    -------
    dict
    """

    if not Path(audio_path).exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    result = MODEL.transcribe(
        audio_path,
        language="fa",
        fp16=False,
        verbose=False,
    )

    full_text = result["text"].strip()

    if not full_text:
        raise ValueError(
            "No text could be extracted from the file."
        )

    segment_lines = []

    confidence_values = []

    for segment in result["segments"]:

        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])

        text = segment["text"].strip()

        segment_lines.append(
            f"[{start} --> {end}]  {text}"
        )

        confidence_values.append(
            segment["avg_logprob"]
        )

    formatted_segments = "\n".join(segment_lines)

    estimated_confidence = round(
        sum(confidence_values) / len(confidence_values),
        4,
    )

    accuracy = None
    corrected_text = None

    if reference_docx_path:

        if not Path(reference_docx_path).exists():
            raise FileNotFoundError(
                f"Reference file not found: {reference_docx_path}"
            )

        reference_text = read_docx_text(
            reference_docx_path
        )

        # Correct transcribed text with GPT before calculating accuracy
        corrected_text = correct_text_with_gpt(full_text)

        accuracy = calculate_accuracy(
            reference_text,
            corrected_text,
        )
    else:
        # Correct text even without reference for RAG system
        corrected_text = correct_text_with_gpt(full_text)

    # Save transcript with metadata for RAG system
    transcript_file = None
    if save_transcript and corrected_text:
        transcript_file = save_transcript_with_metadata(
            audio_path=audio_path,
            corrected_text=corrected_text,
            title=title,
        )

    return {
        "text": full_text,
        "corrected_text_with_qwen": corrected_text,
        "segments": formatted_segments,
        "estimated_confidence": estimated_confidence,
        "accuracy": accuracy,
        "transcript_file": transcript_file,
    }


if __name__ == "__main__":
    import sys
    
    # Example 1: Batch transcribe all audio files in a directory
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        directory = sys.argv[2] if len(sys.argv) > 2 else "."
        print(f"Batch transcribing audio files in: {directory}")
        results = batch_transcribe_directory(directory)
        print(f"\n=== Summary ===")
        print(f"Total files: {results['total_files']}")
        print(f"Transcribed: {results['transcribed']}")
        print(f"Skipped: {results['skipped']}")
        print(f"Failed: {results['failed']}")
    
    # Example 2: Ask a question about recordings
    elif len(sys.argv) > 1 and sys.argv[1] == "ask":
        question = " ".join(sys.argv[2:])
        if not question:
            print("Please provide a question after 'ask'")
            sys.exit(1)
        
        print(f"Question: {question}")
        result = ask_about_recordings(question)
        print(f"\n=== Answer ===\n{result['answer']}")
        print(f"\n=== Sources ===")
        for source in result['sources']:
            print(f"- {source['title']} ({source['audio_filename']})")
    
    # Example 3: Single file transcription (original behavior)
    else:
        result = transcribe_persian(
            audio_path="my voice.mp4",
            reference_docx_path="my voice.docx",
        )

        print("\n===== TIMESTAMPED TRANSCRIPT =====\n")
        print(result["segments"])

        print("\n===== CONFIDENCE =====\n")
        print(result["estimated_confidence"])

        print("\n===== ACCURACY =====\n")
        print(result["accuracy"])

        print("\n===== FULL TEXT =====\n")
        print(result["text"])

        print("\n===== CORRECTED TEXT WITH QWEN =====\n")
        print(result["corrected_text_with_qwen"])
        
        if result["transcript_file"]:
            print(f"\n===== TRANSCRIPT SAVED TO =====\n")
            print(result["transcript_file"])
    