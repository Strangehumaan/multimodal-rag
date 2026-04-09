"""
Multimodal RAG System — Entry Point

Ingestion pipeline  : parse → embed → store
Query pipeline      : retrieve → generate
"""

import os
from ingestion.document_parser import parse_document
from processing.image_processor import process_image
from processing.audio_transcriber import process_audio
from embeddings.embedder import attach_embeddings
from vectordb.store import save_chunks
from retrieval.retriever import retrieve_relevant_chunks
from generation.generator import generate_answer
from config import DOCUMENTS_DIR, IMAGES_DIR, AUDIO_DIR

DOCUMENT_EXTENSIONS = {".pdf", ".docx"}
IMAGE_EXTENSIONS    = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
AUDIO_EXTENSIONS    = {".mp3", ".mp4", ".wav", ".m4a", ".webm", ".ogg"}


def _list_files(directory: str, extensions: set[str]) -> list[str]:
    """Return full paths of all files in a directory matching the given extensions."""
    if not os.path.exists(directory):
        return []
    return [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if os.path.splitext(filename)[1].lower() in extensions
    ]


def _ingest_documents() -> list[dict]:
    """Parse all PDF and DOCX files into text chunks."""
    chunks = []
    for file_path in _list_files(DOCUMENTS_DIR, DOCUMENT_EXTENSIONS):
        print(f"  Parsing document: {os.path.basename(file_path)}")
        chunks.extend(parse_document(file_path))
    return chunks


def _ingest_images() -> list[dict]:
    """Describe all image files using Claude Vision."""
    chunks = []
    for file_path in _list_files(IMAGES_DIR, IMAGE_EXTENSIONS):
        print(f"  Processing image: {os.path.basename(file_path)}")
        chunks.append(process_image(file_path))
    return chunks


def _ingest_audio() -> list[dict]:
    """Transcribe all audio files using Whisper."""
    chunks = []
    for file_path in _list_files(AUDIO_DIR, AUDIO_EXTENSIONS):
        print(f"  Transcribing audio: {os.path.basename(file_path)}")
        chunks.extend(process_audio(file_path))
    return chunks


def ingest() -> None:
    """
    Full ingestion pipeline.
    Collects chunks from all modalities, embeds them, and stores in ChromaDB.
    """
    print("=== Starting Ingestion ===")

    all_chunks = []
    all_chunks.extend(_ingest_documents())
    all_chunks.extend(_ingest_images())
    all_chunks.extend(_ingest_audio())

    if not all_chunks:
        print("No files found in data/ directories. Add files and try again.")
        return

    print(f"\n  Embedding {len(all_chunks)} chunks...")
    all_chunks = attach_embeddings(all_chunks)

    print("  Saving to ChromaDB...")
    save_chunks(all_chunks)

    print(f"\n=== Ingestion Complete: {len(all_chunks)} chunks stored ===")


def query(user_question: str) -> dict:
    """
    Full query pipeline.
    Retrieves relevant chunks and returns Claude's grounded answer with sources.
    """
    print(f"\n=== Query: {user_question} ===")

    chunks = retrieve_relevant_chunks(user_question)
    result = generate_answer(user_question, chunks)

    print(f"\nAnswer:\n{result['answer']}")
    print("\nSources:")
    for source in result["sources"]:
        print(f"  [{source['modality']}] {source['file']} (score: {source['score']})")

    return result


if __name__ == "__main__":
    ingest()
