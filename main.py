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


def ingest_with_progress():
    """
    Full ingestion pipeline that yields (status, message, progress) tuples.
    The UI iterates this generator to display live per-file status updates.

    Yields:
        ("progress", message, fraction)  — ongoing step, fraction is 0.0-1.0
        ("warning",  message)            — no files found
        ("done",     message, 1.0)       — ingestion complete
    """
    all_chunks  = []
    doc_files   = _list_files(DOCUMENTS_DIR, DOCUMENT_EXTENSIONS)
    image_files = _list_files(IMAGES_DIR, IMAGE_EXTENSIONS)
    audio_files = _list_files(AUDIO_DIR, AUDIO_EXTENSIONS)
    total_files = len(doc_files) + len(image_files) + len(audio_files)

    if total_files == 0:
        yield ("warning", "No files found in data/ directories.")
        return

    completed = 0

    for file_path in doc_files:
        name = os.path.basename(file_path)
        yield ("progress", f"Parsing document: {name}", completed / total_files)
        all_chunks.extend(parse_document(file_path))
        completed += 1
        yield ("progress", f"Done: {name}", completed / total_files)

    for file_path in image_files:
        name = os.path.basename(file_path)
        yield ("progress", f"Describing image: {name} (calling Claude Vision...)", completed / total_files)
        all_chunks.append(process_image(file_path))
        completed += 1
        yield ("progress", f"Done: {name}", completed / total_files)

    for file_path in audio_files:
        name = os.path.basename(file_path)
        yield ("progress", f"Transcribing audio: {name} (calling Whisper...)", completed / total_files)
        all_chunks.extend(process_audio(file_path))
        completed += 1
        yield ("progress", f"Done: {name}", completed / total_files)

    yield ("progress", f"Embedding {len(all_chunks)} chunks with Voyage AI...", 0.90)
    all_chunks = attach_embeddings(all_chunks)

    yield ("progress", "Saving to ChromaDB...", 0.97)
    save_chunks(all_chunks)

    yield ("done", f"{len(all_chunks)} chunks stored from {total_files} file(s)", 1.0)


def ingest() -> None:
    """CLI entry point — runs ingestion and prints progress to stdout."""
    for update in ingest_with_progress():
        print(update[1])


def query(user_question: str) -> dict:
    """
    Full query pipeline.
    Retrieves relevant chunks and returns Claude's grounded answer with sources.
    """
    chunks = retrieve_relevant_chunks(user_question)
    return generate_answer(user_question, chunks)


if __name__ == "__main__":
    ingest()
