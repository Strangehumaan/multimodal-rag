import os
from pypdf import PdfReader
from docx import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP

# Target this many chunks per document regardless of size
TARGET_CHUNKS = 400


def extract_text_from_pdf(file_path: str) -> str:
    """Read all pages of a PDF and return a single joined text string."""
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_text_from_docx(file_path: str) -> str:
    """Read all paragraphs of a DOCX and return a single joined text string."""
    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def _calculate_chunk_size(text: str) -> tuple[int, int]:
    """
    Dynamically calculate chunk size so large documents don't explode
    into thousands of chunks while small documents stay precise.
    Returns (chunk_size, chunk_overlap).
    """
    estimated_chunks_at_default = len(text) / CHUNK_SIZE

    if estimated_chunks_at_default <= TARGET_CHUNKS:
        # Small document — use default settings for maximum precision
        return CHUNK_SIZE, CHUNK_OVERLAP

    # Large document — scale chunk size up to hit TARGET_CHUNKS
    dynamic_chunk_size = int(len(text) / TARGET_CHUNKS)
    dynamic_overlap    = int(dynamic_chunk_size * 0.10)  # Always 10% overlap
    return dynamic_chunk_size, dynamic_overlap


def chunk_text(text: str, source_file: str) -> list[dict]:
    """
    Slide a window across text using dynamically calculated chunk size.
    Returns a list of chunk dicts each carrying source metadata.
    """
    chunk_size, chunk_overlap = _calculate_chunk_size(text)
    step        = chunk_size - chunk_overlap
    chunks      = []
    chunk_index = 0

    for start in range(0, len(text), step):
        chunk_slice = text[start : start + chunk_size]

        # Skip chunks that are too small to be meaningful
        if len(chunk_slice.strip()) < 50:
            continue

        chunks.append({
            "text":        chunk_slice,
            "source_file": os.path.basename(source_file),
            "modality":    "text",
            "chunk_index": chunk_index,
        })
        chunk_index += 1

    return chunks


def parse_document(file_path: str) -> list[dict]:
    """
    Public entry point. Detects file type, extracts text, returns chunks.
    Supports .pdf and .docx only.
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        raw_text = extract_text_from_pdf(file_path)
    elif extension == ".docx":
        raw_text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported document type: {extension}")

    return chunk_text(raw_text, file_path)
