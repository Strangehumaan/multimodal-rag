"""
Run this to find exactly where ingestion is slow.
Usage: python test_speed.py path/to/your/file.pdf
"""

import sys
import time


def timer(label: str):
    """Context manager that prints how long a block took."""
    class _Timer:
        def __enter__(self):
            self.start = time.time()
            print(f"  ⏳ {label}...")
            return self

        def __exit__(self, *args):
            elapsed = time.time() - self.start
            print(f"  ✓  {label} done in {elapsed:.2f}s")

    return _Timer()


def diagnose(file_path: str):
    print(f"\n=== Speed Diagnosis: {file_path} ===\n")

    # Step 1 — Parsing
    with timer("Parsing document"):
        from ingestion.document_parser import parse_document
        chunks = parse_document(file_path)
    print(f"     → {len(chunks)} chunks produced\n")

    # Step 2 — Embedding (this is usually the culprit)
    with timer(f"Embedding {len(chunks)} chunks via Voyage AI"):
        from embeddings.embedder import attach_embeddings
        chunks = attach_embeddings(chunks)
    print(f"     → {len(chunks)} chunks embedded\n")

    # Step 3 — Saving to ChromaDB
    with timer("Saving to ChromaDB"):
        from vectordb.store import save_chunks
        save_chunks(chunks)
    print(f"     → Saved\n")

    print("=== Diagnosis Complete ===")
    print("The slowest step above is your bottleneck.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_speed.py path/to/your/file.pdf")
        sys.exit(1)
    diagnose(sys.argv[1])
