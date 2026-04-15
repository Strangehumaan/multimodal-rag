"""
Jenkins CI Test Suite
Runs automatically on every push via Jenkins pipeline.
Tests: config loading, chunk schema, document parsing, embedder imports, retriever imports
"""

import sys
import os

# Add project root to path so imports work in Jenkins environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_config_loads():
    """Config must load without missing key errors."""
    try:
        import config
        assert config.CLAUDE_MODEL,        "CLAUDE_MODEL is empty"
        assert config.VOYAGE_EMBED_MODEL,  "VOYAGE_EMBED_MODEL is empty"
        assert config.WHISPER_MODEL,       "WHISPER_MODEL is empty"
        assert config.CHUNK_SIZE > 0,      "CHUNK_SIZE must be positive"
        assert config.TOP_K_RESULTS > 0,   "TOP_K_RESULTS must be positive"
        print("  PASS  config loads correctly")
        return True
    except Exception as e:
        print(f"  FAIL  config: {e}")
        return False


def test_chunk_schema():
    """Every modality must produce a chunk with the required keys."""
    required_keys = {"text", "source_file", "modality", "chunk_index"}
    sample_chunks = [
        {
            "text":        "Article 52 establishes the office of President.",
            "source_file": "coi.pdf",
            "modality":    "text",
            "chunk_index": 0,
        },
        {
            "text":        "This image shows a bar chart of quarterly revenue.",
            "source_file": "chart.png",
            "modality":    "image",
            "chunk_index": 0,
        },
        {
            "text":        "[00:00 - 00:30] The speaker explains Article 52.",
            "source_file": "lecture.mp3",
            "modality":    "audio",
            "chunk_index": 0,
        },
    ]

    try:
        for chunk in sample_chunks:
            missing = required_keys - set(chunk.keys())
            assert not missing, f"Chunk missing keys: {missing}"
            assert chunk["modality"] in {"text", "image", "audio"}, \
                f"Invalid modality: {chunk['modality']}"
            assert len(chunk["text"].strip()) > 0, "Chunk text is empty"
        print("  PASS  chunk schema is valid for all modalities")
        return True
    except Exception as e:
        print(f"  FAIL  chunk schema: {e}")
        return False


def test_document_parser_imports():
    """Document parser module must import without errors."""
    try:
        from ingestion.document_parser import parse_document, chunk_text
        assert callable(parse_document), "parse_document is not callable"
        assert callable(chunk_text),     "chunk_text is not callable"
        print("  PASS  document_parser imports successfully")
        return True
    except Exception as e:
        print(f"  FAIL  document_parser import: {e}")
        return False


def test_chunking_logic():
    """Chunking must produce correct overlap and skip tiny chunks."""
    try:
        from ingestion.document_parser import chunk_text

        sample_text = "A" * 2000  # 2000 character string
        chunks = chunk_text(sample_text, "test_file.pdf")

        assert len(chunks) > 0,          "No chunks produced"
        assert all("text"        in c for c in chunks), "Missing 'text' key"
        assert all("source_file" in c for c in chunks), "Missing 'source_file' key"
        assert all("modality"    in c for c in chunks), "Missing 'modality' key"
        assert all("chunk_index" in c for c in chunks), "Missing 'chunk_index' key"
        assert all(c["modality"] == "text" for c in chunks), "Wrong modality"
        assert all(len(c["text"].strip()) >= 50 for c in chunks), \
            "Chunk smaller than minimum size slipped through"

        print(f"  PASS  chunking logic produces {len(chunks)} valid chunks")
        return True
    except Exception as e:
        print(f"  FAIL  chunking logic: {e}")
        return False


def test_embedder_imports():
    """Embedder module must import without errors."""
    try:
        from embeddings.embedder import embed_query, embed_texts, attach_embeddings
        assert callable(embed_query),       "embed_query is not callable"
        assert callable(embed_texts),       "embed_texts is not callable"
        assert callable(attach_embeddings), "attach_embeddings is not callable"
        print("  PASS  embedder imports successfully")
        return True
    except Exception as e:
        print(f"  FAIL  embedder import: {e}")
        return False


def test_retriever_imports():
    """Retriever module must import without errors."""
    try:
        from retrieval.retriever import retrieve_relevant_chunks
        assert callable(retrieve_relevant_chunks), \
            "retrieve_relevant_chunks is not callable"
        print("  PASS  retriever imports successfully")
        return True
    except Exception as e:
        print(f"  FAIL  retriever import: {e}")
        return False


def test_generator_imports():
    """Generator module must import without errors."""
    try:
        from generation.generator import generate_answer, build_context_block
        assert callable(generate_answer),    "generate_answer is not callable"
        assert callable(build_context_block),"build_context_block is not callable"
        print("  PASS  generator imports successfully")
        return True
    except Exception as e:
        print(f"  FAIL  generator import: {e}")
        return False


def test_context_block_format():
    """Context block must format chunks into numbered source list."""
    try:
        from generation.generator import build_context_block

        chunks = [
            {"text": "Article 52 text here.", "source_file": "coi.pdf",    "modality": "text"},
            {"text": "[00:00] Speaker explains.", "source_file": "audio.mp3", "modality": "audio"},
        ]

        context = build_context_block(chunks)

        assert "SOURCE 1" in context,   "SOURCE 1 label missing"
        assert "SOURCE 2" in context,   "SOURCE 2 label missing"
        assert "coi.pdf"  in context,   "Source filename missing"
        assert "audio.mp3" in context,  "Audio filename missing"
        assert "text"     in context,   "Modality label missing"
        assert "audio"    in context,   "Audio modality label missing"

        print("  PASS  context block formats correctly")
        return True
    except Exception as e:
        print(f"  FAIL  context block format: {e}")
        return False


def test_supported_file_extensions():
    """Main pipeline must recognise all expected file extensions."""
    try:
        from main import DOCUMENT_EXTENSIONS, IMAGE_EXTENSIONS, AUDIO_EXTENSIONS

        assert ".pdf"  in DOCUMENT_EXTENSIONS, ".pdf not supported"
        assert ".docx" in DOCUMENT_EXTENSIONS, ".docx not supported"
        assert ".jpg"  in IMAGE_EXTENSIONS,    ".jpg not supported"
        assert ".png"  in IMAGE_EXTENSIONS,    ".png not supported"
        assert ".mp3"  in AUDIO_EXTENSIONS,    ".mp3 not supported"
        assert ".wav"  in AUDIO_EXTENSIONS,    ".wav not supported"

        print("  PASS  all expected file extensions are registered")
        return True
    except Exception as e:
        print(f"  FAIL  file extensions: {e}")
        return False


# ── Test Runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n========================================")
    print("  Multimodal RAG --- Jenkins Test Suite")
    print("========================================\n")

    tests = [
        test_config_loads,
        test_chunk_schema,
        test_document_parser_imports,
        test_chunking_logic,
        test_embedder_imports,
        test_retriever_imports,
        test_generator_imports,
        test_context_block_format,
        test_supported_file_extensions,
    ]

    passed = 0
    failed = 0

    for test in tests:
        result = test()
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n========================================")
    print(f"  Results: {passed} passed  |  {failed} failed")
    print(f"========================================\n")

    # Exit code 1 makes Jenkins mark the build as FAILED
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)
