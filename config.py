import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Fetch a required environment variable, raising clearly if it's missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value


# --- API Keys ---
ANTHROPIC_API_KEY = _require("ANTHROPIC_API_KEY")
VOYAGE_API_KEY    = _require("VOYAGE_API_KEY")
OPENAI_API_KEY    = _require("OPENAI_API_KEY")

# --- Model Names ---
CLAUDE_MODEL        = "claude-opus-4-5"          # Used for answer generation and image description
VOYAGE_EMBED_MODEL  = "voyage-3.5"      # Used for embedding all modalities
WHISPER_MODEL       = "whisper-1"                # Used for audio transcription

# --- ChromaDB ---
CHROMA_DB_PATH       = "chroma_store"            # Local folder where vectors are persisted
CHROMA_COLLECTION    = "multimodal_rag"          # Name of the collection inside ChromaDB

# --- Chunking ---
CHUNK_SIZE           = 500                       # Max characters per text chunk
CHUNK_OVERLAP        = 50                        # Characters shared between adjacent chunks

# --- Retrieval ---
TOP_K_RESULTS        = 5                         # Number of chunks to retrieve per query

# --- Data Directories ---
DOCUMENTS_DIR        = "data/documents"
IMAGES_DIR           = "data/images"
AUDIO_DIR            = "data/audio"