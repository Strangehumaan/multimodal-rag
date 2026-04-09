import chromadb
from config import CHROMA_DB_PATH, CHROMA_COLLECTION

_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection. Uses cosine similarity for search."""
    return _client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def save_chunks(chunks: list[dict]) -> None:
    """
    Store a list of embedded chunks into ChromaDB.
    Each chunk needs: text, embedding, source_file, modality, chunk_index.
    """
    collection = get_collection()

    ids         = [_make_chunk_id(chunk) for chunk in chunks]
    embeddings  = [chunk["embedding"] for chunk in chunks]
    documents   = [chunk["text"] for chunk in chunks]
    metadatas   = [_extract_metadata(chunk) for chunk in chunks]

    # upsert = insert if new, update if ID already exists
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def _make_chunk_id(chunk: dict) -> str:
    """Build a unique, deterministic ID from source file + modality + index."""
    return f"{chunk['source_file']}_{chunk['modality']}_{chunk['chunk_index']}"


def _extract_metadata(chunk: dict) -> dict:
    """Pull only the metadata fields ChromaDB needs to store alongside the vector."""
    return {
        "source_file": chunk["source_file"],
        "modality":    chunk["modality"],
        "chunk_index": chunk["chunk_index"],
    }
