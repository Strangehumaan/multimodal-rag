from vectordb.store import get_collection
from embeddings.embedder import embed_query
from config import TOP_K_RESULTS

# Non-text modalities get a guaranteed slot in results if they score above this
MODALITY_BOOST_THRESHOLD = 0.3


def retrieve_relevant_chunks(query: str) -> list[dict]:
    """
    Public entry point. Embeds the query, searches ChromaDB,
    and returns the top matching chunks with modality boosting applied.

    Modality boosting ensures audio and image chunks are never drowned
    out by a large PDF — each modality gets at least one slot if it
    scores above the relevance threshold.
    """
    query_vector = embed_query(query)
    collection   = get_collection()

    # Fetch more candidates than needed so boosting has chunks to pick from
    candidate_count = TOP_K_RESULTS * 6

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(candidate_count, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    all_chunks = _format_results(results)
    return _apply_modality_boost(all_chunks)


def _apply_modality_boost(chunks: list[dict]) -> list[dict]:
    """
    Guarantee at least one chunk per modality (audio, image, text) if that
    modality has any chunk scoring above MODALITY_BOOST_THRESHOLD.
    Remaining slots are filled by top scoring chunks regardless of modality.
    """
    # Separate chunks by modality
    by_modality: dict[str, list[dict]] = {}
    for chunk in chunks:
        modality = chunk["modality"]
        if modality not in by_modality:
            by_modality[modality] = []
        by_modality[modality].append(chunk)

    boosted    = []
    used_ids   = set()

    # Give each non-text modality its guaranteed best slot first
    for modality in ["audio", "image"]:
        if modality not in by_modality:
            continue
        best = by_modality[modality][0]  # Already sorted by score descending
        if best["score"] >= MODALITY_BOOST_THRESHOLD:
            boosted.append(best)
            used_ids.add(_chunk_id(best))

    # Fill remaining slots with top scoring chunks (any modality)
    remaining_slots = TOP_K_RESULTS - len(boosted)
    for chunk in chunks:
        if remaining_slots == 0:
            break
        if _chunk_id(chunk) not in used_ids:
            boosted.append(chunk)
            used_ids.add(_chunk_id(chunk))
            remaining_slots -= 1

    return boosted


def _chunk_id(chunk: dict) -> str:
    """Build a unique identifier for a chunk to track it across sets."""
    return f"{chunk['source_file']}_{chunk['modality']}_{chunk['chunk_index']}"


def _format_results(raw_results: dict) -> list[dict]:
    """
    Flatten ChromaDB's nested result structure into a clean list of chunk dicts.
    ChromaDB returns lists-of-lists because it supports multi-query — we sent one query.
    """
    documents = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    return [
        {
            "text":        document,
            "source_file": metadata["source_file"],
            "modality":    metadata["modality"],
            "chunk_index": metadata["chunk_index"],
            "score":       round(1 - distance, 4),
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]
