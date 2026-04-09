import time
import voyageai
from config import VOYAGE_API_KEY, VOYAGE_EMBED_MODEL

# Voyage AI recommends batches of 128 or fewer inputs per request
BATCH_SIZE = 128

# Retry settings for connection errors
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

_client = voyageai.Client(api_key=VOYAGE_API_KEY)


def _embed_with_retry(texts: list[str], input_type: str) -> list[list[float]]:
    """
    Call Voyage AI embed with automatic retries on connection errors.
    Raises the final error if all retries are exhausted.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = _client.embed(texts, model=VOYAGE_EMBED_MODEL, input_type=input_type)
            return result.embeddings
        except Exception as error:
            last_error = error
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)  # 2s, 4s, 6s backoff

    raise last_error


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Send a list of strings to Voyage AI in batches.
    Returns a matching list of embedding vectors.
    """
    all_embeddings = []

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start : batch_start + BATCH_SIZE]
        embeddings = _embed_with_retry(batch, input_type="document")
        all_embeddings.extend(embeddings)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single search query.
    Uses input_type='query' — Voyage AI optimizes differently for queries vs documents.
    """
    embeddings = _embed_with_retry([query], input_type="query")
    return embeddings[0]


def attach_embeddings(chunks: list[dict]) -> list[dict]:
    """
    Public entry point. Takes a list of chunk dicts, adds an 'embedding'
    key to each one, and returns the enriched list.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks
