import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the user's question using ONLY the sources provided below. "
    "For every claim you make, cite the source using the format [filename | modality]. "
    "If the answer cannot be found in the provided sources, say: "
    "'I could not find an answer in the provided documents.'"
)


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered source list for the prompt."""
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        header = f"SOURCE {index} [{chunk['source_file']} | {chunk['modality']}]:"
        lines.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(lines)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Build a grounded prompt from retrieved chunks and call Claude.
    Returns the answer text and the source list used to generate it.
    """
    context_block = build_context_block(chunks)

    user_message = (
        f"{context_block}\n\n"
        f"QUESTION: {query}"
    )

    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    return {
        "answer":  response.content[0].text,
        "sources": _summarise_sources(chunks),
    }


def _summarise_sources(chunks: list[dict]) -> list[dict]:
    """Build a clean deduplicated source list from the retrieved chunks."""
    seen = set()
    sources = []

    for chunk in chunks:
        key = (chunk["source_file"], chunk["modality"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "file":     chunk["source_file"],
                "modality": chunk["modality"],
                "score":    chunk["score"],
            })

    return sources
