import os
from openai import OpenAI
from config import OPENAI_API_KEY, WHISPER_MODEL

SUPPORTED_FORMATS = {".mp3", ".mp4", ".wav", ".m4a", ".webm", ".ogg"}

# Group Whisper segments into blocks of this many seconds
AUDIO_CHUNK_SECONDS = 30

_client = OpenAI(api_key=OPENAI_API_KEY)


def transcribe_audio_file(file_path: str) -> list[dict]:
    """
    Send audio file to Whisper API and return list of timed segments.
    Each segment is a dict with: text, start (seconds), end (seconds).
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {extension}")

    with open(file_path, "rb") as audio_file:
        response = _client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
            response_format="verbose_json",  # Returns per-segment timestamps
        )

    return [
        {
            "text":  segment.text.strip(),
            "start": segment.start,
            "end":   segment.end,
        }
        for segment in response.segments
    ]


def group_segments_into_chunks(
    segments: list[dict],
    source_file: str
) -> list[dict]:
    """
    Group Whisper segments into fixed-duration time blocks.
    Each block becomes one chunk with a timestamp label in its text.
    """
    chunks = []
    current_texts = []
    current_start = None
    chunk_index = 0

    for segment in segments:
        if current_start is None:
            current_start = segment["start"]

        current_texts.append(segment["text"])
        elapsed = segment["end"] - current_start

        # Flush the current block when it reaches AUDIO_CHUNK_SECONDS
        if elapsed >= AUDIO_CHUNK_SECONDS:
            chunks.append(
                _build_audio_chunk(
                    texts=current_texts,
                    start=current_start,
                    end=segment["end"],
                    source_file=source_file,
                    chunk_index=chunk_index,
                )
            )
            current_texts = []
            current_start = None
            chunk_index += 1

    # Flush any remaining segments that didn't fill a full block
    if current_texts and current_start is not None:
        last_end = segments[-1]["end"]
        chunks.append(
            _build_audio_chunk(
                texts=current_texts,
                start=current_start,
                end=last_end,
                source_file=source_file,
                chunk_index=chunk_index,
            )
        )

    return chunks


def _build_audio_chunk(
    texts: list[str],
    start: float,
    end: float,
    source_file: str,
    chunk_index: int,
) -> dict:
    """Assemble a single audio chunk dict from a group of segments."""
    timestamp = f"[{_format_time(start)} - {_format_time(end)}]"
    combined_text = " ".join(texts)

    return {
        "text":        f"{timestamp} {combined_text}",
        "source_file": os.path.basename(source_file),
        "modality":    "audio",
        "chunk_index": chunk_index,
    }


def _format_time(seconds: float) -> str:
    """Convert raw seconds to MM:SS string for human-readable timestamps."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def process_audio(file_path: str) -> list[dict]:
    """
    Public entry point. Transcribes an audio file and returns
    a list of timed chunks matching the format of all other modalities.
    """
    segments = transcribe_audio_file(file_path)
    return group_segments_into_chunks(segments, file_path)
