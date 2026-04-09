import os
import base64
import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

SUPPORTED_FORMATS = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

DESCRIPTION_PROMPT = (
    "Describe this image in detail. "
    "If it contains charts, tables, or diagrams, describe the data and labels precisely. "
    "If it contains text, transcribe it. "
    "Be thorough — your description will be used to answer questions about this image."
)


def load_image_as_base64(file_path: str) -> tuple[str, str]:
    """
    Read an image file from disk and return (base64_string, media_type).
    Raises ValueError for unsupported formats.
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {extension}")

    media_type = SUPPORTED_FORMATS[extension]

    with open(file_path, "rb") as image_file:
        base64_string = base64.standard_b64encode(image_file.read()).decode("utf-8")

    return base64_string, media_type


def describe_image_with_claude(base64_string: str, media_type: str) -> str:
    """Send a base64 image to Claude Vision and return its text description."""
    message = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": media_type,
                            "data":       base64_string,
                        },
                    },
                    {
                        "type": "text",
                        "text": DESCRIPTION_PROMPT,
                    },
                ],
            }
        ],
    )
    return message.content[0].text


def process_image(file_path: str) -> dict:
    """
    Public entry point. Loads an image, describes it with Claude Vision,
    and returns a single chunk dict matching the format of document chunks.
    """
    base64_string, media_type = load_image_as_base64(file_path)
    description = describe_image_with_claude(base64_string, media_type)

    return {
        "text":        description,
        "source_file": os.path.basename(file_path),
        "modality":    "image",
        "chunk_index": 0,
    }
