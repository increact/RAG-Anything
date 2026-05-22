"""
VLM image description for standalone image uploads.

Standalone — does not depend on LightRAG or the modal processors. Used by the
upload paths to attach a natural-language description of an uploaded image to
the parse result.
"""
import logging
from pathlib import Path

from raganything.utils import encode_image_to_base64

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp",
})


def is_image_file(filename: str) -> bool:
    """True when `filename` has a recognised image extension."""
    if not filename:
        return False
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


_SYSTEM_PROMPT = (
    "You are an assistant that describes images accurately and concisely. "
    "Describe only what is visible; do not speculate."
)
_USER_PROMPT = (
    "Describe this image in detail. Cover any visible text, objects, people, "
    "charts, diagrams, layout, and the overall context or purpose of the image."
)


async def describe_image(image_path: str, vision_model_func) -> str:
    """Return a VLM-generated description of the image, or "" on any failure.

    Never raises — a failed description must not fail the parse request.
    `encode_image_to_base64` already enforces the MAX_IMAGE_SIZE_MB cap and
    returns "" on oversize / read failure.
    """
    if vision_model_func is None:
        return ""
    image_b64 = encode_image_to_base64(image_path)
    if not image_b64:
        return ""
    try:
        result = await vision_model_func(
            _USER_PROMPT,
            image_data=image_b64,
            system_prompt=_SYSTEM_PROMPT,
        )
        return (result or "").strip()
    except Exception as e:
        logger.warning("VLM image description failed for %s: %s", image_path, e)
        return ""
