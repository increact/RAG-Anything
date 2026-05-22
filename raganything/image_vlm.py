"""
VLM image description for standalone image uploads.

Standalone — does not depend on LightRAG or the modal processors. Used by the
upload paths to attach a natural-language description of an uploaded image to
the parse result.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp",
})


def is_image_file(filename: str) -> bool:
    """True when `filename` has a recognised image extension."""
    if not filename:
        return False
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS
