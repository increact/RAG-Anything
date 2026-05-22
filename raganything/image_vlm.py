"""
VLM image description for standalone image uploads.

Standalone — does not depend on LightRAG or the modal processors. Used by the
upload paths to attach a natural-language description of an uploaded image to
the parse result.
"""
import logging
import os
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


def build_vision_model_func():
    """Build a vision_model_func for image VLM calls, or return None.

    Returns None when VISION_BINDING_API_KEY is unset — the feature then stays
    inert. The vision provider is configured independently of the main LLM and
    defaults to OpenRouter.
    """
    vision_key = os.getenv("VISION_BINDING_API_KEY")
    if not vision_key:
        logger.info("VISION_BINDING_API_KEY not set — image VLM disabled")
        return None
    vision_host = os.getenv("VISION_BINDING_HOST", "https://openrouter.ai/api/v1")
    vision_model = os.getenv("VISION_MODEL", "openai/gpt-4o-mini")

    def vision_model_func(prompt, system_prompt=None, history_messages=None,
                          image_data=None, messages=None, **kwargs):
        # Lazy import so importing this module does not require lightrag.
        from lightrag.llm.openai import openai_complete_if_cache

        if messages:
            return openai_complete_if_cache(
                vision_model, "", system_prompt=None, history_messages=[],
                messages=messages, api_key=vision_key, base_url=vision_host,
                **kwargs,
            )
        if image_data:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                ],
            })
            return openai_complete_if_cache(
                vision_model, "", system_prompt=None, history_messages=[],
                messages=msgs, api_key=vision_key, base_url=vision_host, **kwargs,
            )
        return openai_complete_if_cache(
            vision_model, prompt, system_prompt=system_prompt,
            history_messages=history_messages or [], api_key=vision_key,
            base_url=vision_host, **kwargs,
        )

    logger.info("Image VLM enabled: model=%s host=%s", vision_model, vision_host)
    return vision_model_func
