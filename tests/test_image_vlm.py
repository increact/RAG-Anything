"""Tests for raganything.image_vlm — VLM image description helpers.

Mocks vision_model_func; no network, no real VLM call. Uses
pytest.importorskip for the raganything package, like the sibling test files.
"""
import pytest

image_vlm = pytest.importorskip("raganything.image_vlm")


@pytest.mark.parametrize("filename", [
    "photo.jpg", "PHOTO.JPG", "scan.jpeg", "diagram.png",
    "old.bmp", "fax.tiff", "fax.tif", "anim.gif", "pic.webp",
])
def test_is_image_file_accepts_image_extensions(filename):
    assert image_vlm.is_image_file(filename) is True


@pytest.mark.parametrize("filename", [
    "report.pdf", "notes.docx", "data.xlsx", "readme.md",
    "noextension", "", "archive.zip",
])
def test_is_image_file_rejects_non_images(filename):
    assert image_vlm.is_image_file(filename) is False


async def test_describe_image_returns_description(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG fake image bytes")
    calls = []

    async def fake_vision(prompt, image_data=None, system_prompt=None, **kwargs):
        calls.append({"prompt": prompt, "image_data": image_data,
                      "system_prompt": system_prompt})
        return "  A red square on a white background.  "

    result = await image_vlm.describe_image(str(img), fake_vision)
    assert result == "A red square on a white background."
    assert len(calls) == 1
    assert calls[0]["image_data"]        # base64 string was passed
    assert calls[0]["system_prompt"]     # system prompt was passed


async def test_describe_image_none_func_returns_empty(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"data")
    assert await image_vlm.describe_image(str(img), None) == ""


async def test_describe_image_graceful_on_vlm_error(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"data")

    async def failing_vision(prompt, **kwargs):
        raise RuntimeError("VLM API down")

    assert await image_vlm.describe_image(str(img), failing_vision) == ""


async def test_describe_image_oversized_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_IMAGE_SIZE_MB", "0")
    img = tmp_path / "x.png"
    img.write_bytes(b"some bytes")

    async def fake_vision(prompt, **kwargs):
        return "should not be reached"

    assert await image_vlm.describe_image(str(img), fake_vision) == ""


def test_build_vision_model_func_none_without_key(monkeypatch):
    monkeypatch.delenv("VISION_BINDING_API_KEY", raising=False)
    assert image_vlm.build_vision_model_func() is None


def test_build_vision_model_func_returns_callable_with_key(monkeypatch):
    monkeypatch.setenv("VISION_BINDING_API_KEY", "test-key")
    func = image_vlm.build_vision_model_func()
    assert callable(func)
