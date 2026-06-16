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


# --- detect_image_mime / is_image_by_content: magic-byte sniff -----------

@pytest.mark.parametrize("magic,expected_mime", [
    (b"\xff\xd8\xff\xe0\x00\x10JFIF",          "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8,        "image/png"),
    (b"GIF89a" + b"\x00" * 6,                   "image/gif"),
    (b"GIF87a" + b"\x00" * 6,                   "image/gif"),
    (b"BM" + b"\x00" * 10,                      "image/bmp"),
    (b"RIFF\x00\x00\x00\x00WEBP",               "image/webp"),
    (b"II*\x00" + b"\x00" * 8,                  "image/tiff"),
    (b"MM\x00*" + b"\x00" * 8,                  "image/tiff"),
])
def test_detect_image_mime_returns_mime(tmp_path, magic, expected_mime):
    # File suffix is `.pdf` on purpose — the bug we're fixing was that the
    # async queue path falls back to .pdf when the s3_url has no extension,
    # and the old `mimetypes.guess_type` returned "application/pdf".
    f = tmp_path / "anything.pdf"
    f.write_bytes(magic + b"\x00" * 16)
    assert image_vlm.detect_image_mime(str(f)) == expected_mime
    assert image_vlm.is_image_by_content(str(f)) is True


@pytest.mark.parametrize("payload", [
    b"%PDF-1.7 hello",         # actual PDF
    b"hello world",            # plain text
    b"",                       # empty file
    b"\x00\x01\x02",          # under 4 bytes
])
def test_detect_image_mime_rejects_non_images(tmp_path, payload):
    f = tmp_path / "x.png"     # named .png but actually not an image
    f.write_bytes(payload)
    assert image_vlm.detect_image_mime(str(f)) is None
    assert image_vlm.is_image_by_content(str(f)) is False


def test_detect_image_mime_returns_none_for_missing_file(tmp_path):
    assert image_vlm.detect_image_mime(str(tmp_path / "nope.png")) is None
    assert image_vlm.is_image_by_content(str(tmp_path / "nope.png")) is False


async def test_describe_image_returns_description(tmp_path):
    # Suffix is .pdf on purpose — describe_image must sniff bytes for MIME.
    img = tmp_path / "anything.pdf"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake image bytes" + b"\x00" * 16)
    calls = []

    async def fake_vision(prompt, image_data=None, system_prompt=None,
                          image_mime=None, **kwargs):
        calls.append({"prompt": prompt, "image_data": image_data,
                      "system_prompt": system_prompt, "image_mime": image_mime})
        return "  A red square on a white background.  "

    result = await image_vlm.describe_image(str(img), fake_vision)
    assert result == "A red square on a white background."
    assert len(calls) == 1
    assert calls[0]["image_data"]                   # base64 string was passed
    assert calls[0]["system_prompt"]                # system prompt was passed
    assert calls[0]["image_mime"] == "image/png"    # real MIME type, not jpeg


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
