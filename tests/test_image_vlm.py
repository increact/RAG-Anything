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
