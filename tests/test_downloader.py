"""
tests/test_downloader.py — Unit Tests for utils/downloader.py
==============================================================

Downloading real images from the internet in tests is bad practice:
    - Tests become slow and flaky (depends on network)
    - They hit real CDN servers unnecessarily
    - They cannot run in offline / CI environments

Instead, we use unittest.mock to replace the actual HTTP call with a fake
that returns controlled data. This technique is called "mocking" and is
fundamental to writing reliable unit tests for code that makes network calls.

The key mock target here is `utils.downloader._fetch_image`, which is the
function that actually makes the HTTP request. We replace it with a function
that returns either fake image bytes or None, depending on the test scenario.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image

from utils.database import initialize_db, insert_images, update_download
from utils.downloader import (
    _safe_filename,
    _build_save_path,
    _validate_and_get_dimensions,
    download_all_pending,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_fake_image_bytes(width: int = 400, height: int = 300) -> bytes:
    """
    Create a minimal valid JPEG in memory using Pillow.

    We create a solid-colour image (dark blue) and save it to a BytesIO buffer.
    This gives us real image bytes that Pillow can open and validate.
    """
    img = Image.new("RGB", (width, height), color=(26, 26, 46))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def conn():
    """Fresh in-memory database for each test."""
    connection = initialize_db(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def sample_record():
    """One scraped image record that has not been downloaded yet."""
    return {
        "source": "behance",
        "keyword": "dark minimalism",
        "title": "Test Project",
        "owner": "designer",
        "source_url": "https://www.behance.net/gallery/999",
        "image_url": "https://mir-s3-cdn-cf.behance.net/test_image.jpg",
    }


# ─── Tests: _safe_filename ────────────────────────────────────────────────────

def test_safe_filename_is_deterministic():
    """
    The same URL should always produce the same filename.
    This is important for deduplication — we do not want to download the same
    image twice just because the filename changed.
    """
    url = "https://example.com/image.jpg?v=123"
    assert _safe_filename(url) == _safe_filename(url)


def test_safe_filename_preserves_extension():
    """The image file extension should be preserved in the output filename."""
    assert _safe_filename("https://example.com/photo.png").endswith(".png")
    assert _safe_filename("https://example.com/photo.jpg").endswith(".jpg")
    assert _safe_filename("https://example.com/photo.webp").endswith(".webp")


def test_safe_filename_defaults_to_jpg_for_unknown_extension():
    """If no recognisable extension is found, the filename should end in .jpg."""
    assert _safe_filename("https://example.com/imagedata?format=binary").endswith(".jpg")


def test_safe_filename_strips_query_string():
    """
    Query strings like ?width=1200&quality=85 are common in CDN URLs.
    They should not affect the filename or the extension.
    """
    name = _safe_filename("https://cdn.example.com/img.jpg?width=1200")
    assert name.endswith(".jpg")
    assert "?" not in name
    assert "=" not in name


# ─── Tests: _build_save_path ──────────────────────────────────────────────────

def test_build_save_path_structure(tmp_path):
    """
    _build_save_path should produce:
        {base_dir}/{source}/{safe_keyword}/{filename}

    where the keyword has spaces replaced with underscores and is lowercased.
    """
    base = str(tmp_path)
    path = _build_save_path(base, "behance", "Dark Minimalism", "abc123.jpg")

    assert path.startswith(base)
    assert "behance" in path
    assert "dark_minimalism" in path  # spaces → underscores, lowercase
    assert path.endswith("abc123.jpg")


def test_build_save_path_creates_directory(tmp_path):
    """The function should create the directory if it does not exist."""
    base = str(tmp_path / "images")
    _build_save_path(base, "pinterest", "editorial fashion", "img.jpg")
    expected_dir = os.path.join(base, "pinterest", "editorial_fashion")
    assert os.path.isdir(expected_dir)


# ─── Tests: _validate_and_get_dimensions ──────────────────────────────────────

def test_validate_returns_dimensions_for_valid_image():
    """A valid JPEG should return its (width, height)."""
    fake_bytes = make_fake_image_bytes(800, 600)
    result = _validate_and_get_dimensions(fake_bytes)
    assert result == (800, 600)


def test_validate_returns_none_for_invalid_bytes():
    """
    Random bytes that are not a valid image should return None.
    This guards against CDNs returning HTML error pages with a 200 status.
    """
    garbage = b"this is not an image <html>404 Not Found</html>"
    result = _validate_and_get_dimensions(garbage)
    assert result is None


def test_validate_returns_none_for_empty_bytes():
    """Empty bytes should also return None."""
    result = _validate_and_get_dimensions(b"")
    assert result is None


# ─── Tests: download_all_pending (with mocked HTTP) ───────────────────────────

def test_download_all_pending_success(conn, sample_record, tmp_path):
    """
    When the HTTP fetch returns valid image bytes, the image should be saved
    to disk and the database record should be updated.

    We mock _fetch_image to return fake JPEG bytes without making a real
    HTTP request.
    """
    insert_images(conn, [sample_record])
    images_dir = str(tmp_path / "images")
    fake_bytes = make_fake_image_bytes(600, 400)

    with patch("utils.downloader._fetch_image", return_value=fake_bytes):
        stats = download_all_pending(conn, images_dir=images_dir, delay=0)

    assert stats["succeeded"] == 1
    assert stats["failed"]    == 0

    # Verify the database was updated
    row = conn.execute("SELECT * FROM images").fetchone()
    assert row["local_path"]      is not None
    assert row["width_px"]        == 600
    assert row["height_px"]       == 400
    assert row["file_size_bytes"] == len(fake_bytes)
    assert row["downloaded_at"]   is not None

    # Verify the file was actually saved to disk
    assert os.path.exists(row["local_path"])


def test_download_all_pending_fetch_failure(conn, sample_record, tmp_path):
    """
    When _fetch_image returns None (network error, 403, etc.), the download
    should be counted as failed but the loop should continue.
    """
    insert_images(conn, [sample_record])
    images_dir = str(tmp_path / "images")

    with patch("utils.downloader._fetch_image", return_value=None):
        stats = download_all_pending(conn, images_dir=images_dir, delay=0)

    assert stats["failed"]    == 1
    assert stats["succeeded"] == 0

    # Database should NOT have been updated
    row = conn.execute("SELECT * FROM images").fetchone()
    assert row["local_path"] is None


def test_download_all_pending_skips_tiny_images(conn, tmp_path):
    """
    Images smaller than 100x100 should be skipped (not downloaded).
    These are likely placeholder images or thumbnails.
    """
    record = {
        "source": "pinterest",
        "keyword": "dark minimalism",
        "image_url": "https://example.com/tiny.jpg",
    }
    insert_images(conn, [record])
    images_dir = str(tmp_path / "images")

    # Return a 50x50 image — below the minimum threshold
    tiny_bytes = make_fake_image_bytes(50, 50)

    with patch("utils.downloader._fetch_image", return_value=tiny_bytes):
        stats = download_all_pending(conn, images_dir=images_dir, delay=0)

    assert stats["skipped"] == 1
    assert stats["succeeded"] == 0


def test_download_all_pending_no_work(conn):
    """
    When there are no pending images, the function should return zeros
    without raising any errors.
    """
    stats = download_all_pending(conn, delay=0)

    assert stats["attempted"] == 0
    assert stats["succeeded"] == 0


def test_download_all_pending_idempotent(conn, sample_record, tmp_path):
    """
    Running download_all_pending() twice should not re-download images
    that were already downloaded in the first run.
    """
    insert_images(conn, [sample_record])
    images_dir = str(tmp_path / "images")
    fake_bytes = make_fake_image_bytes(300, 200)

    with patch("utils.downloader._fetch_image", return_value=fake_bytes):
        stats_first  = download_all_pending(conn, images_dir=images_dir, delay=0)
        stats_second = download_all_pending(conn, images_dir=images_dir, delay=0)

    assert stats_first["succeeded"]  == 1
    # Second run should have nothing to do
    assert stats_second["attempted"] == 0
