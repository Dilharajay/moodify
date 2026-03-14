"""
tests/test_database.py — Unit Tests for utils/database.py
===========================================================

These tests use an in-memory SQLite database (":memory:") so nothing is written
to disk and every test starts with a clean slate.

In-memory databases are perfect for unit tests:
    - No cleanup needed between tests
    - Runs instantly (no file I/O)
    - Completely isolated from production data
"""

import json
import pytest
import sqlite3
from datetime import datetime

from utils.database import (
    initialize_db,
    insert_images,
    log_scrape_job,
    get_undownloaded,
    get_unprocessed,
    get_all,
    update_download,
    update_colors,
    get_stats,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def conn():
    """
    Pytest fixture: provides a fresh in-memory database for each test.

    The 'yield' pattern means:
        - Code before yield = setup (runs before the test)
        - Code after yield  = teardown (runs after the test, even if it fails)
    """
    connection = initialize_db(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def sample_records():
    """
    A set of sample image records that look like real scraper output.
    Used across multiple tests.
    """
    return [
        {
            "source": "behance",
            "keyword": "dark minimalism",
            "title": "Dark Echoes",
            "owner": "some_designer",
            "source_url": "https://www.behance.net/gallery/123",
            "image_url": "https://mir-s3-cdn-cf.behance.net/img1.jpg",
            "scraped_at": datetime.utcnow().isoformat(),
        },
        {
            "source": "pinterest",
            "keyword": "dark minimalism",
            "title": "Midnight Palette",
            "owner": "pinuser",
            "source_url": "https://www.pinterest.com/pin/456",
            "image_url": "https://i.pinimg.com/736x/img2.jpg",
            "scraped_at": datetime.utcnow().isoformat(),
        },
    ]


# ─── Tests: initialize_db ─────────────────────────────────────────────────────

def test_initialize_db_creates_tables(conn):
    """
    initialize_db() should create 'images' and 'scrape_jobs' tables.
    We query sqlite_master to check that the tables exist.
    """
    tables = {
        row[0] for row in
        conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "images" in tables
    assert "scrape_jobs" in tables


def test_initialize_db_is_idempotent():
    """
    Calling initialize_db() twice on the same connection should not raise an error.
    This is guaranteed by CREATE TABLE IF NOT EXISTS, but we test it explicitly.
    """
    conn = initialize_db(":memory:")
    # Second call should not raise
    initialize_db(":memory:")
    conn.close()


# ─── Tests: insert_images ─────────────────────────────────────────────────────

def test_insert_images_basic(conn, sample_records):
    """
    insert_images() should insert all records and return the correct count.
    """
    inserted = insert_images(conn, sample_records)
    assert inserted == 2


def test_insert_images_deduplication(conn, sample_records):
    """
    Inserting the same records twice should not create duplicates.
    The second insert should return 0 (nothing new was added).
    """
    insert_images(conn, sample_records)
    inserted_again = insert_images(conn, sample_records)
    assert inserted_again == 0

    # Total count should still be 2
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    assert total == 2


def test_insert_images_partial_duplicate(conn, sample_records):
    """
    If we insert 2 records then try to insert 3 (including those 2 again),
    only 1 new record should be inserted.
    """
    insert_images(conn, sample_records)

    new_record = {
        "source": "behance",
        "keyword": "dark minimalism",
        "title": "Brand New Project",
        "owner": "new_designer",
        "source_url": "https://www.behance.net/gallery/789",
        "image_url": "https://mir-s3-cdn-cf.behance.net/new_img.jpg",
    }
    inserted = insert_images(conn, sample_records + [new_record])
    assert inserted == 1


def test_insert_images_fields_stored_correctly(conn, sample_records):
    """
    After insert, the row should have the correct values for all fields.
    """
    insert_images(conn, sample_records)
    row = conn.execute(
        "SELECT * FROM images WHERE source = 'behance'"
    ).fetchone()

    assert row["source"]  == "behance"
    assert row["keyword"] == "dark minimalism"
    assert row["title"]   == "Dark Echoes"
    assert row["owner"]   == "some_designer"
    assert row["image_url"] == "https://mir-s3-cdn-cf.behance.net/img1.jpg"
    # Download and color fields should be NULL initially
    assert row["local_path"]       is None
    assert row["dominant_colors"]  is None
    assert row["color_histogram"]  is None


# ─── Tests: get_undownloaded ──────────────────────────────────────────────────

def test_get_undownloaded_returns_all_before_download(conn, sample_records):
    """
    Before any downloads, get_undownloaded() should return all records.
    """
    insert_images(conn, sample_records)
    pending = get_undownloaded(conn)
    assert len(pending) == 2


def test_get_undownloaded_filters_by_keyword(conn, sample_records):
    """
    get_undownloaded(keyword=...) should only return records for that keyword.
    """
    # Add a record with a different keyword
    extra = [{
        "source": "behance",
        "keyword": "editorial fashion",
        "title": "Fashion Week",
        "owner": "fashionista",
        "image_url": "https://mir-s3-cdn-cf.behance.net/fashion.jpg",
    }]
    insert_images(conn, sample_records + extra)

    pending = get_undownloaded(conn, keyword="editorial fashion")
    assert len(pending) == 1
    assert pending[0]["keyword"] == "editorial fashion"


def test_get_undownloaded_excludes_downloaded(conn, sample_records):
    """
    After marking a record as downloaded, it should not appear in get_undownloaded().
    """
    insert_images(conn, sample_records)
    row_id = conn.execute("SELECT id FROM images LIMIT 1").fetchone()[0]

    update_download(conn, row_id, "data/images/behance/img.jpg", 51200, 800, 600)

    pending = get_undownloaded(conn)
    # Only 1 should remain pending
    assert len(pending) == 1
    assert pending[0]["id"] != row_id


# ─── Tests: update_download ───────────────────────────────────────────────────

def test_update_download_sets_all_fields(conn, sample_records):
    """
    update_download() should set local_path, downloaded_at, file_size_bytes,
    width_px, and height_px on the correct row.
    """
    insert_images(conn, sample_records)
    row_id = conn.execute("SELECT id FROM images LIMIT 1").fetchone()[0]

    update_download(
        conn,
        image_id=row_id,
        local_path="data/images/behance/dark_minimalism/abc.jpg",
        file_size_bytes=102400,
        width_px=1200,
        height_px=800,
    )

    row = conn.execute("SELECT * FROM images WHERE id = ?", (row_id,)).fetchone()
    assert row["local_path"]       == "data/images/behance/dark_minimalism/abc.jpg"
    assert row["file_size_bytes"]  == 102400
    assert row["width_px"]         == 1200
    assert row["height_px"]        == 800
    assert row["downloaded_at"]    is not None


# ─── Tests: get_unprocessed ───────────────────────────────────────────────────

def test_get_unprocessed_requires_download_first(conn, sample_records):
    """
    get_unprocessed() should return 0 results when nothing has been downloaded
    (because we only process images we actually have on disk).
    """
    insert_images(conn, sample_records)
    pending = get_unprocessed(conn)
    assert len(pending) == 0


def test_get_unprocessed_returns_downloaded_without_colors(conn, sample_records):
    """
    After downloading but before color extraction, get_unprocessed() should
    return the downloaded images.
    """
    insert_images(conn, sample_records)
    row_id = conn.execute("SELECT id FROM images LIMIT 1").fetchone()[0]
    update_download(conn, row_id, "data/images/test.jpg", 50000, 800, 600)

    pending = get_unprocessed(conn)
    assert len(pending) == 1
    assert pending[0]["id"] == row_id


# ─── Tests: update_colors ─────────────────────────────────────────────────────

def test_update_colors_stores_json(conn, sample_records):
    """
    update_colors() should serialize the palette and histogram as JSON strings
    and store them in the database. Reading them back and parsing with json.loads()
    should return the original values.
    """
    insert_images(conn, sample_records)
    row_id = conn.execute("SELECT id FROM images LIMIT 1").fetchone()[0]

    palette   = ["#1A1A2E", "#E94560", "#0F3460", "#533483", "#16213E", "#FFFFFF"]
    histogram = [0.1] * 48

    update_colors(conn, row_id, palette, histogram)

    row = conn.execute("SELECT * FROM images WHERE id = ?", (row_id,)).fetchone()

    # Both are stored as JSON strings
    stored_palette    = json.loads(row["dominant_colors"])
    stored_histogram  = json.loads(row["color_histogram"])

    assert stored_palette   == palette
    assert stored_histogram == histogram
    assert row["colors_extracted_at"] is not None


# ─── Tests: get_stats ─────────────────────────────────────────────────────────

def test_get_stats_empty_database(conn):
    """
    Stats on an empty database should return zeros without errors.
    """
    stats = get_stats(conn)
    assert stats["total"]            == 0
    assert stats["downloaded"]       == 0
    assert stats["colors_extracted"] == 0
    assert stats["pending_download"] == 0


def test_get_stats_after_insert(conn, sample_records):
    """
    After inserting records, the stats should reflect the current state.
    """
    insert_images(conn, sample_records)
    stats = get_stats(conn)

    assert stats["total"]            == 2
    assert stats["downloaded"]       == 0
    assert stats["pending_download"] == 2
    assert stats["by_source"]["behance"]  == 1
    assert stats["by_source"]["pinterest"] == 1
