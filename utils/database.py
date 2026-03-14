"""
utils/database.py — SQLite Storage Layer
=========================================

This module manages everything related to the database: creating the schema,
inserting scraped records, updating them as images are downloaded and processed,
and querying for records that still need work.

Why SQLite instead of JSON files?
----------------------------------
Phase 2 stored results as separate JSON files per scrape run. This worked for
exploration but becomes awkward when we need to:
  - Track which images have been downloaded (vs pending)
  - Track which images have had color features extracted
  - Avoid duplicates across multiple scrape runs
  - Query "give me all Behance images for keyword X that are not yet downloaded"

SQLite solves all of this in a single file (data/aestheteai.db) using standard SQL.
It is built into Python — no installation or server needed.

Schema Overview
---------------
images        — one row per scraped image, with lifecycle columns for download
                and color extraction status
scrape_jobs   — one row per scrape run, for tracking history
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional


# ─── Schema ──────────────────────────────────────────────────────────────────

# SQL to create the images table.
# We use TEXT for timestamps (ISO 8601 strings) rather than DATETIME because
# SQLite stores all DATETIME as TEXT internally anyway, and being explicit
# avoids confusion. The UNIQUE constraint on image_url prevents duplicate rows
# if the same image is scraped twice.
CREATE_IMAGES_TABLE = """
CREATE TABLE IF NOT EXISTS images (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source              TEXT NOT NULL,      -- 'behance' or 'pinterest'
    keyword             TEXT NOT NULL,      -- the search keyword used
    title               TEXT,               -- project/pin title (may be NULL if scraper missed it)
    owner               TEXT,               -- creator name
    source_url          TEXT,               -- the page URL (not the direct image URL)
    image_url           TEXT UNIQUE NOT NULL, -- direct URL to the image file

    scraped_at          TEXT NOT NULL,      -- ISO timestamp of when scraping happened

    -- Download lifecycle fields (populated by utils/downloader.py)
    local_path          TEXT,               -- relative path under data/images/, e.g. 'behance/dark_minimalism/abc123.jpg'
    downloaded_at       TEXT,               -- ISO timestamp of download
    file_size_bytes     INTEGER,            -- file size after download
    width_px            INTEGER,            -- image width in pixels
    height_px           INTEGER,            -- image height in pixels

    -- Color feature fields (populated by utils/color_extractor.py)
    dominant_colors     TEXT,               -- JSON array of HEX strings, e.g. '["#1A1A2E", "#E94560"]'
    color_histogram     TEXT,               -- JSON array of 48 floats (normalized LAB histogram)
    colors_extracted_at TEXT                -- ISO timestamp of color extraction
);
"""

# Track each scrape run for history and debugging.
CREATE_SCRAPE_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS scrape_jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword     TEXT NOT NULL,
    source      TEXT NOT NULL,      -- 'behance', 'pinterest', or 'all'
    pages       INTEGER,
    scraped_at  TEXT NOT NULL,
    image_count INTEGER             -- how many images were inserted in this run
);
"""


# ─── Connection ───────────────────────────────────────────────────────────────

def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Open a connection to the SQLite database at db_path.

    We enable row_factory = sqlite3.Row so that query results can be accessed
    by column name (like a dict) rather than only by index. This makes the
    code much more readable downstream.

    Example:
        conn = get_connection("data/aestheteai.db")
        row = conn.execute("SELECT * FROM images LIMIT 1").fetchone()
        print(row["title"])   # works because of row_factory
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # access columns by name, not index
    conn.execute("PRAGMA journal_mode=WAL")  # WAL mode is faster for concurrent reads
    return conn


def initialize_db(db_path: str) -> sqlite3.Connection:
    """
    Create the database file and all tables if they do not already exist.
    Safe to call multiple times — CREATE TABLE IF NOT EXISTS is idempotent.

    Returns an open connection to the database.
    """
    conn = get_connection(db_path)
    conn.execute(CREATE_IMAGES_TABLE)
    conn.execute(CREATE_SCRAPE_JOBS_TABLE)
    conn.commit()
    return conn


# ─── Insert ───────────────────────────────────────────────────────────────────

def insert_images(conn: sqlite3.Connection, records: list[dict]) -> int:
    """
    Insert a batch of scraped image records into the images table.

    Uses INSERT OR IGNORE so that if a record with the same image_url already
    exists (e.g. from a previous scrape of the same keyword), it is silently
    skipped rather than raising an error.

    Parameters
    ----------
    conn    : open SQLite connection
    records : list of dicts, each with keys matching the images table columns.
              The minimal required keys are: source, keyword, image_url.

    Returns
    -------
    int — number of rows actually inserted (duplicates excluded)
    """
    now = datetime.utcnow().isoformat()

    # Build rows, filling in defaults for missing fields.
    # Using .get() with None defaults means scraper output doesn't need to
    # include every column — only the ones it has.
    rows = [
        (
            record.get("source"),
            record.get("keyword"),
            record.get("title"),
            record.get("owner"),
            record.get("source_url"),
            record.get("image_url"),
            record.get("scraped_at", now),
        )
        for record in records
    ]

    before = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    conn.executemany(
        """
        INSERT OR IGNORE INTO images
            (source, keyword, title, owner, source_url, image_url, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()

    after = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    inserted = after - before
    return inserted


def log_scrape_job(conn: sqlite3.Connection, keyword: str, source: str,
                   pages: int, image_count: int) -> None:
    """
    Record a completed scrape run in the scrape_jobs table.
    Useful for auditing how much data was collected and when.
    """
    conn.execute(
        """
        INSERT INTO scrape_jobs (keyword, source, pages, scraped_at, image_count)
        VALUES (?, ?, ?, ?, ?)
        """,
        (keyword, source, pages, datetime.utcnow().isoformat(), image_count),
    )
    conn.commit()


# ─── Query ────────────────────────────────────────────────────────────────────

def get_undownloaded(conn: sqlite3.Connection,
                     keyword: Optional[str] = None) -> list[sqlite3.Row]:
    """
    Return all images that have a scraped image_url but no local_path yet.
    These are the images that the downloader needs to fetch.

    Optionally filter to a specific keyword.
    """
    if keyword:
        return conn.execute(
            "SELECT * FROM images WHERE local_path IS NULL AND keyword = ?",
            (keyword,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM images WHERE local_path IS NULL"
    ).fetchall()


def get_unprocessed(conn: sqlite3.Connection,
                    keyword: Optional[str] = None) -> list[sqlite3.Row]:
    """
    Return all images that have been downloaded (local_path is set)
    but have not yet had color features extracted (dominant_colors is NULL).

    These are the images that the color extractor needs to process.
    """
    if keyword:
        return conn.execute(
            """
            SELECT * FROM images
            WHERE local_path IS NOT NULL
              AND dominant_colors IS NULL
              AND keyword = ?
            """,
            (keyword,),
        ).fetchall()
    return conn.execute(
        """
        SELECT * FROM images
        WHERE local_path IS NOT NULL
          AND dominant_colors IS NULL
        """
    ).fetchall()


def get_all(conn: sqlite3.Connection,
            keyword: Optional[str] = None,
            source: Optional[str] = None) -> list[sqlite3.Row]:
    """
    Return all image records, optionally filtered by keyword and/or source.
    """
    query = "SELECT * FROM images WHERE 1=1"
    params = []
    if keyword:
        query += " AND keyword = ?"
        params.append(keyword)
    if source:
        query += " AND source = ?"
        params.append(source)
    return conn.execute(query, params).fetchall()


# ─── Update ───────────────────────────────────────────────────────────────────

def update_download(conn: sqlite3.Connection,
                    image_id: int,
                    local_path: str,
                    file_size_bytes: int,
                    width_px: int,
                    height_px: int) -> None:
    """
    After successfully downloading an image, store its local path and metadata.

    Parameters
    ----------
    image_id        : the row's primary key
    local_path      : relative path, e.g. 'data/images/behance/dark_minimalism/abc.jpg'
    file_size_bytes : file size in bytes
    width_px        : image width in pixels (from Pillow)
    height_px       : image height in pixels (from Pillow)
    """
    conn.execute(
        """
        UPDATE images
        SET local_path      = ?,
            downloaded_at   = ?,
            file_size_bytes = ?,
            width_px        = ?,
            height_px       = ?
        WHERE id = ?
        """,
        (local_path, datetime.utcnow().isoformat(),
         file_size_bytes, width_px, height_px, image_id),
    )
    conn.commit()


def update_colors(conn: sqlite3.Connection,
                  image_id: int,
                  dominant_colors: list[str],
                  color_histogram: list[float]) -> None:
    """
    After extracting color features from a downloaded image, store them.

    dominant_colors is a list like ["#1A1A2E", "#E94560", ...].
    color_histogram is a 48-float list.

    Both are serialized as JSON strings for storage in SQLite TEXT columns.
    To read them back, use json.loads(row["dominant_colors"]).
    """
    conn.execute(
        """
        UPDATE images
        SET dominant_colors     = ?,
            color_histogram     = ?,
            colors_extracted_at = ?
        WHERE id = ?
        """,
        (
            json.dumps(dominant_colors),
            json.dumps(color_histogram),
            datetime.utcnow().isoformat(),
            image_id,
        ),
    )
    conn.commit()


# ─── Stats ────────────────────────────────────────────────────────────────────

def get_stats(conn: sqlite3.Connection) -> dict:
    """
    Return a summary of the current database state.

    Useful for printing a progress dashboard at the end of each pipeline step
    so you always know how much work is left.

    Example output:
        {
            "total": 420,
            "downloaded": 390,
            "colors_extracted": 350,
            "pending_download": 30,
            "pending_colors": 40,
            "by_source": {"behance": 210, "pinterest": 210}
        }
    """
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    downloaded = conn.execute(
        "SELECT COUNT(*) FROM images WHERE local_path IS NOT NULL"
    ).fetchone()[0]
    colors_done = conn.execute(
        "SELECT COUNT(*) FROM images WHERE dominant_colors IS NOT NULL"
    ).fetchone()[0]

    # Count per source
    source_rows = conn.execute(
        "SELECT source, COUNT(*) as cnt FROM images GROUP BY source"
    ).fetchall()
    by_source = {row["source"]: row["cnt"] for row in source_rows}

    return {
        "total": total,
        "downloaded": downloaded,
        "colors_extracted": colors_done,
        "pending_download": total - downloaded,
        "pending_colors": downloaded - colors_done,
        "by_source": by_source,
    }


# ─── Phase 4: Schema Migration ────────────────────────────────────────────────

def migrate_schema_phase4(conn: sqlite3.Connection) -> None:
    """
    Add Phase 4 embedding path columns to the images table.

    Why ALTER TABLE instead of putting these in CREATE_IMAGES_TABLE?
    Because existing databases (created in Phase 3) already have the images
    table without these columns. ALTER TABLE ADD COLUMN is the safe, additive
    way to extend an existing schema without losing any data.

    This function is idempotent — calling it on a database that already has
    these columns will silently do nothing (the try/except catches the
    "duplicate column" error from SQLite).

    Columns added:
        image_embedding_path  — path to the .npy file for this image's CLIP embedding
        text_embedding_path   — path to the .npy file for the keyword text embedding
                                (same for all images sharing a keyword)
    """
    for column_def in [
        "ALTER TABLE images ADD COLUMN image_embedding_path TEXT",
        "ALTER TABLE images ADD COLUMN text_embedding_path  TEXT",
    ]:
        try:
            conn.execute(column_def)
        except sqlite3.OperationalError:
            # Column already exists — safe to ignore
            pass
    conn.commit()


# ─── Phase 4: Query helpers ───────────────────────────────────────────────────

def get_unembedded(conn: sqlite3.Connection,
                   keyword: str = None) -> list:
    """
    Return all images that have been downloaded but do not yet have a
    CLIP image embedding (image_embedding_path is NULL).

    These are the images the embedder needs to process next.
    """
    if keyword:
        return conn.execute(
            """
            SELECT * FROM images
            WHERE local_path IS NOT NULL
              AND image_embedding_path IS NULL
              AND keyword = ?
            """,
            (keyword,),
        ).fetchall()
    return conn.execute(
        """
        SELECT * FROM images
        WHERE local_path IS NOT NULL
          AND image_embedding_path IS NULL
        """
    ).fetchall()


def get_embedded(conn: sqlite3.Connection,
                 keyword: str = None) -> list:
    """
    Return all images that have a CLIP image embedding.
    Used by build_feature_matrix() to load embeddings for clustering.
    """
    if keyword:
        return conn.execute(
            """
            SELECT * FROM images
            WHERE image_embedding_path IS NOT NULL
              AND keyword = ?
            """,
            (keyword,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM images WHERE image_embedding_path IS NOT NULL"
    ).fetchall()


def update_embedding(conn: sqlite3.Connection,
                     image_id: int,
                     image_embedding_path: str) -> None:
    """
    After saving a CLIP image embedding to disk, record its path in the DB.

    Parameters
    ----------
    image_id             : the row's primary key
    image_embedding_path : relative path to the .npy file,
                           e.g. 'data/embeddings/images/abc123def456.npy'
    """
    conn.execute(
        """
        UPDATE images
        SET image_embedding_path = ?
        WHERE id = ?
        """,
        (image_embedding_path, image_id),
    )
    conn.commit()


# ─── Phase 5: Schema Migration ────────────────────────────────────────────────

def migrate_schema_phase5(conn: sqlite3.Connection) -> None:
    """
    Add Phase 5 clustering columns to the images table.

    cluster_id    — integer cluster label from HDBSCAN (-1 = noise)
    cluster_label — human-readable label e.g. "cinematic deep navy"

    Same additive pattern as Phase 4: safe to call on any existing DB,
    duplicate column errors are silently caught.
    """
    for column_def in [
        "ALTER TABLE images ADD COLUMN cluster_id    INTEGER",
        "ALTER TABLE images ADD COLUMN cluster_label TEXT",
    ]:
        try:
            conn.execute(column_def)
        except sqlite3.OperationalError:
            pass
    conn.commit()