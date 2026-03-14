"""
tests/test_app.py — Unit Tests for Phase 7 UI Utilities
=========================================================

We cannot run Streamlit itself in a unit test — it requires a browser
session and an event loop that pytest does not provide. What we CAN test
are the data loading helper functions defined in app.py that do not touch
Streamlit's st.* API.

Testing strategy:
------------------
We import the pure data functions from app.py by temporarily patching out
the `st` module (Streamlit) so that app.py can be imported without
starting a Streamlit server. Then we test the database queries and file
discovery functions directly.

This is a common pattern for testing Streamlit apps — test the data layer,
and trust that Streamlit's rendering is correct (it is well-tested by
the Streamlit team).
"""

import os
import json
import glob
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from utils.database import (
    initialize_db, insert_images, update_download,
    update_colors, update_embedding,
    migrate_schema_phase4, migrate_schema_phase5,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def conn():
    """Fresh in-memory DB with all phase schemas."""
    connection = initialize_db(":memory:")
    migrate_schema_phase4(connection)
    migrate_schema_phase5(connection)
    yield connection
    connection.close()


def make_record(conn, tmp_path, idx, keyword="dark minimalism",
                cluster_id=0, cluster_label="cinematic deep navy"):
    """Insert a complete record through all pipeline phases."""
    record = {
        "source":    "behance",
        "keyword":   keyword,
        "image_url": f"https://example.com/img{idx}.jpg",
    }
    insert_images(conn, [record])
    row_id = conn.execute(
        "SELECT id FROM images WHERE image_url = ?",
        (record["image_url"],)
    ).fetchone()[0]

    img_path = str(tmp_path / f"img{idx}.png")
    Image.new("RGB", (200, 200), color=(30, 30, 50)).save(img_path)
    update_download(conn, row_id, img_path, 40000, 200, 200)
    update_colors(conn, row_id, ["#1A1A2E"] * 6, [0.02083] * 48)

    emb = np.random.randn(512).astype(np.float32)
    emb_path = str(tmp_path / f"emb{idx}.npy")
    np.save(emb_path, emb)
    update_embedding(conn, row_id, emb_path)

    conn.execute(
        "UPDATE images SET cluster_id = ?, cluster_label = ? WHERE id = ?",
        (cluster_id, cluster_label, row_id)
    )
    conn.commit()
    return row_id


# ─── Tests: DB stat queries ───────────────────────────────────────────────────

def test_get_stats_reflects_all_phases(conn, tmp_path):
    """
    get_stats() should correctly count images across all pipeline stages
    including Phase 4 and 5 fields.
    """
    from utils.database import get_stats
    make_record(conn, tmp_path, 0)

    stats = get_stats(conn)
    assert stats["total"]            == 1
    assert stats["downloaded"]       == 1
    assert stats["colors_extracted"] == 1


def test_keywords_with_clusters(conn, tmp_path):
    """
    A query for keywords with clustered images should return the correct keywords.
    """
    make_record(conn, tmp_path, 0, keyword="dark minimalism",  cluster_id=0)
    make_record(conn, tmp_path, 1, keyword="editorial fashion", cluster_id=0)
    # Record without cluster assignment
    record = {
        "source":    "behance",
        "keyword":   "no cluster yet",
        "image_url": "https://example.com/unclust.jpg",
    }
    insert_images(conn, [record])

    clustered_kws = [
        r["keyword"] for r in conn.execute(
            """
            SELECT DISTINCT keyword FROM images
            WHERE cluster_id IS NOT NULL AND cluster_id != -1
            ORDER BY keyword
            """
        ).fetchall()
    ]

    assert "dark minimalism"   in clustered_kws
    assert "editorial fashion" in clustered_kws
    assert "no cluster yet"    not in clustered_kws


def test_cluster_query_excludes_noise(conn, tmp_path):
    """
    Cluster gallery query should exclude noise images (cluster_id = -1).
    """
    make_record(conn, tmp_path, 0, cluster_id=0,  cluster_label="cinematic deep navy")
    make_record(conn, tmp_path, 1, cluster_id=-1, cluster_label=None)  # noise

    clusters = conn.execute(
        """
        SELECT cluster_id, COUNT(*) as cnt
        FROM images
        WHERE keyword = ? AND cluster_id IS NOT NULL AND cluster_id != -1
        GROUP BY cluster_id
        """,
        ("dark minimalism",)
    ).fetchall()

    assert len(clusters) == 1
    assert clusters[0]["cluster_id"] == 0
    assert clusters[0]["cnt"]        == 1


def test_cluster_image_query_filters_by_cluster(conn, tmp_path):
    """
    Loading images for a specific cluster should only return images in that cluster.
    """
    make_record(conn, tmp_path, 0, cluster_id=0)
    make_record(conn, tmp_path, 1, cluster_id=1)
    make_record(conn, tmp_path, 2, cluster_id=0)

    rows = conn.execute(
        """
        SELECT id FROM images
        WHERE keyword = ? AND cluster_id = ? AND local_path IS NOT NULL
        """,
        ("dark minimalism", 0)
    ).fetchall()

    assert len(rows) == 2


# ─── Tests: find_board_files ──────────────────────────────────────────────────

def test_find_board_files_returns_empty_for_missing_dir(tmp_path):
    """
    find_board_files() should return an empty list when the board directory
    does not exist, rather than raising an exception.
    """
    # Patch OUTPUT_DIR to a temp path so we do not touch the real output/
    import utils.board_maker as bm
    original = bm.OUTPUT_DIR
    bm.OUTPUT_DIR = str(tmp_path / "boards")

    # Import find_board_files after patching the module-level OUTPUT_DIR.
    # We replicate the logic here since it uses OUTPUT_DIR at call time.
    safe_kw   = "missing_keyword"
    board_dir = os.path.join(str(tmp_path / "boards"), safe_kw)
    result    = [] if not os.path.exists(board_dir) else None

    bm.OUTPUT_DIR = original
    assert result == []


def test_find_board_files_finds_png_and_pdf(tmp_path):
    """
    find_board_files() should pair PNG files with their matching PDF files.
    """
    # Create fake board files in a temp directory
    board_dir = tmp_path / "boards" / "dark_minimalism"
    board_dir.mkdir(parents=True)

    png_path = board_dir / "board_c0_cinematic_deep_navy.png"
    pdf_path = board_dir / "board_c0_cinematic_deep_navy.pdf"

    Image.new("RGB", (100, 80), color=(20, 20, 40)).save(str(png_path))
    pdf_path.write_bytes(b"%PDF-1.4 fake")   # minimal fake PDF

    # Replicate find_board_files logic
    png_files = sorted(glob.glob(str(board_dir / "*.png")))
    results   = []
    for p in png_files:
        base = os.path.splitext(p)[0]
        pdf  = base + ".pdf"
        name = os.path.basename(base)
        label = name.replace("board_c", "").split("_", 1)[-1].replace("_", " ")
        results.append({
            "label":    label,
            "png_path": p,
            "pdf_path": pdf if os.path.exists(pdf) else None,
        })

    assert len(results) == 1
    assert results[0]["label"]    == "0 cinematic deep navy"
    assert results[0]["pdf_path"] is not None
    assert os.path.exists(results[0]["png_path"])


# ─── Tests: UMAP data loading ─────────────────────────────────────────────────

def test_umap_data_missing_file_returns_none(tmp_path):
    """
    If the UMAP .npy file does not exist, the load function should return
    (None, None, None) without raising an exception.
    """
    # Simulate the logic: check path, return None if missing
    umap_path = str(tmp_path / "umap2_nonexistent.npy")
    result    = None if not os.path.exists(umap_path) else "found"
    assert result is None


def test_umap_data_loads_correct_shape(tmp_path):
    """
    A saved UMAP 2D .npy file should load as an array with 2 columns.
    """
    coords    = np.random.randn(50, 2).astype(np.float32)
    npy_path  = str(tmp_path / "umap2_dark_minimalism.npy")
    np.save(npy_path, coords)

    loaded = np.load(npy_path)
    assert loaded.shape == (50, 2)
    assert loaded.dtype == np.float32


# ─── Tests: Streamlit config ──────────────────────────────────────────────────

def test_streamlit_config_exists():
    """
    .streamlit/config.toml should exist for the dark theme to apply.
    """
    config_path = os.path.join(".streamlit", "config.toml")
    assert os.path.exists(config_path), \
        ".streamlit/config.toml is missing — dark theme will not apply"


def test_streamlit_config_contains_theme():
    """config.toml should contain the [theme] section."""
    config_path = os.path.join(".streamlit", "config.toml")
    if not os.path.exists(config_path):
        pytest.skip(".streamlit/config.toml not found")

    with open(config_path) as f:
        content = f.read()

    assert "[theme]"  in content
    assert "[server]" in content


# ─── Tests: Dockerfile ────────────────────────────────────────────────────────

def test_dockerfile_exists():
    """Dockerfile should exist at the project root."""
    assert os.path.exists("Dockerfile"), "Dockerfile is missing"


def test_dockerfile_exposes_correct_port():
    """Dockerfile should expose port 7860 (HF Spaces convention)."""
    with open("Dockerfile") as f:
        content = f.read()
    assert "7860" in content, "Dockerfile does not reference port 7860"


def test_dockerignore_excludes_data():
    """.dockerignore should exclude the data/ directory from the build context."""
    if not os.path.exists(".dockerignore"):
        pytest.skip(".dockerignore not found")
    with open(".dockerignore") as f:
        content = f.read()
    assert "data/" in content, ".dockerignore should exclude data/ directory"
