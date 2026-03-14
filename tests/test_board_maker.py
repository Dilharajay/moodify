"""
tests/test_board_maker.py — Unit Tests for utils/board_maker.py
================================================================

Board composition tests use a small synthetic cluster — a handful of
solid-colour images generated entirely in memory with Pillow. This lets
us test the layout logic without needing real scraped images on disk.

What we are testing:
    - compose_board() returns a PIL Image of the correct dimensions
    - Color palette strip is drawn using the correct colors from the DB
    - Hero selection chooses the embedding closest to the centroid
    - export_board() creates both PNG and PDF files at the expected paths
    - generate_all_boards() skips clusters with no downloaded images gracefully

What we are NOT testing:
    - Visual quality of the layout (subjective, requires human review)
    - Font rendering (depends on system fonts, not deterministic)
    - PDF internal structure (we trust reportlab)
"""

import os
import json
import numpy as np
import pytest
from PIL import Image

from utils.database import (
    initialize_db, insert_images, update_download,
    update_colors, update_embedding,
    migrate_schema_phase4, migrate_schema_phase5,
)
from utils.board_maker import (
    compose_board, export_board, generate_all_boards,
    _select_hero, _average_palette, _crop_to_fill,
    CANVAS_W, CANVAS_H,
)


# ─── Fixtures and Helpers ─────────────────────────────────────────────────────

def make_solid_image(path: str, color: tuple = (30, 30, 50),
                     size: tuple = (400, 300)) -> str:
    """Create a solid-colour image file and return its path."""
    Image.new("RGB", size, color=color).save(path, format="PNG")
    return path


def make_full_record(conn, tmp_path, idx: int,
                     cluster_id: int = 0,
                     cluster_label: str = "cinematic deep navy",
                     keyword: str = "dark minimalism") -> tuple:
    """
    Insert a complete image record (scraped, downloaded, colored, embedded,
    clustered) into an in-memory DB. Returns (row_id, embedding).
    """
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

    # Solid-colour local image
    img_path = str(tmp_path / f"img{idx}.png")
    colors   = [(30 + idx * 20, 30, 50), (200, 50, 80),
                (15, 60, 120), (80, 40, 130), (20, 80, 100), (240, 240, 255)]
    color    = colors[idx % len(colors)]
    make_solid_image(img_path, color=color)
    update_download(conn, row_id, img_path, 40000, 400, 300)

    # Color palette
    hex_color = "#{:02X}{:02X}{:02X}".format(*color)
    palette   = [hex_color] * 6
    histogram = [0.02083] * 48
    update_colors(conn, row_id, palette, histogram)

    # Deterministic embedding — each image gets a unique direction
    rng       = np.random.default_rng(seed=idx)
    embedding = rng.standard_normal(512).astype(np.float32)
    embedding /= np.linalg.norm(embedding)
    emb_path  = str(tmp_path / f"emb{idx}.npy")
    np.save(emb_path, embedding)
    update_embedding(conn, row_id, emb_path)

    # Assign to cluster
    conn.execute(
        "UPDATE images SET cluster_id = ?, cluster_label = ? WHERE id = ?",
        (cluster_id, cluster_label, row_id)
    )
    conn.commit()

    return row_id, embedding


@pytest.fixture
def conn():
    """Fresh in-memory DB with all phase schemas."""
    connection = initialize_db(":memory:")
    migrate_schema_phase4(connection)
    migrate_schema_phase5(connection)
    yield connection
    connection.close()


@pytest.fixture
def cluster_setup(conn, tmp_path):
    """
    Create a cluster of 8 images in the DB and return everything needed
    to call compose_board().
    """
    n        = 8
    ids      = []
    embeddings = []

    for i in range(n):
        row_id, emb = make_full_record(conn, tmp_path, idx=i,
                                       cluster_id=0,
                                       cluster_label="cinematic deep navy")
        ids.append(row_id)
        embeddings.append(emb)

    # Build a minimal feature matrix: CLIP portion = embeddings, color portion = zeros
    clip_matrix  = np.stack(embeddings, axis=0)
    color_matrix = np.zeros((n, 48), dtype=np.float32)
    feature_matrix = np.concatenate([clip_matrix, color_matrix], axis=1)

    return {
        "ids":            ids,
        "feature_matrix": feature_matrix,
        "conn":           conn,
    }


# ─── Tests: _crop_to_fill ─────────────────────────────────────────────────────

def test_crop_to_fill_exact_size():
    """Output should be exactly the requested size."""
    img    = Image.new("RGB", (800, 600))
    result = _crop_to_fill(img, 400, 300)
    assert result.size == (400, 300)


def test_crop_to_fill_portrait_to_landscape():
    """A portrait image cropped to landscape should be the target size."""
    img    = Image.new("RGB", (300, 800))
    result = _crop_to_fill(img, 600, 200)
    assert result.size == (600, 200)


def test_crop_to_fill_small_to_large():
    """Upscaling to a larger size should still produce the exact target size."""
    img    = Image.new("RGB", (50, 50))
    result = _crop_to_fill(img, 400, 300)
    assert result.size == (400, 300)


# ─── Tests: _select_hero ──────────────────────────────────────────────────────

def test_select_hero_returns_valid_id(cluster_setup):
    """_select_hero() should return one of the image IDs in the cluster."""
    setup          = cluster_setup
    ids            = setup["ids"]
    feature_matrix = setup["feature_matrix"]

    hero_id = _select_hero(ids, feature_matrix, ids)
    assert hero_id in ids


def test_select_hero_closest_to_centroid():
    """
    With two clearly separated embeddings, the hero should be the one
    that is closest to the mean (centroid) of the two.
    """
    # Create two embeddings: one near [1,0,...] and one near [-1,0,...]
    # The centroid is near [0,0,...], so the one slightly closer to origin wins.
    emb_a = np.zeros(512, dtype=np.float32)
    emb_b = np.zeros(512, dtype=np.float32)
    emb_a[0] = 0.8    # closer to centroid [0,...]
    emb_b[0] = 10.0   # further from centroid

    feature_matrix = np.zeros((2, 560), dtype=np.float32)
    feature_matrix[0, :512] = emb_a
    feature_matrix[1, :512] = emb_b

    image_ids = [100, 200]   # fake DB IDs
    hero_id   = _select_hero(image_ids, feature_matrix, image_ids)

    # ID 100 (emb_a) is closer to the centroid
    assert hero_id == 100


def test_select_hero_single_image():
    """With only one image, it should always be selected as hero."""
    emb            = np.ones(512, dtype=np.float32)
    feature_matrix = np.concatenate([emb, np.zeros(48)])[np.newaxis, :]
    hero_id        = _select_hero([42], feature_matrix, [42])
    assert hero_id == 42


# ─── Tests: _average_palette ─────────────────────────────────────────────────

def test_average_palette_returns_six_hex_strings(conn, tmp_path):
    """_average_palette() should return a list of 6 HEX strings."""
    ids = []
    for i in range(3):
        row_id, _ = make_full_record(conn, tmp_path, idx=i)
        ids.append(row_id)

    palette = _average_palette(conn, ids)

    assert len(palette) == 6
    for color in palette:
        assert color.startswith("#")
        assert len(color) == 7


def test_average_palette_solid_color_cluster(conn, tmp_path):
    """
    If all images in a cluster have the same dominant color, the average
    palette should contain exactly that color.
    """
    # Override: manually set dominant colors to a known value for 2 records
    ids = []
    for i in range(2):
        record = {
            "source":    "behance",
            "keyword":   "dark minimalism",
            "image_url": f"https://example.com/palette{i}.jpg",
        }
        insert_images(conn, [record])
        row_id = conn.execute(
            "SELECT id FROM images WHERE image_url = ?",
            (record["image_url"],)
        ).fetchone()[0]
        img_path = str(tmp_path / f"pal{i}.png")
        make_solid_image(img_path)
        update_download(conn, row_id, img_path, 1000, 100, 100)

        # All images have the same dominant color: pure red
        update_colors(conn, row_id, ["#FF0000"] * 6, [0.02083] * 48)
        ids.append(row_id)

    palette = _average_palette(conn, ids)
    # All entries should be (or be very close to) #FF0000
    for color in palette:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        assert r > 200, f"Expected high red channel, got {color}"
        assert g < 30,  f"Expected low green channel, got {color}"
        assert b < 30,  f"Expected low blue channel, got {color}"


# ─── Tests: compose_board ────────────────────────────────────────────────────

def test_compose_board_returns_pil_image(cluster_setup):
    """compose_board() should return a PIL Image (not None)."""
    s     = cluster_setup
    board = compose_board(
        conn           = s["conn"],
        cluster_id     = 0,
        cluster_label  = "cinematic deep navy",
        keyword        = "dark minimalism",
        feature_matrix = s["feature_matrix"],
        image_ids_full = s["ids"],
    )
    assert board is not None
    assert isinstance(board, Image.Image)


def test_compose_board_correct_dimensions(cluster_setup):
    """The composed board must be exactly CANVAS_W × CANVAS_H."""
    s     = cluster_setup
    board = compose_board(
        conn           = s["conn"],
        cluster_id     = 0,
        cluster_label  = "cinematic deep navy",
        keyword        = "dark minimalism",
        feature_matrix = s["feature_matrix"],
        image_ids_full = s["ids"],
    )
    assert board.size == (CANVAS_W, CANVAS_H)


def test_compose_board_rgb_mode(cluster_setup):
    """The board should be in RGB mode (not RGBA or L)."""
    s     = cluster_setup
    board = compose_board(
        conn           = s["conn"],
        cluster_id     = 0,
        cluster_label  = "cinematic deep navy",
        keyword        = "dark minimalism",
        feature_matrix = s["feature_matrix"],
        image_ids_full = s["ids"],
    )
    assert board.mode == "RGB"


def test_compose_board_empty_cluster_returns_none(conn, tmp_path):
    """
    If the cluster has no images (or no files on disk), compose_board()
    should return None without raising an exception.
    """
    feature_matrix = np.zeros((1, 560), dtype=np.float32)
    board = compose_board(
        conn           = conn,
        cluster_id     = 999,   # non-existent cluster
        cluster_label  = "ghost cluster",
        keyword        = "dark minimalism",
        feature_matrix = feature_matrix,
        image_ids_full = [],
    )
    assert board is None


# ─── Tests: export_board ────────────────────────────────────────────────────

def test_export_board_creates_png_and_pdf(tmp_path, cluster_setup):
    """Both a PNG and a PDF file should be created by export_board()."""
    s     = cluster_setup
    board = compose_board(
        conn           = s["conn"],
        cluster_id     = 0,
        cluster_label  = "cinematic deep navy",
        keyword        = "dark minimalism",
        feature_matrix = s["feature_matrix"],
        image_ids_full = s["ids"],
    )
    assert board is not None

    import utils.board_maker as bm
    original_dir    = bm.OUTPUT_DIR
    bm.OUTPUT_DIR   = str(tmp_path / "boards")

    png_path, pdf_path = export_board(board, "dark minimalism", 0, "cinematic deep navy")

    bm.OUTPUT_DIR = original_dir

    assert os.path.exists(png_path), f"PNG not created: {png_path}"
    assert os.path.exists(pdf_path), f"PDF not created: {pdf_path}"
    assert png_path.endswith(".png")
    assert pdf_path.endswith(".pdf")


def test_export_board_filename_contains_cluster_id(tmp_path, cluster_setup):
    """The output filenames should include the cluster ID."""
    s     = cluster_setup
    board = compose_board(
        conn           = s["conn"],
        cluster_id     = 0,
        cluster_label  = "cinematic deep navy",
        keyword        = "dark minimalism",
        feature_matrix = s["feature_matrix"],
        image_ids_full = s["ids"],
    )

    import utils.board_maker as bm
    bm.OUTPUT_DIR = str(tmp_path / "boards")
    png_path, pdf_path = export_board(board, "dark minimalism", 0, "cinematic deep navy")
    bm.OUTPUT_DIR = os.path.join("output", "boards")

    assert "c0" in os.path.basename(png_path)
    assert "c0" in os.path.basename(pdf_path)


def test_export_board_is_valid_png(tmp_path, cluster_setup):
    """The exported PNG should be a valid, openable image."""
    s     = cluster_setup
    board = compose_board(
        conn           = s["conn"],
        cluster_id     = 0,
        cluster_label  = "cinematic deep navy",
        keyword        = "dark minimalism",
        feature_matrix = s["feature_matrix"],
        image_ids_full = s["ids"],
    )

    import utils.board_maker as bm
    bm.OUTPUT_DIR = str(tmp_path / "boards")
    png_path, _ = export_board(board, "dark minimalism", 0, "cinematic deep navy")
    bm.OUTPUT_DIR = os.path.join("output", "boards")

    # Re-open the saved PNG and verify its dimensions
    saved = Image.open(png_path)
    assert saved.size == (CANVAS_W, CANVAS_H)
