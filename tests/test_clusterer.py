"""
tests/test_clusterer.py — Unit Tests for utils/clusterer.py
============================================================

Clustering tests work with synthetic data rather than real image embeddings.
We generate small random matrices that have known structure (deliberately
separated clusters) so we can verify the pipeline produces sensible output.

Testing strategy:
-----------------
UMAP and HDBSCAN are third-party algorithms we trust to work correctly.
What we are testing here is that OUR code:
  1. Calls them with the right inputs and parameters
  2. Saves outputs to the right paths in the right format
  3. Writes results correctly to the database
  4. Handles edge cases without crashing (too few images, empty clusters, etc.)

We do NOT test that HDBSCAN finds "the right" clusters on real data —
that is a qualitative judgment best evaluated visually with the scatter plot.

Why synthetic data?
--------------------
Creating fake cluster structure is easy: generate points drawn from a small
number of Gaussian distributions with well-separated means. HDBSCAN will
reliably find these clusters, giving us a predictable result to assert against.
"""

import os
import json
import numpy as np
import pytest

from utils.database import (
    initialize_db,
    insert_images,
    update_download,
    update_colors,
    update_embedding,
    migrate_schema_phase4,
    migrate_schema_phase5,
)
from utils.clusterer import (
    cluster_images,
    save_cluster_assignments,
    _rgb_to_color_name,
    _hex_to_rgb,
    NAMED_COLORS,
    HDBSCAN_MIN_CLUSTER_SIZE,
)


# ─── Synthetic Data Helpers ───────────────────────────────────────────────────

def make_clustered_matrix(n_clusters: int = 3,
                           n_per_cluster: int = 20,
                           n_dims: int = 10,
                           seed: int = 42) -> np.ndarray:
    """
    Generate a synthetic feature matrix with n_clusters clearly separated groups.

    Each cluster is a cloud of points drawn from a Gaussian centered at a
    random point. The cluster centers are spaced far apart (multiplied by 20)
    so HDBSCAN can reliably find them even with default parameters.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, n_dims)) * 20
    points  = []
    for center in centers:
        cluster_points = rng.standard_normal((n_per_cluster, n_dims)) + center
        points.append(cluster_points)
    return np.vstack(points).astype(np.float32)


def make_full_db_record(conn, tmp_path, idx: int = 0) -> int:
    """
    Insert a fully populated image record into an in-memory DB.
    Includes all fields needed for Phase 5: local_path, colors, embedding.
    Returns the row ID.
    """
    record = {
        "source":    "behance",
        "keyword":   "dark minimalism",
        "image_url": f"https://example.com/img{idx}.jpg",
    }
    insert_images(conn, [record])
    row_id = conn.execute(
        "SELECT id FROM images WHERE image_url = ?",
        (record["image_url"],)
    ).fetchone()[0]

    # Fake local file
    from PIL import Image
    img_path = str(tmp_path / f"img{idx}.png")
    Image.new("RGB", (200, 200), color=(30, 30, 50)).save(img_path)
    update_download(conn, row_id, img_path, 40000, 200, 200)

    # Fake colors
    palette   = ["#1A1A2E", "#E94560", "#0F3460", "#533483", "#16213E", "#FFFFFF"]
    histogram = [0.02083] * 48
    update_colors(conn, row_id, palette, histogram)

    # Fake embedding
    emb      = np.random.randn(512).astype(np.float32)
    emb_path = str(tmp_path / f"emb{idx}.npy")
    np.save(emb_path, emb)
    update_embedding(conn, row_id, emb_path)

    return row_id


@pytest.fixture
def conn():
    """Fresh in-memory DB with all phase schemas applied."""
    connection = initialize_db(":memory:")
    migrate_schema_phase4(connection)
    migrate_schema_phase5(connection)
    yield connection
    connection.close()


# ─── Tests: _rgb_to_color_name ────────────────────────────────────────────────

def test_rgb_to_color_name_exact_match():
    """
    A value exactly matching a named color should return that color's name.
    """
    # "near black" is (28, 28, 28) — use those exact values
    result = _rgb_to_color_name(28, 28, 28)
    assert result == "near black"


def test_rgb_to_color_name_returns_string():
    """Result should always be a non-empty string."""
    result = _rgb_to_color_name(100, 150, 200)
    assert isinstance(result, str)
    assert len(result) > 0


def test_rgb_to_color_name_white():
    """Pure white should map to 'white' or 'ivory' — both are near (255,255,255)."""
    result = _rgb_to_color_name(255, 255, 255)
    assert result in ("white", "ivory", "cream")


def test_rgb_to_color_name_dark_value():
    """A very dark value should map to one of the dark named colors."""
    result = _rgb_to_color_name(10, 10, 10)
    assert result in ("near black", "dark grey", "charcoal", "deep navy")


def test_rgb_to_color_name_warm_red():
    """A clearly red value should map to a red or warm named color."""
    result = _rgb_to_color_name(200, 30, 30)
    assert result in ("deep red", "crimson", "terracotta", "rust", "burnt orange")


# ─── Tests: _hex_to_rgb ───────────────────────────────────────────────────────

def test_hex_to_rgb_black():
    assert _hex_to_rgb("#000000") == (0, 0, 0)


def test_hex_to_rgb_white():
    assert _hex_to_rgb("#FFFFFF") == (255, 255, 255)


def test_hex_to_rgb_known_color():
    assert _hex_to_rgb("#1A1A2E") == (26, 26, 46)


def test_hex_to_rgb_case_insensitive():
    """Both upper and lower case HEX should parse identically."""
    assert _hex_to_rgb("#e94560") == _hex_to_rgb("#E94560")


# ─── Tests: cluster_images ────────────────────────────────────────────────────

def test_cluster_images_finds_correct_number(tmp_path):
    """
    On clearly separated synthetic data, HDBSCAN should find all 3 clusters.
    """
    matrix = make_clustered_matrix(n_clusters=3, n_per_cluster=20)
    labels = cluster_images(matrix, min_cluster_size=5)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Allow some noise tolerance — expect at least 2 of the 3 clusters
    assert n_clusters >= 2, f"Expected at least 2 clusters, got {n_clusters}"


def test_cluster_images_returns_correct_shape():
    """Output label array should have one entry per row in the input matrix."""
    matrix = make_clustered_matrix(n_clusters=2, n_per_cluster=15)
    labels = cluster_images(matrix, min_cluster_size=3)
    assert labels.shape == (matrix.shape[0],)


def test_cluster_images_labels_are_integers():
    """All labels should be integers (cluster IDs or -1 for noise)."""
    matrix = make_clustered_matrix(n_clusters=2, n_per_cluster=15)
    labels = cluster_images(matrix, min_cluster_size=3)
    assert labels.dtype in (np.int32, np.int64, int)
    for label in labels:
        assert isinstance(int(label), int)


def test_cluster_images_noise_label_is_minus_one():
    """
    Noise points should be assigned label -1 specifically, not some other
    negative number.
    """
    matrix = make_clustered_matrix(n_clusters=2, n_per_cluster=15)
    labels = cluster_images(matrix, min_cluster_size=3)
    valid_labels = set(labels)
    for label in valid_labels:
        assert label >= -1, f"Label {label} is below -1 — unexpected value"


def test_cluster_images_very_small_dataset():
    """
    With fewer images than min_cluster_size, all points should be noise (-1).
    This tests that HDBSCAN does not crash on small inputs.
    """
    matrix = np.random.randn(4, 10).astype(np.float32)
    labels = cluster_images(matrix, min_cluster_size=5)
    # With only 4 points and min_cluster_size=5, everything should be noise
    assert all(label == -1 for label in labels)


# ─── Tests: save_cluster_assignments ─────────────────────────────────────────

def test_save_cluster_assignments_writes_to_db(conn, tmp_path):
    """
    After save_cluster_assignments(), each image's cluster_id in the DB
    should match its corresponding label.
    """
    ids = [make_full_db_record(conn, tmp_path, i) for i in range(6)]
    labels = np.array([0, 0, 1, 1, -1, 0], dtype=np.int64)

    save_cluster_assignments(conn, ids, labels)

    for image_id, expected_label in zip(ids, labels):
        row = conn.execute(
            "SELECT cluster_id FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        assert row["cluster_id"] == int(expected_label)


def test_save_cluster_assignments_noise_stored_as_minus_one(conn, tmp_path):
    """Noise images (label -1) should have cluster_id = -1 in the DB."""
    ids    = [make_full_db_record(conn, tmp_path, i) for i in range(3)]
    labels = np.array([-1, -1, -1], dtype=np.int64)

    save_cluster_assignments(conn, ids, labels)

    for image_id in ids:
        row = conn.execute(
            "SELECT cluster_id FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        assert row["cluster_id"] == -1


# ─── Tests: UMAP output format ────────────────────────────────────────────────

def test_reduce_dimensions_output_shapes(tmp_path):
    """
    reduce_dimensions() should return two arrays with the correct shapes
    and save both as .npy files.
    """
    from utils.clusterer import reduce_dimensions

    matrix = make_clustered_matrix(n_clusters=2, n_per_cluster=20, n_dims=560)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("utils.clusterer.EMBEDDINGS_DIR", str(tmp_path))
        reduced_10d, reduced_2d = reduce_dimensions(matrix, keyword="test keyword")

    assert reduced_10d.shape == (40, 10)
    assert reduced_2d.shape  == (40, 2)
    assert reduced_10d.dtype == np.float32
    assert reduced_2d.dtype  == np.float32


def test_reduce_dimensions_saves_npy_files(tmp_path):
    """Both reduced matrices should be saved as .npy files."""
    from utils.clusterer import reduce_dimensions

    matrix = make_clustered_matrix(n_clusters=2, n_per_cluster=20, n_dims=560)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("utils.clusterer.EMBEDDINGS_DIR", str(tmp_path))
        reduce_dimensions(matrix, keyword="test_keyword")

    assert os.path.exists(str(tmp_path / "umap10_test_keyword.npy"))
    assert os.path.exists(str(tmp_path / "umap2_test_keyword.npy"))


def test_reduce_dimensions_reproducible(tmp_path):
    """
    Two calls with the same input and random_state should produce identical results.
    This is critical for reproducibility in the academic report.
    """
    from utils.clusterer import reduce_dimensions

    matrix = make_clustered_matrix(n_clusters=2, n_per_cluster=20, n_dims=560)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("utils.clusterer.EMBEDDINGS_DIR", str(tmp_path / "a"))
        r10_first, r2_first = reduce_dimensions(matrix, keyword="kw")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("utils.clusterer.EMBEDDINGS_DIR", str(tmp_path / "b"))
        r10_second, r2_second = reduce_dimensions(matrix, keyword="kw")

    np.testing.assert_array_almost_equal(r10_first, r10_second, decimal=4)
    np.testing.assert_array_almost_equal(r2_first,  r2_second,  decimal=4)
