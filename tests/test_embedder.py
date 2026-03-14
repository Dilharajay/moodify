"""
tests/test_embedder.py — Unit Tests for utils/embedder.py
==========================================================

Testing an ML embedding model presents a challenge: loading the real CLIP model
takes several seconds and requires ~600MB of downloaded weights. Running the full
model in every test would make the test suite very slow and require network access
on first run.

We use two strategies to handle this:

1. MOCKING for unit tests that check our code's logic (not the model's output).
   We replace CLIPModel and CLIPProcessor with lightweight fakes that return
   deterministic random tensors. This tests things like:
     - Do we correctly L2-normalize the output?
     - Do we save the .npy file to the right path?
     - Do we update the database correctly?

2. INTEGRATION-STYLE tests that do load the real model are marked with
   @pytest.mark.slow and skipped by default. Run them with:
       pytest -m slow
   This lets the normal test suite (pytest) stay fast while still having
   a way to test real model behavior.

What is unittest.mock.patch?
-----------------------------
`patch` temporarily replaces a real object with a fake (Mock) during a test.
The syntax:
    with patch("utils.embedder.CLIPModel.from_pretrained") as mock_model:
        mock_model.return_value = FakeModel()

means "inside this block, whenever code calls CLIPModel.from_pretrained(),
give it FakeModel() instead of the real thing."

The 'with' block ensures the original is restored after the test, even if
the test fails. This is critical — a failed mock restoration could corrupt
other tests in the suite.
"""

import os
import json
import numpy as np
import pytest
from io import BytesIO
from PIL import Image
from unittest.mock import patch, MagicMock

import torch

from utils.database import initialize_db, insert_images, update_download, update_colors
from utils.embedder import (
    cosine_similarity,
    build_feature_matrix,
    EMBEDDING_DIM,
    IMAGE_EMBEDDINGS_DIR,
    TEXT_EMBEDDINGS_DIR,
    EMBEDDINGS_DIR,
)


# ─── Fake Model Infrastructure ────────────────────────────────────────────────

def make_fake_embedding(seed: int = 42) -> torch.Tensor:
    """
    Create a deterministic fake 512-dim embedding tensor.

    Using a fixed seed makes the value reproducible across test runs.
    We return a raw (un-normalized) tensor to test that our code normalizes it.
    Shape: (1, 512) because CLIP returns batch dimension even for single inputs.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randn(1, EMBEDDING_DIM, generator=rng)


class FakeCLIPModel:
    """
    Minimal fake that mimics the CLIPModel interface our code uses.

    Our code only calls:
        model.get_image_features(**inputs)
        model.get_text_features(**inputs)
        model.eval()

    So FakeCLIPModel only needs to implement these three methods.
    The returned tensors are random but deterministic (fixed seed).
    """
    def eval(self):
        return self   # eval() returns self in real PyTorch models

    def get_image_features(self, **kwargs):
        return make_fake_embedding(seed=1)

    def get_text_features(self, **kwargs):
        return make_fake_embedding(seed=2)


class FakeCLIPProcessor:
    """
    Minimal fake that mimics CLIPProcessor.
    Our code calls processor(images=...) and processor(text=...).
    We just return an empty dict — the fake model ignores the inputs anyway.
    """
    def __call__(self, images=None, text=None, return_tensors=None, padding=None):
        return {}


def make_test_image(tmp_path, filename="test.png",
                    color=(26, 26, 46), size=(300, 300)) -> str:
    """Create a solid-colour test image and return its path."""
    img  = Image.new("RGB", size, color=color)
    path = str(tmp_path / filename)
    img.save(path, format="PNG")
    return path


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def conn():
    """Fresh in-memory database with Phase 4 schema."""
    from utils.database import migrate_schema_phase4
    connection = initialize_db(":memory:")
    migrate_schema_phase4(connection)
    yield connection
    connection.close()


@pytest.fixture
def downloaded_record(conn, tmp_path):
    """
    A database record that has been scraped, downloaded, and had colors extracted.
    This is the state required before embedding can happen.
    """
    # Insert the scrape record
    record = {
        "source": "behance",
        "keyword": "dark minimalism",
        "title": "Test Project",
        "owner": "designer",
        "image_url": "https://example.com/test.jpg",
    }
    insert_images(conn, [record])
    row_id = conn.execute("SELECT id FROM images LIMIT 1").fetchone()[0]

    # Create and register a local image file
    img_path = make_test_image(tmp_path, "test.png")
    update_download(conn, row_id, img_path, 50000, 300, 300)

    # Register a color histogram (as would be done by Phase 3)
    palette   = ["#1A1A2E"] * 6
    histogram = [0.02083] * 48   # uniform distribution, sums to 1.0 per channel
    update_colors(conn, row_id, palette, histogram)

    return row_id


# ─── Tests: cosine_similarity ─────────────────────────────────────────────────

def test_cosine_similarity_identical_vectors():
    """
    The cosine similarity of a vector with itself should be exactly 1.0.
    """
    v = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal_vectors():
    """
    Orthogonal vectors (at 90 degrees) have cosine similarity 0.0.
    """
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_opposite_vectors():
    """
    Opposite vectors (pointing in exactly opposite directions) have similarity -1.0.
    """
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_similarity_range():
    """
    Cosine similarity should always be in the range [-1.0, 1.0].
    """
    rng = np.random.default_rng(seed=0)
    for _ in range(100):
        a = rng.standard_normal(512).astype(np.float32)
        b = rng.standard_normal(512).astype(np.float32)
        sim = cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0, f"Similarity {sim} is out of range"


def test_cosine_similarity_zero_vector():
    """
    If either vector is all zeros, similarity should return 0.0 without crashing.
    (Division by zero protection.)
    """
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)
    assert cosine_similarity(a, b) == 0.0
    assert cosine_similarity(b, a) == 0.0


# ─── Tests: encode_image (with mocked model) ─────────────────────────────────

def test_encode_image_returns_normalized_array(tmp_path):
    """
    encode_image() should return a float32 array of shape (512,) with L2 norm ≈ 1.0.
    We mock the CLIP model so no weights are loaded.
    """
    from utils.embedder import encode_image

    img_path = make_test_image(tmp_path)

    with patch("utils.embedder.CLIPModel.from_pretrained",
               return_value=FakeCLIPModel()), \
         patch("utils.embedder.CLIPProcessor.from_pretrained",
               return_value=FakeCLIPProcessor()):
        # Reset the cached model so our mock is picked up
        import utils.embedder as embedder_module
        embedder_module._model     = None
        embedder_module._processor = None

        model, processor = embedder_module.load_clip_model()
        embedding = encode_image(img_path, model, processor)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape   == (EMBEDDING_DIM,)
    assert embedding.dtype   == np.float32
    # L2 norm of a normalized vector should be very close to 1.0
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)


# ─── Tests: encode_images_batch (with mocked model) ──────────────────────────

def test_encode_images_batch_saves_npy_file(conn, downloaded_record, tmp_path):
    """
    After encoding, a .npy file should exist on disk and the DB should have
    the path recorded in image_embedding_path.
    """
    from utils.embedder import encode_images_batch
    import utils.embedder as embedder_module

    with patch("utils.embedder.CLIPModel.from_pretrained",
               return_value=FakeCLIPModel()), \
         patch("utils.embedder.CLIPProcessor.from_pretrained",
               return_value=FakeCLIPProcessor()):
        embedder_module._model     = None
        embedder_module._processor = None

        # Override the embeddings directory so we don't write to the real filesystem
        with patch("utils.embedder.IMAGE_EMBEDDINGS_DIR", str(tmp_path / "embeddings" / "images")):
            stats = encode_images_batch(conn, keyword="dark minimalism")

    assert stats["succeeded"] == 1
    assert stats["failed"]    == 0

    # Check the DB was updated
    row = conn.execute(
        "SELECT image_embedding_path FROM images WHERE id = ?",
        (downloaded_record,)
    ).fetchone()
    assert row["image_embedding_path"] is not None

    # Check the file exists and is a valid numpy array
    npy_path = row["image_embedding_path"]
    assert os.path.exists(npy_path)
    loaded = np.load(npy_path)
    assert loaded.shape == (EMBEDDING_DIM,)
    assert loaded.dtype == np.float32


def test_encode_images_batch_idempotent(conn, downloaded_record, tmp_path):
    """
    Running encode_images_batch() twice should not re-process already-embedded images.
    """
    from utils.embedder import encode_images_batch
    import utils.embedder as embedder_module

    with patch("utils.embedder.CLIPModel.from_pretrained",
               return_value=FakeCLIPModel()), \
         patch("utils.embedder.CLIPProcessor.from_pretrained",
               return_value=FakeCLIPProcessor()):
        embedder_module._model     = None
        embedder_module._processor = None

        with patch("utils.embedder.IMAGE_EMBEDDINGS_DIR", str(tmp_path / "embeddings" / "images")):
            stats_first  = encode_images_batch(conn, keyword="dark minimalism")
            stats_second = encode_images_batch(conn, keyword="dark minimalism")

    assert stats_first["succeeded"]  == 1
    assert stats_second["attempted"] == 0   # Nothing to do on second run


def test_encode_images_batch_no_pending(conn):
    """
    With no downloaded images, the batch encoder should return zeros without error.
    """
    from utils.embedder import encode_images_batch
    stats = encode_images_batch(conn, keyword="nonexistent keyword")
    assert stats["attempted"] == 0


# ─── Tests: build_feature_matrix ─────────────────────────────────────────────

def test_build_feature_matrix_correct_shape(conn, downloaded_record, tmp_path):
    """
    The feature matrix should have shape (N_images, 560) — 512 CLIP + 48 histogram.
    """
    from utils.embedder import build_feature_matrix
    from utils.database import update_embedding

    # Create a fake .npy embedding file and register it in the DB
    fake_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    emb_path = str(tmp_path / "fake_emb.npy")
    np.save(emb_path, fake_embedding)
    update_embedding(conn, downloaded_record, emb_path)

    with patch("utils.embedder.EMBEDDINGS_DIR", str(tmp_path)):
        matrix, image_ids, npy_path = build_feature_matrix(
            conn, keyword="dark minimalism"
        )

    assert matrix is not None
    assert matrix.shape == (1, EMBEDDING_DIM + 48)  # 512 + 48 = 560
    assert matrix.dtype == np.float32
    assert len(image_ids) == 1
    assert image_ids[0] == downloaded_record


def test_build_feature_matrix_saves_file(conn, downloaded_record, tmp_path):
    """
    build_feature_matrix() should save the matrix as a .npy file on disk.
    """
    from utils.embedder import build_feature_matrix
    from utils.database import update_embedding

    fake_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    emb_path = str(tmp_path / "fake_emb.npy")
    np.save(emb_path, fake_embedding)
    update_embedding(conn, downloaded_record, emb_path)

    with patch("utils.embedder.EMBEDDINGS_DIR", str(tmp_path)):
        matrix, image_ids, npy_path = build_feature_matrix(
            conn, keyword="dark minimalism"
        )

    assert npy_path is not None
    assert os.path.exists(npy_path)

    # Load it back and confirm it matches
    loaded = np.load(npy_path)
    np.testing.assert_array_equal(loaded, matrix)


def test_build_feature_matrix_no_embeddings(conn):
    """
    If no images have embeddings, build_feature_matrix should return None
    without crashing.
    """
    from utils.embedder import build_feature_matrix
    matrix, image_ids, npy_path = build_feature_matrix(conn, keyword="missing")
    assert matrix    is None
    assert image_ids == []
    assert npy_path  is None


def test_build_feature_matrix_weights_applied(conn, downloaded_record, tmp_path):
    """
    Different weight_clip and weight_color values should produce different matrices.
    This verifies that the weights are actually being applied, not ignored.
    """
    from utils.embedder import build_feature_matrix
    from utils.database import update_embedding

    fake_embedding = np.ones(EMBEDDING_DIM, dtype=np.float32)  # all-ones for easy math
    emb_path = str(tmp_path / "fake_emb.npy")
    np.save(emb_path, fake_embedding)
    update_embedding(conn, downloaded_record, emb_path)

    with patch("utils.embedder.EMBEDDINGS_DIR", str(tmp_path / "a")):
        matrix_default, _, _ = build_feature_matrix(
            conn, keyword="dark minimalism", weight_clip=1.0, weight_color=0.5
        )

    # Re-register (update_embedding is idempotent)
    update_embedding(conn, downloaded_record, emb_path)

    with patch("utils.embedder.EMBEDDINGS_DIR", str(tmp_path / "b")):
        matrix_equal, _, _ = build_feature_matrix(
            conn, keyword="dark minimalism", weight_clip=1.0, weight_color=1.0
        )

    assert matrix_default is not None
    assert matrix_equal   is not None
    # The color segment (last 48 values) should differ between the two matrices
    assert not np.allclose(matrix_default[0, 512:], matrix_equal[0, 512:])
