"""
utils/embedder.py — CLIP Embedding Pipeline
=============================================

This module uses the CLIP model (openai/clip-vit-base-patch32) to convert
images and text into 512-dimensional float32 vectors in a shared semantic space.

What is CLIP?
--------------
CLIP (Contrastive Language-Image Pretraining) is a neural network trained by
OpenAI on 400 million image-text pairs scraped from the internet. The key
insight of CLIP is that it trains an image encoder and a text encoder jointly,
so that matching image-text pairs end up close together in the embedding space,
and non-matching pairs end up far apart.

This shared space is extremely useful for us because:
  - Images that look aesthetically similar have similar embeddings
  - The keyword "dark minimalism" has an embedding close to images that
    actually look dark and minimal
  - We can measure image-to-image similarity AND image-to-text similarity
    using the same distance metric (cosine similarity)

Model choice: openai/clip-vit-base-patch32
-------------------------------------------
  - "vit" = Vision Transformer architecture (not CNN-based)
  - "base" = the smaller of OpenAI's released models (vs "large")
  - "patch32" = each image is divided into 32x32 pixel patches
  - Output: 512-dimensional embedding per image or text input
  - Runs on CPU in ~0.5–2s per image (acceptable for a batch job)
  - HuggingFace model card: https://huggingface.co/openai/clip-vit-base-patch32

Storage design: .npy files + DB path reference
-----------------------------------------------
We save each embedding as a NumPy .npy file rather than storing the raw float
array in SQLite. Here is why:

  SQLite TEXT approach (wrong for this):
    - A 512-dim float32 array as JSON string = ~4KB of text per row
    - Loading it requires json.loads() + list-to-array conversion every time
    - No batch loading — must query row by row

  .npy file approach (correct):
    - Binary format, ~2KB per file (float32 precision, compact)
    - np.load() is instant — memory-mapped if needed
    - All embeddings for a keyword can be loaded in one np.load() call
    - Compatible with scikit-learn, FAISS, and every other ML library

The DB stores just the file path (a short string), keeping the DB lean while
giving us a queryable index of what has been embedded.

Directory layout:
    data/embeddings/
        images/
            {md5_hash}.npy         <- one file per image
        text/
            {safe_keyword}.npy     <- one file per keyword
        features_{safe_keyword}.npy  <- unified matrix (CLIP + histogram)
"""

import os
import hashlib
import numpy as np
from typing import Optional
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

from utils.database import get_unembedded, get_embedded, update_embedding


# ─── Tensor Extraction Helper ─────────────────────────────────────────────────

def _to_numpy(model_output) -> np.ndarray:
    """
    Safely extract a flat float32 NumPy array from a CLIP model output.

    Why this helper exists:
    ------------------------
    The transformers library has changed its return types across versions.
    In some versions, get_image_features() and get_text_features() return a
    plain torch.Tensor. In others (particularly newer builds), they may return
    a BaseModelOutputWithPooling named-tuple object that wraps the tensor.

    This helper handles both cases in one place so that encode_image() and
    encode_keyword() do not need to duplicate the logic.

    Extraction priority:
        1. If the output has a .pooler_output attribute (BaseModelOutputWithPooling),
           use that — it is the CLS token embedding, which is what CLIP uses.
        2. If it has a .last_hidden_state, take the first token (CLS) from it.
        3. Otherwise assume it is a plain tensor and call .squeeze() directly.

    In all cases the result is squeezed to remove the batch dimension (we always
    process one item at a time) and cast to float32.
    """
    if hasattr(model_output, "pooler_output"):
        # BaseModelOutputWithPooling — use the pooled representation
        tensor = model_output.pooler_output
    elif hasattr(model_output, "last_hidden_state"):
        # Fallback: take the CLS token (position 0) from the last hidden state
        tensor = model_output.last_hidden_state[:, 0, :]
    else:
        # Plain tensor — the expected path for most transformers versions
        tensor = model_output

    return tensor.squeeze().numpy().astype(np.float32)


# ─── Constants ────────────────────────────────────────────────────────────────

# The HuggingFace model identifier for CLIP base-patch32.
# On first run this will download ~600MB of model weights to ~/.cache/huggingface/.
# Subsequent runs load from the local cache (fast).
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Embedding output dimension for this model variant.
EMBEDDING_DIM = 512

# Directory layout for embeddings on disk.
EMBEDDINGS_DIR        = "data/embeddings"
IMAGE_EMBEDDINGS_DIR  = os.path.join(EMBEDDINGS_DIR, "images")
TEXT_EMBEDDINGS_DIR   = os.path.join(EMBEDDINGS_DIR, "text")


# ─── Model Loading ────────────────────────────────────────────────────────────

# Module-level cache so the model is only loaded once per Python process.
# Loading CLIP takes a few seconds; we do not want to repeat it for every batch.
_model     = None
_processor = None


def load_clip_model() -> tuple:
    """
    Load the CLIP model and processor, using a module-level cache so they are
    only initialized once even if this function is called multiple times.

    Returns
    -------
    (model, processor) — the CLIPModel and CLIPProcessor instances

    What is a CLIPProcessor?
    ------------------------
    CLIPProcessor is a combined pre-processor that handles both images and text.
    For images, it resizes and normalizes pixel values to the range the model
    expects. For text, it tokenizes the string into integer token IDs. Both
    outputs are PyTorch tensors ready to pass into the model.
    """
    global _model, _processor

    if _model is None:
        print(f"[Embedder] Loading CLIP model: {CLIP_MODEL_NAME}")
        print(f"           (First run downloads ~600MB to ~/.cache/huggingface/)")

        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME)

        # Put model in evaluation mode. This disables dropout layers and other
        # training-specific behaviours, making inference deterministic and slightly
        # faster. Always do this before inference — never skip it.
        _model.eval()

        print(f"[Embedder] Model loaded successfully.\n")

    return _model, _processor


# ─── Image Encoding ───────────────────────────────────────────────────────────

def encode_image(image_path: str,
                 model: CLIPModel,
                 processor: CLIPProcessor) -> np.ndarray:
    """
    Encode a single image into a 512-dimensional CLIP embedding.

    Steps:
    1. Open the image with Pillow and convert to RGB (CLIP expects 3-channel input).
    2. Pass it through CLIPProcessor to resize, normalize, and convert to a tensor.
    3. Pass the tensor through the CLIP vision encoder.
    4. L2-normalize the output embedding.
    5. Return as a float32 NumPy array.

    Why L2 normalization?
    ----------------------
    L2 normalization scales the vector so its magnitude (length) is exactly 1.0.
    After normalization, the dot product of two vectors equals their cosine
    similarity. This matters because cosine similarity (angle between vectors)
    is the standard distance metric in embedding spaces — it is invariant to
    magnitude, which makes embeddings from images of different brightness or
    contrast directly comparable.

    Parameters
    ----------
    image_path : path to a local image file
    model      : loaded CLIPModel instance
    processor  : loaded CLIPProcessor instance

    Returns
    -------
    np.ndarray of shape (512,) and dtype float32
    """
    # Load image — convert to RGB to handle PNG with alpha channel (RGBA)
    # and greyscale images, which CLIP cannot process natively.
    image = Image.open(image_path).convert("RGB")

    # Preprocess: resize to 224x224, normalize pixel values to [-1, 1]
    # return_tensors="pt" means "return PyTorch tensors"
    inputs = processor(images=image, return_tensors="pt")

    # No gradient computation needed for inference.
    # torch.no_grad() reduces memory usage and speeds things up.
    with torch.no_grad():
        # get_image_features() runs the vision encoder (the ViT part of CLIP)
        # and returns the embedding before the final projection layer.
        embedding = model.get_image_features(**inputs)

    # Convert to a flat float32 NumPy array using the safe helper.
    # _to_numpy() handles both plain-tensor and BaseModelOutputWithPooling
    # return types, which vary across transformers versions.
    embedding_np = _to_numpy(embedding)
    norm = np.linalg.norm(embedding_np)
    if norm > 0:
        embedding_np = embedding_np / norm

    return embedding_np


def encode_images_batch(conn,
                        keyword: Optional[str] = None,
                        batch_size: int = 16) -> dict:
    """
    Encode all downloaded images that do not yet have a CLIP embedding.

    Processes images in batches to balance memory usage and speed. Each
    embedding is saved as a .npy file and the DB is updated with the path.

    Parameters
    ----------
    conn       : open SQLite connection
    keyword    : if provided, only embed images for this keyword
    batch_size : number of images to process before printing a progress update
                 (not a GPU batch — we process one image at a time on CPU)

    Returns
    -------
    dict with keys: attempted, succeeded, failed
    """
    os.makedirs(IMAGE_EMBEDDINGS_DIR, exist_ok=True)

    pending = get_unembedded(conn, keyword=keyword)

    if not pending:
        print("[Embedder] No pending images. All downloaded images are already embedded.")
        return {"attempted": 0, "succeeded": 0, "failed": 0}

    print(f"[Embedder] Found {len(pending)} images to embed...")
    model, processor = load_clip_model()

    stats = {"attempted": 0, "succeeded": 0, "failed": 0}

    for i, row in enumerate(pending, start=1):
        image_id   = row["id"]
        local_path = row["local_path"]

        print(f"  [{i}/{len(pending)}] {os.path.basename(local_path)} ...", end=" ")
        stats["attempted"] += 1

        # Guard: file may have been deleted after DB was updated
        if not os.path.exists(local_path):
            print("SKIPPED — file not found on disk")
            stats["failed"] += 1
            continue

        try:
            # Encode the image
            embedding = encode_image(local_path, model, processor)

            # Build the output filename from an MD5 hash of the local path.
            # Using the path (not the image URL) ensures the filename is stable
            # even if the source URL changes.
            name_hash = hashlib.md5(local_path.encode()).hexdigest()
            npy_path  = os.path.join(IMAGE_EMBEDDINGS_DIR, f"{name_hash}.npy")

            # Save embedding to disk as a .npy file
            np.save(npy_path, embedding)

            # Record the path in the database
            update_embedding(conn, image_id, npy_path)

            print(f"OK — shape {embedding.shape}, norm {np.linalg.norm(embedding):.4f}")
            stats["succeeded"] += 1

        except Exception as e:
            print(f"FAILED — {type(e).__name__}: {e}")
            stats["failed"] += 1

    print(f"\n[Embedder] Image embedding complete.")
    print(f"           Succeeded : {stats['succeeded']}")
    print(f"           Failed    : {stats['failed']}")

    return stats


# ─── Text (Keyword) Encoding ──────────────────────────────────────────────────

def encode_keyword(keyword: str,
                   model: CLIPModel,
                   processor: CLIPProcessor) -> np.ndarray:
    """
    Encode a text keyword into a 512-dimensional CLIP text embedding.

    CLIP's text encoder processes a short string (up to 77 tokens) and returns
    an embedding in the same 512-dim space as image embeddings. This is the
    "language-image pretraining" part — images and text that belong together
    end up close in this shared space.

    We wrap the keyword in a natural-language template ("a photo of {keyword}")
    because CLIP was trained on captions, not bare keyword strings. Using a
    template like this consistently outperforms bare keywords.

    Parameters
    ----------
    keyword   : the search keyword, e.g. "dark minimalism"
    model     : loaded CLIPModel instance
    processor : loaded CLIPProcessor instance

    Returns
    -------
    np.ndarray of shape (512,) and dtype float32
    """
    # Template wrapping — "a photo of X" is the standard CLIP prompt template.
    # It primes the model to think about visual appearance rather than abstract concepts.
    prompt = f"a photo of {keyword}"

    inputs = processor(text=[prompt], return_tensors="pt", padding=True)

    with torch.no_grad():
        # get_text_features() runs the text encoder (a Transformer)
        embedding = model.get_text_features(**inputs)

    embedding_np = _to_numpy(embedding)

    # L2 normalize — same reason as image embeddings (enables cosine similarity)
    norm = np.linalg.norm(embedding_np)
    if norm > 0:
        embedding_np = embedding_np / norm

    return embedding_np


def save_keyword_embedding(keyword: str) -> str:
    """
    Encode a keyword and save its embedding to disk.

    Returns the path to the saved .npy file.
    Safe to call repeatedly — if the file already exists, it is overwritten
    (text embeddings are cheap to compute and deterministic).

    Parameters
    ----------
    keyword : the search keyword, e.g. "dark minimalism"

    Returns
    -------
    str — path to the saved .npy file
    """
    os.makedirs(TEXT_EMBEDDINGS_DIR, exist_ok=True)

    model, processor = load_clip_model()
    embedding = encode_keyword(keyword, model, processor)

    # Build a filesystem-safe filename from the keyword
    safe_keyword = keyword.strip().lower().replace(" ", "_")
    npy_path = os.path.join(TEXT_EMBEDDINGS_DIR, f"{safe_keyword}.npy")

    np.save(npy_path, embedding)
    print(f"[Embedder] Text embedding saved → {npy_path}")

    return npy_path


# ─── Unified Feature Matrix ───────────────────────────────────────────────────

def build_feature_matrix(conn,
                         keyword: str,
                         weight_clip: float = 1.0,
                         weight_color: float = 0.5) -> tuple:
    """
    Build a unified feature matrix for all embedded images matching a keyword.

    For each image, this concatenates:
        - Its 512-dim CLIP image embedding (normalized)
        - Its 48-dim color histogram from Phase 3 (normalized, then scaled)

    The two feature groups are given different weights before concatenation
    so that the clustering in Phase 5 is not dominated by one signal.
    The defaults (CLIP weight=1.0, color weight=0.5) give CLIP twice the
    influence of the color histogram, which reflects CLIP's richer representation.

    The matrix is saved to disk as a .npy file for use in Phase 5.

    Parameters
    ----------
    conn         : open SQLite connection
    keyword      : filter to images with this keyword
    weight_clip  : scalar multiplier applied to the CLIP embedding before concat
    weight_color : scalar multiplier applied to the color histogram before concat

    Returns
    -------
    (feature_matrix, image_ids, npy_path)
        feature_matrix : np.ndarray of shape (N, 560), dtype float32
        image_ids      : list of DB row IDs in the same row order as the matrix
        npy_path       : path where the matrix was saved

    Why return image_ids?
    ----------------------
    Phase 5 will assign each row of the matrix to a cluster. To know which
    database record belongs to which cluster, we need the row-to-ID mapping.
    """
    import json

    rows = get_embedded(conn, keyword=keyword)

    if not rows:
        print(f"[Embedder] No embedded images found for keyword '{keyword}'.")
        return None, [], None

    print(f"[Embedder] Building feature matrix for '{keyword}' — {len(rows)} images...")

    feature_vectors = []
    image_ids       = []

    for row in rows:
        image_id       = row["id"]
        embedding_path = row["image_embedding_path"]
        color_hist_str = row["color_histogram"]

        # Skip rows with missing data — they should not be included in clustering
        if not embedding_path or not os.path.exists(embedding_path):
            continue
        if not color_hist_str:
            continue

        # Load CLIP image embedding from .npy file
        clip_embedding = np.load(embedding_path).astype(np.float32)

        # Parse color histogram from JSON string stored in DB
        color_histogram = np.array(json.loads(color_hist_str), dtype=np.float32)

        # Apply weights and concatenate into a single feature vector.
        # Shape: (512,) + (48,) → (560,)
        weighted_clip  = clip_embedding   * weight_clip
        weighted_color = color_histogram  * weight_color
        unified_vector = np.concatenate([weighted_clip, weighted_color])

        feature_vectors.append(unified_vector)
        image_ids.append(image_id)

    if not feature_vectors:
        print(f"[Embedder] Could not build matrix — check that images have both embeddings and color histograms.")
        return None, [], None

    # Stack into a 2D matrix: shape (N_images, 560)
    feature_matrix = np.stack(feature_vectors, axis=0)

    # Save to disk
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    safe_keyword = keyword.strip().lower().replace(" ", "_")
    npy_path     = os.path.join(EMBEDDINGS_DIR, f"features_{safe_keyword}.npy")
    np.save(npy_path, feature_matrix)

    print(f"[Embedder] Feature matrix shape: {feature_matrix.shape}")
    print(f"           Saved → {npy_path}")

    return feature_matrix, image_ids, npy_path


# ─── Similarity Utilities ────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalized embedding vectors.

    Cosine similarity = dot product of two unit vectors.
    Range: -1.0 (opposite) to 1.0 (identical).
    For L2-normalized CLIP embeddings, typical similar-image values
    are in the 0.7–0.99 range.

    This is a standalone utility — useful for debugging or exploring the
    embedding space during development.

    Example:
        emb_a = np.load("data/embeddings/images/abc.npy")
        emb_b = np.load("data/embeddings/images/def.npy")
        sim   = cosine_similarity(emb_a, emb_b)
        print(f"Similarity: {sim:.3f}")
    """
    # Both vectors should already be L2-normalized from encode_image().
    # We re-normalize here defensively in case they were loaded from disk
    # and somehow have a non-unit norm.
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a / norm_a, b / norm_b))


def find_most_similar(query_path: str,
                      conn,
                      keyword: Optional[str] = None,
                      top_k: int = 5) -> list[dict]:
    """
    Find the top-K most similar images to a query image using cosine similarity
    over CLIP embeddings.

    This is a brute-force linear search — fine for hundreds of images.
    Phase 5 will use HDBSCAN for proper clustering, but this function is
    useful for sanity-checking the embeddings during development.

    Parameters
    ----------
    query_path : local path to the query image
    conn       : open SQLite connection
    keyword    : if provided, only search within this keyword's images
    top_k      : how many similar images to return

    Returns
    -------
    list of dicts: [{"id": ..., "local_path": ..., "similarity": ...}, ...]
    sorted by similarity descending
    """
    model, processor = load_clip_model()
    query_embedding  = encode_image(query_path, model, processor)

    candidates = get_embedded(conn, keyword=keyword)
    results    = []

    for row in candidates:
        emb_path = row["image_embedding_path"]
        if not emb_path or not os.path.exists(emb_path):
            continue

        candidate_embedding = np.load(emb_path)
        sim = cosine_similarity(query_embedding, candidate_embedding)

        results.append({
            "id":         row["id"],
            "local_path": row["local_path"],
            "title":      row["title"],
            "source":     row["source"],
            "similarity": sim,
        })

    # Sort by similarity descending, return top K
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]