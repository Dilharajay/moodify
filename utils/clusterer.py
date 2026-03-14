"""
utils/clusterer.py — UMAP Reduction + HDBSCAN Clustering + Auto-Labelling
==========================================================================

This module is the analytical heart of AestheteAI. It takes the 560-dimensional
feature matrix built in Phase 4 (512-dim CLIP embedding + 48-dim LAB histogram)
and turns it into a set of named aesthetic clusters.

The pipeline runs in three stages:

  Stage 1 — Dimensionality Reduction (UMAP)
  ------------------------------------------
  560 dimensions is far too high for a clustering algorithm to navigate efficiently.
  UMAP reduces the matrix to 10D for clustering and 2D for visualization.

  Why UMAP over PCA?
  PCA is a linear method — it finds directions in the data that maximize variance.
  UMAP is non-linear — it learns a manifold (a curved lower-dimensional surface)
  that preserves the local neighborhood relationships between points. For visual
  aesthetics, local neighborhoods are what matter: two images that look stylistically
  similar should remain close together after reduction, even if they sit in very
  different parts of the raw 560D space.

  Stage 2 — Clustering (HDBSCAN)
  --------------------------------
  HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
  finds clusters by looking for regions of high point density separated by low-density
  gaps. Unlike K-Means, it does not require you to specify how many clusters to find
  in advance — the number emerges from the data itself.

  HDBSCAN also handles noise: images that do not belong to any coherent aesthetic
  group are assigned cluster_id = -1 rather than being forced into the nearest
  cluster. This is important for mood boards — a noisy or ambiguous image should
  not pollute a coherent aesthetic group.

  Stage 3 — Automatic Labelling
  --------------------------------
  Each cluster gets a human-readable label combining:
    - A color name: nearest named color to the cluster's average dominant color
    - A mood word: chosen by CLIP cosine similarity between the cluster's embedding
      centroid and a curated vocabulary of aesthetic mood words

  Result: labels like "cinematic navy", "warm editorial", "stark geometric"
"""

import os
import json
import numpy as np
from typing import Optional

import umap
import hdbscan
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed on a server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from utils.database import get_embedded, get_connection


# ─── Constants ────────────────────────────────────────────────────────────────

EMBEDDINGS_DIR = "data/embeddings"
CLUSTERS_DIR   = "data/clusters"
PLOTS_DIR      = "output/plots"

# UMAP hyperparameters.
# n_neighbors: how many neighbors to consider when learning the manifold.
#   Higher = more global structure preserved, lower = more local detail.
#   15 is the UMAP default and works well for image datasets of a few hundred images.
# min_dist: minimum distance between points in the reduced space.
#   Lower values allow tighter clusters, which is what we want for mood boards.
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
UMAP_RANDOM_STATE = 42   # fixed seed for reproducibility

# HDBSCAN hyperparameters.
# min_cluster_size: the minimum number of images to form a cluster.
#   With 300 images, 5 is a reasonable minimum — smaller than this would produce
#   clusters too small to generate a meaningful mood board.
# min_samples: controls how conservative the clustering is.
#   Higher = fewer, more reliable clusters. None defaults to min_cluster_size.
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES      = 3

# Curated mood word vocabulary for cluster labelling.
# These 20 words were chosen to span the aesthetic spectrum commonly seen in
# design portfolios: from dark/moody to light/airy, from minimal to complex.
MOOD_WORDS = [
    "cinematic", "editorial", "minimal", "geometric", "organic",
    "industrial", "ethereal", "stark", "warm", "melancholic",
    "futuristic", "vintage", "luxurious", "raw", "serene",
    "dramatic", "playful", "mysterious", "elegant", "bold",
]

# Named color palette for color-to-name lookup.
# Each entry is (name, R, G, B). The lookup finds the nearest named color
# to a given RGB value using Euclidean distance.
NAMED_COLORS = [
    ("ivory",       255, 255, 240),
    ("cream",       255, 253, 208),
    ("white",       255, 255, 255),
    ("light grey",  211, 211, 211),
    ("silver",      192, 192, 192),
    ("mid grey",    128, 128, 128),
    ("charcoal",     54,  69,  79),
    ("slate",        70,  80,  90),
    ("dark grey",    64,  64,  64),
    ("near black",   28,  28,  28),
    ("deep navy",    13,  27,  42),
    ("navy",         23,  37,  84),
    ("cobalt",        0,  71, 171),
    ("sky blue",     87, 160, 211),
    ("steel blue",   70, 130, 180),
    ("teal",          0, 128, 128),
    ("forest green", 34,  85,  34),
    ("olive",        85,  90,  40),
    ("sage",        106, 133,  88),
    ("warm beige",  210, 180, 140),
    ("sand",        194, 178, 128),
    ("terracotta",  204,  78,  92),
    ("rust",        183,  65,  14),
    ("burnt orange",191,  87,   0),
    ("amber",       255, 191,   0),
    ("gold",        212, 175,  55),
    ("blush",       255, 182, 193),
    ("mauve",       153, 119, 123),
    ("dusty rose",  199, 143, 143),
    ("deep red",    139,   0,   0),
    ("crimson",     220,  20,  60),
    ("plum",        142,  69, 133),
    ("violet",      238, 130, 238),
    ("deep purple",  75,   0, 130),
    ("lavender",    230, 230, 250),
]


# ─── Stage 1: UMAP Dimensionality Reduction ───────────────────────────────────

def reduce_dimensions(feature_matrix: np.ndarray,
                      keyword: str,
                      n_components_cluster: int = 10,
                      n_components_vis: int = 2) -> tuple:
    """
    Reduce a high-dimensional feature matrix using UMAP.

    Two reductions are performed:
        - n_components_cluster (default 10): used as input to HDBSCAN
        - n_components_vis (default 2): used only for scatter plot visualization

    Both use the same UMAP hyperparameters and random seed for consistency,
    but are separate fits. Using 2D for visualization does NOT affect the
    clustering quality.

    Parameters
    ----------
    feature_matrix        : np.ndarray of shape (N, 560)
    keyword               : used for output filenames
    n_components_cluster  : target dimensions for clustering (default: 10)
    n_components_vis      : target dimensions for visualization (default: 2)

    Returns
    -------
    (reduced_cluster, reduced_vis)
        reduced_cluster : np.ndarray shape (N, n_components_cluster)
        reduced_vis     : np.ndarray shape (N, n_components_vis)
    """
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    safe_keyword = keyword.strip().lower().replace(" ", "_")
    n = feature_matrix.shape[0]

    # UMAP requires at least n_neighbors + 1 samples.
    # For small datasets, we reduce n_neighbors automatically.
    n_neighbors = min(UMAP_N_NEIGHBORS, n - 1)

    print(f"[Clusterer] Reducing {n} images from {feature_matrix.shape[1]}D "
          f"to {n_components_cluster}D (clustering) and {n_components_vis}D (visualization)...")

    # ── 10D reduction for clustering ──
    reducer_cluster = umap.UMAP(
        n_components  = n_components_cluster,
        n_neighbors   = n_neighbors,
        min_dist      = UMAP_MIN_DIST,
        metric        = "cosine",     # cosine distance suits normalized embeddings
        random_state  = UMAP_RANDOM_STATE,
        verbose       = False,
    )
    reduced_cluster = reducer_cluster.fit_transform(feature_matrix).astype(np.float32)

    # ── 2D reduction for visualization ──
    reducer_vis = umap.UMAP(
        n_components  = n_components_vis,
        n_neighbors   = n_neighbors,
        min_dist      = UMAP_MIN_DIST,
        metric        = "cosine",
        random_state  = UMAP_RANDOM_STATE,
        verbose       = False,
    )
    reduced_vis = reducer_vis.fit_transform(feature_matrix).astype(np.float32)

    # Save both to disk
    cluster_path = os.path.join(EMBEDDINGS_DIR, f"umap{n_components_cluster}_{safe_keyword}.npy")
    vis_path     = os.path.join(EMBEDDINGS_DIR, f"umap{n_components_vis}_{safe_keyword}.npy")
    np.save(cluster_path, reduced_cluster)
    np.save(vis_path,     reduced_vis)

    print(f"[Clusterer] Saved {n_components_cluster}D → {cluster_path}")
    print(f"[Clusterer] Saved {n_components_vis}D  → {vis_path}")

    return reduced_cluster, reduced_vis


# ─── Stage 2: HDBSCAN Clustering ──────────────────────────────────────────────

def cluster_images(reduced_matrix: np.ndarray,
                   min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
                   min_samples: int      = HDBSCAN_MIN_SAMPLES) -> np.ndarray:
    """
    Run HDBSCAN on the UMAP-reduced feature matrix.

    Returns an array of integer cluster labels, one per image.
    Label -1 means the image was classified as noise (no cluster).

    How HDBSCAN works (simplified):
    --------------------------------
    1. Build a minimum spanning tree over all points weighted by their
       "mutual reachability distance" (a density-adjusted distance).
    2. Condense the tree by removing branches with fewer than min_cluster_size points.
    3. Extract stable clusters from the condensed tree using an "excess of mass" criterion.
    4. Assign each remaining point to its cluster, or -1 if it falls in a low-density region.

    Why HDBSCAN over DBSCAN?
    DBSCAN uses a fixed epsilon radius, which means it either misses sparse clusters
    or over-merges dense ones when the data has varying densities (which image
    embeddings always do). HDBSCAN adapts the density threshold per region.

    Parameters
    ----------
    reduced_matrix   : np.ndarray of shape (N, n_components), the UMAP output
    min_cluster_size : minimum images to form a valid cluster
    min_samples      : controls outlier sensitivity (lower = more points in clusters)

    Returns
    -------
    np.ndarray of shape (N,) with integer cluster labels (>= -1)
    """
    print(f"\n[Clusterer] Running HDBSCAN "
          f"(min_cluster_size={min_cluster_size}, min_samples={min_samples})...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = min_cluster_size,
        min_samples      = min_samples,
        metric           = "euclidean",   # euclidean on the UMAP-reduced space
        cluster_selection_method = "eom", # excess of mass — finds stable clusters
    )
    labels = clusterer.fit_predict(reduced_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())

    print(f"[Clusterer] Found {n_clusters} cluster(s), {n_noise} noise point(s)")
    for cid in sorted(set(labels)):
        count = int((labels == cid).sum())
        label_str = f"Cluster {cid}" if cid >= 0 else "Noise"
        print(f"            {label_str:<12}: {count} images")

    return labels


def save_cluster_assignments(conn,
                              image_ids: list,
                              labels: np.ndarray) -> None:
    """
    Write HDBSCAN cluster labels to the database.

    Parameters
    ----------
    conn      : open SQLite connection
    image_ids : list of DB row IDs in the same order as the feature matrix rows
    labels    : np.ndarray of cluster labels from cluster_images()
    """
    for image_id, label in zip(image_ids, labels):
        conn.execute(
            "UPDATE images SET cluster_id = ? WHERE id = ?",
            (int(label), image_id),
        )
    conn.commit()
    print(f"[Clusterer] Cluster assignments saved to database.")


# ─── Stage 3: Automatic Cluster Labelling ─────────────────────────────────────

def _rgb_to_color_name(r: float, g: float, b: float) -> str:
    """
    Find the closest named color to an RGB value by Euclidean distance.

    Parameters
    ----------
    r, g, b : float values in [0, 255]

    Returns
    -------
    str — e.g. "deep navy", "charcoal", "warm beige"
    """
    best_name = "grey"
    best_dist = float("inf")

    for name, nr, ng, nb in NAMED_COLORS:
        dist = (r - nr)**2 + (g - ng)**2 + (b - nb)**2
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name


def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert a HEX color string like '#1A2E3F' to (R, G, B) floats."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _get_cluster_color_name(conn, image_ids_in_cluster: list) -> str:
    """
    Derive a color name for a cluster by averaging the dominant colors of
    all images in the cluster, then finding the nearest named color.

    Each image has up to N_DOMINANT_COLORS HEX strings stored as a JSON
    array in the DB. We take the first (most dominant) color from each
    image, average the RGB values across the cluster, then look up the
    nearest name.
    """
    r_vals, g_vals, b_vals = [], [], []

    for image_id in image_ids_in_cluster:
        row = conn.execute(
            "SELECT dominant_colors FROM images WHERE id = ?",
            (image_id,)
        ).fetchone()

        if not row or not row["dominant_colors"]:
            continue

        colors = json.loads(row["dominant_colors"])
        if not colors:
            continue

        # Use the single most dominant color from each image
        r, g, b = _hex_to_rgb(colors[0])
        r_vals.append(r)
        g_vals.append(g)
        b_vals.append(b)

    if not r_vals:
        return "grey"

    avg_r = float(np.mean(r_vals))
    avg_g = float(np.mean(g_vals))
    avg_b = float(np.mean(b_vals))

    return _rgb_to_color_name(avg_r, avg_g, avg_b)


def _get_mood_word(cluster_embedding_centroid: np.ndarray) -> str:
    """
    Find the mood word whose CLIP text embedding is most similar to the
    cluster's embedding centroid.

    The centroid is the average of all image CLIP embeddings in the cluster.
    We compare it to each mood word's pre-computed text embedding using
    cosine similarity. The word with the highest similarity wins.

    This works because CLIP embeddings are in a shared image-text space:
    images that look "cinematic" will have embeddings close to the text
    embedding of "cinematic".

    Parameters
    ----------
    cluster_embedding_centroid : np.ndarray shape (512,), the mean CLIP
                                 embedding of all images in this cluster

    Returns
    -------
    str — e.g. "cinematic", "minimal", "warm"
    """
    from transformers import CLIPProcessor, CLIPModel
    import torch
    from utils.embedder import load_clip_model, _to_numpy

    model, processor = load_clip_model()

    best_word = MOOD_WORDS[0]
    best_sim  = -float("inf")

    for word in MOOD_WORDS:
        prompt = f"a photo of {word} aesthetics"
        inputs = processor(text=[prompt], return_tensors="pt", padding=True)

        with torch.no_grad():
            text_output = model.get_text_features(**inputs)

        text_emb = _to_numpy(text_output).astype(np.float32)

        # Normalize both vectors before dot product (= cosine similarity)
        c_norm = cluster_embedding_centroid / (np.linalg.norm(cluster_embedding_centroid) + 1e-8)
        t_norm = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        sim = float(np.dot(c_norm, t_norm))

        if sim > best_sim:
            best_sim  = sim
            best_word = word

    return best_word


def label_all_clusters(conn,
                        image_ids: list,
                        labels: np.ndarray,
                        feature_matrix: np.ndarray) -> dict:
    """
    Generate and save a human-readable label for every cluster.

    Label format: "{mood_word} {color_name}"
    Examples: "cinematic deep navy", "warm terracotta", "stark charcoal"

    Noise images (label == -1) are skipped.

    Parameters
    ----------
    conn           : open SQLite connection
    image_ids      : list of DB row IDs in feature matrix row order
    labels         : cluster label array from cluster_images()
    feature_matrix : the (N, 560) unified feature matrix — used to compute
                     cluster centroids from the CLIP portion (first 512 dims)

    Returns
    -------
    dict mapping cluster_id (int) to label string
    """
    # The first 512 dimensions of the feature matrix are the CLIP embeddings
    # (before weighting). We use these for centroid-based mood word selection.
    clip_portion = feature_matrix[:, :512]

    cluster_ids = sorted(set(labels))
    label_map   = {}

    print("\n[Clusterer] Generating cluster labels...")

    for cid in cluster_ids:
        if cid == -1:
            continue   # noise — no label needed

        # Indices of all images belonging to this cluster
        mask      = labels == cid
        idx_list  = [image_ids[i] for i, m in enumerate(mask) if m]

        # Compute centroid of CLIP embeddings for this cluster
        clip_vecs = clip_portion[mask]
        centroid  = clip_vecs.mean(axis=0)

        # Get color name from dominant colors stored in DB
        color_name = _get_cluster_color_name(conn, idx_list)

        # Get mood word from CLIP text similarity
        mood_word  = _get_mood_word(centroid)

        cluster_label = f"{mood_word} {color_name}"
        label_map[cid] = cluster_label

        # Save label to all images in this cluster
        for image_id in idx_list:
            conn.execute(
                "UPDATE images SET cluster_label = ? WHERE id = ?",
                (cluster_label, image_id),
            )

        print(f"  Cluster {cid:>2} ({int(mask.sum())} images) → \"{cluster_label}\"")

    conn.commit()
    return label_map


# ─── Output Generation ────────────────────────────────────────────────────────

def plot_clusters(reduced_2d: np.ndarray,
                  labels: np.ndarray,
                  label_map: dict,
                  keyword: str) -> str:
    """
    Generate a 2D scatter plot of the clusters and save it as a PNG.

    Each cluster is plotted in a distinct color with its label in the legend.
    Noise points (label -1) are shown in light grey.

    Parameters
    ----------
    reduced_2d : np.ndarray shape (N, 2), the UMAP 2D projection
    labels     : cluster label array from cluster_images()
    label_map  : dict mapping cluster_id to label string
    keyword    : used in the title and filename

    Returns
    -------
    str — path to the saved PNG file
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    safe_keyword = keyword.strip().lower().replace(" ", "_")
    plot_path    = os.path.join(PLOTS_DIR, f"clusters_{safe_keyword}.png")

    # Color palette for up to 20 clusters.
    # These are visually distinct colors chosen to work on a dark background.
    CLUSTER_COLORS = [
        "#E94560", "#0F3460", "#533483", "#2ECC71", "#F39C12",
        "#1ABC9C", "#E74C3C", "#3498DB", "#9B59B6", "#F1C40F",
        "#E67E22", "#2980B9", "#27AE60", "#8E44AD", "#C0392B",
        "#16A085", "#D35400", "#2C3E50", "#7F8C8D", "#BDC3C7",
    ]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0D1117")   # dark background
    ax.set_facecolor("#0D1117")

    unique_ids = sorted(set(labels))
    legend_handles = []

    for i, cid in enumerate(unique_ids):
        mask = labels == cid

        if cid == -1:
            # Noise points — muted grey, no label, small dots
            ax.scatter(
                reduced_2d[mask, 0], reduced_2d[mask, 1],
                c="#444444", s=12, alpha=0.4, zorder=1,
            )
            noise_patch = mpatches.Patch(color="#444444", label=f"noise ({mask.sum()})")
            legend_handles.append(noise_patch)
        else:
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            label = label_map.get(cid, f"cluster {cid}")

            ax.scatter(
                reduced_2d[mask, 0], reduced_2d[mask, 1],
                c=color, s=28, alpha=0.85, zorder=2, edgecolors="none",
            )

            # Cluster centroid annotation — show the label at the median position
            cx = float(np.median(reduced_2d[mask, 0]))
            cy = float(np.median(reduced_2d[mask, 1]))
            ax.annotate(
                label,
                (cx, cy),
                fontsize=7.5,
                color="white",
                alpha=0.9,
                ha="center",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#00000088", edgecolor="none"),
            )

            cluster_patch = mpatches.Patch(color=color, label=f"{label} ({mask.sum()})")
            legend_handles.append(cluster_patch)

    ax.set_title(
        f'AestheteAI — "{keyword}" Clusters',
        color="white", fontsize=13, pad=14, fontfamily="monospace",
    )
    ax.tick_params(colors="#555555")
    ax.spines[:].set_color("#222222")

    legend = ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        framealpha=0.3,
        facecolor="#1A1A2E",
        edgecolor="#333333",
        labelcolor="white",
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"[Clusterer] Scatter plot saved → {plot_path}")
    return plot_path


def save_cluster_previews(conn,
                          image_ids: list,
                          labels: np.ndarray,
                          label_map: dict,
                          keyword: str,
                          n_preview: int = 9) -> list:
    """
    Save a grid of sample images for each cluster as a PNG preview.

    For each cluster, we pick up to n_preview representative images
    and tile them into a square grid. The cluster label is printed
    as a title. Useful for visually validating cluster quality.

    Parameters
    ----------
    conn       : open SQLite connection
    image_ids  : list of DB row IDs in feature matrix row order
    labels     : cluster label array
    label_map  : dict mapping cluster_id to label string
    keyword    : used in output filenames
    n_preview  : max images per preview grid (default: 9 = 3x3 grid)

    Returns
    -------
    list of paths to saved PNG preview files
    """
    os.makedirs(CLUSTERS_DIR, exist_ok=True)
    safe_keyword = keyword.strip().lower().replace(" ", "_")
    saved_paths  = []

    for cid in sorted(set(labels)):
        if cid == -1:
            continue

        # Get image IDs for this cluster
        mask     = labels == cid
        idx_list = [image_ids[i] for i, m in enumerate(mask) if m]
        sample   = idx_list[:n_preview]

        # Load image paths from DB
        img_paths = []
        for image_id in sample:
            row = conn.execute(
                "SELECT local_path FROM images WHERE id = ?", (image_id,)
            ).fetchone()
            if row and row["local_path"] and os.path.exists(row["local_path"]):
                img_paths.append(row["local_path"])

        if not img_paths:
            continue

        # Determine grid dimensions (as square as possible)
        n     = len(img_paths)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        thumb_size = 200   # pixels per thumbnail
        fig, axes  = plt.subplots(nrows, ncols,
                                   figsize=(ncols * thumb_size / 72,
                                            nrows * thumb_size / 72 + 0.6))
        fig.patch.set_facecolor("#0D1117")
        axes = np.array(axes).flatten() if n > 1 else [axes]

        label = label_map.get(cid, f"cluster {cid}")
        fig.suptitle(f'"{label}"  ({int(mask.sum())} images)',
                     color="white", fontsize=9, fontfamily="monospace", y=0.98)

        for ax_idx, ax in enumerate(axes):
            ax.set_facecolor("#0D1117")
            ax.axis("off")
            if ax_idx < len(img_paths):
                try:
                    img = Image.open(img_paths[ax_idx]).convert("RGB")
                    img.thumbnail((thumb_size, thumb_size))
                    ax.imshow(np.array(img))
                except Exception:
                    pass   # skip unreadable images silently

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(CLUSTERS_DIR, f"preview_{safe_keyword}_c{cid}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()

        print(f"[Clusterer] Preview saved → {out_path}")
        saved_paths.append(out_path)

    return saved_paths


# ─── High-Level Runner ────────────────────────────────────────────────────────

def run_clustering_pipeline(conn,
                             keyword: str,
                             min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE) -> dict:
    """
    Run the full Phase 5 pipeline for a keyword:
        1. Load the feature matrix from disk (built in Phase 4)
        2. UMAP reduction (10D for clustering, 2D for visualization)
        3. HDBSCAN clustering
        4. Save cluster assignments to DB
        5. Auto-label each cluster
        6. Generate scatter plot and per-cluster image previews

    Parameters
    ----------
    conn             : open SQLite connection
    keyword          : the search keyword to cluster
    min_cluster_size : passed to HDBSCAN (default: 5)

    Returns
    -------
    dict with keys: n_clusters, n_noise, label_map, plot_path, preview_paths
    """
    safe_keyword   = keyword.strip().lower().replace(" ", "_")
    feature_path   = os.path.join(EMBEDDINGS_DIR, f"features_{safe_keyword}.npy")

    # ── Load feature matrix ──
    if not os.path.exists(feature_path):
        print(f"[Clusterer] Feature matrix not found: {feature_path}")
        print(f"            Run: python main.py --action build_features --keyword \"{keyword}\"")
        return {}

    feature_matrix = np.load(feature_path)
    print(f"\n[Clusterer] Loaded feature matrix: {feature_matrix.shape}")

    # ── Load the image IDs in the same row order ──
    # get_embedded() returns rows in the same order every time because SQLite
    # returns rows in insertion order by default. This matches how build_feature_matrix()
    # built the matrix, so row i in the matrix corresponds to image_ids[i].
    rows      = get_embedded(conn, keyword=keyword)
    image_ids = [row["id"] for row in rows
                 if row["image_embedding_path"] and row["color_histogram"]]

    if len(image_ids) != feature_matrix.shape[0]:
        print(f"[Clusterer] Warning: {len(image_ids)} embedded records in DB "
              f"but feature matrix has {feature_matrix.shape[0]} rows.")
        print(f"            Re-run build_features to rebuild the matrix.")
        return {}

    # ── Stage 1: UMAP ──
    reduced_10d, reduced_2d = reduce_dimensions(feature_matrix, keyword)

    # ── Stage 2: HDBSCAN ──
    labels = cluster_images(reduced_10d, min_cluster_size=min_cluster_size)
    save_cluster_assignments(conn, image_ids, labels)

    # ── Stage 3: Auto-labelling ──
    label_map = label_all_clusters(conn, image_ids, labels, feature_matrix)

    # ── Output: scatter plot ──
    plot_path = plot_clusters(reduced_2d, labels, label_map, keyword)

    # ── Output: per-cluster previews ──
    preview_paths = save_cluster_previews(conn, image_ids, labels, label_map, keyword)

    # Summary
    n_clusters = len(label_map)
    n_noise    = int((labels == -1).sum())

    print(f"\n{'='*55}")
    print(f"  Phase 5 Complete — '{keyword}'")
    print(f"  Clusters found    : {n_clusters}")
    print(f"  Noise images      : {n_noise}")
    print(f"  Scatter plot      : {plot_path}")
    print(f"  Cluster previews  : {len(preview_paths)} files in {CLUSTERS_DIR}/")
    print(f"{'='*55}\n")

    return {
        "n_clusters":    n_clusters,
        "n_noise":       n_noise,
        "label_map":     label_map,
        "plot_path":     plot_path,
        "preview_paths": preview_paths,
    }
