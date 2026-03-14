"""
utils/color_extractor.py — Color Feature Extraction Pipeline
=============================================================

Extracts two color features from each downloaded image:

1. Dominant Color Palette
   Six HEX color strings representing the most visually dominant colors,
   found using K-Means clustering in the LAB color space.

2. Color Histogram Vector
   A 48-dimensional list of floats representing the distribution of colors
   across the image. Used as a numerical feature vector in Phase 5 clustering.

Why LAB color space?
---------------------
RGB is the format used by screens: each pixel is a mix of red, green, and blue
values from 0–255. It is convenient for storage but not for color analysis
because RGB distances do not match human perception. For example, two shades of
blue that look nearly identical to the human eye may have a larger RGB distance
than two colors (say, purple and orange) that look very different.

LAB (CIELAB) is designed to fix this:
  - L (Lightness): 0 = black, 100 = white
  - a (green–red axis): negative = green, positive = red/magenta
  - b (blue–yellow axis): negative = blue, positive = yellow

In LAB space, the Euclidean distance between two points corresponds closely to
how different those colors look to a human. This makes K-Means clusters in LAB
produce color groups that are perceptually coherent — exactly what we want for
mood board generation.

Why both a palette and a histogram?
-------------------------------------
The palette tells you WHAT the dominant colors are.
The histogram tells you HOW colors are distributed across the whole image.

Consider two images that both have dark blue as their dominant color:
  - Image A: entirely dark blue — calm, monochromatic
  - Image B: mostly dark blue with bright orange accents — high contrast

Both would have the same dominant color but completely different histograms.
In Phase 5, combining both features (plus CLIP embeddings) will allow HDBSCAN
to separate these two images into different mood clusters, which is the correct
aesthetic grouping.
"""

import json
import os
from typing import Optional

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans

from utils.database import get_unprocessed, update_colors


# ─── Constants ────────────────────────────────────────────────────────────────

# Number of dominant colors to extract per image.
# 6 gives a rich palette without being overwhelming in the UI.
N_DOMINANT_COLORS = 6

# Number of histogram bins per LAB channel.
# 16 bins × 3 channels = 48-dimensional vector.
# Fewer bins = coarser distribution (faster, less precise).
# More bins = finer distribution (slower, more precise, but sparse).
# 16 is a good balance for images in a mood board context.
HISTOGRAM_BINS = 16

# Resize images to this before analysis. Full-resolution analysis is wasteful —
# color distribution is well-represented at a much smaller size. 150x150 gives
# 22,500 pixels to cluster, which is plenty for K-Means.
ANALYSIS_SIZE = (150, 150)


# ─── Core Extraction Functions ────────────────────────────────────────────────

def extract_dominant_colors(image_path: str,
                             n_colors: int = N_DOMINANT_COLORS) -> list[str]:
    """
    Extract the N most visually dominant colors from an image.

    Algorithm:
    1. Open the image and resize it to ANALYSIS_SIZE for speed.
    2. Flatten all pixels into a (N, 3) array of RGB values.
    3. Convert from RGB to LAB color space.
    4. Run K-Means with n_colors clusters in LAB space.
    5. Sort clusters by how many pixels belong to them (largest = most dominant).
    6. Convert cluster centers back from LAB to RGB, then to HEX strings.

    Parameters
    ----------
    image_path : path to a local image file
    n_colors   : how many dominant colors to extract (default: 6)

    Returns
    -------
    list of HEX strings, most dominant first, e.g. ["#1A1A2E", "#E94560", ...]
    """
    # --- Load and resize ---
    img = Image.open(image_path).convert("RGB")
    img = img.resize(ANALYSIS_SIZE, Image.LANCZOS)

    # Convert to a (22500, 3) float array in [0, 1] range.
    # scikit-image's rgb2lab expects values in [0, 1], not [0, 255].
    pixels_rgb = np.array(img).reshape(-1, 3).astype(np.float32) / 255.0

    # --- Convert RGB → LAB ---
    # rgb2lab expects shape (H, W, 3), so we reshape, convert, then flatten back.
    pixels_lab = rgb2lab(pixels_rgb.reshape(1, -1, 3)).reshape(-1, 3)

    # --- K-Means clustering in LAB space ---
    # n_init=10: run K-Means 10 times with different random seeds and keep the best.
    # random_state=42: makes results reproducible.
    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(pixels_lab)

    # --- Sort clusters by dominance ---
    # kmeans.labels_ tells us which cluster each pixel belongs to.
    # We count pixels per cluster, then sort from largest to smallest.
    pixel_counts  = np.bincount(kmeans.labels_)
    sorted_indices = np.argsort(pixel_counts)[::-1]   # descending order

    # --- Convert cluster centers back to HEX ---
    # Take the sorted cluster centers (in LAB), convert to RGB, then to HEX.
    centers_lab = kmeans.cluster_centers_[sorted_indices]

    # lab2rgb also expects shape (H, W, 3)
    centers_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)

    # Clip to [0, 1] to handle tiny floating-point errors from LAB→RGB conversion,
    # then scale to [0, 255] for HEX formatting.
    centers_rgb = np.clip(centers_rgb, 0.0, 1.0)
    centers_uint8 = (centers_rgb * 255).astype(np.uint8)

    hex_colors = [
        "#{:02X}{:02X}{:02X}".format(r, g, b)
        for r, g, b in centers_uint8
    ]

    return hex_colors


def extract_color_histogram(image_path: str,
                             bins: int = HISTOGRAM_BINS) -> list[float]:
    """
    Extract a normalized color histogram in LAB space.

    The image is split into its three LAB channels. For each channel, we count
    how many pixels fall into each of the `bins` equal-width bins. The result
    is normalized so each channel's histogram sums to 1.0.

    The three histograms are concatenated into a single flat vector of length
    bins × 3 (default: 48).

    This vector can later be used directly as a feature in a clustering
    algorithm (Phase 5) alongside CLIP embeddings.

    LAB channel ranges used for binning:
        L : 0   to 100  (lightness)
        a : -128 to 127  (green–red)
        b : -128 to 127  (blue–yellow)

    Parameters
    ----------
    image_path : path to a local image file
    bins       : number of bins per channel (default: 16)

    Returns
    -------
    list of floats with length bins*3, e.g. [0.02, 0.15, ..., 0.04]
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(ANALYSIS_SIZE, Image.LANCZOS)

    pixels_rgb = np.array(img).reshape(-1, 3).astype(np.float32) / 255.0
    pixels_lab = rgb2lab(pixels_rgb.reshape(1, -1, 3)).reshape(-1, 3)

    # Define the valid range for each LAB channel.
    # These are the theoretical min/max values for CIELAB.
    channel_ranges = [
        (0.0,    100.0),   # L: lightness
        (-128.0, 127.0),   # a: green–red axis
        (-128.0, 127.0),   # b: blue–yellow axis
    ]

    histogram_vector = []

    for channel_idx, (low, high) in enumerate(channel_ranges):
        channel_values = pixels_lab[:, channel_idx]

        # np.histogram returns (counts, bin_edges).
        # We only need the counts.
        counts, _ = np.histogram(channel_values, bins=bins, range=(low, high))
        counts = counts.astype(np.float32)

        # Normalize: divide by total pixels so the histogram sums to 1.
        # This makes histograms comparable across images of different sizes.
        total = counts.sum()
        if total > 0:
            counts /= total

        histogram_vector.extend(counts.tolist())

    return histogram_vector   # 48-dimensional list of floats


# ─── Batch Processing ─────────────────────────────────────────────────────────

def process_all_pending(conn,
                        keyword: Optional[str] = None) -> dict:
    """
    Extract color features for all downloaded images that have not yet
    been processed.

    For each image, this calls extract_dominant_colors() and
    extract_color_histogram(), then updates the database with the results.

    Parameters
    ----------
    conn    : open SQLite connection
    keyword : if provided, only process images for this keyword

    Returns
    -------
    dict with keys: attempted, succeeded, failed
    """
    pending = get_unprocessed(conn, keyword=keyword)

    if not pending:
        print("[ColorExtractor] No pending images. All downloaded images have been processed.")
        return {"attempted": 0, "succeeded": 0, "failed": 0}

    print(f"[ColorExtractor] Found {len(pending)} images to process...")

    stats = {"attempted": 0, "succeeded": 0, "failed": 0}

    for i, row in enumerate(pending, start=1):
        image_id   = row["id"]
        local_path = row["local_path"]

        print(f"  [{i}/{len(pending)}] {os.path.basename(local_path)} ...", end=" ")

        stats["attempted"] += 1

        # Guard: make sure the file actually exists on disk.
        # It is possible the file was deleted after the DB was updated.
        if not os.path.exists(local_path):
            print(f"SKIPPED — file not found on disk")
            stats["failed"] += 1
            continue

        try:
            # Extract both features
            palette   = extract_dominant_colors(local_path)
            histogram = extract_color_histogram(local_path)

            # Persist to database
            update_colors(conn, image_id, palette, histogram)

            # Show the palette as HEX codes inline for visual feedback
            print(f"OK — {' '.join(palette)}")
            stats["succeeded"] += 1

        except Exception as e:
            # Catch all exceptions so one bad image doesn't crash the whole run.
            # Print the error and continue.
            print(f"FAILED — {type(e).__name__}: {e}")
            stats["failed"] += 1

    print(f"\n[ColorExtractor] Done.")
    print(f"                 Succeeded : {stats['succeeded']}")
    print(f"                 Failed    : {stats['failed']}")

    return stats


# ─── Preview Helper ───────────────────────────────────────────────────────────

def print_palette(hex_colors: list[str]) -> None:
    """
    Print a color palette to the terminal as labeled HEX values.

    Useful for quick visual inspection during development.
    Terminal colors are approximated using ANSI escape codes.

    Example output:
        Palette: #1A1A2E  #E94560  #0F3460  #533483  #16213E  #FFFFFF
    """
    print("Palette:", " ".join(hex_colors))


def palette_from_image(image_path: str) -> None:
    """
    Convenience function — extract and print the palette for a single image.
    Great for testing the extractor on one image before running the full batch.

    Usage:
        python -c "from utils.color_extractor import palette_from_image; palette_from_image('data/images/behance/dark_minimalism/abc123.jpg')"
    """
    palette = extract_dominant_colors(image_path)
    print(f"Image  : {image_path}")
    print_palette(palette)
    histogram = extract_color_histogram(image_path)
    print(f"Histogram ({len(histogram)}-dim): [{', '.join(f'{v:.3f}' for v in histogram[:6])} ...]")