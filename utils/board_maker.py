"""
utils/board_maker.py — Mood Board Layout Engine + Export
=========================================================

This module takes a cluster of images from Phase 5 and composes them into
a designed mood board using the hero + grid layout pattern.

Layout anatomy (2400 × 1600 px canvas):
----------------------------------------

  ┌────────────────────────────────────────────────────────┐
  │  HEADER BAR  — cluster label title (left) + meta (right) │  80px
  ├──────────────────────────┬─────────────────────────────┤
  │                          │  IMG  │  IMG  │  IMG  │      │
  │       HERO IMAGE         ├───────┼───────┼───────┤      │
  │    (cluster centroid)    │  IMG  │  IMG  │  IMG  │      │  1400px
  │                          │                             │
  ├──────────────────────────┴─────────────────────────────┤
  │  ██  ██  ██  ██  ██  ██   COLOR PALETTE STRIP          │  80px
  ├────────────────────────────────────────────────────────┤
  │  keyword • N images • AestheteAI                        │  40px
  └────────────────────────────────────────────────────────┘
  Total: 2400 × 1600 px

Hero image selection:
---------------------
The hero is the image whose CLIP embedding is closest to the cluster centroid
(the mean embedding of all images in the cluster). This makes the hero the most
"representative" image of the cluster's aesthetic — not just the largest or first.

Color palette strip:
--------------------
The palette strip shows the 6 average dominant colors across all cluster images.
Each color in the strip is the mean RGB value of the corresponding rank position
across all images. So "most dominant color slot 1" is the average of each image's
most dominant color, giving a representative cluster palette.

Canvas size rationale:
----------------------
2400 × 1600 px at 200dpi = an A4 landscape page (297 × 210mm).
This matches the standard design deliverable format. At 200dpi the images
are sharp enough for print while keeping file sizes manageable.
"""

import os
import json
import numpy as np
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

# reportlab is used for PDF export — it accepts PIL images via BytesIO
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO

# ─── Canvas Constants ─────────────────────────────────────────────────────────

CANVAS_W = 2400
CANVAS_H = 1600

HEADER_H = 80  # cluster label bar at top
PALETTE_H = 80  # color strip above footer
FOOTER_H = 40  # keyword / count / branding bar

# Image panel height = canvas minus header, palette, footer
PANEL_H = CANVAS_H - HEADER_H - PALETTE_H - FOOTER_H  # 1400px

# Hero image takes 60% of canvas width; grid takes the remaining 40%
HERO_W = int(CANVAS_W * 0.60)  # 1440px
GRID_W = CANVAS_W - HERO_W  # 960px
GRID_COLS = 2
GRID_ROWS = 3

# Supporting grid: 2 columns × 3 rows = 6 images maximum
GRID_CELL_W = GRID_W // GRID_COLS  # 480px
GRID_CELL_H = PANEL_H // GRID_ROWS  # ~466px

# Narrow gap between grid cells (purely aesthetic)
GAP = 4

# Color palette
BG_COLOR = (13, 13, 13)  # near-black background
HEADER_COLOR = (18, 18, 28)  # slightly lighter for header
FOOTER_COLOR = (10, 10, 18)  # slightly darker for footer
TEXT_PRIMARY = (255, 255, 255)  # white
TEXT_SECONDARY = (160, 160, 170)  # muted grey

OUTPUT_DIR = os.path.join("output", "boards")


# ─── Font Helper ──────────────────────────────────────────────────────────────

def _get_font(size: int, bold: bool = False):
    """
    Load a font for drawing text on the canvas.

    We try to load a system font first. If none is available (which can happen
    in minimal server environments), Pillow's built-in bitmap font is used as
    a fallback. The built-in font ignores the size parameter — this is a known
    Pillow limitation and is acceptable as a fallback.
    """
    font_candidates = [
        # Windows
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFPro-Regular.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    bold_candidates = [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    candidates = bold_candidates if bold else font_candidates
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Fallback to Pillow's built-in font (no size control)
    return ImageFont.load_default()


# ─── Image Helpers ────────────────────────────────────────────────────────────

def _crop_to_fill(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Resize and center-crop an image to exactly (target_w, target_h).

    This is the "cover" resize strategy used by CSS background-size: cover.
    The image is scaled up (or down) so that its smaller dimension matches
    the target, then the excess is cropped from both sides equally.

    We use LANCZOS resampling, which is the highest quality downscaling filter
    available in Pillow — it uses a sinc function to minimize aliasing.
    """
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def _select_hero(image_ids: list,
                 feature_matrix: np.ndarray,
                 image_ids_full: list) -> int:
    """
    Select the hero image as the one whose CLIP embedding is closest to the
    cluster centroid.

    Parameters
    ----------
    image_ids      : DB row IDs of images in this cluster (subset)
    feature_matrix : the full (N, 560) feature matrix for the keyword
    image_ids_full : the full list of image IDs corresponding to matrix rows

    Returns
    -------
    int — DB row ID of the selected hero image
    """
    # Find the rows in the feature matrix that belong to this cluster
    id_to_row = {img_id: i for i, img_id in enumerate(image_ids_full)}
    cluster_rows = [id_to_row[img_id] for img_id in image_ids
                    if img_id in id_to_row]

    if not cluster_rows:
        return image_ids[0]

    # Compute centroid from CLIP portion (first 512 dims) of the feature matrix
    clip_vecs = feature_matrix[cluster_rows, :512]
    centroid = clip_vecs.mean(axis=0)

    # Find the image closest to the centroid
    best_id = image_ids[0]
    best_dist = float("inf")
    for img_id, row_idx in zip(image_ids, cluster_rows):
        dist = float(np.linalg.norm(feature_matrix[row_idx, :512] - centroid))
        if dist < best_dist:
            best_dist = dist
            best_id = img_id

    return best_id


def _average_palette(conn, image_ids: list) -> list:
    """
    Compute the average dominant color palette across all images in a cluster.

    For each color rank (0–5), the RGB values at that rank are averaged across
    all images, then converted back to a HEX string.

    This gives a "cluster palette" where slot 0 is the average of each image's
    most dominant color, slot 1 is the average of the second-most dominant, etc.
    """
    n_colors = 6
    rgb_sums = [[0.0, 0.0, 0.0] for _ in range(n_colors)]
    rgb_counts = [0] * n_colors

    for image_id in image_ids:
        row = conn.execute(
            "SELECT dominant_colors FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        if not row or not row["dominant_colors"]:
            continue
        colors = json.loads(row["dominant_colors"])
        for rank, hex_color in enumerate(colors[:n_colors]):
            h = hex_color.lstrip("#")
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            rgb_sums[rank][0] += r
            rgb_sums[rank][1] += g
            rgb_sums[rank][2] += b
            rgb_counts[rank] += 1

    avg_palette = []
    for rank in range(n_colors):
        if rgb_counts[rank] == 0:
            avg_palette.append("#888888")
        else:
            r = int(rgb_sums[rank][0] / rgb_counts[rank])
            g = int(rgb_sums[rank][1] / rgb_counts[rank])
            b = int(rgb_sums[rank][2] / rgb_counts[rank])
            avg_palette.append(f"#{r:02X}{g:02X}{b:02X}")

    return avg_palette


# ─── Board Composition ────────────────────────────────────────────────────────

def compose_board(conn,
                  cluster_id: int,
                  cluster_label: str,
                  keyword: str,
                  feature_matrix: np.ndarray,
                  image_ids_full: list) -> Optional[Image.Image]:
    """
    Compose a single mood board for a cluster.

    Parameters
    ----------
    conn            : open SQLite connection
    cluster_id      : integer cluster ID from Phase 5
    cluster_label   : human-readable label e.g. "cinematic deep navy"
    keyword         : the search keyword (shown in metadata footer)
    feature_matrix  : full (N, 560) feature matrix for this keyword
    image_ids_full  : full list of DB row IDs matching feature_matrix rows

    Returns
    -------
    PIL.Image.Image — the composed board, or None if the cluster has no images
    """
    # ── Gather cluster images ──
    rows = conn.execute(
        """
        SELECT id, local_path
        FROM images
        WHERE cluster_id = ?
          AND local_path IS NOT NULL
        ORDER BY id
        """,
        (cluster_id,)
    ).fetchall()

    if not rows:
        print(f"[BoardMaker] Cluster {cluster_id} has no downloaded images — skipping.")
        return None

    image_ids = [r["id"] for r in rows]
    img_paths = [r["local_path"] for r in rows]

    # Filter to paths that exist on disk
    valid = [(iid, p) for iid, p in zip(image_ids, img_paths) if os.path.exists(p)]
    if not valid:
        print(f"[BoardMaker] Cluster {cluster_id}: no image files found on disk — skipping.")
        return None

    image_ids = [v[0] for v in valid]
    img_paths = [v[1] for v in valid]

    print(f"[BoardMaker] Composing cluster {cluster_id} "
          f"({len(image_ids)} images) — \"{cluster_label}\"")

    # ── Create canvas ──
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # ── Header bar ──
    draw.rectangle([(0, 0), (CANVAS_W, HEADER_H)], fill=HEADER_COLOR)
    font_title = _get_font(32, bold=True)
    font_meta = _get_font(20)

    # Cluster label — left-aligned in header
    draw.text((32, HEADER_H // 2), cluster_label.upper(),
              font=font_title, fill=TEXT_PRIMARY, anchor="lm")

    # Image count — right-aligned in header
    meta_text = f"{len(image_ids)} images"
    draw.text((CANVAS_W - 32, HEADER_H // 2), meta_text,
              font=font_meta, fill=TEXT_SECONDARY, anchor="rm")

    # ── Hero image ──
    hero_id = _select_hero(image_ids, feature_matrix, image_ids_full)
    hero_path = img_paths[image_ids.index(hero_id)] if hero_id in image_ids else img_paths[0]

    try:
        hero_img = Image.open(hero_path).convert("RGB")
        hero_img = _crop_to_fill(hero_img, HERO_W, PANEL_H)
        canvas.paste(hero_img, (0, HEADER_H))
    except Exception as e:
        print(f"[BoardMaker] Could not load hero image: {e}")
        draw.rectangle([(0, HEADER_H), (HERO_W, HEADER_H + PANEL_H)],
                       fill=(20, 20, 30))

    # Subtle hero overlay — thin right edge to separate from grid
    draw.rectangle([(HERO_W - GAP, HEADER_H), (HERO_W, HEADER_H + PANEL_H)],
                   fill=BG_COLOR)

    # ── Supporting grid ──
    # Use all images except the hero, up to GRID_COLS * GRID_ROWS = 6
    support_paths = [p for iid, p in zip(image_ids, img_paths) if iid != hero_id]
    support_paths = support_paths[: GRID_COLS * GRID_ROWS]

    for idx, path in enumerate(support_paths):
        col = idx % GRID_COLS
        row = idx // GRID_COLS

        x = HERO_W + col * GRID_CELL_W + (GAP if col > 0 else 0)
        y = HEADER_H + row * GRID_CELL_H + (GAP if row > 0 else 0)
        w = GRID_CELL_W - (GAP if col > 0 else 0)
        h = GRID_CELL_H - (GAP if row > 0 else 0)

        try:
            img = Image.open(path).convert("RGB")
            img = _crop_to_fill(img, w, h)
            canvas.paste(img, (x, y))
        except Exception:
            draw.rectangle([(x, y), (x + w, y + h)], fill=(20, 20, 30))

    # ── Color palette strip ──
    palette_y = HEADER_H + PANEL_H
    draw.rectangle([(0, palette_y), (CANVAS_W, palette_y + PALETTE_H)],
                   fill=HEADER_COLOR)

    palette = _average_palette(conn, image_ids)
    swatch_w = CANVAS_W // len(palette)

    for i, hex_color in enumerate(palette):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        sx = i * swatch_w
        # Swatch block — slightly inset top and bottom for a refined look
        draw.rectangle(
            [(sx + 8, palette_y + 12), (sx + swatch_w - 8, palette_y + PALETTE_H - 12)],
            fill=(r, g, b)
        )
        # HEX label below each swatch
        draw.text(
            (sx + swatch_w // 2, palette_y + PALETTE_H - 6),
            hex_color.upper(),
            font=_get_font(16), fill=TEXT_SECONDARY, anchor="mb"
        )

    # ── Footer bar ──
    footer_y = palette_y + PALETTE_H
    draw.rectangle([(0, footer_y), (CANVAS_W, CANVAS_H)], fill=FOOTER_COLOR)

    footer_font = _get_font(18)
    footer_text = f"{keyword}  •  {len(image_ids)} images  •  AestheteAI"
    draw.text((CANVAS_W // 2, footer_y + FOOTER_H // 2), footer_text,
              font=footer_font, fill=TEXT_SECONDARY, anchor="mm")

    return canvas
