"""
utils/downloader.py — Image Downloader
========================================

Downloads all images that have been scraped into the database but not yet saved
locally. Each image is fetched via HTTP, validated, saved to disk, and the
database record is updated with the local path and image dimensions.

Design decisions explained
---------------------------
Sequential downloads (not concurrent):
  We use a simple for-loop with a delay rather than threads or asyncio.
  This is intentional — it is easier to reason about, easier to debug,
  and polite to the servers we are downloading from. If you need speed
  later, look into concurrent.futures.ThreadPoolExecutor.

Idempotency:
  Before downloading, we check if local_path is already set in the DB.
  Running this script twice will not re-download or overwrite anything.
  This means it is safe to run again after a partial failure.

Error handling:
  Some URLs will be dead links or return 403/404 after scraping.
  Rather than crashing the whole run, errors are logged to a file
  and the loop continues.

Directory structure:
  data/images/
    behance/
      dark_minimalism/
        <filename>.jpg
    pinterest/
      dark_minimalism/
        <filename>.jpg
"""

import os
import time
import hashlib
import logging
import requests
from PIL import Image
from io import BytesIO
from typing import Optional
from datetime import datetime

from utils.database import get_undownloaded, update_download


# ─── Logging Setup ────────────────────────────────────────────────────────────

# Log download errors to a file so we can review them without them cluttering
# the terminal output. Append mode ('a') means re-runs add to the same log.
ERROR_LOG_PATH = "data/raw/download_errors.log"

def _get_logger() -> logging.Logger:
    os.makedirs("data/raw", exist_ok=True)
    logger = logging.getLogger("downloader")
    if not logger.handlers:
        handler = logging.FileHandler(ERROR_LOG_PATH, mode="a")
        handler.setFormatter(logging.Formatter("%(asctime)s — %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
    return logger


# ─── Helpers ──────────────────────────────────────────────────────────────────

# Realistic browser headers. Some CDNs will return 403 if the User-Agent looks
# like a script. This header set mimics a normal Chrome browser request.
DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}


def _safe_filename(image_url: str) -> str:
    """
    Derive a filesystem-safe filename from an image URL.

    We take an MD5 hash of the URL rather than the URL itself because:
    1. URLs can contain characters that are invalid in filenames (?, =, /)
    2. URLs can be extremely long
    3. The hash is deterministic — same URL always gives the same filename,
       which helps with deduplication

    The extension is extracted from the URL path so images keep their type.
    Defaults to .jpg if no recognisable extension is found.
    """
    # Extract extension from URL path (strip query string first)
    path_part = image_url.split("?")[0]
    ext = os.path.splitext(path_part)[-1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        ext = ".jpg"
    name_hash = hashlib.md5(image_url.encode()).hexdigest()[:16]
    return f"{name_hash}{ext}"


def _build_save_path(base_dir: str, source: str, keyword: str, filename: str) -> str:
    """
    Build the full path where an image will be saved.

    Example:
        _build_save_path("data/images", "behance", "dark minimalism", "abc123.jpg")
        → "data/images/behance/dark_minimalism/abc123.jpg"

    The keyword is converted to a safe directory name (spaces → underscores,
    lowercase) so it is friendly across all operating systems.
    """
    safe_keyword = keyword.strip().lower().replace(" ", "_")
    folder = os.path.join(base_dir, source, safe_keyword)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)


def _fetch_image(url: str, timeout: int = 15) -> Optional[bytes]:
    """
    Fetch image bytes from a URL. Returns None on any failure.

    timeout=15 seconds is generous but avoids hanging forever on slow CDNs.
    We do not raise exceptions here — callers check for None.
    """
    try:
        response = requests.get(url, headers=DOWNLOAD_HEADERS,
                                timeout=timeout, stream=True)
        response.raise_for_status()
        return response.content
    except requests.RequestException:
        return None


def _validate_and_get_dimensions(image_bytes: bytes) -> Optional[tuple[int, int]]:
    """
    Open the raw bytes as a Pillow Image to confirm it is a valid image
    and get its (width, height).

    Returns None if the bytes are not a valid image (e.g. an HTML error page
    that was returned with a 200 status — this actually happens).
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        img.verify()       # checks the file header is valid
        img = Image.open(BytesIO(image_bytes))   # re-open after verify (verify consumes the stream)
        return img.size    # (width, height)
    except Exception:
        return None


# ─── Main Download Function ───────────────────────────────────────────────────

def download_all_pending(conn,
                         images_dir: str = "data/images",
                         delay: float = 1.0,
                         keyword: Optional[str] = None) -> dict:
    """
    Download all images in the database that have not yet been saved locally.

    Parameters
    ----------
    conn        : open SQLite connection (from database.get_connection)
    images_dir  : root directory for saved images (default: 'data/images')
    delay       : seconds to wait between requests (default: 1.0)
                  Be respectful to servers — do not set this below 0.5
    keyword     : if provided, only download images for this keyword

    Returns
    -------
    dict with keys: attempted, succeeded, failed, skipped
    """
    logger = _get_logger()
    pending = get_undownloaded(conn, keyword=keyword)

    if not pending:
        print("[Downloader] No pending images. All images are already downloaded.")
        return {"attempted": 0, "succeeded": 0, "failed": 0, "skipped": 0}

    print(f"[Downloader] Found {len(pending)} images to download...")
    print(f"             Delay between requests: {delay}s")
    print(f"             Saving to: {images_dir}/\n")

    stats = {"attempted": 0, "succeeded": 0, "failed": 0, "skipped": 0}

    for i, row in enumerate(pending, start=1):
        image_url = row["image_url"]
        source    = row["source"]
        keyword_  = row["keyword"]
        image_id  = row["id"]

        # Progress indicator — print every record so it is easy to see progress
        print(f"  [{i}/{len(pending)}] Downloading from {source} — {image_url[:70]}...")

        stats["attempted"] += 1

        # --- Fetch ---
        image_bytes = _fetch_image(image_url)
        if image_bytes is None:
            print(f"           FAILED — could not fetch URL")
            logger.warning(f"FETCH_FAILED | id={image_id} | url={image_url}")
            stats["failed"] += 1
            time.sleep(delay)
            continue

        # --- Validate ---
        dimensions = _validate_and_get_dimensions(image_bytes)
        if dimensions is None:
            print(f"           FAILED — response was not a valid image")
            logger.warning(f"INVALID_IMAGE | id={image_id} | url={image_url}")
            stats["failed"] += 1
            time.sleep(delay)
            continue

        width_px, height_px = dimensions

        # Skip very small images (icons, thumbnails, placeholders)
        # 100x100 is a reasonable minimum — mood board images should be substantial
        if width_px < 100 or height_px < 100:
            print(f"           SKIPPED — image too small ({width_px}x{height_px}px)")
            logger.warning(f"TOO_SMALL | id={image_id} | size={width_px}x{height_px} | url={image_url}")
            stats["skipped"] += 1
            time.sleep(delay)
            continue

        # --- Save to disk ---
        filename  = _safe_filename(image_url)
        save_path = _build_save_path(images_dir, source, keyword_, filename)

        with open(save_path, "wb") as f:
            f.write(image_bytes)

        # --- Update database ---
        update_download(
            conn,
            image_id    = image_id,
            local_path  = save_path,
            file_size_bytes = len(image_bytes),
            width_px    = width_px,
            height_px   = height_px,
        )

        print(f"           OK — {width_px}x{height_px}px, {len(image_bytes)//1024}KB → {save_path}")
        stats["succeeded"] += 1

        # Polite delay between requests
        time.sleep(delay)

    # Summary
    print(f"\n[Downloader] Done.")
    print(f"             Succeeded : {stats['succeeded']}")
    print(f"             Failed    : {stats['failed']}")
    print(f"             Skipped   : {stats['skipped']}")
    if stats["failed"] > 0:
        print(f"             Error log : {ERROR_LOG_PATH}")

    return stats