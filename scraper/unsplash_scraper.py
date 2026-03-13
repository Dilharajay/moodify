"""
Phase 1 (Alt) — Unsplash Static Scraper
Used to validate the full Phase 1 pipeline since Behance requires
a real browser (handled in Phase 2 with Playwright).

Unsplash embeds image data in a <script id="__NEXT_DATA__"> JSON blob
which is accessible without JavaScript rendering — making it ideal
for static scraping.
"""

import requests
import json
import re
from bs4 import BeautifulSoup
from typing import Optional
from config import BASE_HEADERS, REQUEST_DELAY, MAX_IMAGES_PER_KEYWORD
from utils.helpers import get_random_headers, polite_delay, logger

UNSPLASH_SEARCH_URL = "https://unsplash.com/s/photos"

session = requests.Session()


def fetch_page(url: str, params: dict = {}) -> Optional[str]:
    """
    Fetch raw HTML from Unsplash.
    Uses a session to persist cookies across requests.
    """
    headers = get_random_headers(BASE_HEADERS)
    try:
        response = session.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        logger.info(f"Fetched: {response.url} | Status: {response.status_code}")
        return response.text
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None


def extract_projects_from_html(html: str) -> list[dict]:
    """
    Parse Unsplash HTML and extract image records.

    Unsplash injects all page data into a <script id="__NEXT_DATA__">
    JSON blob. We parse that directly — no fragile CSS selectors needed.
    """
    soup = BeautifulSoup(html, "lxml")
    projects = []

    # Strategy 1: Parse __NEXT_DATA__ JSON blob (primary)
    next_data_tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if next_data_tag and next_data_tag.string:
        try:
            data = json.loads(next_data_tag.string)

            # Navigate the nested JSON to find photo results
            # Path: props -> pageProps -> initialAppState -> search -> photos -> results
            photos = (
                data.get("props", {})
                    .get("pageProps", {})
                    .get("initialAppState", {})
                    .get("search", {})
                    .get("photos", {})
                    .get("results", [])
            )

            for photo in photos[:MAX_IMAGES_PER_KEYWORD]:
                # Pick the best available image size (regular > small > thumb)
                urls = photo.get("urls", {})
                image_url = urls.get("regular") or urls.get("small") or urls.get("thumb")

                if not image_url:
                    continue

                user = photo.get("user", {})
                projects.append({
                    "image_url": image_url,
                    "title": photo.get("alt_description") or photo.get("description") or "Untitled",
                    "owner": user.get("name") or user.get("username") or "Unknown",
                    "source": "unsplash",
                    "keyword": None,  # filled in by caller
                    "likes": photo.get("likes", 0),
                    "unsplash_id": photo.get("id", "")
                })

            if projects:
                logger.info(f"Extracted {len(projects)} photos from __NEXT_DATA__ JSON")
                return projects

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse __NEXT_DATA__: {e}")

    # Strategy 2: Fallback — find image tags with Unsplash CDN URLs
    logger.info("Falling back to HTML img tag parsing...")
    imgs = soup.find_all("img", src=re.compile(r"images\.unsplash\.com"))
    for img in imgs:
        src = img.get("src", "")
        alt = img.get("alt", "Untitled")
        if src:
            projects.append({
                "image_url": src,
                "title": alt,
                "owner": "Unknown",
                "source": "unsplash",
                "keyword": None,
                "likes": 0,
                "unsplash_id": ""
            })

    logger.info(f"Fallback extracted {len(projects)} photos from img tags")
    return projects


def scrape_unsplash(keyword: str, pages: int = 3) -> list[dict]:
    """
    Main scraping function for Unsplash.
    Iterates over multiple pages and returns combined image records.

    Args:
        keyword: Search term (e.g. "dark minimalism")
        pages:   Number of pages to scrape (default: 3)

    Returns:
        List of image dicts with image_url, title, owner, source, keyword
    """
    all_projects = []

    for page in range(1, pages + 1):
        logger.info(f"Scraping Unsplash | Keyword: '{keyword}' | Page: {page}")

        # Unsplash uses keyword in the URL path
        encoded_keyword = keyword.strip().replace(" ", "-")
        url = f"{UNSPLASH_SEARCH_URL}/{encoded_keyword}"
        params = {"page": page} if page > 1 else {}

        html = fetch_page(url, params)
        if not html:
            logger.warning(f"Skipping page {page} — no HTML returned")
            continue

        projects = extract_projects_from_html(html)

        # Tag each record with the original keyword
        for p in projects:
            p["keyword"] = keyword

        all_projects.extend(projects)
        logger.info(f"Running total: {len(all_projects)} photos")

        if len(all_projects) >= MAX_IMAGES_PER_KEYWORD:
            logger.info("Reached max image limit. Stopping.")
            break

        if page < pages:
            polite_delay(base=REQUEST_DELAY)

    return all_projects[:MAX_IMAGES_PER_KEYWORD]