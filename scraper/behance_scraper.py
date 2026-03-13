"""
Phase 1 — Behance Static Scraper
Scrapes project image URLs and metadata from Behance search results.

NOTE: Behance renders most content via JavaScript. This static scraper
targets the initial HTML payload and any embedded JSON data Behance
injects into the page. Phase 2 will upgrade this to Playwright for
full dynamic rendering.
"""

import requests
import json
import re
from bs4 import BeautifulSoup
from typing import Optional
from config import BEHANCE_SEARCH_URL, BASE_HEADERS, REQUEST_DELAY, MAX_IMAGES_PER_KEYWORD
from utils.helpers import get_random_headers, polite_delay, logger


def fetch_page(url: str, params: dict) -> Optional[str]:
    """
    Fetch raw HTML for a given URL and query params.
    Returns HTML string or None on failure.
    """
    headers = get_random_headers(BASE_HEADERS)
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        logger.info(f"Fetched: {response.url} | Status: {response.status_code}")
        return response.text
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None


def extract_projects_from_html(html: str) -> list[dict]:
    """
    Parse Behance search HTML to extract project metadata.
    Behance embeds a JSON blob in a <script> tag — we target that first,
    then fall back to meta tag parsing.
    """
    soup = BeautifulSoup(html, "lxml")
    projects = []

    # Strategy 1: Extract embedded JSON from script tags
    # Behance injects window.__INITIAL_STATE__ or similar
    scripts = soup.find_all("script")
    for script in scripts:
        if script.string and "ImageUrl" in script.string:
            # Look for image URL patterns inside inline JS
            urls = re.findall(r'"(https://mir-s3[^"]+\.(?:jpg|jpeg|png|webp))"', script.string)
            titles = re.findall(r'"name"\s*:\s*"([^"]+)"', script.string)
            owners = re.findall(r'"display_name"\s*:\s*"([^"]+)"', script.string)

            for i, url in enumerate(urls[:MAX_IMAGES_PER_KEYWORD]):
                projects.append({
                    "image_url": url,
                    "title": titles[i] if i < len(titles) else "Unknown",
                    "owner": owners[i] if i < len(owners) else "Unknown",
                    "source": "behance",
                    "keyword": None  # filled in by caller
                })

            if projects:
                logger.info(f"Extracted {len(projects)} projects from inline JSON")
                return projects

    # Strategy 2: Fallback — scrape Open Graph / meta image tags from project cards
    cards = soup.select("div.ProjectCoverNeue-root")
    for card in cards:
        img_tag = card.find("img")
        title_tag = card.find("p", class_=re.compile("title", re.I))
        owner_tag = card.find("a", class_=re.compile("owner", re.I))

        if img_tag and img_tag.get("src"):
            projects.append({
                "image_url": img_tag["src"],
                "title": title_tag.text.strip() if title_tag else "Unknown",
                "owner": owner_tag.text.strip() if owner_tag else "Unknown",
                "source": "behance",
                "keyword": None
            })

    logger.info(f"Fallback extracted {len(projects)} projects from HTML cards")
    return projects


def scrape_behance(keyword: str, pages: int = 3) -> list[dict]:
    """
    Main scraping function for Behance.
    Iterates over multiple pages and returns combined results.

    Args:
        keyword: Search term (e.g. "dark minimalism")
        pages: Number of search result pages to scrape

    Returns:
        List of project dicts with image_url, title, owner, source, keyword
    """
    all_projects = []

    for page in range(1, pages + 1):
        logger.info(f"Scraping Behance | Keyword: '{keyword}' | Page: {page}")

        params = {
            "search_text": keyword,
            "page": page,
            "sort": "recommended"
        }

        html = fetch_page(BEHANCE_SEARCH_URL, params)
        if not html:
            logger.warning(f"Skipping page {page} — no HTML returned")
            continue

        projects = extract_projects_from_html(html)

        # Tag each record with the keyword
        for p in projects:
            p["keyword"] = keyword

        all_projects.extend(projects)
        logger.info(f"Running total: {len(all_projects)} projects")

        if len(all_projects) >= MAX_IMAGES_PER_KEYWORD:
            logger.info("Reached max image limit. Stopping.")
            break

        if page < pages:
            polite_delay(base=REQUEST_DELAY)

    return all_projects[:MAX_IMAGES_PER_KEYWORD]
