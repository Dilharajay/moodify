"""
Phase 2 — Behance Dynamic Scraper
Uses Playwright to launch a real Chromium browser, bypassing bot detection.
Implements infinite scroll to load more images per page.
"""

import json
import re
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from config import MAX_IMAGES_PER_KEYWORD, REQUEST_DELAY
from utils.helpers import polite_delay, logger

BEHANCE_BASE_URL = "https://www.behance.net/search/projects"

def find_projects_recursive(data):
    """Recursively search for the list of projects inside the Behance JSON blob."""
    if isinstance(data, dict):
        # Standard signature of a Behance project item: has "covers", "name", "owners"
        if "projects" in data and isinstance(data["projects"], list) and data["projects"] and isinstance(data["projects"][0], dict) and "covers" in data["projects"][0]:
            return data["projects"]
        if "items" in data and isinstance(data["items"], list) and data["items"] and isinstance(data["items"][0], dict) and "covers" in data["items"][0]:
            return data["items"]
        for k, v in data.items():
            result = find_projects_recursive(v)
            if result:
                return result
    elif isinstance(data, list):
        if data and isinstance(data[0], dict) and "covers" in data[0] and "name" in data[0]:
            return data
        for item in data:
            result = find_projects_recursive(item)
            if result:
                return result
    return None

def extract_from_page(page) -> list[dict]:
    """
    Extract image records from a fully rendered Behance page.
    Targets the large inline JSON blob injected by Behance.
    Falls back to scraping rendered img tags if JSON is unavailable.
    """
    projects = []

    # Strategy 1 — parse inline JSON blob
    try:
        scripts = page.eval_on_selector_all("script", "els => els.map(e => e.textContent)")
        blob = None
        for s in scripts:
            if s and "mir-s3" in s and len(s) > 10000:
                blob = s
                break
                
        if blob:
            data = json.loads(blob)
            results = find_projects_recursive(data) or []
            
            for item in results:
                covers = item.get("covers", {})
                # Prefer highest resolution available
                image_url = (
                    covers.get("original") or
                    covers.get("max_1400") or 
                    covers.get("115") or
                    covers.get("max_808") or
                    covers.get("404") or
                    covers.get("202")
                )
                if not image_url:
                    continue
                projects.append({
                    "image_url": image_url,
                    "title": item.get("name", "Untitled"),
                    "owner": item.get("owners", [{}])[0].get("display_name", "Unknown"),
                    "source": "behance",
                    "keyword": None
                })

            if projects:
                # Deduplicate by image_url
                unique_projects = {p["image_url"]: p for p in projects}.values()
                logger.info(f"Extracted {len(unique_projects)} projects from inline JSON blob")
                return list(unique_projects)

    except Exception as e:
        logger.warning(f"Inline JSON blob parse failed: {e}")

    # Strategy 2 — fallback to rendered img tags
    logger.info("Falling back to rendered img tag extraction...")
    cards = page.query_selector_all("div[class*='ProjectCoverNeue']")
    for card in cards:
        img = card.query_selector("img")
        title_el = card.query_selector("p[class*='title'], span[class*='title']")
        owner_el = card.query_selector("a[class*='owner'], span[class*='owner']")

        if img:
            src = img.get_attribute("src") or img.get_attribute("data-src") or ""
            if src:
                projects.append({
                    "image_url": src,
                    "title": title_el.inner_text().strip() if title_el else "Untitled",
                    "owner": owner_el.inner_text().strip() if owner_el else "Unknown",
                    "source": "behance",
                    "keyword": None
                })

    # Deduplicate
    unique_projects = {p["image_url"]: p for p in projects}.values()
    logger.info(f"Fallback extracted {len(unique_projects)} projects")
    return list(unique_projects)


def scroll_to_load(page, target: int = 50):
    """
    Scroll down repeatedly to trigger lazy loading until
    we have enough images or stop finding new ones.
    """
    previous_count = 0
    stall_count = 0

    while True:
        current_cards = page.query_selector_all("div[class*='ProjectCoverNeue']")
        current_count = len(current_cards)
        logger.info(f"Cards visible: {current_count}")

        if current_count >= target:
            break

        if current_count == previous_count:
            stall_count += 1
            if stall_count >= 3:
                logger.info("No new cards loading, stopping scroll.")
                break
        else:
            stall_count = 0

        previous_count = current_count
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(2000)


def scrape_behance(keyword: str, pages: int = 3) -> list[dict]:
    """
    Main Behance scraper using Playwright.

    Args:
        keyword: Search term (e.g. "dark minimalism")
        pages:   Number of search result pages to scrape

    Returns:
        List of image dicts with image_url, title, owner, source, keyword
    """
    all_projects = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = context.new_page()

        # Block images and fonts to speed up page loads
        page.route("**/*.{png,jpg,jpeg,webp,gif,woff,woff2}", lambda route: route.abort())

        for page_num in range(1, pages + 1):
            logger.info(f"Scraping Behance | Keyword: '{keyword}' | Page: {page_num}")

            encoded_keyword = keyword.strip().replace(" ", "%20")
            url = f"{BEHANCE_BASE_URL}/{encoded_keyword}?sort=recommended&tracking_source=typeahead_search_direct"

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(3000)

                # Scroll to load more cards
                scroll_to_load(page, target=MAX_IMAGES_PER_KEYWORD)

                projects = extract_from_page(page)

                for p_item in projects:
                    p_item["keyword"] = keyword

                all_projects.extend(projects)
                logger.info(f"Running total: {len(all_projects)} projects")

                if len(all_projects) >= MAX_IMAGES_PER_KEYWORD:
                    logger.info("Reached max image limit. Stopping.")
                    break

                if page_num < pages:
                    polite_delay(base=REQUEST_DELAY)

            except PlaywrightTimeout:
                logger.warning(f"Timeout on page {page_num}, skipping.")
                continue

        browser.close()

    return all_projects[:MAX_IMAGES_PER_KEYWORD]