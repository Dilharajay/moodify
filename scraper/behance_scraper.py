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


def extract_from_page(page) -> list[dict]:
    """
    Extract image records from a fully rendered Behance page.
    Targets the __NEXT_DATA__ JSON blob injected by Behance.
    Falls back to scraping rendered img tags if JSON is unavailable.
    """
    projects = []

    # Strategy 1 — parse __NEXT_DATA__ JSON blob
    try:
        next_data_raw = page.eval_on_selector(
            "script#__NEXT_DATA__",
            "el => el.textContent"
        )
        if next_data_raw:
            data = json.loads(next_data_raw)
            results = (
                data.get("props", {})
                    .get("pageProps", {})
                    .get("dehydratedState", {})
                    .get("queries", [{}])[0]
                    .get("state", {})
                    .get("data", {})
                    .get("search", {})
                    .get("content", {})
                    .get("projects", {})
                    .get("items", [])
            )
            for item in results:
                covers = item.get("covers", {})
                image_url = (
                    covers.get("original") or
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
                logger.info(f"Extracted {len(projects)} projects from __NEXT_DATA__")
                return projects

    except Exception as e:
        logger.warning(f"__NEXT_DATA__ parse failed: {e}")

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

    logger.info(f"Fallback extracted {len(projects)} projects")
    return projects


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
            url = f"{BEHANCE_BASE_URL}/{encoded_keyword}?sort=recommended&page={page_num}"

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