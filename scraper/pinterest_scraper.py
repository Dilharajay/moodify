"""
Phase 2 — Pinterest Dynamic Scraper
Uses Playwright to launch a real Chromium browser, intercepting API calls to find pin data.
"""

import json
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from config import MAX_IMAGES_PER_KEYWORD, REQUEST_DELAY
from utils.helpers import polite_delay, logger

PINTEREST_BASE_URL = "https://www.pinterest.com"

def extract_from_page(page, captured_responses: list) -> list[dict]:
    """
    Extract pin records from captured API responses.
    Falls back to scraping rendered img tags if API responses are unavailable.
    """
    projects = []
    
    # Strategy 1 — Parse captured API responses
    try:
        for response_body in captured_responses:
            data_node = response_body.get("resource_response", {}).get("data")
            results = []
            if isinstance(data_node, dict):
                results = data_node.get("results", [])
            elif isinstance(data_node, list):
                results = data_node
                
            for item in results:
                images = item.get("images", {})
                
                # Get the largest available image
                image_info = images.get("orig") or images.get("736x") or images.get("474x")
                if not image_info:
                    continue
                    
                image_url = image_info.get("url")
                if not image_url:
                    continue
                    
                title = item.get("title") or item.get("grid_title") or item.get("description") or "Untitled"
                pinner = item.get("pinner", {})
                owner = pinner.get("full_name") or pinner.get("username") or "Unknown"
                
                projects.append({
                    "image_url": image_url,
                    "title": title.strip() if title.strip() else "Untitled",
                    "owner": owner.strip() if owner.strip() else "Unknown",
                    "source": "pinterest",
                    "keyword": None
                })
        
        if projects:
            # Deduplicate by image_url
            unique_projects = {p["image_url"]: p for p in projects}.values()
            logger.info(f"Extracted {len(unique_projects)} projects from API responses")
            return list(unique_projects)
            
    except Exception as e:
        logger.warning(f"API response parse failed: {e}")

    # Strategy 2 — fallback to rendered img tags
    logger.info("Falling back to rendered img tag extraction...")
    cards = page.query_selector_all("div[data-test-id='pin']")
    for card in cards:
        img = card.query_selector("img")
        title_el = card.query_selector("div[title]")

        if img:
            src = img.get_attribute("src")
            if src:
                # Replace smaller resolution with 736x if possible
                if "236x" in src:
                    src = src.replace("236x", "736x")
                    
                projects.append({
                    "image_url": src,
                    "title": title_el.get_attribute("title") if title_el else img.get_attribute("alt") or "Untitled",
                    "owner": "Unknown",
                    "source": "pinterest",
                    "keyword": None
                })

    # Deduplicate
    unique_projects = {p["image_url"]: p for p in projects}.values()
    logger.info(f"Fallback extracted {len(unique_projects)} projects")
    return list(unique_projects)


def scroll_to_load(page, target: int = 50):
    """
    Scroll down repeatedly to trigger lazy loading.
    """
    previous_count = 0
    stall_count = 0

    while True:
        current_cards = page.query_selector_all("div[data-test-id='pin']")
        current_count = len(current_cards)
        logger.info(f"Pins visible: {current_count}")

        if current_count >= target:
            break

        if current_count == previous_count:
            stall_count += 1
            if stall_count >= 5:
                logger.info("No new pins loading, stopping scroll.")
                break
        else:
            stall_count = 0

        previous_count = current_count
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(2000)


def scrape_pinterest(keyword: str, pages: int = 1) -> list[dict]:
    """
    Main Pinterest scraper using Playwright API interception.
    """
    all_projects = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = context.new_page()
        
        captured_responses = []

        def handle_response(response):
            url = response.url
            if "pinterest.com" in url and any(x in url for x in ["BaseSearch", "search/pins", "resource/Search", "search?", "api/v3"]):
                try:
                    body = response.json()
                    captured_responses.append(body)
                except Exception:
                    pass
        
        page.on("response", handle_response)

        encoded_keyword = keyword.strip().replace(" ", "%20")
        url = f"{PINTEREST_BASE_URL}/search/pins/?q={encoded_keyword}&rs=typed"

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(3000)

            # Scroll to load more cards
            scroll_to_load(page, target=MAX_IMAGES_PER_KEYWORD)

            projects = extract_from_page(page, captured_responses)

            for p_item in projects:
                p_item["keyword"] = keyword

            all_projects.extend(projects)
            logger.info(f"Running total: {len(all_projects)} projects")

        except PlaywrightTimeout:
            logger.warning(f"Timeout on Pinterest scraping, skipping.")

        browser.close()

    return all_projects[:MAX_IMAGES_PER_KEYWORD]