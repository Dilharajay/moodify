"""
Phase 2 — Pinterest Dynamic Scraper
Uses Playwright to scrape Pinterest search results.
Pinterest is heavily JavaScript rendered and requires
browser automation for any meaningful data extraction.
"""

import json
import re
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from config import MAX_IMAGES_PER_KEYWORD, REQUEST_DELAY
from utils.helpers import polite_delay, logger

PINTEREST_SEARCH_URL = "https://www.pinterest.com/search/pins"


def extract_pins_from_page(page) -> list[dict]:
    """
    Extract pin image records from a rendered Pinterest page.
    Pinterest injects pin data into <script> tags as JSON.
    Falls back to scraping rendered pin grid if JSON unavailable.
    """
    pins = []

    # Strategy 1 — look for embedded JSON in script tags
    try:
        scripts = page.eval_on_selector_all(
            "script[type='application/json'], script[id*='initial']",
            "els => els.map(el => el.textContent)"
        )
        for script_text in scripts:
            if not script_text or "images" not in script_text:
                continue
            try:
                data = json.loads(script_text)
                # Pinterest structures vary — walk the tree looking for pin objects
                raw = json.dumps(data)
                # Match 736x image URLs (Pinterest's standard display size)
                image_urls = re.findall(
                    r'"url"\s*:\s*"(https://i\.pinimg\.com/[^"]*736x[^"]*\.(?:jpg|jpeg|png|webp))"',
                    raw
                )
                titles = re.findall(r'"description"\s*:\s*"([^"]{5,200})"', raw)
                owners = re.findall(r'"full_name"\s*:\s*"([^"]+)"', raw)

                for i, url in enumerate(image_urls[:MAX_IMAGES_PER_KEYWORD]):
                    pins.append({
                        "image_url": url,
                        "title": titles[i].strip() if i < len(titles) else "Untitled",
                        "owner": owners[i].strip() if i < len(owners) else "Unknown",
                        "source": "pinterest",
                        "keyword": None
                    })

                if pins:
                    logger.info(f"Extracted {len(pins)} pins from embedded JSON")
                    return pins

            except json.JSONDecodeError:
                continue

    except Exception as e:
        logger.warning(f"JSON extraction failed: {e}")

    # Strategy 2 — fallback to rendered pin grid img tags
    logger.info("Falling back to rendered img tag extraction...")
    pin_imgs = page.query_selector_all("div[data-test-id='pin'] img, img[src*='pinimg.com']")
    for img in pin_imgs:
        src = img.get_attribute("src") or img.get_attribute("data-src") or ""
        alt = img.get_attribute("alt") or "Untitled"
        # Upgrade to 736x resolution if smaller size was returned
        src = re.sub(r'/\d+x/', '/736x/', src)
        if src and "pinimg.com" in src:
            pins.append({
                "image_url": src,
                "title": alt.strip(),
                "owner": "Unknown",
                "source": "pinterest",
                "keyword": None
            })

    logger.info(f"Fallback extracted {len(pins)} pins")
    return pins


def scroll_to_load(page, target: int = 50):
    """
    Scroll down to trigger Pinterest's infinite feed until
    we reach the target count or stop finding new pins.
    """
    previous_count = 0
    stall_count = 0

    while True:
        current_pins = page.query_selector_all(
            "div[data-test-id='pin'], img[src*='pinimg.com']"
        )
        current_count = len(current_pins)
        logger.info(f"Pins visible: {current_count}")

        if current_count >= target:
            break

        if current_count == previous_count:
            stall_count += 1
            if stall_count >= 3:
                logger.info("No new pins loading, stopping scroll.")
                break
        else:
            stall_count = 0

        previous_count = current_count
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(2500)


def scrape_pinterest(keyword: str, pages: int = 3) -> list[dict]:
    """
    Main Pinterest scraper using Playwright.

    Pinterest does not use traditional pagination — it uses an
    infinite scroll feed. The `pages` argument controls how many
    scroll cycles are attempted rather than literal page numbers.

    Args:
        keyword: Search term (e.g. "dark minimalism")
        pages:   Number of scroll cycles (each loads ~20 new pins)

    Returns:
        List of pin dicts with image_url, title, owner, source, keyword
    """
    all_pins = []

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

        # Block heavy assets to speed up loading
        page.route("**/*.{woff,woff2,ttf}", lambda route: route.abort())

        logger.info(f"Scraping Pinterest | Keyword: '{keyword}'")

        encoded_keyword = keyword.strip().replace(" ", "%20")
        url = f"{PINTEREST_SEARCH_URL}/?q={encoded_keyword}&rs=typed"

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(3000)

            # Dismiss login popup if it appears
            try:
                close_btn = page.query_selector("[data-test-id='closeup-close-button'], button[aria-label='Close']")
                if close_btn:
                    close_btn.click()
                    page.wait_for_timeout(1000)
                    logger.info("Dismissed login popup")
            except Exception:
                pass

            # Scroll to load pins up to our target
            scroll_to_load(page, target=MAX_IMAGES_PER_KEYWORD)

            pins = extract_pins_from_page(page)
            for pin in pins:
                pin["keyword"] = keyword

            all_pins.extend(pins)
            logger.info(f"Total pins collected: {len(all_pins)}")

        except PlaywrightTimeout:
            logger.warning("Timeout loading Pinterest page.")

        browser.close()

    return all_pins[:MAX_IMAGES_PER_KEYWORD]