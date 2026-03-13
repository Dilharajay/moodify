import time
import random
import logging
from fake_useragent import UserAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

ua = UserAgent()

def get_random_headers(base_headers: dict) -> dict:
    """Rotate user agents to reduce detection risk."""
    headers = base_headers.copy()
    headers["User-Agent"] = ua.random
    return headers

def polite_delay(base: float = 2.0, jitter: float = 1.5):
    """Sleep for base + random jitter seconds between requests."""
    delay = base + random.uniform(0, jitter)
    logger.info(f"Waiting {delay:.1f}s before next request...")
    time.sleep(delay)

def sanitize_keyword(keyword: str) -> str:
    """Convert keyword to a safe filename-friendly string."""
    return keyword.strip().lower().replace(" ", "_")
