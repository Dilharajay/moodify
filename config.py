import os
from dotenv import load_dotenv

load_dotenv()

# --- Scraper Settings ---
BASE_HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

REQUEST_DELAY = 2          # seconds between requests (be polite)
MAX_IMAGES_PER_KEYWORD = 100

# --- Paths ---
DATA_RAW_DIR = "data/raw"
DATA_IMAGES_DIR = "data/images"
OUTPUT_DIR = "output"

# --- Behance ---
BEHANCE_SEARCH_URL = "https://www.behance.net/search/projects"
