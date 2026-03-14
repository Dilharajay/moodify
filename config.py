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
DATA_RAW_DIR    = "data/raw"
DATA_IMAGES_DIR = "data/images"
OUTPUT_DIR      = "output"

# SQLite database file.
# All scraped records, download status, and color features live here.
DB_PATH = "data/aestheteai.db"

# --- Behance ---
BEHANCE_SEARCH_URL = "https://www.behance.net/search/projects"

# --- Phase 3: Download Settings ---
# Delay in seconds between image download requests.
# Be polite to CDN servers — do not set this below 0.5.
DOWNLOAD_DELAY = 1.0

# --- Phase 3: Color Extraction Settings ---
# Number of dominant colors to extract per image (K-Means clusters).
N_DOMINANT_COLORS = 6

# Number of histogram bins per LAB channel (L, a, b).
# Total histogram vector size = HISTOGRAM_BINS * 3.
HISTOGRAM_BINS = 16

# --- Phase 4: Embedding Settings ---
# HuggingFace model identifier for CLIP.
# First run downloads ~600MB of weights to ~/.cache/huggingface/ — subsequent
# runs load from cache and are fast.
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Output dimension of the CLIP embedding vector for this model variant.
EMBEDDING_DIM = 512

# Root directory for all embedding files (.npy).
EMBEDDINGS_DIR = "data/embeddings"

# Weights for the unified feature matrix (CLIP embedding + color histogram).
# CLIP gets full weight; color histogram gets half weight so CLIP dominates
# without completely suppressing color signal.
FEATURE_WEIGHT_CLIP  = 1.0
FEATURE_WEIGHT_COLOR = 0.5