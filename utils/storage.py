import json
import os
from datetime import datetime
from config import DATA_RAW_DIR

def save_to_json(data: list[dict], keyword: str) -> str:
    """
    Save a list of image records to a JSON file.
    Returns the file path.
    """
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    safe_keyword = keyword.strip().lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_keyword}_{timestamp}.json"
    filepath = os.path.join(DATA_RAW_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[Storage] Saved {len(data)} records to {filepath}")
    return filepath

def load_from_json(filepath: str) -> list[dict]:
    """Load records from an existing JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
