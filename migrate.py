"""
migrate.py — Phase 2 JSON → SQLite Migration
=============================================

This is a one-time script that reads all the JSON files produced by the Phase 2
scrapers (stored in data/raw/) and imports their records into the new SQLite
database introduced in Phase 3.

When to run this:
    Run it once after Phase 3 is set up if you already have Phase 2 JSON data
    you want to carry forward. After this, the scrapers write directly to SQLite
    and this script is no longer needed.

What it does:
    1. Finds all .json files in data/raw/ that look like scraper output
    2. Loads each file and extracts the list of image records
    3. Inserts them into the database, skipping duplicates (by image_url)
    4. Prints a summary of how many records were imported vs skipped

Run with:
    python migrate.py

    Optional flags:
    --dry-run    Print what would be imported without writing to the database
    --raw-dir    Override the default data/raw/ directory
    --db         Override the default database path
"""

import os
import json
import argparse
import glob

from utils.database import initialize_db, insert_images
from config import DATA_RAW_DIR, DB_PATH


def find_json_files(raw_dir: str) -> list[str]:
    """
    Find all .json files in the raw data directory.

    We skip files that start with 'debug_' because those are output from
    the debug extractor scripts, not scraper output files.
    """
    all_json = glob.glob(os.path.join(raw_dir, "*.json"))
    scraper_files = [
        f for f in all_json
        if not os.path.basename(f).startswith("debug_")
    ]
    return sorted(scraper_files)


def load_records_from_file(filepath: str) -> list[dict]:
    """
    Load image records from a JSON file.

    The Phase 2 scraper saves a list of dicts at the top level.
    If the file has a different structure (e.g. a wrapped object),
    we try to extract a list from it gracefully.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Phase 2 output is a plain list — this is the expected case.
    if isinstance(data, list):
        return data

    # Some debug files may be dicts with a list inside.
    if isinstance(data, dict):
        for key in ("images", "results", "data", "records"):
            if isinstance(data.get(key), list):
                return data[key]

    return []


def migrate(raw_dir: str, db_path: str, dry_run: bool = False) -> None:
    """
    Main migration function.

    Parameters
    ----------
    raw_dir  : directory containing Phase 2 JSON files
    db_path  : path to the target SQLite database
    dry_run  : if True, print what would be imported without writing
    """
    print(f"\n{'='*55}")
    print(f"  AestheteAI — Phase 2 JSON → SQLite Migration")
    print(f"{'='*55}")
    print(f"  Source directory : {raw_dir}")
    print(f"  Target database  : {db_path}")
    print(f"  Mode             : {'DRY RUN (no writes)' if dry_run else 'LIVE'}")
    print()

    # Find all JSON files to process
    json_files = find_json_files(raw_dir)

    if not json_files:
        print(f"[Migrate] No scraper JSON files found in {raw_dir}/")
        print("          Make sure you have run the Phase 2 scraper first.")
        return

    print(f"[Migrate] Found {len(json_files)} JSON file(s):\n")
    for f in json_files:
        print(f"          {f}")
    print()

    # Initialize the database (creates tables if they do not exist)
    if not dry_run:
        conn = initialize_db(db_path)

    total_records = 0
    total_inserted = 0
    total_skipped = 0

    for filepath in json_files:
        records = load_records_from_file(filepath)
        filename = os.path.basename(filepath)

        if not records:
            print(f"  [SKIP] {filename} — no records found or unrecognised format")
            continue

        print(f"  [FILE] {filename} — {len(records)} records")

        if dry_run:
            # Just print a preview of the first record
            if records:
                first = records[0]
                print(f"         First record keys: {list(first.keys())}")
                print(f"         Example URL: {str(first.get('image_url', '?'))[:80]}")
            total_records += len(records)
            continue

        # Live insert — duplicates are silently skipped by INSERT OR IGNORE
        inserted = insert_images(conn, records)
        skipped  = len(records) - inserted

        print(f"         Inserted: {inserted}  |  Skipped (duplicates): {skipped}")

        total_records  += len(records)
        total_inserted += inserted
        total_skipped  += skipped

    # Summary
    print(f"\n{'='*55}")
    if dry_run:
        print(f"  DRY RUN complete. Would process {total_records} records.")
        print(f"  Run without --dry-run to perform the actual migration.")
    else:
        print(f"  Migration complete.")
        print(f"  Total records processed : {total_records}")
        print(f"  Inserted into DB        : {total_inserted}")
        print(f"  Skipped (duplicates)    : {total_skipped}")
        print(f"  Database                : {db_path}")
        conn.close()
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate Phase 2 JSON scraper output into the SQLite database."
    )
    parser.add_argument(
        "--raw-dir",
        default=DATA_RAW_DIR,
        help=f"Directory containing Phase 2 JSON files (default: {DATA_RAW_DIR})"
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help=f"Path to the target SQLite database (default: {DB_PATH})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the migration without writing to the database"
    )
    args = parser.parse_args()

    migrate(raw_dir=args.raw_dir, db_path=args.db, dry_run=args.dry_run)