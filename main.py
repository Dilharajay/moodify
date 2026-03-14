"""
AestheteAI — Main Pipeline Entry Point
Phase 3: Scrape → Store → Download → Extract Colors

This script is the central runner for AestheteAI. Each phase of the pipeline
is exposed as an --action argument, so you can run individual steps or chain
them together.

Available actions:
    scrape          Scrape Behance and/or Pinterest and store results in SQLite
    download        Download all pending images from the database
    extract_colors  Extract color features from all downloaded images
    pipeline        Run all three actions in sequence (scrape → download → extract_colors)
    stats           Print a summary of the current database state

Usage examples:
    # Scrape and run the full pipeline in one command
    python main.py --action pipeline --keyword "dark minimalism" --pages 3 --source all

    # Only scrape (writes to DB, no downloads yet)
    python main.py --action scrape --keyword "editorial fashion" --pages 2 --source behance

    # Only download images that are already in the DB
    python main.py --action download

    # Only extract colors from already-downloaded images
    python main.py --action extract_colors

    # See current database stats
    python main.py --action stats
"""

import argparse
import sys

from scraper.behance_scraper import scrape_behance
from scraper.pinterest_scraper import scrape_pinterest
from utils.database import initialize_db, insert_images, log_scrape_job, get_stats
from utils.downloader import download_all_pending
from utils.color_extractor import process_all_pending
from config import DB_PATH, DOWNLOAD_DELAY


def action_scrape(conn, args):
    if not args.keyword:
        print("[Error] --keyword is required for the scrape action.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  AestheteAI — Phase 3 | Scrape")
    print(f"  Keyword : {args.keyword}")
    print(f"  Pages   : {args.pages}")
    print(f"  Source  : {args.source}")
    print(f"{'='*55}\n")

    all_results = []

    if args.source in ("behance", "all"):
        print("[Behance] Launching browser...")
        results = scrape_behance(keyword=args.keyword, pages=args.pages)
        for r in results:
            r["keyword"] = args.keyword
        print(f"  Found {len(results)} images from Behance\n")
        all_results.extend(results)
        log_scrape_job(conn, args.keyword, "behance", args.pages, len(results))

    if args.source in ("pinterest", "all"):
        print("[Pinterest] Launching browser...")
        results = scrape_pinterest(keyword=args.keyword, pages=args.pages)
        for r in results:
            r["keyword"] = args.keyword
        print(f"  Found {len(results)} images from Pinterest\n")
        all_results.extend(results)
        log_scrape_job(conn, args.keyword, "pinterest", args.pages, len(results))

    if all_results:
        inserted = insert_images(conn, all_results)
        skipped  = len(all_results) - inserted
        print(f"[DB] Inserted {inserted} new records  |  Skipped {skipped} duplicates")
    else:
        print("[Warning] No results returned from scrapers.")


def action_download(conn, args):
    print(f"\n{'='*55}")
    print(f"  AestheteAI — Phase 3 | Download Images")
    print(f"{'='*55}\n")
    keyword_filter = getattr(args, "keyword", None)
    download_all_pending(conn, delay=DOWNLOAD_DELAY, keyword=keyword_filter)


def action_extract_colors(conn, args):
    print(f"\n{'='*55}")
    print(f"  AestheteAI — Phase 3 | Extract Colors")
    print(f"{'='*55}\n")
    keyword_filter = getattr(args, "keyword", None)
    process_all_pending(conn, keyword=keyword_filter)


def action_stats(conn, _args):
    stats = get_stats(conn)
    print(f"\n{'='*55}")
    print(f"  AestheteAI — Database Stats")
    print(f"{'='*55}")
    print(f"  Total images scraped      : {stats['total']}")
    print(f"  Downloaded                : {stats['downloaded']}")
    print(f"  Colors extracted          : {stats['colors_extracted']}")
    print(f"  Pending download          : {stats['pending_download']}")
    print(f"  Pending color extraction  : {stats['pending_colors']}")
    if stats["by_source"]:
        print(f"\n  By source:")
        for source, count in stats["by_source"].items():
            print(f"    {source:<20}: {count}")
    print(f"{'='*55}\n")


def action_pipeline(conn, args):
    action_scrape(conn, args)
    action_download(conn, args)
    action_extract_colors(conn, args)
    action_stats(conn, args)


ACTION_MAP = {
    "scrape":         action_scrape,
    "download":       action_download,
    "extract_colors": action_extract_colors,
    "pipeline":       action_pipeline,
    "stats":          action_stats,
}



# ─── Phase 4 additions ────────────────────────────────────────────────────────
# These are appended here and merged into ACTION_MAP at the bottom of the file.
# In production, you would refactor the whole file rather than appending,
# but for a learning project this keeps each phase's changes clearly visible.

from utils.database import migrate_schema_phase4
from utils.embedder import encode_images_batch, save_keyword_embedding, build_feature_matrix


def action_embed(conn, args):
    """
    Encode all downloaded images using CLIP and save embeddings to disk.
    Also encodes the keyword as a text embedding.
    """
    # Ensure Phase 4 columns exist (safe to call on any existing DB)
    migrate_schema_phase4(conn)

    print(f"\n{'='*55}")
    print(f"  AestheteAI — Phase 4 | CLIP Image Embeddings")
    print(f"{'='*55}\n")

    keyword_filter = getattr(args, "keyword", None)
    encode_images_batch(conn, keyword=keyword_filter)

    # Also encode the keyword as a text embedding if one was provided
    if keyword_filter:
        print()
        save_keyword_embedding(keyword_filter)


def action_build_features(conn, args):
    """
    Build the unified feature matrix (CLIP + color histogram) for a keyword.
    This is the input to Phase 5 clustering.
    """
    if not args.keyword:
        print("[Error] --keyword is required for the build_features action.")
        return

    print(f"\n{'='*55}")
    print(f"  AestheteAI — Phase 4 | Build Feature Matrix")
    print(f"{'='*55}\n")

    build_feature_matrix(conn, keyword=args.keyword)


# Extend the action map with Phase 4 entries
ACTION_MAP["embed"]          = action_embed
ACTION_MAP["build_features"] = action_build_features

# Update pipeline to include Phase 4 steps
_original_pipeline = ACTION_MAP["pipeline"]

def action_pipeline_phase4(conn, args):
    """Full pipeline: scrape → download → colors → embed → build_features."""
    migrate_schema_phase4(conn)
    _original_pipeline(conn, args)
    action_embed(conn, args)
    action_build_features(conn, args)

ACTION_MAP["pipeline"] = action_pipeline_phase4

def main():
    parser = argparse.ArgumentParser(
        description="AestheteAI — Keyword-driven mood board pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--action", choices=list(ACTION_MAP.keys()),
                        default="pipeline",
                        help="Which pipeline step to run (default: pipeline)")
    parser.add_argument("--keyword", type=str, default=None,
                        help="Search keyword (required for scrape and pipeline)")
    parser.add_argument("--pages",   type=int, default=3,
                        help="Pages to scrape (default: 3)")
    parser.add_argument("--source",  choices=["behance", "pinterest", "all"],
                        default="all", help="Platform to scrape (default: all)")
    parser.add_argument("--db",      default=DB_PATH,
                        help=f"SQLite database path (default: {DB_PATH})")
    args = parser.parse_args()

    conn = initialize_db(args.db)
    try:
        ACTION_MAP[args.action](conn, args)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
