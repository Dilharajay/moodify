"""
AestheteAI — Main Entry Point
Phase 1: Run keyword scraping and save results to JSON
"""

import argparse
from scraper.behance_scraper import scrape_behance
from scraper.unsplash_scraper import scrape_unsplash
from utils.storage import save_to_json

def main():
    parser = argparse.ArgumentParser(description="AestheteAI — Mood Board Scraper")
    parser.add_argument("--keyword", type=str, required=True, help="Search keyword (e.g. 'dark minimalism')")
    parser.add_argument("--pages", type=int, default=3, help="Number of pages to scrape (default: 3)")
    parser.add_argument(
        "--source",
        type=str,
        choices=["behance", "unsplash", "all"],
        default="unsplash",
        help="Which platform to scrape (default: unsplash)"
    )
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  AestheteAI | Phase 1 — Scraper")
    print(f"  Keyword : {args.keyword}")
    print(f"  Pages   : {args.pages}")
    print(f"  Source  : {args.source}")
    print(f"{'='*50}\n")

    all_results = []

    # --- Unsplash ---
    if args.source in ("unsplash", "all"):
        print("[Unsplash] Scraping...")
        results = scrape_unsplash(keyword=args.keyword, pages=args.pages)
        print(f"  Found {len(results)} images from Unsplash\n")
        all_results.extend(results)

    # --- Behance ---
    if args.source in ("behance", "all"):
        print("[Behance] Scraping...")
        results = scrape_behance(keyword=args.keyword, pages=args.pages)
        print(f"  Found {len(results)} images from Behance\n")
        all_results.extend(results)

    # --- Save ---
    if all_results:
        path = save_to_json(all_results, keyword=args.keyword)
        print(f"\nDone. {len(all_results)} total images saved to: {path}")
    else:
        print("\nNo results found. Try a different keyword or check your connection.")


if __name__ == "__main__":
    main()