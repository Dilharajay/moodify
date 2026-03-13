"""
Moodify — Main Entry Point
Phase 1: Run keyword scraping and save results to JSON
"""

import argparse
from scraper.behance_scraper import scrape_behance
from utils.storage import save_to_json

def main():
    parser = argparse.ArgumentParser(description="Moodify — Mood Board Scraper")
    parser.add_argument("--keyword", type=str, required=True, help="Search keyword (e.g. 'dark minimalism')")
    parser.add_argument("--pages", type=int, default=3, help="Number of pages to scrape (default: 3)")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  AestheteAI | Phase 1 — Behance Scraper")
    print(f"  Keyword : {args.keyword}")
    print(f"  Pages   : {args.pages}")
    print(f"{'='*50}\n")

    # --- Behance ---
    print("[1/1] Scraping Behance...")
    behance_results = scrape_behance(keyword=args.keyword, pages=args.pages)
    print(f"  Found {len(behance_results)} images from Behance\n")

    # --- Save ---
    if behance_results:
        path = save_to_json(behance_results, keyword=args.keyword)
        print(f"\nDone. Results saved to: {path}")
    else:
        print("\nNo results found. Try a different keyword or check your connection.")

if __name__ == "__main__":
    main()
