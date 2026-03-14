"""
Tests for scraper/behance_scraper.py (Phase 2)

What we test:
- find_projects_recursive() correctly extracts projects from JSON blobs
- extract_from_page() pulls image records from mocked Playwright page
- scrape_behance() respects MAX_IMAGES_PER_KEYWORD limit
- scrape_behance() tags all records with the correct keyword

Note: Playwright and network calls are mocked.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from scraper.behance_scraper import find_projects_recursive, extract_from_page, scrape_behance

# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_JSON_BLOB = {
    "nodes": [
        {
            "name": "Project 1",
            "owners": [{"displayName": "Owner 1"}],
            "covers": {
                "allAvailable": [
                    {"url": "https://mir-s3.behance.net/p1_original.jpg", "width": 2000}
                ]
            }
        },
        {
            "name": "Project 2",
            "owners": [{"displayName": "Owner 2"}],
            "covers": {
                "808": "https://mir-s3.behance.net/p2_808.jpg"
            }
        }
    ]
}

# ── find_projects_recursive ───────────────────────────────────────────────────

def test_find_projects_recursive_success():
    data = {"some_key": {"nodes": MOCK_JSON_BLOB["nodes"]}}
    results = find_projects_recursive(data)
    assert results is not None
    assert len(results) == 2
    assert results[0]["name"] == "Project 1"

def test_find_projects_recursive_failure():
    data = {"no_projects": "here"}
    results = find_projects_recursive(data)
    assert results is None


# ── extract_from_page ────────────────────────────────────────────────────────

def test_extract_from_page_json_strategy():
    mock_page = MagicMock()
    # Mocking scripts on the page
    # Behance scraper expects a long string (> 10000 chars) containing 'mir-s3'
    # Use a large dictionary to ensure it passes the length check and contains 'mir-s3'
    large_data = {
        "nodes": MOCK_JSON_BLOB["nodes"],
        "extra": "x" * 10000
    }
    blob_content = json.dumps(large_data)
    
    mock_page.eval_on_selector_all.return_value = [
        "console.log('hello')",
        blob_content
    ]
    
    results = extract_from_page(mock_page)
    assert len(results) == 2
    assert results[0]["title"] == "Project 1"
    assert results[0]["owner"] == "Owner 1"
    assert results[0]["source"] == "behance"
    assert "p1_original.jpg" in results[0]["image_url"]

def test_extract_from_page_fallback_strategy():
    mock_page = MagicMock()
    # Strategy 1 fails (no blob)
    mock_page.eval_on_selector_all.return_value = []
    
    # Strategy 2: query_selector_all for cards
    mock_card = MagicMock()
    mock_img = MagicMock()
    mock_img.get_attribute.return_value = "https://mir-s3.behance.net/fallback.jpg"
    
    # Mock elements for title and owner
    mock_title_el = MagicMock()
    mock_title_el.inner_text.return_value.strip.return_value = "Fallback Title"
    
    mock_owner_el = MagicMock()
    mock_owner_el.inner_text.return_value.strip.return_value = "Fallback Owner"
    
    def side_effect(sel):
        if "img" in sel: return mock_img
        if "title" in sel: return mock_title_el
        if "owner" in sel: return mock_owner_el
        return None

    mock_card.query_selector.side_effect = side_effect
    
    mock_page.query_selector_all.return_value = [mock_card]
    
    results = extract_from_page(mock_page)
    assert len(results) == 1
    assert results[0]["title"] == "Fallback Title"
    assert "fallback.jpg" in results[0]["image_url"]


# ── scrape_behance ────────────────────────────────────────────────────────────

@patch("scraper.behance_scraper.sync_playwright")
@patch("scraper.behance_scraper.extract_from_page")
@patch("scraper.behance_scraper.scroll_to_load")
def test_scrape_behance_calls_playwright(mock_scroll, mock_extract, mock_pw):
    # Setup mocks for Playwright context
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    
    mock_pw.return_value.__enter__.return_value.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page
    
    mock_extract.return_value = [
        {"image_url": "url1", "title": "T1", "owner": "O1", "source": "behance"}
    ]
    
    results = scrape_behance(keyword="test", pages=1)
    
    assert len(results) == 1
    assert results[0]["keyword"] == "test"
    mock_page.goto.assert_called()
    mock_extract.assert_called_with(mock_page)

@patch("scraper.behance_scraper.sync_playwright")
@patch("scraper.behance_scraper.extract_from_page")
def test_scrape_behance_respects_limit(mock_extract, mock_pw):
    # Setup mocks
    mock_browser = MagicMock()
    mock_pw.return_value.__enter__.return_value.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value.new_page.return_value = MagicMock()

    mock_extract.return_value = [
        {"image_url": f"url{i}", "title": "T", "owner": "O", "source": "behance"}
        for i in range(10)
    ]
    
    with patch("scraper.behance_scraper.MAX_IMAGES_PER_KEYWORD", 5):
        results = scrape_behance(keyword="test", pages=1)
        
    assert len(results) == 5
