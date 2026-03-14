"""
Tests for scraper/pinterest_scraper.py (Phase 2)

What we test:
- extract_from_page() pulls image records from mocked API responses
- extract_from_page() falls back to img tags if API fails
- scrape_pinterest() runs the Playwright lifecycle
"""

import pytest
from unittest.mock import patch, MagicMock
from scraper.pinterest_scraper import extract_from_page, scrape_pinterest

# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_API_RESPONSE = {
    "resource_response": {
        "data": [
            {
                "images": {"orig": {"url": "https://i.pinimg.com/orig/1.jpg"}},
                "title": "Pin 1",
                "pinner": {"full_name": "User 1"}
            },
            {
                "images": {"736x": {"url": "https://i.pinimg.com/736x/2.jpg"}},
                "grid_title": "Pin 2",
                "pinner": {"username": "user2"}
            }
        ]
    }
}

# ── extract_from_page ────────────────────────────────────────────────────────

def test_extract_from_page_api_strategy():
    mock_page = MagicMock()
    captured = [MOCK_API_RESPONSE]
    
    results = extract_from_page(mock_page, captured)
    assert len(results) == 2
    assert results[0]["title"] == "Pin 1"
    assert results[1]["owner"] == "user2"
    assert "orig/1.jpg" in results[0]["image_url"]

def test_extract_from_page_fallback_strategy():
    mock_page = MagicMock()
    # Strategy 1 fails (empty captured)
    captured = []
    
    # Strategy 2: img tags
    mock_card = MagicMock()
    mock_img = MagicMock()
    mock_img.get_attribute.side_effect = lambda attr: "https://i.pinimg.com/236x/fallback.jpg" if attr == "src" else "Alt text"
    mock_card.query_selector.return_value = mock_img
    
    mock_page.query_selector_all.return_value = [mock_card]
    
    results = extract_from_page(mock_page, captured)
    assert len(results) == 1
    # Fallback should replace 236x with 736x
    assert "736x/fallback.jpg" in results[0]["image_url"]


# ── scrape_pinterest ──────────────────────────────────────────────────────────

@patch("scraper.pinterest_scraper.sync_playwright")
@patch("scraper.pinterest_scraper.extract_from_page")
@patch("scraper.pinterest_scraper.scroll_to_load")
def test_scrape_pinterest_calls_playwright(mock_scroll, mock_extract, mock_pw):
    # Setup mocks
    mock_browser = MagicMock()
    mock_pw.return_value.__enter__.return_value.chromium.launch.return_value = mock_browser
    mock_page = mock_browser.new_context.return_value.new_page.return_value
    
    mock_extract.return_value = [
        {"image_url": "url1", "title": "T1", "owner": "O1", "source": "pinterest"}
    ]
    
    results = scrape_pinterest(keyword="moody", pages=1)
    
    assert len(results) == 1
    assert results[0]["keyword"] == "moody"
    mock_page.goto.assert_called()
    mock_extract.assert_called()
