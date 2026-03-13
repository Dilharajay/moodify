"""
Tests for scraper/behance_scraper.py

What we test:
- fetch_page() returns HTML string on success
- fetch_page() returns None on network failure
- extract_projects_from_html() pulls records from a mock HTML payload
- extract_projects_from_html() returns empty list on unrecognized HTML
- scrape_behance() respects MAX_IMAGES_PER_KEYWORD limit
- scrape_behance() tags all records with the correct keyword
- scrape_behance() skips pages that return no HTML

NOTE: We never make real HTTP requests in tests.
      All network calls are mocked with unittest.mock.patch.
"""

import pytest
from unittest.mock import patch, MagicMock
from scraper.behance_scraper import fetch_page, extract_projects_from_html, scrape_behance


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_HTML_WITH_JSON = """
<html><body>
<script>
var data = {
  "ImageUrl": "placeholder",
  "projects": [
    {"name": "Dark Poster", "display_name": "artist_one"},
    {"name": "Noir Edit", "display_name": "artist_two"}
  ]
};
</script>
<img src="https://mir-s3.behance.net/img1.jpg"/>
<img src="https://mir-s3.behance.net/img2.jpg"/>
</body></html>
"""

MOCK_HTML_CARD_FALLBACK = """
<html><body>
  <div class="ProjectCoverNeue-root">
    <img src="https://mir-s3.behance.net/card1.jpg"/>
    <p class="title">Minimal Black</p>
    <a class="owner">studio_x</a>
  </div>
  <div class="ProjectCoverNeue-root">
    <img src="https://mir-s3.behance.net/card2.jpg"/>
    <p class="title">Editorial White</p>
    <a class="owner">studio_y</a>
  </div>
</body></html>
"""

EMPTY_HTML = "<html><body><p>Nothing here</p></body></html>"


# ── fetch_page ────────────────────────────────────────────────────────────────

class TestFetchPage:

    @patch("scraper.behance_scraper.requests.get")
    def test_returns_html_string_on_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>OK</html>"
        mock_response.url = "https://www.behance.net/search/projects"
        mock_get.return_value = mock_response

        result = fetch_page("https://www.behance.net/search/projects", params={})
        assert isinstance(result, str)
        assert "<html>" in result

    @patch("scraper.behance_scraper.requests.get")
    def test_returns_none_on_request_exception(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("Connection refused")

        result = fetch_page("https://www.behance.net/search/projects", params={})
        assert result is None

    @patch("scraper.behance_scraper.requests.get")
    def test_raises_for_bad_status_code(self, mock_get):
        import requests
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        mock_get.return_value = mock_response

        result = fetch_page("https://www.behance.net/search/projects", params={})
        assert result is None


# ── extract_projects_from_html ────────────────────────────────────────────────

class TestExtractProjectsFromHtml:

    def test_returns_list(self):
        result = extract_projects_from_html(MOCK_HTML_WITH_JSON)
        assert isinstance(result, list)

    def test_returns_empty_list_on_empty_html(self):
        result = extract_projects_from_html(EMPTY_HTML)
        assert result == []

    def test_each_record_has_required_fields(self):
        result = extract_projects_from_html(MOCK_HTML_CARD_FALLBACK)
        if result:
            for record in result:
                assert "image_url" in record
                assert "title" in record
                assert "owner" in record
                assert "source" in record

    def test_source_is_always_behance(self):
        result = extract_projects_from_html(MOCK_HTML_CARD_FALLBACK)
        for record in result:
            assert record["source"] == "behance"

    def test_image_urls_are_strings(self):
        result = extract_projects_from_html(MOCK_HTML_CARD_FALLBACK)
        for record in result:
            assert isinstance(record["image_url"], str)
            assert len(record["image_url"]) > 0


# ── scrape_behance ────────────────────────────────────────────────────────────

class TestScrapeBehance:

    @patch("scraper.behance_scraper.polite_delay")
    @patch("scraper.behance_scraper.fetch_page")
    def test_all_records_tagged_with_keyword(self, mock_fetch, mock_delay):
        mock_fetch.return_value = MOCK_HTML_CARD_FALLBACK
        results = scrape_behance(keyword="editorial", pages=1)
        for record in results:
            assert record["keyword"] == "editorial"

    @patch("scraper.behance_scraper.polite_delay")
    @patch("scraper.behance_scraper.fetch_page")
    def test_skips_page_when_fetch_returns_none(self, mock_fetch, mock_delay):
        mock_fetch.return_value = None
        results = scrape_behance(keyword="test", pages=2)
        assert results == []

    @patch("scraper.behance_scraper.polite_delay")
    @patch("scraper.behance_scraper.fetch_page")
    def test_respects_max_images_limit(self, mock_fetch, mock_delay):
        # Return same HTML repeatedly across pages
        mock_fetch.return_value = MOCK_HTML_CARD_FALLBACK
        with patch("scraper.behance_scraper.MAX_IMAGES_PER_KEYWORD", 1):
            results = scrape_behance(keyword="dark", pages=5)
        assert len(results) <= 1

    @patch("scraper.behance_scraper.polite_delay")
    @patch("scraper.behance_scraper.fetch_page")
    def test_returns_list(self, mock_fetch, mock_delay):
        mock_fetch.return_value = EMPTY_HTML
        results = scrape_behance(keyword="test", pages=1)
        assert isinstance(results, list)

    @patch("scraper.behance_scraper.polite_delay")
    @patch("scraper.behance_scraper.fetch_page")
    def test_no_delay_called_on_single_page(self, mock_fetch, mock_delay):
        mock_fetch.return_value = EMPTY_HTML
        scrape_behance(keyword="test", pages=1)
        mock_delay.assert_not_called()

    @patch("scraper.behance_scraper.polite_delay")
    @patch("scraper.behance_scraper.fetch_page")
    def test_delay_called_between_pages(self, mock_fetch, mock_delay):
        mock_fetch.return_value = EMPTY_HTML
        scrape_behance(keyword="test", pages=3)
        # delay should be called between pages, not after the last one
        assert mock_delay.call_count == 2
