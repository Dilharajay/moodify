"""
Tests for main.py (Phase 2)

What we test:
- CLI handles --keyword and --source arguments
- Both Behance and Pinterest scrapers are called by default
- Only specific scraper is called if --source is set
- save_to_json is called if any results are found
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import importlib

SAMPLE_RECORDS = [
    {"image_url": "url1", "title": "T1", "owner": "O1", "source": "behance", "keyword": "k"}
]

class TestCLIArguments:

    def test_missing_keyword_exits(self):
        with patch("sys.argv", ["main.py"]):
            import main
            importlib.reload(main)
            with pytest.raises(SystemExit):
                main.main()

    def test_help_flag(self):
        with patch("sys.argv", ["main.py", "--help"]):
            import main
            importlib.reload(main)
            with pytest.raises(SystemExit):
                main.main()

class TestMainPipeline:

    @patch("utils.storage.save_to_json")
    @patch("scraper.behance_scraper.scrape_behance")
    @patch("scraper.pinterest_scraper.scrape_pinterest")
    def test_both_scrapers_called_by_default(self, mock_p_scrape, mock_b_scrape, mock_save):
        mock_b_scrape.return_value = SAMPLE_RECORDS
        mock_p_scrape.return_value = SAMPLE_RECORDS
        mock_save.return_value = "path/to/file.json"

        with patch("sys.argv", ["main.py", "--keyword", "test"]):
            import main
            importlib.reload(main)
            main.main()

        mock_b_scrape.assert_called()
        mock_p_scrape.assert_called()
        mock_save.assert_called_once()

    @patch("utils.storage.save_to_json")
    @patch("scraper.behance_scraper.scrape_behance")
    @patch("scraper.pinterest_scraper.scrape_pinterest")
    def test_only_behance_called_if_source_behance(self, mock_p_scrape, mock_b_scrape, mock_save):
        mock_b_scrape.return_value = SAMPLE_RECORDS
        mock_p_scrape.return_value = []
        
        with patch("sys.argv", ["main.py", "--keyword", "test", "--source", "behance"]):
            import main
            importlib.reload(main)
            main.main()

        mock_b_scrape.assert_called()
        mock_p_scrape.assert_not_called()

    @patch("utils.storage.save_to_json")
    @patch("scraper.behance_scraper.scrape_behance")
    @patch("scraper.pinterest_scraper.scrape_pinterest")
    def test_only_pinterest_called_if_source_pinterest(self, mock_p_scrape, mock_b_scrape, mock_save):
        mock_p_scrape.return_value = SAMPLE_RECORDS
        mock_b_scrape.return_value = []
        
        with patch("sys.argv", ["main.py", "--keyword", "test", "--source", "pinterest"]):
            import main
            importlib.reload(main)
            main.main()

        mock_p_scrape.assert_called()
        mock_b_scrape.assert_not_called()

    @patch("utils.storage.save_to_json")
    @patch("scraper.behance_scraper.scrape_behance")
    @patch("scraper.pinterest_scraper.scrape_pinterest")
    def test_no_save_if_no_results(self, mock_p_scrape, mock_b_scrape, mock_save):
        mock_p_scrape.return_value = []
        mock_b_scrape.return_value = []
        
        with patch("sys.argv", ["main.py", "--keyword", "test"]):
            import main
            importlib.reload(main)
            main.main()

        mock_save.assert_not_called()
