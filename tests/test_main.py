"""
Tests for main.py (CLI entry point)

What we test:
- CLI exits with error when --keyword is missing
- CLI runs successfully with valid --keyword argument
- CLI calls scrape_behance() and save_to_json() with correct arguments
- CLI prints a summary to stdout
- CLI handles empty scrape results gracefully without crashing
"""

import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
import subprocess
import sys


SAMPLE_RECORDS = [
    {
        "image_url": "https://mir-s3.behance.net/img1.jpg",
        "title": "Dark Edit",
        "owner": "artist_a",
        "source": "behance",
        "keyword": "dark minimalism"
    }
]


# ── CLI argument validation ───────────────────────────────────────────────────

class TestCLIArguments:

    def test_missing_keyword_exits_with_error(self):
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True, text=True
        )
        assert result.returncode != 0

    def test_help_flag_works(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "--keyword" in result.stdout

    def test_keyword_argument_accepted(self):
        with patch("scraper.behance_scraper.scrape_behance", return_value=[]):
            result = subprocess.run(
                [sys.executable, "main.py", "--keyword", "test"],
                capture_output=True, text=True
            )
        # should not crash even with no results
        assert result.returncode == 0


# ── Core pipeline wiring ──────────────────────────────────────────────────────

class TestMainPipeline:

    @patch("utils.storage.save_to_json")
    @patch("scraper.behance_scraper.scrape_behance")
    def test_scrape_behance_called_with_correct_keyword(self, mock_scrape, mock_save):
        mock_scrape.return_value = SAMPLE_RECORDS
        mock_save.return_value = "data/raw/dark_minimalism_20240101.json"

        with patch("sys.argv", ["main.py", "--keyword", "dark minimalism", "--pages", "2"]):
            import importlib
            import main
            importlib.reload(main)
            main.main()

        mock_scrape.assert_called_once_with(keyword="dark minimalism", pages=2)

    @patch("utils.storage.save_to_json")
    @patch("scraper.behance_scraper.scrape_behance")
    def test_save_to_json_called_when_results_exist(self, mock_scrape, mock_save):
        mock_scrape.return_value = SAMPLE_RECORDS
        mock_save.return_value = "data/raw/test.json"

        with patch("sys.argv", ["main.py", "--keyword", "editorial"]):
            import importlib
            import main
            importlib.reload(main)
            main.main()

        mock_save.assert_called_once()

    @patch("utils.storage.save_to_json")
    @patch("scraper.behance_scraper.scrape_behance")
    def test_save_not_called_on_empty_results(self, mock_scrape, mock_save):
        mock_scrape.return_value = []

        with patch("sys.argv", ["main.py", "--keyword", "nothing"]):
            import importlib
            import main
            importlib.reload(main)
            main.main()

        mock_save.assert_not_called()

    @patch("utils.storage.save_to_json")
    @patch("scraper.behance_scraper.scrape_behance")
    def test_output_includes_keyword_in_summary(self, mock_scrape, mock_save, capsys):
        mock_scrape.return_value = SAMPLE_RECORDS
        mock_save.return_value = "data/raw/editorial.json"

        with patch("sys.argv", ["main.py", "--keyword", "editorial"]):
            import importlib
            import main
            importlib.reload(main)
            main.main()

        captured = capsys.readouterr()
        assert "editorial" in captured.out.lower()
