"""
Tests for utils/storage.py

What we test:
- save_to_json() creates a valid JSON file in data/raw/
- save_to_json() filename contains the keyword and a timestamp
- load_from_json() reads back identical data
- save_to_json() auto-creates output directory if missing
- save_to_json() handles empty list gracefully
"""

import os
import json
import pytest
from unittest.mock import patch
from utils.storage import save_to_json, load_from_json


SAMPLE_RECORDS = [
    {
        "image_url": "https://mir-s3.behance.net/sample1.jpg",
        "title": "Dark Minimal",
        "owner": "artist_one",
        "source": "behance",
        "keyword": "dark minimalism"
    },
    {
        "image_url": "https://mir-s3.behance.net/sample2.jpg",
        "title": "Editorial Black",
        "owner": "artist_two",
        "source": "behance",
        "keyword": "dark minimalism"
    }
]


# ── save_to_json ──────────────────────────────────────────────────────────────

class TestSaveToJson:

    def test_file_is_created(self, tmp_path):
        with patch("utils.storage.DATA_RAW_DIR", str(tmp_path)):
            filepath = save_to_json(SAMPLE_RECORDS, keyword="dark minimalism")
        assert os.path.exists(filepath)

    def test_filename_contains_sanitized_keyword(self, tmp_path):
        with patch("utils.storage.DATA_RAW_DIR", str(tmp_path)):
            filepath = save_to_json(SAMPLE_RECORDS, keyword="dark minimalism")
        assert "dark_minimalism" in os.path.basename(filepath)

    def test_file_has_json_extension(self, tmp_path):
        with patch("utils.storage.DATA_RAW_DIR", str(tmp_path)):
            filepath = save_to_json(SAMPLE_RECORDS, keyword="dark minimalism")
        assert filepath.endswith(".json")

    def test_saved_content_matches_input(self, tmp_path):
        with patch("utils.storage.DATA_RAW_DIR", str(tmp_path)):
            filepath = save_to_json(SAMPLE_RECORDS, keyword="dark minimalism")
        with open(filepath, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == SAMPLE_RECORDS

    def test_record_count_matches(self, tmp_path):
        with patch("utils.storage.DATA_RAW_DIR", str(tmp_path)):
            filepath = save_to_json(SAMPLE_RECORDS, keyword="dark minimalism")
        with open(filepath, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert len(loaded) == len(SAMPLE_RECORDS)

    def test_auto_creates_output_directory(self, tmp_path):
        nested_dir = str(tmp_path / "nested" / "raw")
        with patch("utils.storage.DATA_RAW_DIR", nested_dir):
            filepath = save_to_json(SAMPLE_RECORDS, keyword="test")
        assert os.path.exists(nested_dir)

    def test_empty_list_saves_without_error(self, tmp_path):
        with patch("utils.storage.DATA_RAW_DIR", str(tmp_path)):
            filepath = save_to_json([], keyword="empty")
        with open(filepath, "r") as f:
            loaded = json.load(f)
        assert loaded == []

    def test_returns_filepath_string(self, tmp_path):
        with patch("utils.storage.DATA_RAW_DIR", str(tmp_path)):
            result = save_to_json(SAMPLE_RECORDS, keyword="test")
        assert isinstance(result, str)


# ── load_from_json ────────────────────────────────────────────────────────────

class TestLoadFromJson:

    def test_load_returns_list(self, tmp_path):
        with patch("utils.storage.DATA_RAW_DIR", str(tmp_path)):
            filepath = save_to_json(SAMPLE_RECORDS, keyword="test")
            result = load_from_json(filepath)
        assert isinstance(result, list)

    def test_load_returns_correct_records(self, tmp_path):
        with patch("utils.storage.DATA_RAW_DIR", str(tmp_path)):
            filepath = save_to_json(SAMPLE_RECORDS, keyword="test")
            result = load_from_json(filepath)
        assert result == SAMPLE_RECORDS

    def test_load_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_from_json("/nonexistent/path/file.json")

    def test_record_fields_are_intact(self, tmp_path):
        with patch("utils.storage.DATA_RAW_DIR", str(tmp_path)):
            filepath = save_to_json(SAMPLE_RECORDS, keyword="test")
            result = load_from_json(filepath)
        for record in result:
            assert "image_url" in record
            assert "title" in record
            assert "owner" in record
            assert "source" in record
            assert "keyword" in record
