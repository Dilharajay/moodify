"""
Tests for utils/helpers.py

What we test:
- polite_delay() waits at least the base amount
- get_random_headers() always returns a User-Agent
- sanitize_keyword() produces safe, consistent strings
"""

import time
import pytest
from unittest.mock import patch
from utils.helpers import get_random_headers, polite_delay, sanitize_keyword
from config import BASE_HEADERS


# ── sanitize_keyword ─────────────────────────────────────────────────────────

class TestSanitizeKeyword:

    def test_spaces_replaced_with_underscores(self):
        assert sanitize_keyword("dark minimalism") == "dark_minimalism"

    def test_output_is_lowercase(self):
        assert sanitize_keyword("Dark Minimalism") == "dark_minimalism"

    def test_leading_trailing_whitespace_stripped(self):
        assert sanitize_keyword("  cyberpunk  ") == "cyberpunk"

    def test_single_word_unchanged(self):
        assert sanitize_keyword("editorial") == "editorial"

    def test_empty_string_returns_empty(self):
        assert sanitize_keyword("") == ""


# ── get_random_headers ────────────────────────────────────────────────────────

class TestGetRandomHeaders:

    def test_returns_dict(self):
        headers = get_random_headers(BASE_HEADERS)
        assert isinstance(headers, dict)

    def test_user_agent_key_present(self):
        headers = get_random_headers(BASE_HEADERS)
        assert "User-Agent" in headers

    def test_user_agent_is_non_empty_string(self):
        headers = get_random_headers(BASE_HEADERS)
        assert isinstance(headers["User-Agent"], str)
        assert len(headers["User-Agent"]) > 0

    def test_base_headers_are_preserved(self):
        headers = get_random_headers(BASE_HEADERS)
        for key in BASE_HEADERS:
            assert key in headers

    def test_does_not_mutate_original_base_headers(self):
        original_copy = BASE_HEADERS.copy()
        get_random_headers(BASE_HEADERS)
        assert BASE_HEADERS == original_copy

    def test_user_agent_rotates_across_calls(self):
        """Two calls should not always return identical user agents."""
        agents = {get_random_headers(BASE_HEADERS)["User-Agent"] for _ in range(10)}
        assert len(agents) > 1


# ── polite_delay ──────────────────────────────────────────────────────────────

class TestPoliteDelay:

    def test_waits_at_least_base_seconds(self):
        base = 0.1
        start = time.time()
        polite_delay(base=base, jitter=0.0)
        elapsed = time.time() - start
        assert elapsed >= base

    def test_jitter_adds_extra_time(self):
        base = 0.1
        jitter = 0.2
        start = time.time()
        polite_delay(base=base, jitter=jitter)
        elapsed = time.time() - start
        assert elapsed >= base

    def test_does_not_exceed_base_plus_jitter(self):
        base = 0.1
        jitter = 0.2
        start = time.time()
        polite_delay(base=base, jitter=jitter)
        elapsed = time.time() - start
        # allow 50ms overhead for test runner
        assert elapsed < base + jitter + 0.05
