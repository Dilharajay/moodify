"""
conftest.py — shared pytest fixtures and configuration

Fixtures defined here are automatically available to all test files
without needing to import them.
"""

import pytest
import sys
import os

# Make sure the project root is on the Python path
# so imports like `from utils.helpers import ...` work inside tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_records():
    """A minimal set of image records matching the expected schema."""
    return [
        {
            "image_url": "https://mir-s3.behance.net/img1.jpg",
            "title": "Dark Minimal",
            "owner": "artist_one",
            "source": "behance",
            "keyword": "dark minimalism"
        },
        {
            "image_url": "https://mir-s3.behance.net/img2.jpg",
            "title": "Editorial Noir",
            "owner": "artist_two",
            "source": "behance",
            "keyword": "dark minimalism"
        }
    ]

@pytest.fixture
def empty_records():
    """An empty list for edge case testing."""
    return []
