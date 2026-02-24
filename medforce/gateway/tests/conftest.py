"""
Test configuration for gateway tests.
Uses pytest-asyncio in auto mode for async tests.
"""

import pytest


# Override the strict mode from the root conftest for gateway tests
def pytest_collection_modifyitems(items):
    """Add asyncio marker to all async test functions."""
    for item in items:
        if item.get_closest_marker("asyncio") is None:
            if hasattr(item, "function") and hasattr(item.function, "__wrapped__"):
                pass  # already wrapped
