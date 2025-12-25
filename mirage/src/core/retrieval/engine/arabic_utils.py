"""
Arabic Text Utilities for Retrieval

Re-exports from shared utils module for backward compatibility.
"""

from ...utils.arabic import normalize_arabic, get_arabic_variants

__all__ = ["normalize_arabic", "get_arabic_variants"]
