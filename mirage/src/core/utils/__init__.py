"""Core Utilities Package"""

from .arabic import (
    normalize_arabic,
    get_arabic_variants,
    extract_arabic_keywords,
    is_arabic_text,
)

__all__ = [
    "normalize_arabic",
    "get_arabic_variants",
    "extract_arabic_keywords",
    "is_arabic_text",
]
