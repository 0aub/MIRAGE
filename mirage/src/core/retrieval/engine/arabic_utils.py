"""
Arabic Text Utilities for Retrieval

Functions for normalizing and matching Arabic text with common variations.
"""

import re
from typing import List


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for consistent matching.

    Handles common Arabic text variations:
    - ة ↔ ه (ta marbuta / ha) - critical for entity names
    - أ/إ/آ → ا (alef variants)
    - ى → ي (alef maqsura / ya)
    - Remove diacritics (tashkeel)

    Args:
        text: Arabic text to normalize

    Returns:
        Normalized text for matching
    """
    if not text:
        return text

    # Remove diacritics (tashkeel)
    diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    text = diacritics.sub('', text)

    # Normalize ta marbuta ة → ه (both directions for matching)
    # We convert to ه as it's more common in informal Arabic text
    text = text.replace('ة', 'ه')

    # Normalize alef variants → ا
    text = text.replace('أ', 'ا')
    text = text.replace('إ', 'ا')
    text = text.replace('آ', 'ا')

    # Normalize alef maqsura → ي
    text = text.replace('ى', 'ي')

    return text


def get_arabic_variants(text: str) -> List[str]:
    """
    Generate common Arabic text variants for a phrase.

    Returns both the original and normalized forms,
    plus common variants for better matching.

    Args:
        text: Original Arabic text

    Returns:
        List of text variants to search for
    """
    variants = [text]

    # Add normalized form
    normalized = normalize_arabic(text)
    if normalized != text:
        variants.append(normalized)

    # If text has ه, also try ة (reverse normalization)
    if 'ه' in text:
        reverse = text.replace('ه', 'ة')
        if reverse not in variants:
            variants.append(reverse)

    # If text has ة, also try ه
    if 'ة' in text:
        reverse = text.replace('ة', 'ه')
        if reverse not in variants:
            variants.append(reverse)

    return list(set(variants))
