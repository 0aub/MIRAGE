"""
Shared Arabic Text Utilities

Centralized Arabic text normalization and variant generation.
Used across retrieval, entity extraction, and chat modules.
"""

import re
from typing import List


# Precompiled regex for diacritics removal (performance optimization)
_DIACRITICS_PATTERN = re.compile(r'[\u064B-\u065F\u0670]')

# Character normalization mappings
_ARABIC_CHAR_MAPPINGS = {
    # Ta marbuta → Ha (most common normalization)
    'ة': 'ه',
    # Alef variants → Alef
    'أ': 'ا',
    'إ': 'ا',
    'آ': 'ا',
    # Alef maqsura → Ya
    'ى': 'ي',
}


def normalize_arabic(text: str, remove_tatweel: bool = True) -> str:
    """
    Normalize Arabic text for consistent matching.

    Handles common Arabic text variations:
    - Remove diacritics (tashkeel/harakat)
    - ة → ه (ta marbuta to ha)
    - أ/إ/آ → ا (alef variants to alef)
    - ى → ي (alef maqsura to ya)
    - Optionally remove tatweel (kashida)

    Args:
        text: Arabic text to normalize
        remove_tatweel: Whether to remove tatweel/kashida character

    Returns:
        Normalized text for matching

    Example:
        >>> normalize_arabic("الشَّرِكَة")
        'الشركه'
        >>> normalize_arabic("أحمد")
        'احمد'
    """
    if not text:
        return text

    # Remove diacritics (tashkeel)
    text = _DIACRITICS_PATTERN.sub('', text)

    # Apply character mappings
    for old, new in _ARABIC_CHAR_MAPPINGS.items():
        text = text.replace(old, new)

    # Remove tatweel (stretching character) if requested
    if remove_tatweel:
        text = text.replace('\u0640', '')

    return text


def get_arabic_variants(text: str) -> List[str]:
    """
    Generate common Arabic text variants for a phrase.

    Returns both the original and normalized forms,
    plus common variants for better matching.

    Args:
        text: Original Arabic text

    Returns:
        List of text variants to search for (deduplicated)

    Example:
        >>> get_arabic_variants("شركة")
        ['شركة', 'شركه']
    """
    if not text:
        return [text] if text else []

    variants = {text}  # Use set for deduplication

    # Add normalized form
    normalized = normalize_arabic(text)
    if normalized != text:
        variants.add(normalized)

    # Bidirectional ta marbuta / ha conversion
    if 'ه' in text:
        variants.add(text.replace('ه', 'ة'))
    if 'ة' in text:
        variants.add(text.replace('ة', 'ه'))

    # Also add normalized versions of the variants
    for variant in list(variants):
        norm_variant = normalize_arabic(variant)
        if norm_variant not in variants:
            variants.add(norm_variant)

    return list(variants)


def extract_arabic_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract potential Arabic keywords/entities from text.

    Filters out common Arabic stop words and short words.

    Args:
        text: Text to extract keywords from
        min_length: Minimum word length to include

    Returns:
        List of potential keywords
    """
    if not text:
        return []

    # Common Arabic stop words to filter
    stop_words = {
        'من', 'هي', 'هو', 'ما', 'هل', 'في', 'على', 'إلى', 'عن',
        'التي', 'الذي', 'هذا', 'هذه', 'تلك', 'ذلك', 'كان', 'كانت',
        'أن', 'إن', 'لا', 'نعم', 'أو', 'و', 'ثم', 'لكن', 'بل',
        'كل', 'بعض', 'غير', 'مع', 'عند', 'قبل', 'بعد', 'حتى',
    }

    words = text.split()
    keywords = []

    for word in words:
        # Clean word
        word_clean = word.strip('،.؟!:«»"\'')

        # Skip stop words and short words
        if word_clean.lower() in stop_words:
            continue
        if len(word_clean) < min_length:
            continue

        # Keep definite article words if they're long enough
        if word_clean.startswith('ال') and len(word_clean) <= 4:
            continue

        keywords.append(word_clean)

    return keywords


def is_arabic_text(text: str, threshold: float = 0.5) -> bool:
    """
    Check if text is primarily Arabic.

    Args:
        text: Text to check
        threshold: Minimum ratio of Arabic characters

    Returns:
        True if text is primarily Arabic
    """
    if not text:
        return False

    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))

    total = arabic_chars + latin_chars
    if total == 0:
        return False

    return (arabic_chars / total) >= threshold
