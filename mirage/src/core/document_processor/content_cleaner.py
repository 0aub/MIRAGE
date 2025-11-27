"""
Content Cleaner for YouTube Transcripts and Noisy Text
Removes common noise patterns from auto-generated transcripts
"""

import re
from typing import Dict, Any
from loguru import logger


class ContentCleaner:
    """Clean noisy content from YouTube transcripts and other sources"""

    def __init__(self):
        """Initialize cleaner with noise patterns"""
        # Common YouTube transcript noise patterns
        self.noise_patterns = [
            # Music and sound markers (Arabic)
            r'\[موسيقى\]',
            r'\[تصفيق\]',
            r'\[ضحك\]',
            r'\[صمت\]',
           r'\[صوت\]',

            # Music and sound markers (English)
            r'\[Music\]',
            r'\[Applause\]',
            r'\[Laughter\]',
            r'\[Silence\]',
            r'\[Sound\]',
            r'\[music\]',
            r'\[applause\]',

            # Foreign characters (Thai, Cyrillic, etc.)
            r'[\u0E00-\u0E7F]+',  # Thai
            r'[\u0400-\u04FF]+',  # Cyrillic (only standalone, not in words)

            # Random single characters
            r'\bเฮ\b',
            r'\bKom\b',
            r'\bДа\b',
            r'\baوف\b',

            # Multiple consecutive spaces
            r'\s{3,}',

            # Repeated punctuation
            r'[.]{3,}',
            r'[!]{2,}',
            r'[?]{2,}',
        ]

        # Compile patterns for efficiency
        self.compiled_patterns = [
            (re.compile(pattern, re.UNICODE), ' ') for pattern in self.noise_patterns
        ]

        logger.info("Content cleaner initialized with YouTube transcript denoising patterns")

    def clean_youtube_transcript(self, text: str) -> Dict[str, Any]:
        """
        Clean YouTube transcript from auto-generated noise

        Args:
            text: Raw YouTube transcript

        Returns:
            Dict with cleaned_text, removed_noise, and stats
        """
        if not text:
            return {
                "cleaned_text": "",
                "removed_patterns": [],
                "original_length": 0,
                "cleaned_length": 0,
                "noise_percentage": 0.0
            }

        original_length = len(text)
        cleaned = text
        removed_patterns = []

        # Apply all noise pattern removals
        for pattern, replacement in self.compiled_patterns:
            matches = pattern.findall(cleaned)
            if matches:
                removed_patterns.append({
                    "pattern": pattern.pattern,
                    "count": len(matches),
                    "examples": list(set(matches))[:5]  # Keep max 5 unique examples
                })
                cleaned = pattern.sub(replacement, cleaned)

        # Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)  # Multiple newlines to single
        cleaned = cleaned.strip()

        cleaned_length = len(cleaned)
        noise_percentage = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0.0

        logger.info(f"Cleaned transcript: {original_length} → {cleaned_length} chars ({noise_percentage:.1f}% noise removed)")
        if removed_patterns:
            logger.debug(f"Removed {len(removed_patterns)} noise pattern types")

        return {
            "cleaned_text": cleaned,
            "removed_patterns": removed_patterns,
            "original_length": original_length,
            "cleaned_length": cleaned_length,
            "noise_percentage": round(noise_percentage, 2)
        }

    def clean_text(self, text: str, content_type: str = "general") -> str:
        """
        Clean text based on content type

        Args:
            text: Text to clean
            content_type: Type of content ("youtube", "webpage", "file", "general")

        Returns:
            Cleaned text string
        """
        if content_type == "youtube":
            result = self.clean_youtube_transcript(text)
            return result["cleaned_text"]
        else:
            # Basic cleaning for other content types
            cleaned = text.strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            return cleaned

    def should_clean(self, text: str, content_type: str) -> bool:
        """
        Determine if text needs cleaning based on noise detection

        Args:
            text: Text to check
            content_type: Content type

        Returns:
            True if cleaning is recommended
        """
        if content_type == "youtube":
            # Check for common YouTube noise markers
            youtube_markers = [r'\[موسيقى\]', r'\[Music\]', r'\[تصفيق\]', r'\[Applause\]']
            for marker in youtube_markers:
                if re.search(marker, text):
                    logger.info(f"Detected YouTube noise markers - cleaning recommended")
                    return True

            # Check for high percentage of non-content characters
            content_chars = sum(1 for c in text if c.isalnum() or c in '.,;:!?-')
            total_chars = len(text)
            if total_chars > 0:
                content_ratio = content_chars / total_chars
                if content_ratio < 0.7:  # Less than 70% actual content
                    logger.info(f"Low content ratio ({content_ratio:.1%}) - cleaning recommended")
                    return True

        return False
