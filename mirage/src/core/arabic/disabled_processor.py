"""
MIRAGE V2 Arabic Processing - Disabled Processor
Passthrough processor that does minimal processing.
Used for A/B testing to measure impact of Arabic NLP.
"""

from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from .base_processor import (
    BaseArabicProcessor,
    ProcessedText,
    Token,
    NamedEntity,
    EntityType,
    ArabicDialect,
    ArabicNormalizer,
)


class DisabledProcessor(BaseArabicProcessor):
    """
    Minimal Arabic text processor for baseline comparison.

    This processor:
    - Does NOT perform morphological analysis
    - Does NOT perform NER
    - Does NOT perform dialect detection
    - Only applies basic normalization (optional)
    - Uses simple whitespace tokenization

    Use this processor for A/B testing to measure the impact
    of full Arabic NLP processing on downstream tasks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize disabled processor.

        Args:
            config: Configuration dict with options:
                - apply_normalization: bool (default True)
                - normalization.*: normalization options if enabled
        """
        super().__init__(config)

        # Whether to apply even basic normalization
        self.apply_normalization = self.config.get("apply_normalization", True)

        # Normalization config (minimal by default)
        if self.apply_normalization:
            norm_config = self.config.get("normalization", {})
            self.normalizer = ArabicNormalizer(
                normalize_alef=norm_config.get("alef", True),
                normalize_yaa=norm_config.get("yaa", True),
                normalize_taa_marbuta=norm_config.get("taa_marbuta", False),
                remove_diacritics=norm_config.get("remove_diacritics", True),
                remove_tatweel=norm_config.get("remove_tatweel", True),
            )
        else:
            self.normalizer = None

    def initialize(self) -> None:
        """No initialization needed for disabled processor"""
        logger.info("Disabled Arabic processor initialized (passthrough mode)")
        self._initialized = True

    def process(self, text: str) -> ProcessedText:
        """
        Minimal processing of text.

        Args:
            text: Input text

        Returns:
            ProcessedText with minimal annotations
        """
        self.ensure_initialized()

        # Check if text contains Arabic
        has_arabic = self.is_arabic(text)

        # Normalize if enabled
        normalized = self.normalize(text) if self.apply_normalization else text

        # Simple tokenization
        tokens = self.tokenize(text)

        return ProcessedText(
            original_text=text,
            normalized_text=normalized,
            tokens=tokens,
            entities=[],  # No entity extraction
            dialect=ArabicDialect.UNKNOWN,  # No dialect detection
            is_arabic=has_arabic,
            language_confidence=1.0 if has_arabic else 0.0,
            metadata={
                "processor": "disabled",
                "normalization_applied": self.apply_normalization,
            }
        )

    def normalize(self, text: str) -> str:
        """
        Apply basic normalization if enabled.

        Args:
            text: Input text

        Returns:
            Normalized text (or original if disabled)
        """
        if self.normalizer:
            return self.normalizer.normalize(text)
        return text

    def tokenize(self, text: str) -> List[Token]:
        """
        Simple whitespace tokenization.

        Args:
            text: Input text

        Returns:
            List of Token objects (minimal info)
        """
        tokens = []
        words = text.split()

        for word in words:
            token = Token(
                text=word,
                lemma=None,  # No lemmatization
                pos=None,    # No POS tagging
                is_arabic=self.is_arabic(word),
            )
            tokens.append(token)

        return tokens

    def extract_entities(self, text: str) -> List[NamedEntity]:
        """
        No entity extraction in disabled mode.

        Args:
            text: Input text

        Returns:
            Empty list
        """
        return []

    def detect_dialect(self, text: str) -> Tuple[ArabicDialect, float]:
        """
        No dialect detection in disabled mode.

        Args:
            text: Input text

        Returns:
            Tuple of (UNKNOWN, 0.0)
        """
        return ArabicDialect.UNKNOWN, 0.0

    def close(self) -> None:
        """Nothing to close for disabled processor"""
        super().close()
