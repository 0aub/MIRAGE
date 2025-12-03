"""
MIRAGE V2 Arabic Processing Module
Provides configurable Arabic NLP with multiple backend options.

Usage:
    from core.arabic import get_arabic_processor, ArabicProcessorFactory

    # Get processor from config
    processor = get_arabic_processor()

    # Or create specific processor
    from core.arabic import CAMeLProcessor, StanzaProcessor, DisabledProcessor
    processor = CAMeLProcessor(config)

    # Process text
    result = processor.process("مرحبا بالعالم")
    print(result.normalized_text)
    print(result.entities)
"""

from .base_processor import (
    BaseArabicProcessor,
    ProcessedText,
    Token,
    NamedEntity,
    EntityType,
    ArabicDialect,
    ArabicNormalizer,
    EntityNormalizer,
)

from .camel_processor import CAMeLProcessor
from .stanza_processor import StanzaProcessor
from .disabled_processor import DisabledProcessor
from .factory import ArabicProcessorFactory, get_arabic_processor

__all__ = [
    # Base classes
    "BaseArabicProcessor",
    "ProcessedText",
    "Token",
    "NamedEntity",
    "EntityType",
    "ArabicDialect",
    "ArabicNormalizer",
    "EntityNormalizer",
    # Processors
    "CAMeLProcessor",
    "StanzaProcessor",
    "DisabledProcessor",
    # Factory
    "ArabicProcessorFactory",
    "get_arabic_processor",
]
