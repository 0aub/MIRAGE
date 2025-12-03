"""
MIRAGE V2 Arabic Processing - Factory
Creates Arabic processors based on configuration.
"""

from typing import Dict, Any, Optional, Type
from loguru import logger

from .base_processor import BaseArabicProcessor
from .camel_processor import CAMeLProcessor
from .stanza_processor import StanzaProcessor
from .disabled_processor import DisabledProcessor


class ArabicProcessorFactory:
    """
    Factory for creating Arabic text processors.

    Supports:
    - camel: CAMeL Tools (recommended for Arabic)
    - stanza: Stanford Stanza
    - none/disabled: Minimal processing (for A/B testing)
    """

    # Registry of available processors
    PROCESSORS: Dict[str, Type[BaseArabicProcessor]] = {
        "camel": CAMeLProcessor,
        "stanza": StanzaProcessor,
        "none": DisabledProcessor,
        "disabled": DisabledProcessor,
    }

    @classmethod
    def create(
        cls,
        processor_type: str = "camel",
        config: Optional[Dict[str, Any]] = None
    ) -> BaseArabicProcessor:
        """
        Create an Arabic processor.

        Args:
            processor_type: Type of processor ("camel", "stanza", "none", "disabled")
            config: Processor configuration

        Returns:
            Configured Arabic processor

        Raises:
            ValueError: If processor type is unknown
        """
        processor_type = processor_type.lower()

        if processor_type not in cls.PROCESSORS:
            available = ", ".join(cls.PROCESSORS.keys())
            raise ValueError(
                f"Unknown Arabic processor: {processor_type}. "
                f"Available: {available}"
            )

        processor_class = cls.PROCESSORS[processor_type]
        processor = processor_class(config)

        logger.info(f"Created Arabic processor: {processor_type}")
        return processor

    @classmethod
    def create_from_config(cls, arabic_config: Dict[str, Any]) -> BaseArabicProcessor:
        """
        Create processor from configuration dict.

        Expected config format:
        {
            "enabled": true,
            "processor": "camel",  # or "stanza", "none"
            "camel": { ... },      # CAMeL-specific config
            "stanza": { ... },     # Stanza-specific config
        }

        Args:
            arabic_config: Configuration dictionary

        Returns:
            Configured Arabic processor
        """
        # Check if Arabic processing is enabled
        if not arabic_config.get("enabled", True):
            logger.info("Arabic processing disabled in config")
            return DisabledProcessor({"apply_normalization": False})

        # Get processor type
        processor_type = arabic_config.get("processor", "camel")

        # Get processor-specific config
        if processor_type == "camel":
            config = arabic_config.get("camel", {})
            # Add normalization config
            if "normalization" not in config:
                config["normalization"] = arabic_config.get("normalization", {})
            # Add entity normalization config
            if "entity_normalization" not in config:
                config["entity_normalization"] = arabic_config.get("entity_normalization", {})

        elif processor_type == "stanza":
            config = arabic_config.get("stanza", {})
            if "normalization" not in config:
                config["normalization"] = arabic_config.get("normalization", {})

        else:
            config = {}

        return cls.create(processor_type, config)

    @classmethod
    def list_processors(cls) -> Dict[str, str]:
        """
        List available processors with descriptions.

        Returns:
            Dict of processor_type -> description
        """
        return {
            "camel": "CAMeL Tools - Full Arabic NLP (morphology, NER, dialect)",
            "stanza": "Stanford Stanza - Multilingual NLP with Arabic support",
            "none": "Disabled - Minimal processing for baseline comparison",
            "disabled": "Disabled - Alias for 'none'",
        }

    @classmethod
    def register(
        cls,
        name: str,
        processor_class: Type[BaseArabicProcessor]
    ) -> None:
        """
        Register a custom processor.

        Args:
            name: Processor name
            processor_class: Processor class
        """
        cls.PROCESSORS[name.lower()] = processor_class
        logger.info(f"Registered Arabic processor: {name}")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_arabic_processor: Optional[BaseArabicProcessor] = None


def get_arabic_processor(
    processor_type: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    force_new: bool = False
) -> BaseArabicProcessor:
    """
    Get or create global Arabic processor.

    If no arguments provided, creates processor from V2 config.

    Args:
        processor_type: Processor type (if None, reads from config)
        config: Processor config (if None, reads from config)
        force_new: Force creation of new processor

    Returns:
        Arabic processor instance
    """
    global _arabic_processor

    if _arabic_processor is not None and not force_new:
        return _arabic_processor

    # Load from V2 config if not specified
    if processor_type is None and config is None:
        try:
            from ...config.v2_config_loader import get_v2_config
            v2_config = get_v2_config()
            arabic_config = dict(v2_config.get_arabic_config())

            _arabic_processor = ArabicProcessorFactory.create_from_config(arabic_config)
            return _arabic_processor

        except Exception as e:
            logger.warning(f"Failed to load Arabic config from V2: {e}")
            # Fall back to disabled
            _arabic_processor = DisabledProcessor()
            return _arabic_processor

    # Create with provided arguments
    _arabic_processor = ArabicProcessorFactory.create(
        processor_type or "disabled",
        config
    )
    return _arabic_processor


def reset_arabic_processor() -> None:
    """Reset global Arabic processor"""
    global _arabic_processor

    if _arabic_processor is not None:
        _arabic_processor.close()
        _arabic_processor = None
