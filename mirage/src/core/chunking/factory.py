"""
MIRAGE V2 Chunking - Factory
Creates chunkers based on configuration.
"""

from typing import Dict, Any, Optional, Type
from loguru import logger

from .base_chunker import BaseChunker, ChunkingStrategy
from .token_chunker import TokenChunker
from .semantic_chunker import SemanticChunker
from .sentence_chunker import SentenceChunker
from .recursive_chunker import RecursiveChunker
from .late_chunker import LateChunker


class ChunkerFactory:
    """
    Factory for creating text chunkers.

    Supports all 5 chunking strategies:
    - token: Token-based with overlap (LightRAG-style)
    - semantic: Semantic similarity grouping
    - sentence: Sentence-based with paragraph awareness
    - recursive: Recursive character splitting
    - late: Late chunking (embed first, chunk after)
    """

    # Registry of available chunkers
    CHUNKERS: Dict[str, Type[BaseChunker]] = {
        "token": TokenChunker,
        "semantic": SemanticChunker,
        "sentence": SentenceChunker,
        "recursive": RecursiveChunker,
        "late": LateChunker,
    }

    @classmethod
    def create(
        cls,
        strategy: str = "semantic",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseChunker:
        """
        Create a text chunker.

        Args:
            strategy: Chunking strategy name
            config: Chunker configuration
            **kwargs: Additional arguments passed to chunker

        Returns:
            Configured chunker instance

        Raises:
            ValueError: If strategy is unknown
        """
        strategy = strategy.lower()

        if strategy not in cls.CHUNKERS:
            available = ", ".join(cls.CHUNKERS.keys())
            raise ValueError(
                f"Unknown chunking strategy: {strategy}. "
                f"Available: {available}"
            )

        chunker_class = cls.CHUNKERS[strategy]
        chunker = chunker_class(config, **kwargs)

        logger.info(f"Created chunker: {strategy}")
        return chunker

    @classmethod
    def create_from_config(cls, chunking_config: Dict[str, Any]) -> BaseChunker:
        """
        Create chunker from configuration dict.

        Expected config format:
        {
            "strategy": "semantic",
            "strategies": {
                "token": { ... },
                "semantic": { ... },
                ...
            }
        }

        Args:
            chunking_config: Configuration dictionary

        Returns:
            Configured chunker
        """
        # Get strategy
        strategy = chunking_config.get("strategy", "semantic")

        # Get strategy-specific config
        strategies_config = chunking_config.get("strategies", {})
        strategy_config = strategies_config.get(strategy, {})

        return cls.create(strategy, strategy_config)

    @classmethod
    def list_strategies(cls) -> Dict[str, str]:
        """
        List available chunking strategies with descriptions.

        Returns:
            Dict of strategy_name -> description
        """
        return {
            "token": "Token-based chunking with overlap (LightRAG-style)",
            "semantic": "Semantic similarity grouping using embeddings",
            "sentence": "Sentence-based with paragraph awareness",
            "recursive": "Recursive character splitting with separators",
            "late": "Late chunking - embed full doc, then chunk",
        }

    @classmethod
    def register(
        cls,
        name: str,
        chunker_class: Type[BaseChunker]
    ) -> None:
        """
        Register a custom chunker.

        Args:
            name: Strategy name
            chunker_class: Chunker class
        """
        cls.CHUNKERS[name.lower()] = chunker_class
        logger.info(f"Registered chunker: {name}")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_chunker: Optional[BaseChunker] = None


def get_chunker(
    strategy: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    force_new: bool = False
) -> BaseChunker:
    """
    Get or create global chunker.

    If no arguments provided, creates chunker from V2 config.

    Args:
        strategy: Chunking strategy (if None, reads from config)
        config: Chunker config (if None, reads from config)
        force_new: Force creation of new chunker

    Returns:
        Chunker instance
    """
    global _chunker

    if _chunker is not None and not force_new:
        return _chunker

    # Load from V2 config if not specified
    if strategy is None and config is None:
        try:
            from ...config.v2_config_loader import get_v2_config
            v2_config = get_v2_config()
            chunking_config = dict(v2_config.get_chunking_config())

            _chunker = ChunkerFactory.create_from_config(chunking_config)
            return _chunker

        except Exception as e:
            logger.warning(f"Failed to load chunking config from V2: {e}")
            # Fall back to semantic chunking
            _chunker = SemanticChunker()
            return _chunker

    # Create with provided arguments
    _chunker = ChunkerFactory.create(strategy or "semantic", config)
    return _chunker


def reset_chunker() -> None:
    """Reset global chunker"""
    global _chunker
    _chunker = None
