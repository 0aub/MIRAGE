"""
Chunker Factory
Selects appropriate chunking strategy based on content type
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .semantic_chunker import SemanticChunker
from ..embeddings import JinaEmbedder


class ChunkerFactory:
    """
    Factory for creating content-type-specific chunkers
    """

    def __init__(self, strategies_file: Optional[str] = None):
        """
        Initialize factory with chunking strategies config

        Args:
            strategies_file: Path to chunking_strategies.yaml
        """
        if strategies_file is None:
            # Default to chunking_strategies.yaml in config directory
            config_dir = Path(__file__).parent.parent.parent / "config"
            strategies_file = config_dir / "chunking_strategies.yaml"

        self.strategies_file = Path(strategies_file)
        self._strategies: Dict[str, Any] = {}
        self._load_strategies()

    def _load_strategies(self) -> None:
        """Load chunking strategies from YAML file"""
        try:
            if not self.strategies_file.exists():
                logger.warning(
                    f"Chunking strategies file not found: {self.strategies_file}. "
                    f"Using default semantic chunking."
                )
                self._strategies = {}
                return

            with open(self.strategies_file, 'r', encoding='utf-8') as f:
                self._strategies = yaml.safe_load(f)

            logger.info(f"Loaded chunking strategies from {self.strategies_file}")

        except Exception as e:
            logger.error(f"Failed to load chunking strategies: {e}")
            self._strategies = {}

    def get_chunker(
        self,
        content_type: str,
        embedder: Optional[JinaEmbedder] = None,
    ):
        """
        Get appropriate chunker for content type

        Args:
            content_type: Type of content ("youtube", "webpage", "pdf", "text_file", etc.)
            embedder: Optional embedder instance (created if not provided)

        Returns:
            Appropriate chunker instance
        """
        # Normalize content type
        content_type = content_type.lower()

        # Get strategy config for this content type
        strategy_config = self._strategies.get(content_type)

        if not strategy_config:
            logger.warning(
                f"No chunking strategy found for content type '{content_type}'. "
                f"Using default semantic chunking."
            )
            strategy_config = self._strategies.get("default", {})

        # Determine which chunker to use
        strategy_name = strategy_config.get("strategy", "semantic")

        logger.info(
            f"Creating {strategy_name} chunker for content type '{content_type}'"
        )

        # Create appropriate chunker
        if strategy_name == "semantic":
            return self._create_semantic_chunker(strategy_config, embedder)

        elif strategy_name == "structural":
            # TODO: Implement in Phase 2
            logger.warning(
                f"Structural chunking not yet implemented. "
                f"Falling back to semantic chunking."
            )
            return self._create_semantic_chunker(strategy_config, embedder)

        elif strategy_name == "page_based":
            # TODO: Implement in Phase 3
            logger.warning(
                f"Page-based chunking not yet implemented. "
                f"Falling back to semantic chunking."
            )
            return self._create_semantic_chunker(strategy_config, embedder)

        else:
            logger.error(
                f"Unknown chunking strategy: {strategy_name}. "
                f"Using semantic chunking."
            )
            return self._create_semantic_chunker(strategy_config, embedder)

    def _create_semantic_chunker(
        self,
        config: Dict[str, Any],
        embedder: Optional[JinaEmbedder] = None,
    ) -> SemanticChunker:
        """
        Create SemanticChunker with config parameters

        Args:
            config: Strategy configuration
            embedder: Optional embedder instance

        Returns:
            Configured SemanticChunker instance
        """
        # Extract parameters from config
        chunk_size = config.get("chunk_size", 1000)
        max_chunk_size = config.get("max_chunk_size", 1500)
        similarity_threshold = config.get("similarity_threshold", 0.7)
        min_sentences_per_chunk = config.get("min_sentences_per_chunk", 2)

        logger.debug(
            f"Creating SemanticChunker: "
            f"chunk_size={chunk_size}, "
            f"max={max_chunk_size}, "
            f"threshold={similarity_threshold}"
        )

        return SemanticChunker(
            embedder=embedder,
            chunk_size=chunk_size,
            max_chunk_size=max_chunk_size,
            similarity_threshold=similarity_threshold,
            min_sentences_per_chunk=min_sentences_per_chunk,
        )

    def get_strategy_info(self, content_type: str) -> Dict[str, Any]:
        """
        Get information about chunking strategy for content type

        Args:
            content_type: Type of content

        Returns:
            Strategy configuration dict
        """
        content_type = content_type.lower()
        return self._strategies.get(content_type, self._strategies.get("default", {}))


# Global factory instance
_chunker_factory: Optional[ChunkerFactory] = None


def get_chunker_factory() -> ChunkerFactory:
    """
    Get or create global ChunkerFactory instance

    Returns:
        Global ChunkerFactory instance
    """
    global _chunker_factory
    if _chunker_factory is None:
        _chunker_factory = ChunkerFactory()
    return _chunker_factory
