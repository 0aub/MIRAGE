"""
MIRAGE V2 Chunking Module
Provides 5 configurable chunking strategies for text processing.

Usage:
    from core.chunking import get_chunker, ChunkerFactory

    # Get chunker from config
    chunker = get_chunker()

    # Or create specific chunker
    from core.chunking import TokenChunker, SemanticChunker
    chunker = TokenChunker({"max_tokens": 600})

    # Chunk text
    result = chunker.chunk("Your document text here...")
    for chunk in result.chunks:
        print(f"Chunk {chunk.index}: {chunk.text[:50]}...")

Available Strategies:
    - token: Token-based with overlap (like LightRAG)
    - semantic: Groups sentences by similarity
    - sentence: Groups by sentence count
    - recursive: Hierarchical separator-based
    - late: Embed first, then chunk (Jina approach)
"""

from .base_chunker import (
    BaseChunker,
    ChunkingStrategy,
    Chunk,
    ChunkingResult,
    TextSplitter,
)

from .token_chunker import TokenChunker
from .semantic_chunker import SemanticChunker
from .sentence_chunker import SentenceChunker
from .recursive_chunker import RecursiveChunker
from .late_chunker import LateChunker
from .factory import ChunkerFactory, get_chunker, reset_chunker

__all__ = [
    # Base classes
    "BaseChunker",
    "ChunkingStrategy",
    "Chunk",
    "ChunkingResult",
    "TextSplitter",
    # Chunkers
    "TokenChunker",
    "SemanticChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "LateChunker",
    # Factory
    "ChunkerFactory",
    "get_chunker",
    "reset_chunker",
]
