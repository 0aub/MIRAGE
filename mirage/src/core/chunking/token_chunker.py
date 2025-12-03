"""
MIRAGE V2 Chunking - Token-based Chunker
Splits text into chunks based on token count with overlap.
Similar to LightRAG's approach.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from .base_chunker import (
    BaseChunker,
    ChunkingStrategy,
    Chunk,
    ChunkingResult,
)


class TokenChunker(BaseChunker):
    """
    Token-based text chunker.

    Splits text into chunks of approximately equal token count
    with configurable overlap between chunks.

    This is the most common chunking strategy, used by LightRAG
    with 1200 tokens per chunk.

    Config options:
        - max_tokens: Maximum tokens per chunk (default: 600)
        - overlap_tokens: Token overlap between chunks (default: 50)
        - tokenizer: Tokenizer to use (default: cl100k_base)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.max_tokens = self.config.get("max_tokens", 600)
        self.overlap_tokens = self.config.get("overlap_tokens", 50)

        logger.debug(
            f"TokenChunker initialized: max_tokens={self.max_tokens}, "
            f"overlap={self.overlap_tokens}"
        )

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.TOKEN

    def chunk(self, text: str, **kwargs) -> ChunkingResult:
        """
        Split text into token-based chunks.

        Args:
            text: Input text
            **kwargs: Additional parameters (can override config)

        Returns:
            ChunkingResult with chunks
        """
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        overlap_tokens = kwargs.get("overlap_tokens", self.overlap_tokens)

        # Initialize tokenizer if needed
        if self._tokenizer is None:
            self._init_tokenizer()

        chunks = []
        original_length = len(text)

        if self._tokenizer:
            # Token-aware chunking
            chunks = self._chunk_by_tokens(text, max_tokens, overlap_tokens)
        else:
            # Fallback: character-based estimation
            chars_per_token = 4
            max_chars = max_tokens * chars_per_token
            overlap_chars = overlap_tokens * chars_per_token
            chunks = self._chunk_by_chars(text, max_chars, overlap_chars)

        return ChunkingResult(
            chunks=chunks,
            original_length=original_length,
            strategy=self.strategy.value,
            config={
                "max_tokens": max_tokens,
                "overlap_tokens": overlap_tokens,
            }
        )

    def _chunk_by_tokens(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int
    ) -> List[Chunk]:
        """Chunk text using actual tokenizer"""
        tokens = self._tokenizer.encode(text)
        chunks = []

        if len(tokens) <= max_tokens:
            # Text fits in single chunk
            chunk = self._create_chunk(text, index=0)
            return [chunk]

        # Chunk with overlap
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))

            # Get token slice
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self._tokenizer.decode(chunk_tokens)

            # Calculate character offsets
            if chunks:
                # Approximate start char based on previous chunk
                start_char = chunks[-1].end_char - (overlap_tokens * 4)
            else:
                start_char = 0

            chunk = self._create_chunk(
                text=chunk_text,
                index=chunk_index,
                start_char=start_char,
                token_count_exact=len(chunk_tokens)
            )
            chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - overlap_tokens
            chunk_index += 1

            # Prevent infinite loop
            if start >= len(tokens) - overlap_tokens:
                break

        return chunks

    def _chunk_by_chars(
        self,
        text: str,
        max_chars: int,
        overlap_chars: int
    ) -> List[Chunk]:
        """Fallback: chunk by character count"""
        chunks = []

        if len(text) <= max_chars:
            chunk = self._create_chunk(text, index=0)
            return [chunk]

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + max_chars, len(text))

            # Try to break at word boundary
            if end < len(text):
                # Look for space to break at
                space_pos = text.rfind(' ', start + max_chars // 2, end)
                if space_pos > start:
                    end = space_pos + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = self._create_chunk(
                    text=chunk_text,
                    index=chunk_index,
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move with overlap
            start = end - overlap_chars

            # Prevent infinite loop
            if start >= len(text) - overlap_chars:
                break

        return chunks
