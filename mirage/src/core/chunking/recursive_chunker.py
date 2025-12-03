"""
MIRAGE V2 Chunking - Recursive Character Chunker
Recursively splits text using a hierarchy of separators.
Inspired by LangChain's RecursiveCharacterTextSplitter.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from .base_chunker import (
    BaseChunker,
    ChunkingStrategy,
    Chunk,
    ChunkingResult,
    TextSplitter,
)


class RecursiveChunker(BaseChunker):
    """
    Recursive character-based text chunker.

    Recursively splits text using a hierarchy of separators,
    trying larger separators first (paragraphs, then sentences,
    then words).

    This preserves semantic structure as much as possible while
    respecting size limits.

    Config options:
        - chunk_size: Target chunk size in characters (default: 1000)
        - chunk_overlap: Character overlap between chunks (default: 200)
        - separators: List of separators to try (default: ["\\n\\n", "\\n", ". ", " "])
        - max_tokens: Maximum tokens per chunk (default: 600)
    """

    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraphs
        "\n",    # Lines
        "。",    # Chinese period
        ".",     # Sentences
        "؟",     # Arabic question mark
        ".",     # Period
        " ",     # Words
        "",      # Characters (last resort)
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)
        self.separators = self.config.get("separators", self.DEFAULT_SEPARATORS)
        self.max_tokens = self.config.get("max_tokens", 600)

        logger.debug(
            f"RecursiveChunker initialized: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.RECURSIVE

    def chunk(self, text: str, **kwargs) -> ChunkingResult:
        """
        Recursively split text into chunks.

        Args:
            text: Input text
            **kwargs: Additional parameters

        Returns:
            ChunkingResult with chunks
        """
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        chunk_overlap = kwargs.get("chunk_overlap", self.chunk_overlap)
        separators = kwargs.get("separators", self.separators)

        original_length = len(text)

        # Recursively split
        text_chunks = self._split_text(text, separators, chunk_size, chunk_overlap)

        # Convert to Chunk objects
        chunks = []
        char_offset = 0

        for i, chunk_text in enumerate(text_chunks):
            chunk = self._create_chunk(
                text=chunk_text,
                index=i,
                start_char=char_offset,
            )
            chunks.append(chunk)
            char_offset += len(chunk_text) - chunk_overlap

        return ChunkingResult(
            chunks=chunks,
            original_length=original_length,
            strategy=self.strategy.value,
            config={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        )

    def _split_text(
        self,
        text: str,
        separators: List[str],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Recursively split text using separators"""
        final_chunks = []

        # Find appropriate separator
        separator = separators[-1]  # Default to last (smallest)
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split by separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)  # Character split

        # Merge splits into chunks
        good_splits = []
        for split in splits:
            if split:
                good_splits.append(split)

        chunks = self._merge_splits(
            good_splits,
            separator,
            chunk_size,
            chunk_overlap
        )

        # Recursively split chunks that are too large
        for chunk in chunks:
            if len(chunk) > chunk_size:
                if new_separators:
                    # Try smaller separators
                    sub_chunks = self._split_text(
                        chunk, new_separators, chunk_size, chunk_overlap
                    )
                    final_chunks.extend(sub_chunks)
                else:
                    # Can't split further, keep as is
                    final_chunks.append(chunk)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _merge_splits(
        self,
        splits: List[str],
        separator: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Merge small splits into larger chunks"""
        chunks = []
        current_chunk = []
        current_size = 0

        for split in splits:
            split_size = len(split)

            # Check if adding this split would exceed limit
            new_size = current_size + split_size + len(separator)

            if new_size > chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = separator.join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_size = 0
                overlap_splits = []

                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= chunk_overlap:
                        overlap_splits.insert(0, s)
                        overlap_size += len(s) + len(separator)
                    else:
                        break

                current_chunk = overlap_splits
                current_size = sum(len(s) for s in current_chunk)

            current_chunk.append(split)
            current_size += split_size + len(separator)

        # Last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            chunks.append(chunk_text)

        return chunks
