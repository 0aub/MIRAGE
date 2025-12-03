"""
MIRAGE V2 Chunking - Sentence-based Chunker
Groups sentences into chunks while respecting paragraph boundaries.
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


class SentenceChunker(BaseChunker):
    """
    Sentence-based text chunker.

    Groups a fixed number of sentences into each chunk.
    Optionally respects paragraph boundaries.

    Config options:
        - sentences_per_chunk: Number of sentences per chunk (default: 5)
        - overlap_sentences: Sentence overlap between chunks (default: 1)
        - respect_paragraphs: Don't split paragraphs (default: True)
        - max_chunk_tokens: Maximum tokens per chunk (default: 800)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.sentences_per_chunk = self.config.get("sentences_per_chunk", 5)
        self.overlap_sentences = self.config.get("overlap_sentences", 1)
        self.respect_paragraphs = self.config.get("respect_paragraphs", True)
        self.max_chunk_tokens = self.config.get("max_chunk_tokens", 800)

        logger.debug(
            f"SentenceChunker initialized: sentences_per_chunk={self.sentences_per_chunk}, "
            f"overlap={self.overlap_sentences}, respect_paragraphs={self.respect_paragraphs}"
        )

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.SENTENCE

    def chunk(self, text: str, **kwargs) -> ChunkingResult:
        """
        Split text into sentence-based chunks.

        Args:
            text: Input text
            **kwargs: Additional parameters

        Returns:
            ChunkingResult with chunks
        """
        sentences_per_chunk = kwargs.get("sentences_per_chunk", self.sentences_per_chunk)
        overlap = kwargs.get("overlap_sentences", self.overlap_sentences)
        respect_paragraphs = kwargs.get("respect_paragraphs", self.respect_paragraphs)
        max_tokens = kwargs.get("max_chunk_tokens", self.max_chunk_tokens)

        original_length = len(text)

        if respect_paragraphs:
            chunks = self._chunk_with_paragraphs(
                text, sentences_per_chunk, overlap, max_tokens
            )
        else:
            chunks = self._chunk_sentences(
                text, sentences_per_chunk, overlap, max_tokens
            )

        return ChunkingResult(
            chunks=chunks,
            original_length=original_length,
            strategy=self.strategy.value,
            config={
                "sentences_per_chunk": sentences_per_chunk,
                "overlap_sentences": overlap,
                "respect_paragraphs": respect_paragraphs,
            }
        )

    def _chunk_sentences(
        self,
        text: str,
        sentences_per_chunk: int,
        overlap: int,
        max_tokens: int
    ) -> List[Chunk]:
        """Chunk by sentence count"""
        sentences = TextSplitter.split_sentences(text)

        if len(sentences) <= sentences_per_chunk:
            chunk = self._create_chunk(text, index=0)
            return [chunk]

        chunks = []
        chunk_index = 0
        char_offset = 0
        i = 0

        while i < len(sentences):
            # Get next group of sentences
            end = min(i + sentences_per_chunk, len(sentences))
            group = sentences[i:end]

            # Check token limit
            chunk_text = ' '.join(group)
            tokens = self.count_tokens(chunk_text)

            # Reduce if too long
            while tokens > max_tokens and len(group) > 1:
                group = group[:-1]
                chunk_text = ' '.join(group)
                tokens = self.count_tokens(chunk_text)

            chunk = self._create_chunk(
                text=chunk_text,
                index=chunk_index,
                start_char=char_offset,
                num_sentences=len(group)
            )
            chunks.append(chunk)
            chunk_index += 1
            char_offset += len(chunk_text) + 1

            # Move forward with overlap
            i += len(group) - overlap

            # Prevent infinite loop
            if i <= chunks[-1].metadata.get("start_sentence", 0):
                i = len(group)

        return chunks

    def _chunk_with_paragraphs(
        self,
        text: str,
        sentences_per_chunk: int,
        overlap: int,
        max_tokens: int
    ) -> List[Chunk]:
        """Chunk while respecting paragraph boundaries"""
        paragraphs = TextSplitter.split_paragraphs(text)

        if len(paragraphs) <= 1:
            return self._chunk_sentences(text, sentences_per_chunk, overlap, max_tokens)

        chunks = []
        chunk_index = 0
        char_offset = 0
        current_group = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # Check if paragraph fits
            if current_tokens + para_tokens <= max_tokens:
                current_group.append(para)
                current_tokens += para_tokens

                # Check sentence count
                total_sentences = sum(
                    len(TextSplitter.split_sentences(p)) for p in current_group
                )
                if total_sentences >= sentences_per_chunk:
                    # Finalize chunk
                    chunk_text = '\n\n'.join(current_group)
                    chunk = self._create_chunk(
                        text=chunk_text,
                        index=chunk_index,
                        start_char=char_offset,
                        num_paragraphs=len(current_group),
                        num_sentences=total_sentences
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    char_offset += len(chunk_text) + 2

                    # Reset with overlap (keep last paragraph)
                    if overlap > 0 and current_group:
                        current_group = [current_group[-1]]
                        current_tokens = self.count_tokens(current_group[0])
                    else:
                        current_group = []
                        current_tokens = 0

            else:
                # Paragraph too big to fit
                if current_group:
                    # Finalize current chunk
                    chunk_text = '\n\n'.join(current_group)
                    chunk = self._create_chunk(
                        text=chunk_text,
                        index=chunk_index,
                        start_char=char_offset,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    char_offset += len(chunk_text) + 2

                # Handle large paragraph separately
                if para_tokens > max_tokens:
                    # Split large paragraph by sentences
                    para_chunks = self._chunk_sentences(
                        para, sentences_per_chunk, overlap, max_tokens
                    )
                    for pc in para_chunks:
                        pc.index = chunk_index
                        pc.start_char = char_offset
                        chunks.append(pc)
                        chunk_index += 1
                        char_offset += len(pc.text) + 1
                    current_group = []
                    current_tokens = 0
                else:
                    current_group = [para]
                    current_tokens = para_tokens

        # Last group
        if current_group:
            chunk_text = '\n\n'.join(current_group)
            chunk = self._create_chunk(
                text=chunk_text,
                index=chunk_index,
                start_char=char_offset,
            )
            chunks.append(chunk)

        return chunks
