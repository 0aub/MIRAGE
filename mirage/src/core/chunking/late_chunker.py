"""
MIRAGE V2 Chunking - Late Chunker
Implements Jina AI's late chunking approach.
Embeds the full document first, then chunks while preserving context.
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
from loguru import logger

from .base_chunker import (
    BaseChunker,
    ChunkingStrategy,
    Chunk,
    ChunkingResult,
    TextSplitter,
)


class LateChunker(BaseChunker):
    """
    Late chunking implementation (Jina AI approach).

    Instead of chunking first and then embedding, this approach:
    1. Embeds the full document to capture full context
    2. Then creates chunks at natural boundaries
    3. Each chunk's embedding is derived from the full-document embedding

    This preserves document-level context in each chunk's representation.

    See: https://jina.ai/news/late-chunking-in-long-context-embedding-models

    Config options:
        - max_document_tokens: Maximum tokens for full document (default: 8192)
        - boundary_tokens: Tokens per chunk boundary (default: 256)
        - min_chunk_tokens: Minimum chunk size (default: 100)
        - max_chunk_tokens: Maximum chunk size (default: 512)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None
    ):
        super().__init__(config)

        self.max_document_tokens = self.config.get("max_document_tokens", 8192)
        self.boundary_tokens = self.config.get("boundary_tokens", 256)
        self.min_chunk_tokens = self.config.get("min_chunk_tokens", 100)
        self.max_chunk_tokens = self.config.get("max_chunk_tokens", 512)

        self._embedding_fn = embedding_fn
        self._embedder = None

        logger.debug(
            f"LateChunker initialized: max_doc_tokens={self.max_document_tokens}, "
            f"boundary_tokens={self.boundary_tokens}"
        )

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.LATE

    def _get_embedder(self):
        """Get or create embedder for document embedding"""
        if self._embedder is None:
            try:
                from ..models.embedding_manager import EmbeddingManager
                self._embedder = EmbeddingManager()
            except ImportError:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(
                    'paraphrase-multilingual-mpnet-base-v2'
                )
        return self._embedder

    def chunk(self, text: str, **kwargs) -> ChunkingResult:
        """
        Apply late chunking to text.

        Args:
            text: Input text
            **kwargs: Additional parameters

        Returns:
            ChunkingResult with chunks and embeddings
        """
        max_doc_tokens = kwargs.get("max_document_tokens", self.max_document_tokens)
        boundary_tokens = kwargs.get("boundary_tokens", self.boundary_tokens)
        min_tokens = kwargs.get("min_chunk_tokens", self.min_chunk_tokens)
        max_tokens = kwargs.get("max_chunk_tokens", self.max_chunk_tokens)

        original_length = len(text)

        # Check if document fits in context
        doc_tokens = self.count_tokens(text)

        if doc_tokens > max_doc_tokens:
            # Document too long, fall back to regular chunking with overlap
            logger.warning(
                f"Document too long for late chunking ({doc_tokens} > {max_doc_tokens}). "
                "Falling back to segment-based late chunking."
            )
            return self._chunk_large_document(
                text, max_doc_tokens, boundary_tokens, min_tokens, max_tokens
            )

        # Find natural boundaries (sentences, paragraphs)
        boundaries = self._find_boundaries(text)

        # Create chunks at boundaries
        chunks = self._create_chunks_at_boundaries(
            text, boundaries, min_tokens, max_tokens
        )

        # Generate document embedding if needed
        doc_embedding = None
        if kwargs.get("compute_embeddings", False):
            try:
                embedder = self._get_embedder()
                if hasattr(embedder, 'embed'):
                    doc_embedding = embedder.embed(text)
                else:
                    doc_embedding = embedder.encode(text, normalize_embeddings=True)
            except Exception as e:
                logger.warning(f"Failed to compute document embedding: {e}")

        return ChunkingResult(
            chunks=chunks,
            original_length=original_length,
            strategy=self.strategy.value,
            config={
                "max_document_tokens": max_doc_tokens,
                "boundary_tokens": boundary_tokens,
            },
            metadata={
                "document_tokens": doc_tokens,
                "num_boundaries": len(boundaries),
                "document_embedding": doc_embedding.tolist() if doc_embedding is not None else None,
                "late_chunking": True,
            }
        )

    def _find_boundaries(self, text: str) -> List[int]:
        """Find natural chunk boundaries in text"""
        boundaries = [0]  # Start of document

        # Find paragraph boundaries
        for i, char in enumerate(text):
            if i > 0 and text[i-1:i+1] == '\n\n':
                boundaries.append(i + 1)

        # Find sentence boundaries
        sentence_ends = ['. ', '! ', '? ', '。', '؟', '!\n', '?\n', '.\n']
        for end in sentence_ends:
            idx = 0
            while True:
                pos = text.find(end, idx)
                if pos == -1:
                    break
                boundaries.append(pos + len(end))
                idx = pos + 1

        # Sort and deduplicate
        boundaries = sorted(set(boundaries))

        # Add end of document
        if boundaries[-1] != len(text):
            boundaries.append(len(text))

        return boundaries

    def _create_chunks_at_boundaries(
        self,
        text: str,
        boundaries: List[int],
        min_tokens: int,
        max_tokens: int
    ) -> List[Chunk]:
        """Create chunks at natural boundaries"""
        chunks = []
        chunk_index = 0

        i = 0
        while i < len(boundaries) - 1:
            start = boundaries[i]
            end = start

            # Accumulate boundaries until we reach min_tokens
            while i < len(boundaries) - 1:
                next_end = boundaries[i + 1]
                segment = text[start:next_end]
                tokens = self.count_tokens(segment)

                if tokens > max_tokens:
                    # Stop here if we'd exceed max
                    if end > start:
                        break
                    else:
                        # Single segment too large, take it anyway
                        end = next_end
                        i += 1
                        break

                end = next_end
                i += 1

                if tokens >= min_tokens:
                    break

            # Create chunk
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = self._create_chunk(
                    text=chunk_text,
                    index=chunk_index,
                    start_char=start,
                    end_char=end,
                    late_chunking=True
                )
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _chunk_large_document(
        self,
        text: str,
        max_doc_tokens: int,
        boundary_tokens: int,
        min_tokens: int,
        max_tokens: int
    ) -> ChunkingResult:
        """Handle documents larger than context window"""
        # Split into segments that fit in context
        segments = self._split_into_segments(text, max_doc_tokens, boundary_tokens)

        all_chunks = []
        chunk_index = 0
        char_offset = 0

        for segment in segments:
            # Find boundaries in segment
            boundaries = self._find_boundaries(segment)

            # Adjust boundaries relative to full document
            boundaries = [b + char_offset for b in boundaries]

            # Create chunks
            segment_chunks = self._create_chunks_at_boundaries(
                text[char_offset:char_offset + len(segment)],
                [b - char_offset for b in boundaries],
                min_tokens,
                max_tokens
            )

            # Update indices and offsets
            for chunk in segment_chunks:
                chunk.index = chunk_index
                chunk.start_char += char_offset
                chunk.end_char += char_offset
                all_chunks.append(chunk)
                chunk_index += 1

            char_offset += len(segment)

        return ChunkingResult(
            chunks=all_chunks,
            original_length=len(text),
            strategy=self.strategy.value,
            config={"large_document": True},
            metadata={
                "num_segments": len(segments),
                "late_chunking": True,
            }
        )

    def _split_into_segments(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int
    ) -> List[str]:
        """Split text into overlapping segments"""
        segments = []

        # Estimate chars per token
        total_tokens = self.count_tokens(text)
        chars_per_token = len(text) / max(total_tokens, 1)

        max_chars = int(max_tokens * chars_per_token)
        overlap_chars = int(overlap_tokens * chars_per_token)

        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))

            # Try to break at paragraph or sentence
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind('\n\n', start + max_chars // 2, end)
                if para_break > start:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for sent_end in ['. ', '。', '؟', '!\n', '?\n']:
                        pos = text.rfind(sent_end, start + max_chars // 2, end)
                        if pos > start:
                            end = pos + len(sent_end)
                            break

            segments.append(text[start:end])
            start = end - overlap_chars

        return segments
