"""
MIRAGE V2 Chunking - Semantic Chunker
Groups sentences by semantic similarity using embeddings.
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


class SemanticChunker(BaseChunker):
    """
    Semantic similarity-based text chunker.

    Groups consecutive sentences together based on their
    semantic similarity. Creates chunks that are semantically
    coherent rather than just token-based.

    Config options:
        - similarity_threshold: Min similarity to group sentences (default: 0.7)
        - min_chunk_tokens: Minimum tokens per chunk (default: 100)
        - max_chunk_tokens: Maximum tokens per chunk (default: 800)
        - embedding_fn: Optional embedding function (uses default if not provided)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        embedding_fn: Optional[Callable[[List[str]], np.ndarray]] = None
    ):
        super().__init__(config)

        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.min_chunk_tokens = self.config.get("min_chunk_tokens", 100)
        self.max_chunk_tokens = self.config.get("max_chunk_tokens", 800)

        self._embedding_fn = embedding_fn
        self._embedder = None

        logger.debug(
            f"SemanticChunker initialized: threshold={self.similarity_threshold}, "
            f"min_tokens={self.min_chunk_tokens}, max_tokens={self.max_chunk_tokens}"
        )

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.SEMANTIC

    def _get_embedding_fn(self) -> Callable[[List[str]], np.ndarray]:
        """Get or create embedding function"""
        if self._embedding_fn:
            return self._embedding_fn

        if self._embedder is None:
            try:
                from ..models.embedding_manager import EmbeddingManager
                self._embedder = EmbeddingManager()
            except ImportError:
                # Fallback to sentence-transformers directly
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(
                    'paraphrase-multilingual-mpnet-base-v2'
                )

        if hasattr(self._embedder, 'embed'):
            return lambda texts: self._embedder.embed(texts)
        else:
            return lambda texts: self._embedder.encode(texts, normalize_embeddings=True)

    def chunk(self, text: str, **kwargs) -> ChunkingResult:
        """
        Split text into semantically coherent chunks.

        Args:
            text: Input text
            **kwargs: Additional parameters

        Returns:
            ChunkingResult with semantic chunks
        """
        similarity_threshold = kwargs.get("similarity_threshold", self.similarity_threshold)
        min_tokens = kwargs.get("min_chunk_tokens", self.min_chunk_tokens)
        max_tokens = kwargs.get("max_chunk_tokens", self.max_chunk_tokens)

        original_length = len(text)

        # Split into sentences
        sentences = TextSplitter.split_sentences(text)

        if len(sentences) <= 1:
            # Single sentence or no sentences
            chunk = self._create_chunk(text, index=0)
            return ChunkingResult(
                chunks=[chunk],
                original_length=original_length,
                strategy=self.strategy.value,
                config={"similarity_threshold": similarity_threshold}
            )

        # Get embeddings for all sentences
        try:
            embed_fn = self._get_embedding_fn()
            embeddings = embed_fn(sentences)
        except Exception as e:
            logger.warning(f"Embedding failed, falling back to sentence chunking: {e}")
            return self._fallback_chunk(text, sentences)

        # Group sentences by similarity
        chunks = self._group_by_similarity(
            sentences,
            embeddings,
            similarity_threshold,
            min_tokens,
            max_tokens
        )

        return ChunkingResult(
            chunks=chunks,
            original_length=original_length,
            strategy=self.strategy.value,
            config={
                "similarity_threshold": similarity_threshold,
                "min_chunk_tokens": min_tokens,
                "max_chunk_tokens": max_tokens,
            },
            metadata={
                "num_sentences": len(sentences),
                "avg_similarity": self._compute_avg_similarity(embeddings)
            }
        )

    def _group_by_similarity(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        threshold: float,
        min_tokens: int,
        max_tokens: int
    ) -> List[Chunk]:
        """Group sentences by semantic similarity"""
        chunks = []
        current_group = [sentences[0]]
        current_embedding = embeddings[0]
        chunk_index = 0
        char_offset = 0

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            embedding = embeddings[i]

            # Compute similarity with current group
            similarity = np.dot(current_embedding, embedding)

            # Check if should add to current group
            current_text = ' '.join(current_group + [sentence])
            current_tokens = self.count_tokens(current_text)

            if similarity >= threshold and current_tokens <= max_tokens:
                # Add to current group
                current_group.append(sentence)
                # Update group embedding (running average)
                current_embedding = (current_embedding + embedding) / 2
                current_embedding = current_embedding / np.linalg.norm(current_embedding)

            else:
                # Start new group
                group_text = ' '.join(current_group)
                group_tokens = self.count_tokens(group_text)

                # Check minimum size
                if group_tokens >= min_tokens or len(chunks) == 0:
                    chunk = self._create_chunk(
                        text=group_text,
                        index=chunk_index,
                        start_char=char_offset,
                        avg_similarity=float(similarity)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    char_offset += len(group_text) + 1
                else:
                    # Merge with previous chunk if too small
                    if chunks:
                        chunks[-1] = self._create_chunk(
                            text=chunks[-1].text + ' ' + group_text,
                            index=chunks[-1].index,
                            start_char=chunks[-1].start_char,
                        )

                # Reset for new group
                current_group = [sentence]
                current_embedding = embedding

        # Don't forget last group
        if current_group:
            group_text = ' '.join(current_group)
            chunk = self._create_chunk(
                text=group_text,
                index=chunk_index,
                start_char=char_offset,
            )
            chunks.append(chunk)

        return chunks

    def _compute_avg_similarity(self, embeddings: np.ndarray) -> float:
        """Compute average pairwise similarity"""
        if len(embeddings) <= 1:
            return 1.0

        # Compute similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        return float(np.mean(similarities))

    def _fallback_chunk(
        self,
        text: str,
        sentences: List[str]
    ) -> ChunkingResult:
        """Fallback to simple sentence grouping"""
        chunks = []
        current_group = []
        chunk_index = 0
        char_offset = 0

        for sentence in sentences:
            current_group.append(sentence)
            current_text = ' '.join(current_group)

            if self.count_tokens(current_text) >= self.max_chunk_tokens:
                # Finalize chunk (without last sentence)
                if len(current_group) > 1:
                    chunk_text = ' '.join(current_group[:-1])
                    chunk = self._create_chunk(
                        text=chunk_text,
                        index=chunk_index,
                        start_char=char_offset,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    char_offset += len(chunk_text) + 1
                    current_group = [sentence]
                else:
                    # Single sentence too long
                    chunk = self._create_chunk(
                        text=sentence,
                        index=chunk_index,
                        start_char=char_offset,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    char_offset += len(sentence) + 1
                    current_group = []

        # Last group
        if current_group:
            chunk_text = ' '.join(current_group)
            chunk = self._create_chunk(
                text=chunk_text,
                index=chunk_index,
                start_char=char_offset,
            )
            chunks.append(chunk)

        return ChunkingResult(
            chunks=chunks,
            original_length=len(text),
            strategy=self.strategy.value,
            metadata={"fallback": True}
        )
