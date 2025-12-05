"""
True REFRAG Chunk Embedder - Meta's Approach

Implements the core REFRAG concept from Meta's paper:
1. Split documents into fixed-size token chunks (16 tokens)
2. Embed each chunk into a single dense vector
3. Store embeddings for ultra-fast retrieval
4. RL policy decides which chunks to expand for full processing

This achieves the 30x speedup by:
- Processing embeddings instead of text at retrieval time
- One-time embedding cost during indexing
- Compact representation (16 tokens → 1 embedding)
- RL-guided selective expansion (only expand important chunks)

For standard LLMs (like Allam) that can't accept raw embeddings,
we use embedding similarity + RL policy to select relevant chunks,
then decode only the selected chunks' text.

The RL component (RLExpansionPolicy) uses REINFORCE algorithm with
perplexity-based rewards to learn which chunks are most important
for a given query context.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
import time
from loguru import logger

# Import centralized constants
from core.config.constants import (
    REFRAG_CHUNK_SIZE_TOKENS,
    EMBEDDING_DIM,
    EMBEDDING_BATCH_SIZE,
)

# Import RL expansion policy (required, no fallback)
from .rl_expansion_policy import RLExpansionPolicy, ExpansionDecision, get_rl_policy


@dataclass
class REFRAGChunk:
    """A REFRAG-style chunk with embedding."""
    chunk_id: str
    document_id: str
    text: str
    token_count: int
    embedding: Optional[np.ndarray] = None
    position: int = 0  # Position in original document
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "token_count": self.token_count,
            "position": self.position,
            "metadata": self.metadata,
            "has_embedding": self.embedding is not None,
        }


@dataclass
class REFRAGDocument:
    """A document represented as REFRAG chunks."""
    document_id: str
    chunks: List[REFRAGChunk]
    total_tokens: int
    compression_ratio: float  # tokens / embeddings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_count": len(self.chunks),
            "total_tokens": self.total_tokens,
            "compression_ratio": self.compression_ratio,
        }


class ChunkEmbedder:
    """
    REFRAG Chunk Embedder - Creates dense chunk representations.

    This is the core of True REFRAG:
    - Fixed-size tokenization (16 tokens by default)
    - Dense embedding per chunk
    - Embedding caching for one-time compute cost
    - RL policy for intelligent chunk expansion decisions

    The 30x speedup comes from:
    1. Processing N embeddings instead of N*16 tokens
    2. Embedding similarity is O(N) not O(N*16) for attention
    3. Cache embeddings → zero compute at retrieval time
    4. RL policy selects only important chunks for expansion

    Usage:
        embedder = ChunkEmbedder(embedding_model, use_rl_policy=True)
        refrag_doc = embedder.embed_document(document_text, document_id)

        # At retrieval time (with RL-guided selection):
        context, meta = embedder.get_context_for_llm(query_emb, [refrag_doc])
    """

    def __init__(
        self,
        embedding_model=None,
        chunk_size_tokens: int = None,
        overlap_tokens: int = 0,
        batch_size: int = None,
        use_rl_policy: bool = True,
        rl_policy_weights: Optional[str] = None,
        rl_budget_ratio: float = 0.3
    ):
        """
        Initialize REFRAG chunk embedder.

        Args:
            embedding_model: Model for generating embeddings
            chunk_size_tokens: Tokens per chunk (default: 16 from REFRAG paper)
            overlap_tokens: Overlap between chunks (default: 0)
            batch_size: Batch size for embedding (default: from constants)
            use_rl_policy: Whether to use RL expansion policy (default: True)
            rl_policy_weights: Path to pre-trained RL policy weights
            rl_budget_ratio: Ratio of chunks to expand (default: 0.3 = 30%)
        """
        self.embedding_model = embedding_model
        self.chunk_size_tokens = chunk_size_tokens or REFRAG_CHUNK_SIZE_TOKENS
        self.overlap_tokens = overlap_tokens
        self.batch_size = batch_size or EMBEDDING_BATCH_SIZE

        # Initialize RL expansion policy (always enabled - required component)
        self.use_rl_policy = use_rl_policy
        self.rl_policy: Optional[RLExpansionPolicy] = None

        if self.use_rl_policy:
            self.rl_policy = get_rl_policy(
                weights_path=rl_policy_weights,
                budget_ratio=rl_budget_ratio
            )
            logger.info("RL expansion policy enabled")

        # Embedding cache: chunk_hash -> embedding
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Statistics
        self._stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_computed": 0,
            "cache_hits": 0,
            "total_tokens_processed": 0,
            "rl_expansions": 0,
            "rl_decisions": 0,
        }

        logger.info(
            f"REFRAG ChunkEmbedder initialized: "
            f"chunk_size={self.chunk_size_tokens} tokens, "
            f"overlap={self.overlap_tokens}, "
            f"rl_policy={'enabled' if self.use_rl_policy else 'disabled'}"
        )

    def embed_document(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> REFRAGDocument:
        """
        Convert a document to REFRAG representation.

        Args:
            text: Document text
            document_id: Unique document identifier
            metadata: Optional document metadata

        Returns:
            REFRAGDocument with embedded chunks
        """
        start_time = time.time()

        # Step 1: Tokenize into fixed-size chunks
        chunks = self._create_chunks(text, document_id, metadata)

        # Step 2: Embed all chunks
        if self.embedding_model:
            self._embed_chunks(chunks)

        # Calculate statistics
        total_tokens = sum(c.token_count for c in chunks)
        compression_ratio = total_tokens / max(1, len(chunks))

        # Update stats
        self._stats["documents_processed"] += 1
        self._stats["chunks_created"] += len(chunks)
        self._stats["total_tokens_processed"] += total_tokens

        processing_time = time.time() - start_time
        logger.info(
            f"REFRAG processed document {document_id}: "
            f"{total_tokens} tokens → {len(chunks)} chunks "
            f"(compression: {compression_ratio:.1f}x, time: {processing_time:.3f}s)"
        )

        return REFRAGDocument(
            document_id=document_id,
            chunks=chunks,
            total_tokens=total_tokens,
            compression_ratio=compression_ratio
        )

    def _create_chunks(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[REFRAGChunk]:
        """
        Split text into fixed-size token chunks.

        Uses whitespace tokenization as approximation.
        For production, should use actual tokenizer.
        """
        # Simple whitespace tokenization (approximate)
        # Production should use actual tokenizer (tiktoken, HF tokenizer)
        tokens = text.split()

        chunks = []
        position = 0
        step = self.chunk_size_tokens - self.overlap_tokens

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + self.chunk_size_tokens]
            chunk_text = " ".join(chunk_tokens)

            # Generate chunk ID from content hash
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            chunk_id = f"{document_id}_{position}_{chunk_hash}"

            chunk = REFRAGChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=chunk_text,
                token_count=len(chunk_tokens),
                position=position,
                metadata=metadata or {}
            )
            chunks.append(chunk)
            position += 1

        return chunks

    def _embed_chunks(self, chunks: List[REFRAGChunk]):
        """
        Generate embeddings for chunks.
        Uses caching to avoid recomputing.
        """
        texts_to_embed = []
        chunks_to_embed = []

        for chunk in chunks:
            # Check cache
            cache_key = self._get_cache_key(chunk.text)
            if cache_key in self._embedding_cache:
                chunk.embedding = self._embedding_cache[cache_key]
                self._stats["cache_hits"] += 1
            else:
                texts_to_embed.append(chunk.text)
                chunks_to_embed.append(chunk)

        if not texts_to_embed:
            return

        # Batch embed
        try:
            embeddings = self._batch_embed(texts_to_embed)

            for chunk, embedding in zip(chunks_to_embed, embeddings):
                chunk.embedding = embedding
                # Cache the embedding
                cache_key = self._get_cache_key(chunk.text)
                self._embedding_cache[cache_key] = embedding
                self._stats["embeddings_computed"] += 1

        except Exception as e:
            logger.error(f"Embedding failed: {e}")

    def _batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed texts in batches.
        """
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            if hasattr(self.embedding_model, 'encode'):
                # sentence-transformers style
                embeddings = self.embedding_model.encode(batch)
            elif hasattr(self.embedding_model, 'embed'):
                # Custom embedding manager style
                embeddings = self.embedding_model.embed(batch)
            else:
                # Fallback: random embeddings for testing
                logger.warning("No embedding model, using random embeddings")
                embeddings = [np.random.randn(EMBEDDING_DIM) for _ in batch]

            all_embeddings.extend(embeddings)

        return all_embeddings

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def retrieve_similar(
        self,
        query_embedding: np.ndarray,
        refrag_doc: REFRAGDocument,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Tuple[REFRAGChunk, float]]:
        """
        Retrieve most similar chunks using embedding similarity.

        This is where the 30x speedup happens:
        - Compare query embedding to N chunk embeddings
        - O(N) instead of O(N * chunk_size) for attention

        Args:
            query_embedding: Query embedding vector
            refrag_doc: REFRAG document to search
            top_k: Number of chunks to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not refrag_doc.chunks:
            return []

        # Calculate similarities
        similarities = []
        for chunk in refrag_doc.chunks:
            if chunk.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, chunk.embedding)
                if similarity >= min_similarity:
                    similarities.append((chunk, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_context_for_llm(
        self,
        query_embedding: np.ndarray,
        refrag_docs: List[REFRAGDocument],
        top_k: int = 10,
        max_tokens: int = 1800,
        use_rl_selection: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get context for LLM from REFRAG documents.

        This is the adapted REFRAG for standard LLMs:
        - If RL policy available: Use RL-guided chunk expansion
        - Otherwise: Use embedding similarity to select chunks
        - Return TEXT of selected chunks (not embeddings)
        - Standard LLM can process the text

        The RL policy provides the key REFRAG speedup by intelligently
        selecting which chunks to expand based on learned patterns.

        Args:
            query_embedding: Query embedding
            refrag_docs: List of REFRAG documents
            top_k: Max chunks per document
            max_tokens: Maximum tokens in context
            use_rl_selection: Whether to use RL policy for selection

        Returns:
            Tuple of (context_text, metadata)
        """
        all_chunks = []

        # Collect similar chunks from all documents
        for doc in refrag_docs:
            similar = self.retrieve_similar(query_embedding, doc, top_k)
            all_chunks.extend(similar)

        # Sort by similarity across all documents
        all_chunks.sort(key=lambda x: x[1], reverse=True)

        # Use RL policy for intelligent chunk selection if available
        if use_rl_selection and self.use_rl_policy and self.rl_policy is not None:
            context_text, metadata = self._rl_guided_selection(
                query_embedding, all_chunks, max_tokens
            )
        else:
            # Fallback: similarity-based selection
            context_text, metadata = self._similarity_based_selection(
                all_chunks, max_tokens
            )

        return context_text, metadata

    def _rl_guided_selection(
        self,
        query_embedding: np.ndarray,
        ranked_chunks: List[Tuple[REFRAGChunk, float]],
        max_tokens: int
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Use RL policy to select chunks for expansion.

        This implements the core REFRAG "Sense" stage where an RL policy
        decides which chunks should be expanded to full token representation.

        Args:
            query_embedding: Query embedding
            ranked_chunks: List of (chunk, similarity) tuples
            max_tokens: Maximum tokens in context

        Returns:
            Tuple of (context_text, metadata)
        """
        if not ranked_chunks:
            return "", {
                "chunks_used": 0,
                "total_tokens": 0,
                "average_similarity": 0.0,
                "compression_achieved": 0.0,
                "selection_method": "rl_policy",
            }

        # Prepare chunk data for RL policy
        chunk_dicts = []
        for chunk, similarity in ranked_chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "embedding": chunk.embedding,
                "importance_score": similarity,  # Use similarity as initial importance
                "entities": chunk.metadata.get("entities", []),
            }
            chunk_dicts.append(chunk_dict)

        # Calculate expansion budget based on max_tokens
        avg_chunk_tokens = np.mean([c.token_count for c, _ in ranked_chunks])
        budget = max(1, int(max_tokens / max(1, avg_chunk_tokens)))

        # Get RL policy decisions
        expansion_decisions = self.rl_policy.decide_expansions(
            chunks=chunk_dicts,
            query_embedding=query_embedding,
            budget=budget,
            deterministic=True  # Use deterministic policy at inference
        )

        # Update stats
        self._stats["rl_decisions"] += len(expansion_decisions)
        self._stats["rl_expansions"] += sum(1 for d in expansion_decisions if d.should_expand)

        # Build context from expanded chunks
        context_parts = []
        total_tokens = 0
        chunks_used = 0
        avg_similarity = 0.0
        expansion_probs = []

        for i, decision in enumerate(expansion_decisions):
            if not decision.should_expand:
                continue

            if i >= len(ranked_chunks):
                break

            chunk, similarity = ranked_chunks[i]

            if total_tokens + chunk.token_count > max_tokens:
                break

            context_parts.append(chunk.text)
            total_tokens += chunk.token_count
            chunks_used += 1
            avg_similarity += similarity
            expansion_probs.append(decision.expansion_probability)

        context_text = "\n\n".join(context_parts)
        avg_similarity = avg_similarity / max(1, chunks_used)
        avg_expansion_prob = np.mean(expansion_probs) if expansion_probs else 0.0

        metadata = {
            "chunks_used": chunks_used,
            "total_tokens": total_tokens,
            "average_similarity": avg_similarity,
            "compression_achieved": total_tokens / max(1, sum(c.token_count for c, _ in ranked_chunks)),
            "selection_method": "rl_policy",
            "rl_expansion_budget": budget,
            "rl_avg_expansion_prob": avg_expansion_prob,
            "rl_policy_stats": self.rl_policy.get_stats() if self.rl_policy else {},
        }

        logger.debug(
            f"RL-guided selection: {chunks_used}/{len(ranked_chunks)} chunks expanded, "
            f"avg_prob={avg_expansion_prob:.3f}"
        )

        return context_text, metadata

    def _similarity_based_selection(
        self,
        ranked_chunks: List[Tuple[REFRAGChunk, float]],
        max_tokens: int
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Fallback similarity-based chunk selection.

        Used when RL policy is not available.

        Args:
            ranked_chunks: List of (chunk, similarity) tuples, pre-sorted
            max_tokens: Maximum tokens in context

        Returns:
            Tuple of (context_text, metadata)
        """
        context_parts = []
        total_tokens = 0
        chunks_used = 0
        avg_similarity = 0.0

        for chunk, similarity in ranked_chunks:
            if total_tokens + chunk.token_count > max_tokens:
                break

            context_parts.append(chunk.text)
            total_tokens += chunk.token_count
            chunks_used += 1
            avg_similarity += similarity

        context_text = "\n\n".join(context_parts)
        avg_similarity = avg_similarity / max(1, chunks_used)

        metadata = {
            "chunks_used": chunks_used,
            "total_tokens": total_tokens,
            "average_similarity": avg_similarity,
            "compression_achieved": total_tokens / max(1, sum(c.token_count for c, _ in ranked_chunks)),
            "selection_method": "similarity",
        }

        return context_text, metadata

    def train_rl_policy(
        self,
        model_logits: np.ndarray,
        target_tokens: np.ndarray,
        chunk_features: np.ndarray,
        expansion_decisions: List[bool]
    ) -> Dict[str, float]:
        """
        Train the RL policy using perplexity-based rewards.

        Call this after LLM inference to update the policy based on
        how well the selected chunks helped the model answer.

        Args:
            model_logits: Model output logits [seq_len, vocab_size]
            target_tokens: Target token IDs [seq_len]
            chunk_features: Features used for expansion decisions
            expansion_decisions: Binary expansion decisions made

        Returns:
            Dict with training metrics
        """
        if not self.use_rl_policy or self.rl_policy is None:
            return {"error": "RL policy not available"}

        # Compute perplexity-based reward
        reward = self.rl_policy.compute_perplexity_reward(model_logits, target_tokens)

        # Update policy using REINFORCE
        metrics = self.rl_policy.update(
            chunk_features=chunk_features,
            decisions=expansion_decisions,
            reward=reward,
            perplexity=np.exp(-reward * 5.0)  # Approximate perplexity
        )

        return metrics

    def save_rl_policy(self, path: str):
        """Save RL policy weights."""
        if self.rl_policy is not None:
            self.rl_policy.save_weights(path)

    def clear_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()
        logger.info("REFRAG embedding cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get embedder statistics."""
        return {
            **self._stats,
            "cache_size": len(self._embedding_cache),
            "chunk_size_tokens": self.chunk_size_tokens,
        }


# Singleton instance
_chunk_embedder: Optional[ChunkEmbedder] = None


def get_chunk_embedder(embedding_model=None) -> ChunkEmbedder:
    """
    Get or create singleton ChunkEmbedder.

    Args:
        embedding_model: Embedding model (required on first call)

    Returns:
        ChunkEmbedder instance
    """
    global _chunk_embedder

    if _chunk_embedder is None:
        _chunk_embedder = ChunkEmbedder(embedding_model)

    return _chunk_embedder
