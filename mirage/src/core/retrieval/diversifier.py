"""
Result Diversification - Phase 5 GraphRAG Enhancement

Uses Maximal Marginal Relevance (MMR) to balance relevance and diversity.

Problem: Top-K retrieval results are often semantically similar
Solution: Select results that are both relevant AND diverse

MMR = λ * Sim(query, doc) - (1-λ) * max(Sim(doc, selected_docs))

Where:
- λ = 1.0: Pure relevance (no diversity)
- λ = 0.0: Pure diversity (no relevance)
- λ = 0.7: Good balance (default)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class DiversifiedResult:
    """Result of diversification."""
    results: List[Dict[str, Any]]
    original_count: int
    diversity_score: float        # How diverse the final set is
    relevance_score: float        # Average relevance
    mmr_scores: List[float]       # MMR score for each result


class ResultDiversifier:
    """
    Diversify retrieval results using Maximal Marginal Relevance (MMR).

    Usage:
        diversifier = ResultDiversifier(embedder)

        # Get diverse subset
        diversified = diversifier.diversify(
            query="ما هي جائزة الحكومة الرقمية؟",
            candidates=retrieval_results,
            top_k=5
        )
    """

    def __init__(
        self,
        embedder=None,
        lambda_param: float = 0.7,
        use_cached_embeddings: bool = True
    ):
        """
        Initialize diversifier.

        Args:
            embedder: Embedding model for similarity computation
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
            use_cached_embeddings: Use pre-computed embeddings if available
        """
        self.embedder = embedder
        self.lambda_param = lambda_param
        self.use_cached = use_cached_embeddings

        logger.info(f"ResultDiversifier initialized: lambda={lambda_param}")

    def diversify(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        query_embedding: Optional[np.ndarray] = None
    ) -> DiversifiedResult:
        """
        Select diverse subset using MMR.

        Args:
            query: User query (for embedding)
            candidates: List of candidate results with 'text' key
            top_k: Number of results to return
            query_embedding: Pre-computed query embedding (optional)

        Returns:
            DiversifiedResult with selected diverse results
        """
        if not candidates:
            return DiversifiedResult(
                results=[],
                original_count=0,
                diversity_score=0.0,
                relevance_score=0.0,
                mmr_scores=[]
            )

        if len(candidates) <= top_k:
            # No need to diversify
            return DiversifiedResult(
                results=candidates,
                original_count=len(candidates),
                diversity_score=1.0,
                relevance_score=1.0,
                mmr_scores=[1.0] * len(candidates)
            )

        if self.embedder is None:
            # Fallback: return top-k by score
            logger.warning("No embedder available, returning top-k by score")
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.get('score', 0),
                reverse=True
            )
            return DiversifiedResult(
                results=sorted_candidates[:top_k],
                original_count=len(candidates),
                diversity_score=0.5,
                relevance_score=0.5,
                mmr_scores=[0.5] * top_k
            )

        # Get embeddings
        if query_embedding is None:
            query_embedding = self._get_embedding(query)

        doc_embeddings = self._get_doc_embeddings(candidates)

        # Compute query-document similarities
        query_sims = self._compute_similarities(query_embedding, doc_embeddings)

        # MMR selection
        selected_indices, mmr_scores = self._mmr_select(
            query_sims, doc_embeddings, top_k
        )

        # Build results
        selected_results = [candidates[i] for i in selected_indices]

        # Compute diversity score (average pairwise distance)
        diversity = self._compute_diversity(doc_embeddings, selected_indices)

        # Average relevance
        relevance = np.mean([query_sims[i] for i in selected_indices])

        return DiversifiedResult(
            results=selected_results,
            original_count=len(candidates),
            diversity_score=float(diversity),
            relevance_score=float(relevance),
            mmr_scores=mmr_scores
        )

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if self.embedder is None:
            return np.zeros(768)

        try:
            embedding = self.embedder.embed(text)
            return np.array(embedding)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return np.zeros(768)

    def _get_doc_embeddings(
        self,
        candidates: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Get embeddings for all candidates."""
        # Check for cached embeddings
        if self.use_cached:
            cached = []
            for c in candidates:
                if 'embedding' in c:
                    cached.append(np.array(c['embedding']))
                else:
                    cached.append(None)

            if all(e is not None for e in cached):
                return np.array(cached)

        # Compute embeddings
        if self.embedder is None:
            return np.zeros((len(candidates), 768))

        try:
            texts = [c.get('text', '') for c in candidates]
            embeddings = self.embedder.embed_batch(texts)
            return np.array(embeddings)
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")
            return np.zeros((len(candidates), 768))

    def _compute_similarities(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarities between query and documents."""
        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = doc_embeddings / (
            np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        return similarities

    def _mmr_select(
        self,
        query_similarities: np.ndarray,
        doc_embeddings: np.ndarray,
        top_k: int
    ) -> Tuple[List[int], List[float]]:
        """
        Select documents using MMR algorithm.

        MMR = λ * Sim(query, doc) - (1-λ) * max(Sim(doc, selected))

        Returns:
            (selected_indices, mmr_scores)
        """
        n = len(query_similarities)
        selected = []
        mmr_scores = []
        remaining = set(range(n))

        # Normalize embeddings for pairwise similarity
        doc_norms = doc_embeddings / (
            np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8
        )

        while len(selected) < top_k and remaining:
            best_score = -float('inf')
            best_idx = None

            for idx in remaining:
                # Relevance component
                relevance = query_similarities[idx]

                # Diversity component (max similarity to already selected)
                if selected:
                    selected_embeddings = doc_norms[selected]
                    sims_to_selected = np.dot(selected_embeddings, doc_norms[idx])
                    max_sim = np.max(sims_to_selected)
                else:
                    max_sim = 0.0

                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim

                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                mmr_scores.append(float(best_score))
                remaining.remove(best_idx)
            else:
                break

        return selected, mmr_scores

    def _compute_diversity(
        self,
        doc_embeddings: np.ndarray,
        selected_indices: List[int]
    ) -> float:
        """Compute diversity score for selected set."""
        if len(selected_indices) < 2:
            return 1.0

        selected_embeddings = doc_embeddings[selected_indices]

        # Normalize
        norms = np.linalg.norm(selected_embeddings, axis=1, keepdims=True) + 1e-8
        normalized = selected_embeddings / norms

        # Pairwise similarities
        similarities = np.dot(normalized, normalized.T)

        # Get upper triangle (excluding diagonal)
        n = len(selected_indices)
        upper_triangle = []
        for i in range(n):
            for j in range(i + 1, n):
                upper_triangle.append(similarities[i, j])

        if not upper_triangle:
            return 1.0

        # Diversity = 1 - average similarity
        avg_similarity = np.mean(upper_triangle)
        return float(1.0 - avg_similarity)

    def diversify_by_source(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        source_key: str = 'document_id'
    ) -> List[Dict[str, Any]]:
        """
        Simple diversification by source document.

        Ensures results come from different documents.

        Args:
            candidates: Candidate results
            top_k: Max results
            source_key: Key for document identifier

        Returns:
            Diversified results (one per source, then fill)
        """
        if not candidates:
            return []

        # Group by source
        by_source: Dict[str, List[Dict]] = {}
        for c in candidates:
            source = c.get(source_key, 'unknown')
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(c)

        # Round-robin selection
        selected = []
        sources = list(by_source.keys())
        indices = {s: 0 for s in sources}

        while len(selected) < top_k:
            added = False
            for source in sources:
                if len(selected) >= top_k:
                    break
                if indices[source] < len(by_source[source]):
                    selected.append(by_source[source][indices[source]])
                    indices[source] += 1
                    added = True

            if not added:
                break

        return selected


# Singleton
_diversifier: Optional[ResultDiversifier] = None


def get_diversifier(**kwargs) -> ResultDiversifier:
    """Get or create diversifier singleton."""
    global _diversifier
    if _diversifier is None:
        _diversifier = ResultDiversifier(**kwargs)
    return _diversifier
