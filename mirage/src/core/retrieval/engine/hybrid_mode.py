"""
Hybrid Retrieval Modes

Hybrid, semantic, and mix retrieval strategies.
"""

from typing import Optional
import numpy as np
from loguru import logger

from ..base_retriever import RetrievalMode, RetrievalResult, RetrievalResponse


class HybridModeMixin:
    """Mixin providing hybrid, semantic, and mix retrieval modes"""

    def _hybrid_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """Combines vector + local + global with RRF"""
        vector_response = self._vector_retrieve(query, query_embedding, top_k, **kwargs)
        local_response = self._local_retrieve(query, query_embedding, top_k, **kwargs)
        global_response = self._global_retrieve(query, query_embedding, top_k, **kwargs)

        weights = [
            self.config.mode_weights.get("vector", 0.6),
            self.config.mode_weights.get("local", 0.8),
            self.config.mode_weights.get("global", 0.9)
        ]

        fused = self.fusion.fuse_responses(
            [vector_response, local_response, global_response],
            method=self.config.fusion_method,
            weights=weights
        )

        fused.mode = RetrievalMode.HYBRID
        fused.query = query
        fused.results = fused.results[:top_k]

        return fused

    def _semantic_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """
        Deep semantic matching with cross-encoder re-ranking.

        Strategy:
        1. Retrieve top_k * 3 candidates via vector search
        2. Re-rank using cross-encoder for improved precision
        3. Return top_k highest scoring results
        """
        vector_response = self._vector_retrieve(
            query, query_embedding, top_k * 3, **kwargs
        )

        try:
            from ..reranker import get_reranker

            reranker = get_reranker()

            candidates = [
                {
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "text": r.text,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in vector_response.results
            ]

            reranked = reranker.rerank(query, candidates, top_k=top_k)

            results = [
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    text=r.text,
                    score=r.rerank_score,
                    retrieval_mode="semantic",
                    metadata={
                        "original_score": r.original_score,
                        "rank_change": r.rank_change
                    }
                )
                for r in reranked
            ]

            logger.info(
                f"Semantic reranking: {len(candidates)} candidates -> {len(results)} results"
            )

            return RetrievalResponse(
                results=results,
                query=query,
                mode=RetrievalMode.SEMANTIC,
                total_candidates=len(vector_response.results),
                metadata={
                    "reranked": True,
                    "reranker_model": reranker.model_id
                }
            )

        except Exception as e:
            logger.warning(f"Reranking failed, falling back to vector: {e}")
            vector_response.mode = RetrievalMode.SEMANTIC
            vector_response.results = vector_response.results[:top_k]
            vector_response.metadata = {"reranked": False, "fallback_reason": str(e)}
            return vector_response

    def _retrieve_mix(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """
        Robust fusion: HYBRID as primary, augmented with LOCAL for entity coverage.

        Strategy:
        1. Run HYBRID (combines local+global, most robust)
        2. Add high-quality LOCAL chunks for entity coverage
        3. Skip standalone VECTOR/GLOBAL to avoid their failures
        """
        query_embedding = None
        if self.embedder:
            query_embedding = self.embedder.embed(query)

        results = []
        seen_chunks = set()

        # 1. HYBRID is most robust
        try:
            hybrid_response = self._hybrid_retrieve(query, query_embedding, top_k, **kwargs)
            for r in hybrid_response.results:
                if r.chunk_id and r.chunk_id not in seen_chunks:
                    seen_chunks.add(r.chunk_id)
                    r.retrieval_mode = "mix"
                    r.metadata = r.metadata or {}
                    r.metadata["source_mode"] = "hybrid"
                    results.append(r)
        except Exception as e:
            logger.warning(f"HYBRID mode failed in mix: {e}")

        # 2. Add LOCAL chunks for entity coverage
        try:
            local_response = self._local_retrieve(query, query_embedding, top_k, **kwargs)
            for r in local_response.results:
                if r.chunk_id and r.chunk_id not in seen_chunks and r.score >= 0.75:
                    seen_chunks.add(r.chunk_id)
                    r.retrieval_mode = "mix"
                    r.metadata = r.metadata or {}
                    r.metadata["source_mode"] = "local"
                    results.append(r)
        except Exception as e:
            logger.warning(f"LOCAL mode failed in mix: {e}")

        # Fallback to VECTOR
        if not results:
            try:
                vector_response = self._vector_retrieve(query, query_embedding, top_k, **kwargs)
                vector_response.mode = RetrievalMode.MIX
                return vector_response
            except Exception as e:
                return RetrievalResponse(
                    results=[],
                    query=query,
                    mode=RetrievalMode.MIX,
                    metadata={"error": "All modes failed"}
                )

        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]

        return RetrievalResponse(
            results=results,
            query=query,
            mode=RetrievalMode.MIX,
            total_candidates=len(seen_chunks),
            metadata={
                "fusion_method": "hybrid_plus_local",
                "hybrid_chunks": len([r for r in results if r.metadata.get("source_mode") == "hybrid"]),
                "local_chunks": len([r for r in results if r.metadata.get("source_mode") == "local"])
            }
        )
