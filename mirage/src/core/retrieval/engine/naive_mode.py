"""
Naive Retrieval Mode

Vector search with keyword fallback for Arabic entities.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from ..base_retriever import RetrievalMode, RetrievalResult, RetrievalResponse
from .arabic_utils import get_arabic_variants


class NaiveModeMixin:
    """Mixin providing naive retrieval mode"""

    def _naive_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """
        Vector search with keyword fallback for Arabic entities.

        Strategy:
        1. Primary: Vector similarity search
        2. Fallback: Keyword search for Arabic entity phrases
        3. Merge: Combine and deduplicate results
        """
        if query_embedding is None:
            return RetrievalResponse(
                results=[],
                query=query,
                mode=RetrievalMode.NAIVE,
                metadata={"error": "No embedding available"}
            )

        if self.index_manager is None:
            return RetrievalResponse(
                results=[],
                query=query,
                mode=RetrievalMode.NAIVE,
                metadata={"error": "Index manager not available"}
            )

        # 1. Primary: Vector search
        search_results = self.index_manager.search_chunks(
            query_embedding,
            top_k=top_k,
            score_threshold=self.config.min_score
        )

        # Convert to RetrievalResult
        results = [
            RetrievalResult(
                chunk_id=r.id,
                document_id=r.payload.get("document_id", ""),
                text=r.payload.get("text", ""),
                score=r.score,
                retrieval_mode="naive",
                metadata=r.payload
            )
            for r in search_results
        ]

        # 2. Fallback: Keyword-based search for Arabic entities
        # Vector search may miss chunks due to embedding model limitations for proper nouns
        entity_phrases = self._extract_arabic_entity_phrases(query)
        keyword_chunks = []

        if entity_phrases and self.index_manager:
            try:
                seen_chunks = {r.chunk_id for r in results}
                for phrase in entity_phrases[:5]:
                    if len(phrase) >= 3:
                        # Get Arabic variants (handles ة↔ه, أ/إ/آ→ا)
                        variants = get_arabic_variants(phrase)
                        for variant in variants:
                            keyword_results = self.index_manager.keyword_search(
                                variant, limit=3
                            )
                            for kr in keyword_results:
                                chunk_id = kr.get("chunk_id", kr.get("id", ""))
                                if chunk_id and chunk_id not in seen_chunks:
                                    seen_chunks.add(chunk_id)
                                    keyword_chunks.append(RetrievalResult(
                                        chunk_id=chunk_id,
                                        document_id=kr.get("document_id", ""),
                                        text=kr.get("text", ""),
                                        score=0.88,  # High score for keyword match
                                        retrieval_mode="naive",
                                        metadata={"keyword_match": phrase, "matched_variant": variant}
                                    ))
            except Exception as e:
                logger.debug(f"Keyword search fallback failed: {e}")

        # 3. Merge keyword results into vector results
        if keyword_chunks:
            # Add keyword matches at the beginning (higher priority)
            results = keyword_chunks + results
            logger.debug(f"NAIVE keyword fallback: added {len(keyword_chunks)} chunks")

        # 4. Boost results that contain query terms (improves relevance for English queries)
        query_terms = [t.lower() for t in entity_phrases if len(t) >= 3]
        if query_terms:
            for result in results:
                text_lower = (result.text or "").lower()
                matching_terms = sum(1 for t in query_terms if t in text_lower)
                if matching_terms > 0:
                    # Boost score based on term overlap
                    boost = 0.05 * min(matching_terms, 3)  # Max +0.15 boost
                    result.score = min(0.98, result.score + boost)
                    result.metadata = result.metadata or {}
                    result.metadata["term_boost"] = matching_terms

        # Re-sort by score
        results.sort(key=lambda x: x.score, reverse=True)

        return RetrievalResponse(
            results=results[:top_k],
            query=query,
            mode=RetrievalMode.NAIVE,
            total_candidates=len(search_results) + len(keyword_chunks),
            metadata={"keyword_fallback": len(keyword_chunks) > 0, "term_boosted": len(query_terms) > 0}
        )
