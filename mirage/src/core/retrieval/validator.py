"""
Retrieval Validation and Correction - Phase 4 GraphRAG Enhancement

Inspired by CRAG (Corrective RAG) paper:
1. Assess if retrieved documents are relevant to the query
2. If not relevant, apply corrective strategies:
   - Query expansion (synonyms, related terms)
   - Alternative retrieval mode
   - Increased search depth
3. Return validated results with confidence scores

This prevents garbage answers from garbage retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from loguru import logger

from .base_retriever import RetrievalResult, RetrievalResponse, RetrievalMode


class ValidationStatus(Enum):
    """Status of validation result"""
    VALIDATED = "validated"          # Results passed validation
    CORRECTED = "corrected"          # Results fixed by correction
    PARTIALLY_VALIDATED = "partial"  # Some results valid
    FAILED = "failed"                # Could not validate/correct


@dataclass
class ValidatedResult:
    """Result of validation and correction process."""
    results: List[RetrievalResult]
    status: ValidationStatus
    confidence: float                    # Overall confidence 0-1
    relevance_scores: List[float] = field(default_factory=list)
    corrections_applied: int = 0
    correction_method: Optional[str] = None
    original_count: int = 0
    filtered_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "confidence": round(self.confidence, 3),
            "corrections_applied": self.corrections_applied,
            "correction_method": self.correction_method,
            "original_count": self.original_count,
            "filtered_count": self.filtered_count,
            "avg_relevance": round(np.mean(self.relevance_scores), 3) if self.relevance_scores else 0,
        }


class RetrievalValidator:
    """
    CRAG-style retrieval validator and corrector.

    Usage:
        validator = RetrievalValidator(reranker, retrieval_engine)

        # Validate and correct retrieval results
        validated = validator.validate_and_correct(
            query="ما هي جائزة الحكومة الرقمية؟",
            response=retrieval_response,
            max_corrections=2
        )

        if validated.status == ValidationStatus.VALIDATED:
            # Use validated.results
            pass
        elif validated.status == ValidationStatus.FAILED:
            # Fallback strategy
            pass
    """

    def __init__(
        self,
        reranker=None,
        retrieval_engine=None,
        llm_client=None,
        relevance_threshold: float = 0.5,
        min_relevant_ratio: float = 0.4,
        use_llm_grading: bool = False
    ):
        """
        Initialize validator.

        Args:
            reranker: Cross-encoder reranker for scoring relevance
            retrieval_engine: Engine for retry with different modes
            llm_client: LLM for query expansion (optional)
            relevance_threshold: Min score for a chunk to be relevant
            min_relevant_ratio: Min ratio of relevant chunks to pass
            use_llm_grading: Use LLM for relevance grading (slower but more accurate)
        """
        self.reranker = reranker
        self.retrieval_engine = retrieval_engine
        self.llm_client = llm_client
        self.threshold = relevance_threshold
        self.min_ratio = min_relevant_ratio
        self.use_llm_grading = use_llm_grading

        logger.info(
            f"RetrievalValidator initialized: threshold={self.threshold}, "
            f"min_ratio={self.min_ratio}"
        )

    def validate_and_correct(
        self,
        query: str,
        response: RetrievalResponse,
        max_corrections: int = 2
    ) -> ValidatedResult:
        """
        Validate retrieval results and apply corrections if needed.

        Steps:
        1. Score relevance of each retrieved chunk
        2. If too many irrelevant, apply correction strategies
        3. Return validated results with confidence

        Args:
            query: User query
            response: Retrieval response to validate
            max_corrections: Max correction attempts

        Returns:
            ValidatedResult with filtered/corrected results
        """
        if not response.results:
            return ValidatedResult(
                results=[],
                status=ValidationStatus.FAILED,
                confidence=0.0,
                original_count=0,
                filtered_count=0
            )

        # Step 1: Score relevance
        relevance_scores = self._score_relevance(query, response.results)

        # Step 2: Check if results are acceptable
        relevant_count = sum(1 for s in relevance_scores if s >= self.threshold)
        total = len(response.results)
        relevance_ratio = relevant_count / total if total > 0 else 0

        logger.info(
            f"Validation: {relevant_count}/{total} relevant "
            f"(ratio={relevance_ratio:.2f}, threshold={self.min_ratio})"
        )

        if relevance_ratio >= self.min_ratio:
            # Results are acceptable - filter and return
            filtered = self._filter_by_relevance(response.results, relevance_scores)
            return ValidatedResult(
                results=filtered,
                status=ValidationStatus.VALIDATED,
                confidence=relevance_ratio,
                relevance_scores=relevance_scores,
                original_count=total,
                filtered_count=len(filtered)
            )

        # Step 3: Apply corrections
        if max_corrections > 0:
            corrected = self._apply_corrections(
                query, response, relevance_scores, max_corrections
            )
            if corrected.status != ValidationStatus.FAILED:
                return corrected

        # Return partial results if available
        filtered = self._filter_by_relevance(response.results, relevance_scores)
        if filtered:
            return ValidatedResult(
                results=filtered,
                status=ValidationStatus.PARTIALLY_VALIDATED,
                confidence=relevance_ratio,
                relevance_scores=relevance_scores,
                original_count=total,
                filtered_count=len(filtered)
            )

        # Complete failure - return top results anyway
        return ValidatedResult(
            results=response.results[:3],
            status=ValidationStatus.FAILED,
            confidence=0.2,
            relevance_scores=relevance_scores[:3],
            original_count=total,
            filtered_count=min(3, total),
            metadata={"fallback": True}
        )

    def _score_relevance(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[float]:
        """
        Score relevance of each chunk to query.

        Uses cross-encoder reranker if available, otherwise falls back
        to embedding similarity or simple heuristics.
        """
        if not results:
            return []

        # Try cross-encoder reranking
        if self.reranker:
            try:
                texts = [r.text for r in results]
                scores = self.reranker.score_pairs(query, texts)
                return scores
            except Exception as e:
                logger.warning(f"Reranker scoring failed: {e}")

        # Fallback: use original retrieval scores normalized
        scores = []
        for r in results:
            # Normalize original score to 0-1 range
            score = min(1.0, max(0.0, r.score))
            scores.append(score)

        return scores

    def _filter_by_relevance(
        self,
        results: List[RetrievalResult],
        scores: List[float]
    ) -> List[RetrievalResult]:
        """Filter results keeping only those above threshold."""
        filtered = []
        for result, score in zip(results, scores):
            if score >= self.threshold:
                # Update score with relevance score
                result.score = score
                filtered.append(result)

        # Sort by score descending
        filtered.sort(key=lambda x: x.score, reverse=True)
        return filtered

    def _apply_corrections(
        self,
        query: str,
        response: RetrievalResponse,
        relevance_scores: List[float],
        max_corrections: int
    ) -> ValidatedResult:
        """
        Apply corrective strategies to improve retrieval quality.

        Strategies (in order):
        1. Query expansion - add synonyms and related terms
        2. Alternative mode - try different retrieval mode
        3. Increased depth - fetch more candidates
        """
        corrections = [
            ("query_expansion", self._try_query_expansion),
            ("alternative_mode", self._try_alternative_mode),
            ("increased_depth", self._try_increased_depth),
        ]

        for i, (name, correction_fn) in enumerate(corrections[:max_corrections]):
            try:
                logger.info(f"Trying correction {i+1}/{max_corrections}: {name}")
                new_response = correction_fn(query, response)

                if new_response and new_response.results:
                    new_scores = self._score_relevance(query, new_response.results)
                    relevant_count = sum(1 for s in new_scores if s >= self.threshold)
                    relevance_ratio = relevant_count / len(new_response.results)

                    if relevance_ratio >= self.min_ratio:
                        filtered = self._filter_by_relevance(new_response.results, new_scores)
                        logger.info(
                            f"Correction {name} succeeded: {len(filtered)} relevant results"
                        )
                        return ValidatedResult(
                            results=filtered,
                            status=ValidationStatus.CORRECTED,
                            confidence=relevance_ratio,
                            relevance_scores=new_scores,
                            corrections_applied=i + 1,
                            correction_method=name,
                            original_count=len(response.results),
                            filtered_count=len(filtered)
                        )

            except Exception as e:
                logger.warning(f"Correction {name} failed: {e}")

        # All corrections failed
        return ValidatedResult(
            results=response.results[:3],
            status=ValidationStatus.FAILED,
            confidence=0.3,
            corrections_applied=max_corrections,
            metadata={"all_corrections_failed": True}
        )

    def _try_query_expansion(
        self,
        query: str,
        original: RetrievalResponse
    ) -> Optional[RetrievalResponse]:
        """
        Expand query with synonyms and related terms.

        Uses LLM to generate expanded queries if available,
        otherwise uses simple Arabic/English synonym expansion.
        """
        expanded_queries = []

        # Try LLM expansion
        if self.llm_client:
            try:
                expansion_prompt = f"""وسّع هذا الاستعلام بإضافة مصطلحات مرادفة وذات صلة.
اكتب 2 استعلامات موسعة فقط:

الاستعلام: {query}

1."""
                response = self.llm_client.generate(expansion_prompt, max_tokens=100)
                lines = response.strip().split('\n')
                for line in lines[:2]:
                    # Clean up the line
                    line = line.strip()
                    if line and not line.startswith(('1.', '2.')):
                        expanded_queries.append(line)
                    elif line.startswith(('1.', '2.')):
                        expanded_queries.append(line[2:].strip())
            except Exception as e:
                logger.warning(f"LLM query expansion failed: {e}")

        # Fallback: simple keyword expansion
        if not expanded_queries:
            expanded_queries = self._simple_expansion(query)

        if not expanded_queries or not self.retrieval_engine:
            return None

        # Search with expanded queries
        all_results = []
        seen_chunks = set()

        for eq in expanded_queries[:2]:
            try:
                response = self.retrieval_engine.retrieve(
                    eq,
                    mode=original.mode,
                    top_k=5,
                    auto_route=False
                )
                for r in response.results:
                    if r.chunk_id not in seen_chunks:
                        seen_chunks.add(r.chunk_id)
                        all_results.append(r)
            except Exception as e:
                logger.warning(f"Expanded query search failed: {e}")

        if not all_results:
            return None

        # Merge with original results (deduped)
        for r in original.results:
            if r.chunk_id not in seen_chunks:
                seen_chunks.add(r.chunk_id)
                all_results.append(r)

        return RetrievalResponse(
            results=all_results[:10],
            query=query,
            mode=original.mode,
            metadata={"expanded_queries": expanded_queries}
        )

    def _simple_expansion(self, query: str) -> List[str]:
        """Simple keyword-based query expansion."""
        # Common Arabic synonyms
        expansions = {
            "جائزة": ["مكافأة", "تكريم"],
            "الحكومة": ["الجهات الحكومية", "القطاع العام"],
            "الرقمية": ["التقنية", "الإلكترونية"],
            "برنامج": ["مبادرة", "مشروع"],
            "خدمات": ["خدمة", "منتجات"],
            "شركة": ["مؤسسة", "جهة"],
        }

        expanded = []
        for word, synonyms in expansions.items():
            if word in query:
                for syn in synonyms[:1]:
                    expanded.append(query.replace(word, syn))

        return expanded[:2]

    def _try_alternative_mode(
        self,
        query: str,
        original: RetrievalResponse
    ) -> Optional[RetrievalResponse]:
        """Try retrieval with an alternative mode."""
        if not self.retrieval_engine:
            return None

        # Define mode alternatives
        mode_alternatives = {
            RetrievalMode.NAIVE: [RetrievalMode.LOCAL, RetrievalMode.SEMANTIC],
            RetrievalMode.LOCAL: [RetrievalMode.HYBRID, RetrievalMode.GLOBAL],
            RetrievalMode.GLOBAL: [RetrievalMode.HYBRID, RetrievalMode.LOCAL],
            RetrievalMode.HYBRID: [RetrievalMode.SEMANTIC, RetrievalMode.MIX],
            RetrievalMode.SEMANTIC: [RetrievalMode.HYBRID, RetrievalMode.LOCAL],
        }

        alternatives = mode_alternatives.get(original.mode, [RetrievalMode.HYBRID])

        for alt_mode in alternatives[:1]:  # Try only first alternative
            try:
                logger.info(f"Trying alternative mode: {alt_mode.value}")
                response = self.retrieval_engine.retrieve(
                    query,
                    mode=alt_mode,
                    top_k=10,
                    auto_route=False
                )
                if response.results:
                    return response
            except Exception as e:
                logger.warning(f"Alternative mode {alt_mode.value} failed: {e}")

        return None

    def _try_increased_depth(
        self,
        query: str,
        original: RetrievalResponse
    ) -> Optional[RetrievalResponse]:
        """Try retrieval with more candidates."""
        if not self.retrieval_engine:
            return None

        try:
            # Double the top_k
            original_k = len(original.results)
            new_k = min(original_k * 2, 30)

            response = self.retrieval_engine.retrieve(
                query,
                mode=original.mode,
                top_k=new_k,
                auto_route=False
            )
            return response

        except Exception as e:
            logger.warning(f"Increased depth retrieval failed: {e}")
            return None

    def grade_relevance(
        self,
        query: str,
        chunk_text: str
    ) -> Tuple[float, str]:
        """
        Grade relevance of a single chunk (for detailed analysis).

        Returns:
            (score, explanation) tuple
        """
        if self.use_llm_grading and self.llm_client:
            return self._llm_grade(query, chunk_text)

        # Use reranker score
        if self.reranker:
            try:
                scores = self.reranker.score_pairs(query, [chunk_text])
                score = scores[0]
                if score >= 0.7:
                    explanation = "High relevance - directly answers query"
                elif score >= 0.5:
                    explanation = "Medium relevance - related to query"
                else:
                    explanation = "Low relevance - marginally related"
                return score, explanation
            except Exception:
                pass

        # Fallback: simple overlap
        query_terms = set(query.lower().split())
        chunk_terms = set(chunk_text.lower().split())
        overlap = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0
        return overlap, "Overlap-based score"

    def _llm_grade(
        self,
        query: str,
        chunk_text: str
    ) -> Tuple[float, str]:
        """Use LLM to grade relevance (more accurate but slower)."""
        if not self.llm_client:
            return 0.5, "LLM not available"

        prompt = f"""قيّم مدى صلة هذا النص بالسؤال على مقياس 0-10:

السؤال: {query}

النص: {chunk_text[:500]}

أجب برقم واحد فقط (0-10):"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=10)
            # Extract number from response
            for char in response:
                if char.isdigit():
                    score = int(char) / 10.0
                    return score, f"LLM grade: {int(char)}/10"
            return 0.5, "Could not parse LLM response"
        except Exception as e:
            return 0.5, f"LLM grading failed: {e}"


# Singleton
_validator: Optional[RetrievalValidator] = None


def get_validator(**kwargs) -> RetrievalValidator:
    """Get or create validator singleton."""
    global _validator
    if _validator is None:
        _validator = RetrievalValidator(**kwargs)
    return _validator
