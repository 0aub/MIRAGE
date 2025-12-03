"""
MIRAGE V2 Result Fusion
Implements fusion algorithms for combining results from multiple retrievers.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger

from .base_retriever import RetrievalResult, RetrievalResponse


@dataclass
class FusionConfig:
    """Configuration for result fusion"""
    # RRF parameters
    rrf_k: int = 60  # RRF constant (higher = more weight to lower ranks)

    # Score normalization
    normalize_scores: bool = True

    # Deduplication
    deduplicate: bool = True

    # Weighting
    mode_weights: Optional[Dict[str, float]] = None  # Mode-specific weights


class ResultFusion:
    """
    Fuses results from multiple retrieval modes.

    Supports:
    - Reciprocal Rank Fusion (RRF)
    - Weighted combination
    - Score-based fusion
    - Interleaving
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()

    def reciprocal_rank_fusion(
        self,
        result_lists: List[List[RetrievalResult]],
        weights: Optional[List[float]] = None
    ) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF score = sum(weight_i / (k + rank_i)) for all lists where item appears

        Args:
            result_lists: List of ranked result lists from different retrievers
            weights: Optional weights for each list (default: equal weights)

        Returns:
            Fused and re-ranked results
        """
        k = self.config.rrf_k

        if weights is None:
            weights = [1.0] * len(result_lists)

        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        # Calculate RRF scores
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_data: Dict[str, RetrievalResult] = {}

        for list_idx, results in enumerate(result_lists):
            weight = weights[list_idx]

            for rank, result in enumerate(results, 1):
                chunk_id = result.chunk_id

                # RRF contribution
                rrf_scores[chunk_id] += weight / (k + rank)

                # Keep the result with highest original score
                if chunk_id not in chunk_data or result.score > chunk_data[chunk_id].score:
                    chunk_data[chunk_id] = result

        # Create fused results
        fused = []
        for chunk_id, rrf_score in rrf_scores.items():
            result = chunk_data[chunk_id]
            # Update score to RRF score
            fused.append(RetrievalResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=result.text,
                score=rrf_score,
                retrieval_mode="rrf_fusion",
                metadata={
                    **result.metadata,
                    "original_score": result.score,
                    "original_mode": result.retrieval_mode
                },
                via_entity=result.via_entity,
                via_relationship=result.via_relationship,
                hop_distance=result.hop_distance
            ))

        # Sort by RRF score
        fused.sort(key=lambda x: x.score, reverse=True)

        return fused

    def weighted_fusion(
        self,
        result_lists: List[List[RetrievalResult]],
        weights: List[float]
    ) -> List[RetrievalResult]:
        """
        Combine results using weighted score aggregation.

        Combined score = sum(weight_i * score_i) for all lists where item appears

        Args:
            result_lists: List of result lists
            weights: Weights for each list

        Returns:
            Fused results sorted by combined score
        """
        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        # Aggregate scores
        combined_scores: Dict[str, float] = defaultdict(float)
        chunk_data: Dict[str, RetrievalResult] = {}
        score_counts: Dict[str, int] = defaultdict(int)

        for list_idx, results in enumerate(result_lists):
            weight = weights[list_idx]

            for result in results:
                chunk_id = result.chunk_id

                # Weighted score
                combined_scores[chunk_id] += weight * result.score
                score_counts[chunk_id] += 1

                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = result

        # Normalize by number of times seen (optional boost for appearing in multiple lists)
        if self.config.normalize_scores:
            for chunk_id in combined_scores:
                # Boost items that appear in multiple lists
                boost = 1.0 + 0.1 * (score_counts[chunk_id] - 1)
                combined_scores[chunk_id] *= boost

        # Create fused results
        fused = []
        for chunk_id, combined_score in combined_scores.items():
            result = chunk_data[chunk_id]
            fused.append(RetrievalResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=result.text,
                score=combined_score,
                retrieval_mode="weighted_fusion",
                metadata={
                    **result.metadata,
                    "original_score": result.score,
                    "lists_appeared": score_counts[chunk_id]
                }
            ))

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused

    def interleave_fusion(
        self,
        result_lists: List[List[RetrievalResult]],
        round_robin: bool = True
    ) -> List[RetrievalResult]:
        """
        Interleave results from multiple lists.

        Args:
            result_lists: List of result lists
            round_robin: If True, take one from each list in turn.
                        If False, weighted by score.

        Returns:
            Interleaved results
        """
        if not result_lists:
            return []

        seen = set()
        fused = []

        if round_robin:
            # Simple round-robin
            max_len = max(len(lst) for lst in result_lists)
            for i in range(max_len):
                for lst in result_lists:
                    if i < len(lst):
                        result = lst[i]
                        if result.chunk_id not in seen:
                            seen.add(result.chunk_id)
                            fused.append(result)
        else:
            # Score-weighted interleaving
            indices = [0] * len(result_lists)
            while any(idx < len(lst) for idx, lst in zip(indices, result_lists)):
                # Find list with highest-scoring next item
                best_score = -1
                best_list_idx = -1

                for list_idx, lst in enumerate(result_lists):
                    idx = indices[list_idx]
                    if idx < len(lst):
                        result = lst[idx]
                        if result.chunk_id not in seen and result.score > best_score:
                            best_score = result.score
                            best_list_idx = list_idx

                if best_list_idx == -1:
                    break

                result = result_lists[best_list_idx][indices[best_list_idx]]
                indices[best_list_idx] += 1

                if result.chunk_id not in seen:
                    seen.add(result.chunk_id)
                    fused.append(result)

        return fused

    def fuse_responses(
        self,
        responses: List[RetrievalResponse],
        method: str = "rrf",
        weights: Optional[List[float]] = None
    ) -> RetrievalResponse:
        """
        Fuse multiple RetrievalResponses into one.

        Args:
            responses: List of RetrievalResponses from different modes
            method: Fusion method ("rrf", "weighted", "interleave")
            weights: Optional weights for each response

        Returns:
            Combined RetrievalResponse
        """
        if not responses:
            return RetrievalResponse(
                results=[],
                query="",
                mode=None,
                metadata={"fusion_method": method}
            )

        # Extract result lists
        result_lists = [r.results for r in responses]
        query = responses[0].query

        # Apply fusion
        if method == "rrf":
            fused_results = self.reciprocal_rank_fusion(result_lists, weights)
        elif method == "weighted":
            fused_results = self.weighted_fusion(
                result_lists,
                weights or [1.0] * len(result_lists)
            )
        elif method == "interleave":
            fused_results = self.interleave_fusion(result_lists)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        # Combine metadata
        total_candidates = sum(r.total_candidates for r in responses)
        total_time = sum(r.retrieval_time_ms for r in responses)
        modes_used = [r.mode.value for r in responses if r.mode]

        return RetrievalResponse(
            results=fused_results,
            query=query,
            mode=None,  # Multi-mode fusion
            total_candidates=total_candidates,
            retrieval_time_ms=total_time,
            metadata={
                "fusion_method": method,
                "modes_used": modes_used,
                "num_sources": len(responses)
            }
        )


def rrf_score(rank: int, k: int = 60) -> float:
    """Calculate RRF score for a single rank"""
    return 1.0 / (k + rank)


def combine_with_rrf(
    *ranked_lists: List[Dict[str, Any]],
    id_key: str = "chunk_id",
    score_key: str = "score",
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Utility function to combine ranked lists using RRF.

    Args:
        *ranked_lists: Variable number of ranked result lists
        id_key: Key for item ID in dicts
        score_key: Key for score in output dicts
        k: RRF constant

    Returns:
        Combined and re-ranked list
    """
    rrf_scores = defaultdict(float)
    item_data = {}

    for rank_list in ranked_lists:
        for rank, item in enumerate(rank_list, 1):
            item_id = item[id_key]
            rrf_scores[item_id] += 1.0 / (k + rank)

            if item_id not in item_data:
                item_data[item_id] = item.copy()

    # Update scores and sort
    results = []
    for item_id, rrf_score in rrf_scores.items():
        item = item_data[item_id]
        item[score_key] = rrf_score
        results.append(item)

    results.sort(key=lambda x: x[score_key], reverse=True)
    return results
