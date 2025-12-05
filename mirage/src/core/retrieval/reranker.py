"""
MIRAGE V2 Cross-Encoder Reranker

Uses a cross-encoder model to re-score query-document pairs for improved precision.
Cross-encoders process query and document together, enabling richer interactions
compared to bi-encoders that embed them separately.

Supports multilingual queries (Arabic/English) using models like:
- cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingual)
- BAAI/bge-reranker-base
- cross-encoder/ms-marco-MiniLM-L-6-v2 (English only)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not available for cross-encoder reranking")


@dataclass
class RerankResult:
    """Result from reranking"""
    chunk_id: str
    document_id: str
    text: str
    original_score: float
    rerank_score: float
    rank_change: int  # positive = moved up, negative = moved down
    metadata: Dict[str, Any] = None


class CrossEncoderReranker:
    """
    Cross-encoder based reranker for improved retrieval precision.

    Cross-encoders are more accurate than bi-encoders because they:
    - Process query and document together (not separately)
    - Can model complex query-document interactions
    - Better capture semantic nuances

    Trade-off: Slower than bi-encoder retrieval, so used for top-k reranking.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, candidates, top_k=5)
    """

    DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # Multilingual

    def __init__(
        self,
        model_id: str = None,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 32
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_id: HuggingFace model ID for cross-encoder
            device: Device to run on ('cpu', 'cuda', 'cuda:0')
            max_length: Maximum sequence length
            batch_size: Batch size for inference
        """
        self.model_id = model_id or self.DEFAULT_MODEL
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        self._model = None
        self._loaded = False

    def _load_model(self):
        """Lazy load the cross-encoder model"""
        if self._loaded:
            return

        if not CROSSENCODER_AVAILABLE:
            logger.error("CrossEncoder not available - install sentence-transformers")
            return

        try:
            logger.info(f"Loading cross-encoder: {self.model_id}")
            self._model = CrossEncoder(
                self.model_id,
                max_length=self.max_length,
                device=self.device
            )
            self._loaded = True
            logger.info(f"Cross-encoder loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            self._model = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        score_threshold: float = 0.0
    ) -> List[RerankResult]:
        """
        Rerank candidate documents using cross-encoder.

        Args:
            query: Query string
            candidates: List of candidate dicts with 'text', 'chunk_id', 'document_id', 'score'
            top_k: Return only top-k results (None = return all)
            score_threshold: Minimum rerank score to include

        Returns:
            List of RerankResult sorted by rerank_score descending
        """
        if not candidates:
            return []

        # Load model if not loaded
        self._load_model()

        if self._model is None:
            # Fallback: return candidates sorted by original score
            logger.warning("Cross-encoder not available, using original scores")
            results = [
                RerankResult(
                    chunk_id=c.get("chunk_id", ""),
                    document_id=c.get("document_id", ""),
                    text=c.get("text", ""),
                    original_score=c.get("score", 0.0),
                    rerank_score=c.get("score", 0.0),
                    rank_change=0,
                    metadata=c.get("metadata", {})
                )
                for c in candidates
            ]
            return sorted(results, key=lambda x: x.rerank_score, reverse=True)[:top_k]

        # Prepare pairs for cross-encoder
        pairs = [(query, c.get("text", "")) for c in candidates]

        try:
            # Get cross-encoder scores
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )

            # Create results with original and rerank scores
            results = []
            for i, (candidate, score) in enumerate(zip(candidates, scores)):
                results.append(RerankResult(
                    chunk_id=candidate.get("chunk_id", ""),
                    document_id=candidate.get("document_id", ""),
                    text=candidate.get("text", ""),
                    original_score=candidate.get("score", 0.0),
                    rerank_score=float(score),
                    rank_change=0,  # Will be calculated after sorting
                    metadata=candidate.get("metadata", {})
                ))

            # Sort by rerank score
            results.sort(key=lambda x: x.rerank_score, reverse=True)

            # Calculate rank changes
            original_ranks = {c.get("chunk_id"): i for i, c in enumerate(candidates)}
            for new_rank, result in enumerate(results):
                old_rank = original_ranks.get(result.chunk_id, new_rank)
                result.rank_change = old_rank - new_rank  # positive = moved up

            # Filter by threshold
            results = [r for r in results if r.rerank_score >= score_threshold]

            # Apply top_k
            if top_k is not None:
                results = results[:top_k]

            logger.debug(f"Reranked {len(candidates)} candidates -> {len(results)} results")

            return results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to original order
            return [
                RerankResult(
                    chunk_id=c.get("chunk_id", ""),
                    document_id=c.get("document_id", ""),
                    text=c.get("text", ""),
                    original_score=c.get("score", 0.0),
                    rerank_score=c.get("score", 0.0),
                    rank_change=0,
                    metadata=c.get("metadata", {})
                )
                for c in candidates
            ][:top_k]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_id": self.model_id,
            "loaded": self._loaded,
            "device": self.device,
            "max_length": self.max_length,
            "available": CROSSENCODER_AVAILABLE
        }

    def preload(self) -> bool:
        """
        Preload the cross-encoder model to avoid cold-start latency.
        Call this on application startup for semantic mode.

        Returns:
            True if model loaded successfully
        """
        logger.info("Preloading cross-encoder model...")
        self._load_model()

        if self._loaded:
            # Warm up with a test query
            try:
                self._model.predict([("test query", "test document")])
                logger.info("Cross-encoder preloaded and warmed up")
                return True
            except Exception as e:
                logger.warning(f"Cross-encoder warmup failed: {e}")
                return self._loaded

        return False


# Global instance
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(**kwargs) -> CrossEncoderReranker:
    """Get or create global reranker instance"""
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoderReranker(**kwargs)

    return _reranker


def preload_reranker(**kwargs) -> bool:
    """
    Preload the reranker model on application startup.
    Call this to eliminate cold-start latency for semantic mode.

    Returns:
        True if preload successful
    """
    reranker = get_reranker(**kwargs)
    return reranker.preload()
