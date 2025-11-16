"""
Compression Policy
RL-based policy for deciding what and how much to compress
Balances speed vs information retention
"""

import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger


class CompressionPolicy:
    """
    RL-based policy for selective compression
    Decides which fragments to compress and by how much
    """

    def __init__(
        self,
        base_compression_ratio: float = 0.5,
        importance_threshold: float = 0.7,
        adaptive: bool = True,
    ):
        """
        Initialize compression policy

        Args:
            base_compression_ratio: Default compression ratio (0-1)
            importance_threshold: Threshold for high-importance content
            adaptive: Whether to adapt ratios based on content
        """
        self.base_compression_ratio = base_compression_ratio
        self.importance_threshold = importance_threshold
        self.adaptive = adaptive
        
        # Policy state (for RL adaptation)
        self.compression_history = []
        self.feedback_history = []
        
        logger.info(
            f"Initialized CompressionPolicy "
            f"(base_ratio={base_compression_ratio}, "
            f"threshold={importance_threshold}, adaptive={adaptive})"
        )

    def decide_compression(
        self,
        chunks: List[Dict[str, Any]],
        query_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Decide compression strategy for each chunk

        Args:
            chunks: List of text chunks with metadata
            query_context: Optional query context for relevance scoring

        Returns:
            List of chunks with compression decisions
        """
        if not chunks:
            return []

        logger.debug(f"Computing compression strategy for {len(chunks)} chunks")

        # Score importance of each chunk
        importance_scores = self._score_importance(chunks, query_context)

        # Decide compression ratio for each chunk
        decisions = []
        for i, (chunk, importance) in enumerate(zip(chunks, importance_scores)):
            # Determine compression ratio based on importance
            if importance >= self.importance_threshold:
                # High importance: minimal or no compression
                compression_ratio = 0.9
                action = "minimal"
            elif importance >= 0.5:
                # Medium importance: moderate compression
                compression_ratio = self.base_compression_ratio
                action = "moderate"
            else:
                # Low importance: aggressive compression
                compression_ratio = 0.3
                action = "aggressive"

            decision = {
                **chunk,
                "importance_score": float(importance),
                "compression_ratio": compression_ratio,
                "compression_action": action,
                "should_compress": compression_ratio < 1.0,
            }

            decisions.append(decision)

        # Log statistics
        compress_count = sum(1 for d in decisions if d["should_compress"])
        avg_ratio = np.mean([d["compression_ratio"] for d in decisions])
        
        logger.info(
            f"Compression decisions: {compress_count}/{len(chunks)} chunks, "
            f"avg_ratio={avg_ratio:.2f}"
        )

        return decisions

    def _score_importance(
        self,
        chunks: List[Dict[str, Any]],
        query_context: Optional[str] = None,
    ) -> np.ndarray:
        """
        Score importance of each chunk

        Args:
            chunks: List of chunks
            query_context: Optional query for relevance scoring

        Returns:
            Array of importance scores (0-1)
        """
        scores = []

        for chunk in chunks:
            # Base importance from chunk properties
            importance = 0.5

            # Factor 1: Length (longer chunks often more important)
            char_count = chunk.get("char_count", 0)
            if char_count > 800:
                importance += 0.1
            elif char_count < 200:
                importance -= 0.1

            # Factor 2: Position (first and last chunks often important)
            chunk_index = chunk.get("chunk_index", 0)
            total_chunks = len(chunks)
            if chunk_index == 0 or chunk_index == total_chunks - 1:
                importance += 0.15

            # Factor 3: Sentence count (more sentences = more info density)
            sentence_count = chunk.get("sentence_count", 1)
            if sentence_count >= 5:
                importance += 0.1

            # Factor 4: Query relevance (if query provided)
            if query_context and "embedding" in chunk:
                # TODO: Compute semantic similarity with query
                # For now, use placeholder
                pass

            # Factor 5: Named entity density (if entities extracted)
            if "entities" in chunk:
                entity_density = len(chunk["entities"]) / max(1, char_count / 100)
                importance += min(0.2, entity_density * 0.1)

            # Normalize to [0, 1]
            importance = np.clip(importance, 0.0, 1.0)
            scores.append(importance)

        return np.array(scores)

    def update_policy(
        self,
        compression_result: Dict[str, Any],
        feedback: Dict[str, Any],
    ):
        """
        Update policy based on feedback (RL adaptation)

        Args:
            compression_result: Result of compression
            feedback: Performance feedback (latency, quality, etc.)
        """
        self.compression_history.append(compression_result)
        self.feedback_history.append(feedback)

        # Simple adaptive policy: adjust base ratio based on feedback
        if self.adaptive and len(self.feedback_history) >= 10:
            # Get recent feedback
            recent_feedback = self.feedback_history[-10:]
            
            # Compute average quality and speed metrics
            avg_quality = np.mean([
                f.get("quality_score", 0.5)
                for f in recent_feedback
            ])
            avg_speedup = np.mean([
                f.get("speedup_factor", 1.0)
                for f in recent_feedback
            ])

            # Adjust policy
            if avg_quality < 0.7:
                # Quality too low, reduce compression
                self.base_compression_ratio = min(
                    0.9,
                    self.base_compression_ratio + 0.05,
                )
                logger.info(
                    f"Policy adapted: increased ratio to "
                    f"{self.base_compression_ratio:.2f} (low quality)"
                )
            elif avg_quality > 0.9 and avg_speedup < 5.0:
                # Quality high but not fast enough, increase compression
                self.base_compression_ratio = max(
                    0.3,
                    self.base_compression_ratio - 0.05,
                )
                logger.info(
                    f"Policy adapted: decreased ratio to "
                    f"{self.base_compression_ratio:.2f} (room for compression)"
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get policy statistics"""
        return {
            "base_compression_ratio": self.base_compression_ratio,
            "importance_threshold": self.importance_threshold,
            "adaptive": self.adaptive,
            "compression_count": len(self.compression_history),
            "feedback_count": len(self.feedback_history),
        }
