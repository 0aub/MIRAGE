"""
Retrieval Quality Metrics - Phase 3 GraphRAG Enhancement

Implements standard IR metrics for evaluating retrieval quality:
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- Precision@K, Recall@K, F1@K
- MAP (Mean Average Precision)
- Hit Rate@K

Also includes RAG-specific metrics:
- Answer Relevancy (embedding similarity)
- Context Relevancy
- Faithfulness (answer grounded in context)
"""

import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger
import numpy as np

# Embedding for semantic similarity
try:
    from ..embeddings import JinaEmbedder
    EMBEDDER_AVAILABLE = True
except ImportError:
    EMBEDDER_AVAILABLE = False
    logger.warning("Embedder not available for semantic metrics")


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    mrr: float = 0.0                    # Mean Reciprocal Rank
    ndcg: float = 0.0                   # Normalized DCG
    precision_at_k: Dict[int, float] = field(default_factory=dict)  # P@1, P@3, P@5, P@10
    recall_at_k: Dict[int, float] = field(default_factory=dict)     # R@1, R@3, R@5, R@10
    f1_at_k: Dict[int, float] = field(default_factory=dict)         # F1@1, F1@3, F1@5, F1@10
    map_score: float = 0.0              # Mean Average Precision
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)   # Hit@1, Hit@3, Hit@5, Hit@10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mrr": round(self.mrr, 4),
            "ndcg": round(self.ndcg, 4),
            "map": round(self.map_score, 4),
            "precision": {f"P@{k}": round(v, 4) for k, v in self.precision_at_k.items()},
            "recall": {f"R@{k}": round(v, 4) for k, v in self.recall_at_k.items()},
            "f1": {f"F1@{k}": round(v, 4) for k, v in self.f1_at_k.items()},
            "hit_rate": {f"Hit@{k}": round(v, 4) for k, v in self.hit_rate_at_k.items()},
        }

    def summary(self) -> str:
        """Return a brief summary of key metrics."""
        return (
            f"MRR: {self.mrr:.3f} | NDCG: {self.ndcg:.3f} | "
            f"P@5: {self.precision_at_k.get(5, 0):.3f} | "
            f"Hit@3: {self.hit_rate_at_k.get(3, 0):.3f}"
        )


@dataclass
class RAGMetrics:
    """Container for RAG-specific evaluation metrics."""
    answer_relevancy: float = 0.0       # How relevant is the answer to the query
    context_relevancy: float = 0.0      # How relevant is retrieved context to query
    faithfulness: float = 0.0           # Is answer grounded in context
    context_precision: float = 0.0      # Precision of context retrieval
    context_recall: float = 0.0         # Recall of relevant context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_relevancy": round(self.context_relevancy, 4),
            "faithfulness": round(self.faithfulness, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
        }

    def summary(self) -> str:
        return (
            f"Answer Rel: {self.answer_relevancy:.3f} | "
            f"Context Rel: {self.context_relevancy:.3f} | "
            f"Faithfulness: {self.faithfulness:.3f}"
        )


class RetrievalEvaluator:
    """
    Evaluates retrieval quality using standard IR metrics.

    Usage:
        evaluator = RetrievalEvaluator()

        # Single query evaluation
        metrics = evaluator.evaluate_single(
            retrieved_ids=["doc1", "doc2", "doc3"],
            relevant_ids=["doc1", "doc5"],
            relevance_scores={...}  # Optional graded relevance
        )

        # Batch evaluation
        metrics = evaluator.evaluate_batch(results)
    """

    def __init__(self, k_values: List[int] = None):
        """
        Initialize evaluator.

        Args:
            k_values: K values to compute metrics at (default: [1, 3, 5, 10])
        """
        self.k_values = k_values or [1, 3, 5, 10]
        logger.info(f"RetrievalEvaluator initialized: k_values={self.k_values}")

    def evaluate_single(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> RetrievalMetrics:
        """
        Evaluate a single query's retrieval results.

        Args:
            retrieved_ids: List of retrieved document IDs (in ranked order)
            relevant_ids: Set of truly relevant document IDs
            relevance_scores: Optional dict of doc_id -> relevance score (for graded relevance)

        Returns:
            RetrievalMetrics with all computed metrics
        """
        metrics = RetrievalMetrics()

        if not retrieved_ids or not relevant_ids:
            return metrics

        # Convert to set for faster lookup
        relevant_set = set(relevant_ids)

        # MRR: Reciprocal rank of first relevant document
        metrics.mrr = self._compute_mrr(retrieved_ids, relevant_set)

        # NDCG: Normalized DCG
        if relevance_scores:
            metrics.ndcg = self._compute_ndcg(retrieved_ids, relevance_scores, k=10)
        else:
            # Binary relevance
            binary_scores = {doc_id: 1.0 for doc_id in relevant_set}
            metrics.ndcg = self._compute_ndcg(retrieved_ids, binary_scores, k=10)

        # Precision, Recall, F1 at various K
        for k in self.k_values:
            p_at_k = self._compute_precision_at_k(retrieved_ids, relevant_set, k)
            r_at_k = self._compute_recall_at_k(retrieved_ids, relevant_set, k)

            metrics.precision_at_k[k] = p_at_k
            metrics.recall_at_k[k] = r_at_k
            metrics.f1_at_k[k] = self._compute_f1(p_at_k, r_at_k)
            metrics.hit_rate_at_k[k] = self._compute_hit_rate_at_k(retrieved_ids, relevant_set, k)

        # MAP: Mean Average Precision
        metrics.map_score = self._compute_average_precision(retrieved_ids, relevant_set)

        return metrics

    def evaluate_batch(
        self,
        results: List[Dict[str, Any]]
    ) -> RetrievalMetrics:
        """
        Evaluate multiple queries and return averaged metrics.

        Args:
            results: List of dicts with:
                - 'retrieved_ids': List[str]
                - 'relevant_ids': Set[str]
                - 'relevance_scores': Optional[Dict[str, float]]

        Returns:
            Averaged RetrievalMetrics
        """
        if not results:
            return RetrievalMetrics()

        all_metrics = []
        for r in results:
            m = self.evaluate_single(
                retrieved_ids=r.get('retrieved_ids', []),
                relevant_ids=set(r.get('relevant_ids', [])),
                relevance_scores=r.get('relevance_scores')
            )
            all_metrics.append(m)

        # Average all metrics
        return self._average_metrics(all_metrics)

    def _compute_mrr(self, retrieved_ids: List[str], relevant_set: Set[str]) -> float:
        """Compute Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    def _compute_ndcg(
        self,
        retrieved_ids: List[str],
        relevance_scores: Dict[str, float],
        k: int = 10
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain."""
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

        # Ideal DCG
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    def _compute_precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_set: Set[str],
        k: int
    ) -> float:
        """Compute Precision@K."""
        if k == 0:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return relevant_in_top_k / k

    def _compute_recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_set: Set[str],
        k: int
    ) -> float:
        """Compute Recall@K."""
        if not relevant_set:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return relevant_in_top_k / len(relevant_set)

    def _compute_f1(self, precision: float, recall: float) -> float:
        """Compute F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _compute_hit_rate_at_k(
        self,
        retrieved_ids: List[str],
        relevant_set: Set[str],
        k: int
    ) -> float:
        """Compute Hit Rate@K (1 if any relevant doc in top-k, else 0)."""
        top_k = retrieved_ids[:k]
        return 1.0 if any(doc_id in relevant_set for doc_id in top_k) else 0.0

    def _compute_average_precision(
        self,
        retrieved_ids: List[str],
        relevant_set: Set[str]
    ) -> float:
        """Compute Average Precision (AP)."""
        if not relevant_set:
            return 0.0

        num_relevant = 0
        sum_precision = 0.0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                sum_precision += precision_at_i

        if num_relevant == 0:
            return 0.0
        return sum_precision / len(relevant_set)

    def _average_metrics(self, all_metrics: List[RetrievalMetrics]) -> RetrievalMetrics:
        """Average metrics across multiple queries."""
        n = len(all_metrics)
        if n == 0:
            return RetrievalMetrics()

        avg = RetrievalMetrics()
        avg.mrr = sum(m.mrr for m in all_metrics) / n
        avg.ndcg = sum(m.ndcg for m in all_metrics) / n
        avg.map_score = sum(m.map_score for m in all_metrics) / n

        for k in self.k_values:
            avg.precision_at_k[k] = sum(m.precision_at_k.get(k, 0) for m in all_metrics) / n
            avg.recall_at_k[k] = sum(m.recall_at_k.get(k, 0) for m in all_metrics) / n
            avg.f1_at_k[k] = sum(m.f1_at_k.get(k, 0) for m in all_metrics) / n
            avg.hit_rate_at_k[k] = sum(m.hit_rate_at_k.get(k, 0) for m in all_metrics) / n

        return avg


class RAGEvaluator:
    """
    Evaluates RAG-specific quality metrics.

    Uses embedding similarity for semantic evaluation.
    """

    def __init__(self):
        """Initialize RAG evaluator with embedder."""
        self.embedder = None
        if EMBEDDER_AVAILABLE:
            try:
                self.embedder = JinaEmbedder()
                logger.info("RAGEvaluator initialized with JinaEmbedder")
            except Exception as e:
                logger.warning(f"Failed to initialize embedder: {e}")

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> RAGMetrics:
        """
        Evaluate RAG quality.

        Args:
            query: User query
            answer: Generated answer
            contexts: Retrieved context chunks
            ground_truth: Optional ground truth answer

        Returns:
            RAGMetrics with evaluation scores
        """
        metrics = RAGMetrics()

        if not self.embedder:
            logger.warning("Embedder not available, returning zero metrics")
            return metrics

        try:
            # 1. Answer Relevancy: similarity between query and answer
            metrics.answer_relevancy = self._compute_answer_relevancy(query, answer)

            # 2. Context Relevancy: avg similarity between query and contexts
            metrics.context_relevancy = self._compute_context_relevancy(query, contexts)

            # 3. Faithfulness: is answer grounded in context
            metrics.faithfulness = self._compute_faithfulness(answer, contexts)

            # 4. Context Precision/Recall (if ground truth available)
            if ground_truth:
                metrics.context_precision, metrics.context_recall = \
                    self._compute_context_coverage(answer, ground_truth, contexts)

        except Exception as e:
            logger.error(f"Error computing RAG metrics: {e}")

        return metrics

    def _compute_answer_relevancy(self, query: str, answer: str) -> float:
        """Compute semantic similarity between query and answer."""
        if not answer or not query:
            return 0.0

        embeddings = self.embedder.embed([query, answer])
        if len(embeddings) != 2:
            return 0.0

        return self._cosine_similarity(embeddings[0], embeddings[1])

    def _compute_context_relevancy(self, query: str, contexts: List[str]) -> float:
        """Compute average similarity between query and retrieved contexts."""
        if not contexts:
            return 0.0

        # Embed query and all contexts
        texts = [query] + contexts
        embeddings = self.embedder.embed(texts)

        if len(embeddings) < 2:
            return 0.0

        query_emb = embeddings[0]
        context_embs = embeddings[1:]

        # Average similarity
        similarities = [
            self._cosine_similarity(query_emb, ctx_emb)
            for ctx_emb in context_embs
        ]

        return sum(similarities) / len(similarities)

    def _compute_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Compute faithfulness: is the answer grounded in the retrieved context?

        Uses max similarity between answer and any context as proxy.
        """
        if not answer or not contexts:
            return 0.0

        texts = [answer] + contexts
        embeddings = self.embedder.embed(texts)

        if len(embeddings) < 2:
            return 0.0

        answer_emb = embeddings[0]
        context_embs = embeddings[1:]

        # Max similarity (answer should be similar to at least one context)
        similarities = [
            self._cosine_similarity(answer_emb, ctx_emb)
            for ctx_emb in context_embs
        ]

        return max(similarities)

    def _compute_context_coverage(
        self,
        answer: str,
        ground_truth: str,
        contexts: List[str]
    ) -> Tuple[float, float]:
        """
        Compute how well contexts cover the ground truth.

        Returns:
            (precision, recall) of context coverage
        """
        if not ground_truth or not contexts:
            return 0.0, 0.0

        # Use simple token overlap as proxy
        gt_tokens = set(ground_truth.lower().split())
        answer_tokens = set(answer.lower().split())

        context_tokens = set()
        for ctx in contexts:
            context_tokens.update(ctx.lower().split())

        # Precision: how much of context is in ground truth
        if context_tokens:
            precision = len(gt_tokens.intersection(context_tokens)) / len(context_tokens)
        else:
            precision = 0.0

        # Recall: how much of ground truth is covered by context
        if gt_tokens:
            recall = len(gt_tokens.intersection(context_tokens)) / len(gt_tokens)
        else:
            recall = 0.0

        return precision, recall

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


class MetricsTracker:
    """
    Tracks metrics over time for monitoring retrieval quality.

    Usage:
        tracker = MetricsTracker()
        tracker.record(query_id, retrieval_metrics, rag_metrics)
        stats = tracker.get_stats()
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize tracker.

        Args:
            window_size: Number of recent queries to keep for rolling stats
        """
        self.window_size = window_size
        self.retrieval_history: List[RetrievalMetrics] = []
        self.rag_history: List[RAGMetrics] = []
        self.query_count = 0

    def record(
        self,
        retrieval_metrics: Optional[RetrievalMetrics] = None,
        rag_metrics: Optional[RAGMetrics] = None
    ):
        """Record metrics for a query."""
        self.query_count += 1

        if retrieval_metrics:
            self.retrieval_history.append(retrieval_metrics)
            if len(self.retrieval_history) > self.window_size:
                self.retrieval_history.pop(0)

        if rag_metrics:
            self.rag_history.append(rag_metrics)
            if len(self.rag_history) > self.window_size:
                self.rag_history.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        stats = {
            "total_queries": self.query_count,
            "window_size": len(self.retrieval_history),
        }

        # Retrieval metrics averages
        if self.retrieval_history:
            stats["retrieval"] = {
                "mrr": sum(m.mrr for m in self.retrieval_history) / len(self.retrieval_history),
                "ndcg": sum(m.ndcg for m in self.retrieval_history) / len(self.retrieval_history),
                "map": sum(m.map_score for m in self.retrieval_history) / len(self.retrieval_history),
            }

        # RAG metrics averages
        if self.rag_history:
            stats["rag"] = {
                "answer_relevancy": sum(m.answer_relevancy for m in self.rag_history) / len(self.rag_history),
                "context_relevancy": sum(m.context_relevancy for m in self.rag_history) / len(self.rag_history),
                "faithfulness": sum(m.faithfulness for m in self.rag_history) / len(self.rag_history),
            }

        return stats


# Convenience functions

def compute_mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Compute MRR for a single query."""
    evaluator = RetrievalEvaluator()
    metrics = evaluator.evaluate_single(retrieved_ids, relevant_ids)
    return metrics.mrr


def compute_ndcg(
    retrieved_ids: List[str],
    relevance_scores: Dict[str, float],
    k: int = 10
) -> float:
    """Compute NDCG@K for a single query."""
    evaluator = RetrievalEvaluator()
    metrics = evaluator.evaluate_single(
        retrieved_ids,
        set(relevance_scores.keys()),
        relevance_scores
    )
    return metrics.ndcg


# Singleton evaluators
_retrieval_evaluator: Optional[RetrievalEvaluator] = None
_rag_evaluator: Optional[RAGEvaluator] = None
_metrics_tracker: Optional[MetricsTracker] = None


def get_retrieval_evaluator() -> RetrievalEvaluator:
    """Get or create retrieval evaluator singleton."""
    global _retrieval_evaluator
    if _retrieval_evaluator is None:
        _retrieval_evaluator = RetrievalEvaluator()
    return _retrieval_evaluator


def get_rag_evaluator() -> RAGEvaluator:
    """Get or create RAG evaluator singleton."""
    global _rag_evaluator
    if _rag_evaluator is None:
        _rag_evaluator = RAGEvaluator()
    return _rag_evaluator


def get_metrics_tracker() -> MetricsTracker:
    """Get or create metrics tracker singleton."""
    global _metrics_tracker
    if _metrics_tracker is None:
        _metrics_tracker = MetricsTracker()
    return _metrics_tracker
