"""
Core Evaluation Module - Phase 3 GraphRAG Enhancement

Provides retrieval and RAG quality metrics for monitoring and evaluation.

Usage:
    from core.evaluation import (
        RetrievalEvaluator, RetrievalMetrics,
        RAGEvaluator, RAGMetrics,
        MetricsTracker,
        get_retrieval_evaluator, get_rag_evaluator, get_metrics_tracker
    )

    # Evaluate retrieval
    evaluator = get_retrieval_evaluator()
    metrics = evaluator.evaluate_single(
        retrieved_ids=["doc1", "doc2"],
        relevant_ids={"doc1", "doc5"}
    )
    print(metrics.summary())  # MRR: 1.000 | NDCG: 0.631 | P@5: 0.200 | Hit@3: 1.000

    # Evaluate RAG quality
    rag_eval = get_rag_evaluator()
    rag_metrics = rag_eval.evaluate(query, answer, contexts)
    print(rag_metrics.summary())  # Answer Rel: 0.85 | Context Rel: 0.72 | Faithfulness: 0.88

    # Track metrics over time
    tracker = get_metrics_tracker()
    tracker.record(metrics, rag_metrics)
    stats = tracker.get_stats()
"""

from .retrieval_metrics import (
    # Dataclasses
    RetrievalMetrics,
    RAGMetrics,
    # Evaluators
    RetrievalEvaluator,
    RAGEvaluator,
    MetricsTracker,
    # Convenience functions
    compute_mrr,
    compute_ndcg,
    # Singletons
    get_retrieval_evaluator,
    get_rag_evaluator,
    get_metrics_tracker,
)

__all__ = [
    # Metrics dataclasses
    "RetrievalMetrics",
    "RAGMetrics",
    # Evaluators
    "RetrievalEvaluator",
    "RAGEvaluator",
    "MetricsTracker",
    # Convenience functions
    "compute_mrr",
    "compute_ndcg",
    # Singletons
    "get_retrieval_evaluator",
    "get_rag_evaluator",
    "get_metrics_tracker",
]
