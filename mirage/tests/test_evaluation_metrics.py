"""
Test MIRAGE Evaluation Metrics Module

Tests the retrieval and RAG evaluation metrics including:
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- Precision/Recall@K
- RAG quality metrics (faithfulness, relevancy)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestRetrievalMetrics:
    """Test retrieval evaluation metrics"""

    def test_import_metrics(self):
        """Test that metrics module can be imported"""
        from core.evaluation import (
            RetrievalMetrics,
            RetrievalEvaluator,
            RAGMetrics,
            RAGEvaluator,
            MetricsTracker,
            compute_mrr,
            compute_ndcg,
            get_retrieval_evaluator,
            get_rag_evaluator,
            get_metrics_tracker,
        )
        assert True

    def test_mrr_perfect_ranking(self):
        """Test MRR with perfect ranking (first result is relevant)"""
        from core.evaluation import compute_mrr

        # First result is relevant
        mrr = compute_mrr(
            retrieved_ids=["doc1", "doc2", "doc3"],
            relevant_ids={"doc1"}
        )
        assert mrr == 1.0

    def test_mrr_second_position(self):
        """Test MRR when relevant doc is second"""
        from core.evaluation import compute_mrr

        # Second result is relevant
        mrr = compute_mrr(
            retrieved_ids=["doc0", "doc1", "doc2"],
            relevant_ids={"doc1"}
        )
        assert mrr == 0.5

    def test_mrr_no_relevant(self):
        """Test MRR when no relevant docs are retrieved"""
        from core.evaluation import compute_mrr

        # No relevant results
        mrr = compute_mrr(
            retrieved_ids=["doc0", "doc1", "doc2"],
            relevant_ids={"doc5", "doc6"}
        )
        assert mrr == 0.0

    def test_retrieval_evaluator_single(self):
        """Test RetrievalEvaluator.evaluate_single"""
        from core.evaluation import get_retrieval_evaluator

        evaluator = get_retrieval_evaluator()

        # Perfect retrieval
        metrics = evaluator.evaluate_single(
            retrieved_ids=["doc1", "doc2", "doc3"],
            relevant_ids={"doc1", "doc2"}
        )

        assert metrics.mrr == 1.0  # First result is relevant
        assert metrics.precision_at_5 > 0  # At least some precision
        assert metrics.recall_at_10 > 0  # At least some recall

    def test_retrieval_metrics_summary(self):
        """Test RetrievalMetrics.summary() formatting"""
        from core.evaluation import RetrievalMetrics

        metrics = RetrievalMetrics(
            mrr=0.75,
            ndcg_at_10=0.65,
            precision_at_5=0.40,
            recall_at_10=0.60,
            hit_rate_at_3=1.0
        )

        summary = metrics.summary()
        assert "MRR" in summary
        assert "NDCG" in summary
        assert "0.75" in summary or "0.750" in summary


class TestRAGMetrics:
    """Test RAG evaluation metrics"""

    def test_rag_metrics_dataclass(self):
        """Test RAGMetrics dataclass"""
        from core.evaluation import RAGMetrics

        metrics = RAGMetrics(
            answer_relevancy=0.85,
            context_relevancy=0.72,
            faithfulness=0.90
        )

        assert metrics.answer_relevancy == 0.85
        assert metrics.context_relevancy == 0.72
        assert metrics.faithfulness == 0.90

    def test_rag_metrics_summary(self):
        """Test RAGMetrics.summary() formatting"""
        from core.evaluation import RAGMetrics

        metrics = RAGMetrics(
            answer_relevancy=0.85,
            context_relevancy=0.72,
            faithfulness=0.90
        )

        summary = metrics.summary()
        assert "Relevancy" in summary or "relevancy" in summary.lower()
        assert "Faithfulness" in summary or "faithfulness" in summary.lower()


class TestMetricsTracker:
    """Test metrics tracking over time"""

    def test_tracker_record(self):
        """Test MetricsTracker.record"""
        from core.evaluation import (
            get_metrics_tracker,
            RetrievalMetrics,
            RAGMetrics
        )

        tracker = get_metrics_tracker()

        # Record some metrics
        retrieval_metrics = RetrievalMetrics(
            mrr=0.80,
            ndcg_at_10=0.70,
            precision_at_5=0.60,
            recall_at_10=0.75,
            hit_rate_at_3=1.0
        )

        rag_metrics = RAGMetrics(
            answer_relevancy=0.85,
            context_relevancy=0.75,
            faithfulness=0.88
        )

        tracker.record(retrieval_metrics, rag_metrics)

        # Check stats
        stats = tracker.get_stats()
        assert stats is not None
        assert "retrieval" in stats or len(stats) > 0


class TestNDCG:
    """Test NDCG computation"""

    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking"""
        from core.evaluation import compute_ndcg

        # Perfect ranking - relevant docs first
        ndcg = compute_ndcg(
            retrieved_ids=["doc1", "doc2", "doc3"],
            relevant_ids={"doc1", "doc2"},
            k=3
        )

        # Should be 1.0 for perfect ranking
        assert ndcg == 1.0

    def test_ndcg_worst_ranking(self):
        """Test NDCG with suboptimal ranking"""
        from core.evaluation import compute_ndcg

        # Relevant docs at the end
        ndcg = compute_ndcg(
            retrieved_ids=["doc3", "doc4", "doc1", "doc2"],
            relevant_ids={"doc1", "doc2"},
            k=4
        )

        # Should be less than 1.0 since relevant docs not first
        assert ndcg < 1.0
        assert ndcg > 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
