"""
MIRAGE V2 Evaluation Framework
Comprehensive evaluation for RAG systems.

Usage:
    from evaluation import (
        MetricSuite, EvaluationSample,
        ExperimentRunner, ExperimentConfig
    )

    # Create metrics
    suite = MetricSuite()

    # Create sample
    sample = EvaluationSample(
        query="What is MIRAGE?",
        ground_truth_answer="MIRAGE is a RAG system.",
        retrieved_chunks=[{"text": "..."}],
        generated_answer="MIRAGE is..."
    )

    # Evaluate
    results = suite.evaluate(sample)
    for name, result in results.items():
        print(f"{name}: {result.value:.4f}")
"""

# V2 Metrics
from .metrics.base_metrics import (
    BaseMetric,
    MetricType,
    MetricResult,
    MetricSuite,
    EvaluationSample,
    Precision,
    Recall,
    MRR,
    NDCG,
    ContextRelevance,
    AnswerRelevance,
    Faithfulness,
    GroundTruthSimilarity,
    AnswerCompleteness,
)

# V2 Runners
from .runners import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentResult,
    create_test_dataset,
)

# Legacy exports (for backward compatibility)
try:
    from .metrics import RAGEvaluator, EvaluationMetrics
except ImportError:
    RAGEvaluator = None
    EvaluationMetrics = None

__all__ = [
    # V2 Metrics
    "BaseMetric",
    "MetricType",
    "MetricResult",
    "MetricSuite",
    "EvaluationSample",
    "Precision",
    "Recall",
    "MRR",
    "NDCG",
    "ContextRelevance",
    "AnswerRelevance",
    "Faithfulness",
    "GroundTruthSimilarity",
    "AnswerCompleteness",
    # V2 Runners
    "ExperimentRunner",
    "ExperimentConfig",
    "ExperimentResult",
    "create_test_dataset",
    # Legacy
    "RAGEvaluator",
    "EvaluationMetrics",
]
