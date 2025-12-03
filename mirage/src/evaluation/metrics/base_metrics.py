"""
MIRAGE V2 Evaluation Metrics
Base classes and common metrics for RAG evaluation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from loguru import logger


class MetricType(Enum):
    """Types of evaluation metrics"""
    RETRIEVAL = "retrieval"    # Measures retrieval quality
    GENERATION = "generation"  # Measures answer quality
    END_TO_END = "end_to_end"  # Measures full pipeline


@dataclass
class MetricResult:
    """Result from a metric calculation"""
    name: str
    value: float
    type: MetricType
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"{self.name}: {self.value:.4f}"


@dataclass
class EvaluationSample:
    """Single evaluation sample with ground truth"""
    query: str
    ground_truth_answer: str
    ground_truth_chunks: List[str] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    generated_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMetric(ABC):
    """Abstract base class for all metrics"""

    def __init__(self, name: str, metric_type: MetricType):
        self.name = name
        self.metric_type = metric_type

    @abstractmethod
    def calculate(self, sample: EvaluationSample) -> MetricResult:
        """Calculate metric for a single sample"""
        pass

    def calculate_batch(
        self,
        samples: List[EvaluationSample]
    ) -> Tuple[float, List[MetricResult]]:
        """Calculate metric for multiple samples, return average and individual results"""
        results = [self.calculate(s) for s in samples]
        avg = np.mean([r.value for r in results])
        return avg, results


# =============================================================================
# RETRIEVAL METRICS
# =============================================================================

class Precision(BaseMetric):
    """Precision at K: proportion of retrieved chunks that are relevant"""

    def __init__(self, k: int = 10):
        super().__init__(f"Precision@{k}", MetricType.RETRIEVAL)
        self.k = k

    def calculate(self, sample: EvaluationSample) -> MetricResult:
        retrieved = sample.retrieved_chunks[:self.k]
        if not retrieved:
            return MetricResult(self.name, 0.0, self.metric_type)

        gt_set = set(sample.ground_truth_chunks)
        retrieved_ids = [c.get("chunk_id", c.get("text", "")[:50]) for c in retrieved]

        relevant_count = sum(1 for rid in retrieved_ids if rid in gt_set)
        precision = relevant_count / len(retrieved)

        return MetricResult(
            name=self.name,
            value=precision,
            type=self.metric_type,
            details={
                "retrieved": len(retrieved),
                "relevant": relevant_count
            }
        )


class Recall(BaseMetric):
    """Recall at K: proportion of relevant chunks that were retrieved"""

    def __init__(self, k: int = 10):
        super().__init__(f"Recall@{k}", MetricType.RETRIEVAL)
        self.k = k

    def calculate(self, sample: EvaluationSample) -> MetricResult:
        if not sample.ground_truth_chunks:
            return MetricResult(self.name, 0.0, self.metric_type)

        retrieved = sample.retrieved_chunks[:self.k]
        gt_set = set(sample.ground_truth_chunks)
        retrieved_ids = [c.get("chunk_id", c.get("text", "")[:50]) for c in retrieved]

        relevant_count = sum(1 for rid in retrieved_ids if rid in gt_set)
        recall = relevant_count / len(gt_set)

        return MetricResult(
            name=self.name,
            value=recall,
            type=self.metric_type,
            details={
                "ground_truth": len(gt_set),
                "retrieved_relevant": relevant_count
            }
        )


class MRR(BaseMetric):
    """Mean Reciprocal Rank: average of reciprocal ranks of first relevant chunk"""

    def __init__(self):
        super().__init__("MRR", MetricType.RETRIEVAL)

    def calculate(self, sample: EvaluationSample) -> MetricResult:
        gt_set = set(sample.ground_truth_chunks)
        if not gt_set:
            return MetricResult(self.name, 0.0, self.metric_type)

        for i, chunk in enumerate(sample.retrieved_chunks, 1):
            chunk_id = chunk.get("chunk_id", chunk.get("text", "")[:50])
            if chunk_id in gt_set:
                return MetricResult(
                    name=self.name,
                    value=1.0 / i,
                    type=self.metric_type,
                    details={"first_relevant_rank": i}
                )

        return MetricResult(self.name, 0.0, self.metric_type)


class NDCG(BaseMetric):
    """Normalized Discounted Cumulative Gain"""

    def __init__(self, k: int = 10):
        super().__init__(f"NDCG@{k}", MetricType.RETRIEVAL)
        self.k = k

    def calculate(self, sample: EvaluationSample) -> MetricResult:
        gt_set = set(sample.ground_truth_chunks)
        if not gt_set:
            return MetricResult(self.name, 0.0, self.metric_type)

        retrieved = sample.retrieved_chunks[:self.k]

        # Calculate DCG
        dcg = 0.0
        for i, chunk in enumerate(retrieved, 1):
            chunk_id = chunk.get("chunk_id", chunk.get("text", "")[:50])
            relevance = 1.0 if chunk_id in gt_set else 0.0
            dcg += relevance / np.log2(i + 1)

        # Calculate ideal DCG
        ideal_relevances = [1.0] * min(len(gt_set), self.k)
        ideal_relevances.extend([0.0] * (self.k - len(ideal_relevances)))
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_relevances))

        ndcg = dcg / idcg if idcg > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=ndcg,
            type=self.metric_type,
            details={"dcg": dcg, "idcg": idcg}
        )


class ContextRelevance(BaseMetric):
    """Measures how relevant retrieved context is to the query (using embeddings)"""

    def __init__(self, embedder=None):
        super().__init__("ContextRelevance", MetricType.RETRIEVAL)
        self._embedder = embedder

    def calculate(self, sample: EvaluationSample) -> MetricResult:
        if not sample.retrieved_chunks:
            return MetricResult(self.name, 0.0, self.metric_type)

        if self._embedder is None:
            try:
                from ...core.models.embedding_manager import get_embedding_manager
                self._embedder = get_embedding_manager()
            except ImportError:
                return MetricResult(
                    self.name, 0.5, self.metric_type,
                    details={"error": "No embedder available"}
                )

        # Get query embedding
        query_emb = self._embedder.embed(sample.query)

        # Get chunk embeddings and calculate similarities
        similarities = []
        for chunk in sample.retrieved_chunks[:10]:
            text = chunk.get("text", "")
            if text:
                chunk_emb = self._embedder.embed(text)
                sim = self._embedder.similarity(query_emb, chunk_emb)
                similarities.append(sim)

        avg_sim = np.mean(similarities) if similarities else 0.0

        return MetricResult(
            name=self.name,
            value=avg_sim,
            type=self.metric_type,
            details={"num_chunks": len(similarities)}
        )


# =============================================================================
# GENERATION METRICS
# =============================================================================

class AnswerRelevance(BaseMetric):
    """Measures how relevant the answer is to the query"""

    def __init__(self, embedder=None):
        super().__init__("AnswerRelevance", MetricType.GENERATION)
        self._embedder = embedder

    def calculate(self, sample: EvaluationSample) -> MetricResult:
        if not sample.generated_answer:
            return MetricResult(self.name, 0.0, self.metric_type)

        if self._embedder is None:
            try:
                from ...core.models.embedding_manager import get_embedding_manager
                self._embedder = get_embedding_manager()
            except ImportError:
                return MetricResult(
                    self.name, 0.5, self.metric_type,
                    details={"error": "No embedder available"}
                )

        query_emb = self._embedder.embed(sample.query)
        answer_emb = self._embedder.embed(sample.generated_answer)
        similarity = self._embedder.similarity(query_emb, answer_emb)

        return MetricResult(
            name=self.name,
            value=similarity,
            type=self.metric_type
        )


class Faithfulness(BaseMetric):
    """Measures if the answer is grounded in retrieved context (no hallucination)"""

    def __init__(self, embedder=None, threshold: float = 0.6):
        super().__init__("Faithfulness", MetricType.GENERATION)
        self._embedder = embedder
        self.threshold = threshold

    def calculate(self, sample: EvaluationSample) -> MetricResult:
        if not sample.generated_answer or not sample.retrieved_chunks:
            return MetricResult(self.name, 0.0, self.metric_type)

        if self._embedder is None:
            try:
                from ...core.models.embedding_manager import get_embedding_manager
                self._embedder = get_embedding_manager()
            except ImportError:
                return MetricResult(
                    self.name, 0.5, self.metric_type,
                    details={"error": "No embedder available"}
                )

        # Get answer embedding
        answer_emb = self._embedder.embed(sample.generated_answer)

        # Check similarity with each context chunk
        max_similarity = 0.0
        for chunk in sample.retrieved_chunks:
            text = chunk.get("text", "")
            if text:
                chunk_emb = self._embedder.embed(text)
                sim = self._embedder.similarity(answer_emb, chunk_emb)
                max_similarity = max(max_similarity, sim)

        # Higher similarity = more faithful
        return MetricResult(
            name=self.name,
            value=max_similarity,
            type=self.metric_type,
            details={"max_chunk_similarity": max_similarity}
        )


class GroundTruthSimilarity(BaseMetric):
    """Measures similarity between generated answer and ground truth"""

    def __init__(self, embedder=None):
        super().__init__("GroundTruthSimilarity", MetricType.GENERATION)
        self._embedder = embedder

    def calculate(self, sample: EvaluationSample) -> MetricResult:
        if not sample.generated_answer or not sample.ground_truth_answer:
            return MetricResult(self.name, 0.0, self.metric_type)

        if self._embedder is None:
            try:
                from ...core.models.embedding_manager import get_embedding_manager
                self._embedder = get_embedding_manager()
            except ImportError:
                return MetricResult(
                    self.name, 0.5, self.metric_type,
                    details={"error": "No embedder available"}
                )

        generated_emb = self._embedder.embed(sample.generated_answer)
        gt_emb = self._embedder.embed(sample.ground_truth_answer)
        similarity = self._embedder.similarity(generated_emb, gt_emb)

        return MetricResult(
            name=self.name,
            value=similarity,
            type=self.metric_type
        )


class AnswerCompleteness(BaseMetric):
    """Measures if answer covers key information from context"""

    def __init__(self):
        super().__init__("AnswerCompleteness", MetricType.GENERATION)

    def calculate(self, sample: EvaluationSample) -> MetricResult:
        if not sample.generated_answer:
            return MetricResult(self.name, 0.0, self.metric_type)

        # Simple heuristic: check for citations and answer length
        answer = sample.generated_answer
        has_citations = "[1]" in answer or "[2]" in answer

        # Penalize very short answers
        word_count = len(answer.split())
        length_score = min(word_count / 50, 1.0)  # Normalize to 50 words

        # Combine scores
        score = 0.5 * (1.0 if has_citations else 0.3) + 0.5 * length_score

        return MetricResult(
            name=self.name,
            value=score,
            type=self.metric_type,
            details={
                "has_citations": has_citations,
                "word_count": word_count
            }
        )


# =============================================================================
# METRIC SUITE
# =============================================================================

class MetricSuite:
    """Collection of metrics for comprehensive evaluation"""

    def __init__(self, metrics: Optional[List[BaseMetric]] = None):
        if metrics is None:
            # Default metric suite
            metrics = [
                Precision(k=5),
                Precision(k=10),
                Recall(k=10),
                MRR(),
                NDCG(k=10),
                AnswerRelevance(),
                Faithfulness(),
                AnswerCompleteness(),
            ]
        self.metrics = metrics

    def evaluate(
        self,
        sample: EvaluationSample
    ) -> Dict[str, MetricResult]:
        """Evaluate all metrics for a sample"""
        results = {}
        for metric in self.metrics:
            try:
                result = metric.calculate(sample)
                results[metric.name] = result
            except Exception as e:
                logger.warning(f"Metric {metric.name} failed: {e}")
                results[metric.name] = MetricResult(
                    metric.name, 0.0, metric.metric_type,
                    details={"error": str(e)}
                )
        return results

    def evaluate_batch(
        self,
        samples: List[EvaluationSample]
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all metrics for multiple samples"""
        all_results = {m.name: [] for m in self.metrics}

        for sample in samples:
            results = self.evaluate(sample)
            for name, result in results.items():
                all_results[name].append(result.value)

        # Calculate aggregates
        summary = {}
        for name, values in all_results.items():
            summary[name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values)
            }

        return summary

    def get_retrieval_metrics(self) -> List[BaseMetric]:
        """Get only retrieval metrics"""
        return [m for m in self.metrics if m.metric_type == MetricType.RETRIEVAL]

    def get_generation_metrics(self) -> List[BaseMetric]:
        """Get only generation metrics"""
        return [m for m in self.metrics if m.metric_type == MetricType.GENERATION]
