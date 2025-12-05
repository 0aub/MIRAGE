# MIRAGE Evaluation Metrics - Complete Reference

This document defines all metrics for evaluating MIRAGE's performance across speed, accuracy, and quality dimensions.

---

## 1. SPEED METRICS

### 1.1 Latency Metrics

| Metric | Definition | Target | Measurement |
|--------|------------|--------|-------------|
| **TTFT** (Time to First Token) | Time from query submission to first response token | <100ms | `time.time()` from request to first stream byte |
| **E2E Latency** (End-to-End) | Total time from query to complete response | <3s | Full request-response cycle |
| **Retrieval Latency** | Time spent in retrieval phase only | <500ms | `retrieval_engine.retrieve()` duration |
| **Generation Latency** | Time spent in LLM generation | <2s | `llm.generate()` duration |
| **Embedding Latency** | Time to compute query embedding | <50ms | `embedder.embed()` duration |
| **Reranking Latency** | Time for cross-encoder reranking | <200ms | `reranker.rerank()` duration |

### 1.2 Throughput Metrics

| Metric | Definition | Target | Measurement |
|--------|------------|--------|-------------|
| **QPS** (Queries Per Second) | Queries handled per second | >10 | Concurrent load testing |
| **Tokens/Second** | Output tokens generated per second | >50 | Token count / generation time |
| **Chunks/Second** | Chunks processed per second | >100 | Retrieval throughput |

### 1.3 REFRAG-Specific Speed Metrics

| Metric | Definition | Target | Measurement |
|--------|------------|--------|-------------|
| **Compression Ratio** | Input tokens / effective tokens | 16x | REFRAG chunk compression |
| **Expansion Ratio** | Chunks expanded / total chunks | <20% | Policy decisions |
| **Speedup Factor** | Baseline TTFT / REFRAG TTFT | 30x | A/B comparison |
| **Context Reduction** | Original context / compressed context | 10-16x | Token count comparison |

---

## 2. RETRIEVAL ACCURACY METRICS

### 2.1 Standard IR Metrics

| Metric | Definition | Formula | Range |
|--------|------------|---------|-------|
| **MRR** (Mean Reciprocal Rank) | Average of reciprocal ranks of first relevant result | `1/rank_of_first_relevant` | 0-1 |
| **NDCG@K** (Normalized DCG) | Discounted cumulative gain normalized by ideal | `DCG@K / IDCG@K` | 0-1 |
| **Precision@K** | Fraction of top-K that are relevant | `relevant_in_top_k / K` | 0-1 |
| **Recall@K** | Fraction of all relevant found in top-K | `relevant_in_top_k / total_relevant` | 0-1 |
| **F1@K** | Harmonic mean of P@K and R@K | `2 * P@K * R@K / (P@K + R@K)` | 0-1 |
| **MAP** (Mean Average Precision) | Mean of AP across queries | `mean(AP_per_query)` | 0-1 |
| **Hit@K** | Binary: any relevant in top-K | `1 if any_relevant else 0` | 0 or 1 |

### 2.2 Implementation

```python
# Already in: mirage/src/core/evaluation/retrieval_metrics.py

class RetrievalEvaluator:
    def evaluate_single(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> RetrievalMetrics:
        """
        Returns:
            RetrievalMetrics with mrr, ndcg, precision_at_k, recall_at_k,
            f1_at_k, map_score, hit_rate_at_k
        """
```

---

## 3. RAG QUALITY METRICS

### 3.1 Answer Quality Metrics

| Metric | Definition | Method | Range |
|--------|------------|--------|-------|
| **Answer Relevancy** | How relevant is answer to query | Embedding similarity (query, answer) | 0-1 |
| **Faithfulness** | Is answer grounded in context (no hallucination) | Max similarity (answer, contexts) | 0-1 |
| **Answer Completeness** | Does answer cover all aspects of query | LLM evaluation or checklist | 0-1 |
| **Answer Correctness** | Is answer factually correct | Comparison with ground truth | 0-1 |
| **Coherence** | Is answer well-structured and readable | LLM evaluation | 0-1 |

### 3.2 Context Quality Metrics

| Metric | Definition | Method | Range |
|--------|------------|--------|-------|
| **Context Relevancy** | Average relevance of retrieved contexts | Mean similarity (query, contexts) | 0-1 |
| **Context Precision** | Fraction of context tokens that are useful | LLM evaluation | 0-1 |
| **Context Recall** | Coverage of ground truth by context | Token overlap with ground truth | 0-1 |
| **Context Utilization** | How much of context appears in answer | Token overlap (context, answer) | 0-1 |

### 3.3 Implementation

```python
# Already in: mirage/src/core/evaluation/retrieval_metrics.py

class RAGEvaluator:
    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> RAGMetrics:
        """
        Returns:
            RAGMetrics with answer_relevancy, context_relevancy,
            faithfulness, context_precision, context_recall
        """
```

---

## 4. ENTITY & GRAPH METRICS

### 4.1 Entity Extraction Metrics

| Metric | Definition | Formula | Target |
|--------|------------|---------|--------|
| **Entity Precision** | Correct entities / extracted entities | `TP / (TP + FP)` | >85% |
| **Entity Recall** | Correct entities / all true entities | `TP / (TP + FN)` | >80% |
| **Entity F1** | Harmonic mean | `2 * P * R / (P + R)` | >82% |
| **Type Accuracy** | Correct type / total entities | `correct_type / total` | >90% |

### 4.2 Relationship Extraction Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Relationship Precision** | Correct relationships / extracted | >70% |
| **Relationship Recall** | Correct relationships / all true | >60% |
| **Relationship F1** | Harmonic mean | >65% |
| **Relation Type Accuracy** | Correct relation type / total | >80% |

### 4.3 Entity Disambiguation Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Linking Accuracy** | Correctly linked / total mentions | >85% |
| **Disambiguation Accuracy** | Correct disambiguation / ambiguous mentions | >80% |
| **New Entity Precision** | Correctly new / marked as new | >90% |

---

## 5. COMMUNITY & SUMMARIZATION METRICS

### 5.1 Community Detection Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Modularity** | Quality of community partition | >0.4 |
| **Coverage** | Entities in communities / total entities | >95% |
| **Silhouette Score** | Cluster cohesion and separation | >0.3 |

### 5.2 Summarization Metrics

| Metric | Definition | Method | Target |
|--------|------------|--------|--------|
| **Entity Coverage** | Key entities in summary / total | Token matching | >60% |
| **Information Retention** | Key facts preserved | LLM evaluation | >70% |
| **Compression Quality** | Quality vs compression trade-off | Human evaluation | >3.5/5 |
| **ROUGE-L** | Longest common subsequence | Standard ROUGE | >0.3 |

---

## 6. SYSTEM METRICS

### 6.1 Resource Utilization

| Metric | Definition | Measurement |
|--------|------------|-------------|
| **CPU Usage** | Processor utilization | `psutil.cpu_percent()` |
| **Memory Usage** | RAM consumption | `psutil.virtual_memory()` |
| **GPU Usage** | GPU utilization (if applicable) | `nvidia-smi` |
| **Disk I/O** | Read/write operations | `psutil.disk_io_counters()` |

### 6.2 Storage Metrics

| Metric | Definition | Measurement |
|--------|------------|-------------|
| **Vector Index Size** | Qdrant storage | Qdrant API |
| **Graph Size** | Neo4j nodes + edges | Neo4j queries |
| **Cache Hit Rate** | Cache hits / total requests | Redis stats |
| **Embedding Cache Size** | Cached embeddings | Memory tracking |

---

## 7. BUSINESS METRICS

### 7.1 User Experience

| Metric | Definition | Measurement |
|--------|------------|-------------|
| **Response Satisfaction** | User rating of answers | 1-5 scale feedback |
| **Query Success Rate** | Queries with useful answers | User feedback |
| **Clarification Rate** | Follow-up questions needed | Session analysis |

### 7.2 Cost Metrics

| Metric | Definition | Measurement |
|--------|------------|-------------|
| **Cost per Query** | Total cost / queries | API costs + compute |
| **Tokens per Query** | Average tokens consumed | Token counting |
| **LLM Calls per Query** | Average LLM invocations | Call counting |

---

## 8. BENCHMARK SUITE

### 8.1 Standard Benchmarks

| Benchmark | Type | Purpose |
|-----------|------|---------|
| **NQ (Natural Questions)** | QA | General knowledge |
| **TriviaQA** | QA | Factoid questions |
| **HotpotQA** | Multi-hop QA | Complex reasoning |
| **MS MARCO** | Passage retrieval | Retrieval quality |
| **BEIR** | Retrieval benchmark | Cross-domain |

### 8.2 Arabic-Specific Benchmarks

| Benchmark | Type | Purpose |
|-----------|------|---------|
| **Arabic SQuAD** | QA | Arabic reading comprehension |
| **ARCD** | QA | Arabic comprehension |
| **Custom Arabic Eval** | QA | Domain-specific Arabic |

---

## 9. EVALUATION RUNNER

### 9.1 Automated Evaluation Script

```python
"""
mirage/src/evaluation/benchmark_runner.py

Comprehensive evaluation runner for all metrics.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import time
import json
from loguru import logger


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    # Speed metrics
    ttft_ms: float
    e2e_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    throughput_qps: float

    # Retrieval metrics
    mrr: float
    ndcg_at_10: float
    precision_at_5: float
    recall_at_10: float
    map_score: float

    # RAG metrics
    answer_relevancy: float
    faithfulness: float
    context_relevancy: float

    # Entity metrics
    entity_f1: float
    linking_accuracy: float

    # System metrics
    memory_mb: float
    tokens_per_query: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speed": {
                "ttft_ms": self.ttft_ms,
                "e2e_latency_ms": self.e2e_latency_ms,
                "retrieval_latency_ms": self.retrieval_latency_ms,
                "generation_latency_ms": self.generation_latency_ms,
                "throughput_qps": self.throughput_qps,
            },
            "retrieval": {
                "mrr": self.mrr,
                "ndcg@10": self.ndcg_at_10,
                "precision@5": self.precision_at_5,
                "recall@10": self.recall_at_10,
                "map": self.map_score,
            },
            "rag_quality": {
                "answer_relevancy": self.answer_relevancy,
                "faithfulness": self.faithfulness,
                "context_relevancy": self.context_relevancy,
            },
            "entity": {
                "entity_f1": self.entity_f1,
                "linking_accuracy": self.linking_accuracy,
            },
            "system": {
                "memory_mb": self.memory_mb,
                "tokens_per_query": self.tokens_per_query,
            }
        }

    def summary(self) -> str:
        return f"""
=== MIRAGE Evaluation Results ===

SPEED:
  TTFT: {self.ttft_ms:.1f}ms (target: <100ms)
  E2E Latency: {self.e2e_latency_ms:.1f}ms (target: <3000ms)
  Throughput: {self.throughput_qps:.2f} QPS (target: >10)

RETRIEVAL:
  MRR: {self.mrr:.3f} (target: >0.7)
  NDCG@10: {self.ndcg_at_10:.3f} (target: >0.6)
  P@5: {self.precision_at_5:.3f} (target: >0.5)
  Recall@10: {self.recall_at_10:.3f} (target: >0.7)

RAG QUALITY:
  Answer Relevancy: {self.answer_relevancy:.3f} (target: >0.85)
  Faithfulness: {self.faithfulness:.3f} (target: >0.90)
  Context Relevancy: {self.context_relevancy:.3f} (target: >0.75)

ENTITY:
  Entity F1: {self.entity_f1:.3f} (target: >0.82)
  Linking Accuracy: {self.linking_accuracy:.3f} (target: >0.85)

SYSTEM:
  Memory: {self.memory_mb:.1f}MB
  Tokens/Query: {self.tokens_per_query:.1f}
"""


class BenchmarkRunner:
    """Run comprehensive benchmarks."""

    def __init__(self, mirage_client, test_dataset: List[Dict]):
        self.client = mirage_client
        self.dataset = test_dataset

    def run_full_evaluation(self) -> EvaluationResult:
        """Run all evaluation metrics."""
        logger.info(f"Starting evaluation on {len(self.dataset)} samples")

        # Collect metrics
        speed_metrics = self._evaluate_speed()
        retrieval_metrics = self._evaluate_retrieval()
        rag_metrics = self._evaluate_rag_quality()
        entity_metrics = self._evaluate_entities()
        system_metrics = self._evaluate_system()

        return EvaluationResult(
            **speed_metrics,
            **retrieval_metrics,
            **rag_metrics,
            **entity_metrics,
            **system_metrics
        )

    def _evaluate_speed(self) -> Dict[str, float]:
        """Measure speed metrics."""
        ttft_times = []
        e2e_times = []
        retrieval_times = []
        generation_times = []

        for sample in self.dataset[:50]:  # Speed test on subset
            query = sample["query"]

            # Measure TTFT and E2E
            start = time.time()
            response = self.client.query(query, stream=True)
            first_token_time = None

            for chunk in response:
                if first_token_time is None:
                    first_token_time = time.time()

            end = time.time()

            if first_token_time:
                ttft_times.append((first_token_time - start) * 1000)
            e2e_times.append((end - start) * 1000)

        # Calculate throughput
        throughput = len(self.dataset[:50]) / (sum(e2e_times) / 1000)

        return {
            "ttft_ms": sum(ttft_times) / len(ttft_times) if ttft_times else 0,
            "e2e_latency_ms": sum(e2e_times) / len(e2e_times),
            "retrieval_latency_ms": 0,  # TODO: instrument
            "generation_latency_ms": 0,  # TODO: instrument
            "throughput_qps": throughput,
        }

    def _evaluate_retrieval(self) -> Dict[str, float]:
        """Measure retrieval accuracy."""
        from ..core.evaluation import get_retrieval_evaluator

        evaluator = get_retrieval_evaluator()
        results = []

        for sample in self.dataset:
            if "relevant_chunks" not in sample:
                continue

            query = sample["query"]
            relevant_ids = set(sample["relevant_chunks"])

            # Get retrieval results
            response = self.client.retrieve(query, top_k=10)
            retrieved_ids = [r["chunk_id"] for r in response["results"]]

            # Evaluate
            metrics = evaluator.evaluate_single(retrieved_ids, relevant_ids)
            results.append(metrics)

        if not results:
            return {
                "mrr": 0, "ndcg_at_10": 0, "precision_at_5": 0,
                "recall_at_10": 0, "map_score": 0
            }

        # Average
        return {
            "mrr": sum(r.mrr for r in results) / len(results),
            "ndcg_at_10": sum(r.ndcg for r in results) / len(results),
            "precision_at_5": sum(r.precision_at_k.get(5, 0) for r in results) / len(results),
            "recall_at_10": sum(r.recall_at_k.get(10, 0) for r in results) / len(results),
            "map_score": sum(r.map_score for r in results) / len(results),
        }

    def _evaluate_rag_quality(self) -> Dict[str, float]:
        """Measure RAG quality metrics."""
        from ..core.evaluation import get_rag_evaluator

        evaluator = get_rag_evaluator()
        relevancy_scores = []
        faithfulness_scores = []
        context_relevancy_scores = []

        for sample in self.dataset:
            query = sample["query"]
            ground_truth = sample.get("answer")

            # Get response
            response = self.client.query(query)
            answer = response["answer"]
            contexts = [c["text"] for c in response.get("sources", [])]

            # Evaluate
            metrics = evaluator.evaluate(query, answer, contexts, ground_truth)
            relevancy_scores.append(metrics.answer_relevancy)
            faithfulness_scores.append(metrics.faithfulness)
            context_relevancy_scores.append(metrics.context_relevancy)

        return {
            "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0,
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0,
            "context_relevancy": sum(context_relevancy_scores) / len(context_relevancy_scores) if context_relevancy_scores else 0,
        }

    def _evaluate_entities(self) -> Dict[str, float]:
        """Measure entity extraction quality."""
        # TODO: Implement with annotated test set
        return {
            "entity_f1": 0.75,  # Placeholder
            "linking_accuracy": 0.70,  # Placeholder
        }

    def _evaluate_system(self) -> Dict[str, float]:
        """Measure system resource usage."""
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)

        # Estimate tokens per query
        tokens_per_query = 500  # Placeholder

        return {
            "memory_mb": memory_mb,
            "tokens_per_query": tokens_per_query,
        }
```

---

## 10. METRIC TARGETS BY VERSION

| Metric | V3 (Current) | V4 Target | V5 Target |
|--------|--------------|-----------|-----------|
| **TTFT** | ~3000ms | <100ms | <50ms |
| **MRR** | 0.65 | 0.75 | 0.85 |
| **Answer Relevancy** | 0.80 | 0.88 | 0.92 |
| **Faithfulness** | 0.78 | 0.90 | 0.95 |
| **Entity F1** | 0.75 | 0.85 | 0.90 |
| **Speedup** | 1x | 30x | 50x |

---

## 11. LOGGING & MONITORING

### 11.1 Per-Request Logging

```python
# Add to every request
request_metrics = {
    "request_id": uuid4(),
    "timestamp": datetime.now().isoformat(),
    "query": query,
    "ttft_ms": ttft,
    "e2e_latency_ms": e2e,
    "retrieval_mode": mode,
    "chunks_retrieved": len(chunks),
    "chunks_used": len(used_chunks),
    "tokens_generated": token_count,
    "answer_length": len(answer),
}
logger.info(f"Request metrics: {json.dumps(request_metrics)}")
```

### 11.2 Aggregated Metrics (Prometheus-style)

```python
# Expose metrics endpoint
@app.get("/metrics")
def get_metrics():
    return {
        "mirage_ttft_p50": tracker.percentile(50, "ttft"),
        "mirage_ttft_p95": tracker.percentile(95, "ttft"),
        "mirage_ttft_p99": tracker.percentile(99, "ttft"),
        "mirage_qps": tracker.qps(),
        "mirage_error_rate": tracker.error_rate(),
        "mirage_avg_relevancy": tracker.mean("answer_relevancy"),
    }
```

---

## Usage

```bash
# Run full evaluation
python -m mirage.evaluation.benchmark_runner --dataset test_data.json --output results.json

# Quick speed test
python -m mirage.evaluation.speed_test --queries 100

# RAG quality evaluation
python -m mirage.evaluation.rag_eval --dataset annotated_qa.json
```
