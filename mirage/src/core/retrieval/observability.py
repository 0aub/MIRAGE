"""
RAG Observability - Production-Grade Monitoring and Tracing

Key Features:
1. Query Tracing: Full trace of every query through the system
2. Metrics Collection: Latency, throughput, quality metrics
3. Cost Tracking: Token usage, LLM costs
4. Dashboard Data: Export for monitoring dashboards

This enables:
- Debugging production issues
- Performance optimization
- Cost management
- Quality monitoring
"""

import time
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger
import threading


@dataclass
class TraceStep:
    """A single step in a query trace."""
    name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


@dataclass
class QueryTrace:
    """Complete trace of a query through the system."""
    trace_id: str
    query: str
    start_time: float
    end_time: float = 0.0
    total_duration_ms: float = 0.0
    steps: List[TraceStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    def add_step(self, step: TraceStep):
        """Add a completed step to the trace."""
        self.steps.append(step)

    def get_step_timings(self) -> Dict[str, float]:
        """Get timing breakdown by step."""
        return {step.name: step.duration_ms for step in self.steps}

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "total_duration_ms": self.total_duration_ms,
            "steps": [
                {
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "success": s.success,
                    "error": s.error,
                    "metadata": s.metadata
                }
                for s in self.steps
            ],
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error
        }


@dataclass
class MetricsSnapshot:
    """Snapshot of collected metrics."""
    timestamp: float
    total_queries: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    queries_per_second: float
    error_rate: float
    avg_tokens_per_query: float
    cache_hit_rate: float
    mode_distribution: Dict[str, int] = field(default_factory=dict)


class RAGObservability:
    """
    Production-grade observability for RAG system.

    Features:
    - Query tracing with step breakdown
    - Latency percentile tracking
    - Token and cost tracking
    - Error tracking
    - Mode usage distribution
    """

    def __init__(
        self,
        max_traces: int = 1000,
        metrics_window_seconds: int = 300
    ):
        """
        Initialize observability.

        Args:
            max_traces: Maximum traces to keep in memory
            metrics_window_seconds: Window for rate calculations
        """
        self.max_traces = max_traces
        self.metrics_window = metrics_window_seconds

        # Trace storage (circular buffer)
        self._traces: List[QueryTrace] = []
        self._trace_lock = threading.Lock()

        # Metrics accumulators
        self._latencies: List[float] = []
        self._token_counts: List[int] = []
        self._timestamps: List[float] = []
        self._mode_counts: Dict[str, int] = defaultdict(int)
        self._error_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._metrics_lock = threading.Lock()

        logger.info("RAGObservability initialized")

    def start_trace(self, query: str) -> QueryTrace:
        """Start a new query trace."""
        trace = QueryTrace(
            trace_id=str(uuid.uuid4())[:8],
            query=query,
            start_time=time.time()
        )
        return trace

    def end_trace(self, trace: QueryTrace, success: bool = True, error: str = None):
        """End and store a query trace."""
        trace.end_time = time.time()
        trace.total_duration_ms = (trace.end_time - trace.start_time) * 1000
        trace.success = success
        trace.error = error

        with self._trace_lock:
            self._traces.append(trace)
            if len(self._traces) > self.max_traces:
                self._traces = self._traces[-self.max_traces:]

        with self._metrics_lock:
            self._latencies.append(trace.total_duration_ms)
            self._timestamps.append(trace.end_time)

            if not success:
                self._error_count += 1

            # Prune old data
            self._prune_old_data()

        logger.debug(
            f"Trace {trace.trace_id}: {trace.total_duration_ms:.1f}ms, "
            f"success={success}"
        )

    def record_step(
        self,
        trace: QueryTrace,
        name: str,
        metadata: Dict = None
    ):
        """
        Context manager for recording a step in a trace.

        Usage:
            with obs.record_step(trace, "embedding") as step:
                embedding = embedder.embed(query)
                step.metadata["dim"] = len(embedding)
        """
        return StepRecorder(trace, name, metadata or {})

    def record_mode(self, mode: str):
        """Record retrieval mode usage."""
        with self._metrics_lock:
            self._mode_counts[mode] += 1

    def record_tokens(self, input_tokens: int, output_tokens: int):
        """Record token usage."""
        with self._metrics_lock:
            self._token_counts.append(input_tokens + output_tokens)

    def record_cache_hit(self, hit: bool):
        """Record cache hit/miss."""
        with self._metrics_lock:
            if hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    def get_metrics(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        with self._metrics_lock:
            now = time.time()

            # Prune old data
            self._prune_old_data()

            # Calculate metrics
            total_queries = len(self._latencies)

            if total_queries == 0:
                return MetricsSnapshot(
                    timestamp=now,
                    total_queries=0,
                    avg_latency_ms=0,
                    p50_latency_ms=0,
                    p95_latency_ms=0,
                    p99_latency_ms=0,
                    queries_per_second=0,
                    error_rate=0,
                    avg_tokens_per_query=0,
                    cache_hit_rate=0
                )

            # Latency percentiles
            sorted_latencies = sorted(self._latencies)
            avg_latency = sum(sorted_latencies) / len(sorted_latencies)
            p50 = self._percentile(sorted_latencies, 50)
            p95 = self._percentile(sorted_latencies, 95)
            p99 = self._percentile(sorted_latencies, 99)

            # QPS
            if self._timestamps:
                time_span = now - min(self._timestamps)
                qps = total_queries / time_span if time_span > 0 else 0
            else:
                qps = 0

            # Error rate
            error_rate = self._error_count / total_queries if total_queries > 0 else 0

            # Average tokens
            avg_tokens = (
                sum(self._token_counts) / len(self._token_counts)
                if self._token_counts else 0
            )

            # Cache hit rate
            total_cache = self._cache_hits + self._cache_misses
            cache_hit_rate = self._cache_hits / total_cache if total_cache > 0 else 0

            return MetricsSnapshot(
                timestamp=now,
                total_queries=total_queries,
                avg_latency_ms=avg_latency,
                p50_latency_ms=p50,
                p95_latency_ms=p95,
                p99_latency_ms=p99,
                queries_per_second=qps,
                error_rate=error_rate,
                avg_tokens_per_query=avg_tokens,
                cache_hit_rate=cache_hit_rate,
                mode_distribution=dict(self._mode_counts)
            )

    def get_recent_traces(self, limit: int = 10) -> List[QueryTrace]:
        """Get most recent traces."""
        with self._trace_lock:
            return list(reversed(self._traces[-limit:]))

    def get_slow_traces(
        self,
        threshold_ms: float = 1000,
        limit: int = 10
    ) -> List[QueryTrace]:
        """Get traces slower than threshold."""
        with self._trace_lock:
            slow = [t for t in self._traces if t.total_duration_ms > threshold_ms]
            return sorted(slow, key=lambda t: t.total_duration_ms, reverse=True)[:limit]

    def get_failed_traces(self, limit: int = 10) -> List[QueryTrace]:
        """Get failed traces."""
        with self._trace_lock:
            failed = [t for t in self._traces if not t.success]
            return list(reversed(failed[-limit:]))

    def get_dashboard_data(self) -> Dict:
        """Get data for monitoring dashboard."""
        metrics = self.get_metrics()
        recent_traces = self.get_recent_traces(5)
        slow_traces = self.get_slow_traces(limit=3)

        return {
            "metrics": {
                "total_queries": metrics.total_queries,
                "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                "p95_latency_ms": round(metrics.p95_latency_ms, 2),
                "p99_latency_ms": round(metrics.p99_latency_ms, 2),
                "qps": round(metrics.queries_per_second, 2),
                "error_rate": round(metrics.error_rate * 100, 2),
                "cache_hit_rate": round(metrics.cache_hit_rate * 100, 2),
            },
            "mode_distribution": metrics.mode_distribution,
            "recent_traces": [t.to_dict() for t in recent_traces],
            "slow_traces": [t.to_dict() for t in slow_traces],
        }

    def _prune_old_data(self):
        """Remove data older than metrics window."""
        cutoff = time.time() - self.metrics_window

        if self._timestamps:
            valid_indices = [i for i, t in enumerate(self._timestamps) if t > cutoff]

            if len(valid_indices) < len(self._timestamps):
                self._timestamps = [self._timestamps[i] for i in valid_indices]
                self._latencies = [self._latencies[i] for i in valid_indices if i < len(self._latencies)]
                self._token_counts = [self._token_counts[i] for i in valid_indices if i < len(self._token_counts)]

    def _percentile(self, sorted_data: List[float], percentile: int) -> float:
        """Calculate percentile from sorted data."""
        if not sorted_data:
            return 0.0

        k = (len(sorted_data) - 1) * (percentile / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f

        if f == c:
            return sorted_data[f]

        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    def reset(self):
        """Reset all metrics."""
        with self._trace_lock:
            self._traces.clear()

        with self._metrics_lock:
            self._latencies.clear()
            self._token_counts.clear()
            self._timestamps.clear()
            self._mode_counts.clear()
            self._error_count = 0
            self._cache_hits = 0
            self._cache_misses = 0

        logger.info("Observability reset")


class StepRecorder:
    """Context manager for recording a trace step."""

    def __init__(self, trace: QueryTrace, name: str, metadata: Dict):
        self.trace = trace
        self.step = TraceStep(
            name=name,
            start_time=time.time(),
            metadata=metadata
        )

    def __enter__(self):
        return self.step

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.step.end_time = time.time()
        self.step.duration_ms = (self.step.end_time - self.step.start_time) * 1000

        if exc_type is not None:
            self.step.success = False
            self.step.error = str(exc_val)

        self.trace.add_step(self.step)
        return False  # Don't suppress exceptions


class CostTracker:
    """Track costs for LLM and API usage."""

    # Cost per 1K tokens (approximate)
    COSTS = {
        "embedding": 0.0001,    # $0.0001 per 1K tokens
        "generation": 0.002,   # $0.002 per 1K tokens (small model)
        "reranking": 0.0005,   # $0.0005 per 1K tokens
    }

    def __init__(self):
        self._costs: Dict[str, float] = defaultdict(float)
        self._token_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record_usage(self, operation: str, tokens: int):
        """Record token usage for an operation."""
        with self._lock:
            self._token_counts[operation] += tokens
            cost_per_token = self.COSTS.get(operation, 0.001) / 1000
            self._costs[operation] += tokens * cost_per_token

    def get_costs(self) -> Dict[str, float]:
        """Get cost breakdown by operation."""
        with self._lock:
            return dict(self._costs)

    def get_total_cost(self) -> float:
        """Get total cost."""
        with self._lock:
            return sum(self._costs.values())

    def get_usage(self) -> Dict[str, int]:
        """Get token usage by operation."""
        with self._lock:
            return dict(self._token_counts)

    def reset(self):
        """Reset cost tracking."""
        with self._lock:
            self._costs.clear()
            self._token_counts.clear()


# Singleton instances
_observability: Optional[RAGObservability] = None
_cost_tracker: Optional[CostTracker] = None


def get_observability() -> RAGObservability:
    """Get or create observability singleton."""
    global _observability

    if _observability is None:
        _observability = RAGObservability()

    return _observability


def get_cost_tracker() -> CostTracker:
    """Get or create cost tracker singleton."""
    global _cost_tracker

    if _cost_tracker is None:
        _cost_tracker = CostTracker()

    return _cost_tracker
