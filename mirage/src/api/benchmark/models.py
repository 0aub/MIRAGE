"""
Benchmark Service - Pydantic Models
Request/response models for benchmark API endpoints
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime


# =============================================================================
# BASIC BENCHMARK MODELS
# =============================================================================

class BenchmarkQuery(BaseModel):
    query: str
    ground_truth_answer: Optional[str] = None
    relevant_entities: Optional[List[str]] = None


class BenchmarkConfig(BaseModel):
    use_graph: bool
    use_refrag: bool
    name: str


class BenchmarkResult(BaseModel):
    config_name: str
    query: str

    # Performance Metrics
    total_time_ms: int
    retrieval_time_ms: int
    compression_time_ms: Optional[int]
    generation_time_ms: int

    # Context Metrics
    original_context_length: int
    compressed_context_length: Optional[int]
    compression_ratio: Optional[float]
    speedup_factor: Optional[float]

    # Retrieval Metrics
    nodes_retrieved: int
    chunks_retrieved: int
    entities_found: Optional[List[str]]

    # Response Metrics
    response: str
    response_length: int
    citations_count: int

    # Quality Metrics
    contains_answer: Optional[bool] = None
    entity_precision: Optional[float] = None
    entity_recall: Optional[float] = None


class ComparisonReport(BaseModel):
    query: str
    timestamp: datetime
    configurations: List[BenchmarkResult]
    fastest_config: str
    most_compressed_config: str
    most_entities_config: str
    recommendations: Dict[str, str]


# =============================================================================
# DETAILED BENCHMARK MODELS
# =============================================================================

class DetailedBenchmarkRequest(BaseModel):
    """Request for detailed benchmark"""
    query: str
    retrieval_modes: Optional[List[str]] = None
    compare_refrag: bool = True
    measure_ttft: bool = True
    top_k: int = 5


class RetrievalStats(BaseModel):
    """Detailed retrieval statistics"""
    total_chunks: int
    vector_chunks: int
    graph_1hop_chunks: int
    graph_2hop_chunks: int
    graph_total: int
    entities_used: List[str]


class TimingMetrics(BaseModel):
    """Detailed timing breakdown"""
    total_ms: float
    retrieval_ms: float
    compression_ms: Optional[float] = None
    generation_ms: float
    ttft_ms: Optional[float] = None


class CompressionMetrics(BaseModel):
    """RefRAG compression metrics"""
    enabled: bool
    original_length: int
    compressed_length: int
    compression_ratio: float
    chunks_compressed: int = 0
    strategy: str = "none"


class DetailedBenchmarkResult(BaseModel):
    """Result from detailed benchmark"""
    config_name: str
    retrieval_mode: str
    use_refrag: bool
    timing: TimingMetrics
    retrieval_stats: RetrievalStats
    compression: CompressionMetrics
    response: str
    response_length: int
    contains_keywords: Optional[List[str]] = None


class DetailedComparisonReport(BaseModel):
    """Full comparison report"""
    query: str
    timestamp: datetime
    results: List[DetailedBenchmarkResult]
    fastest_config: str
    best_compression_config: Optional[str] = None
    most_graph_usage_config: str
    recommendations: Dict[str, str]


# =============================================================================
# RAGAS EVALUATION MODELS
# =============================================================================

class RagasEvaluationRequest(BaseModel):
    """Request for RAGAS evaluation"""
    test_case_ids: List[str] = []
    modes: List[str] = ["vector", "local", "global", "hybrid", "mix"]
    compare_refrag: bool = True
    top_k: int = 5


class RagasScores(BaseModel):
    """Individual scores for RAGAS evaluation"""
    chunk_count: float
    entity_match: float
    chunk_content: float
    answer_keywords: float
    ground_truth_similarity: float
    weighted_score: float
    passed: bool


class RagasModeResult(BaseModel):
    """Result for a single mode"""
    mode: str
    answer: str
    scores: RagasScores
    timing_ms: float
    chunks_used: int
    entities_found: List[str]
    refrag_scores: Optional[RagasScores] = None
    refrag_answer: Optional[str] = None
    refrag_timing_ms: Optional[float] = None
    ground_truth_delta: Optional[float] = None
    speed_improvement: Optional[float] = None
    token_savings: Optional[float] = None


class RagasTestResult(BaseModel):
    """Result for a single test case"""
    test_id: str
    query: str
    expected_answer: str
    difficulty: str
    mode_results: List[RagasModeResult]
    best_mode: str
    best_score: float


class RefragImpactAnalysis(BaseModel):
    """Analysis of REFRAG impact on quality"""
    modes_improved: int
    modes_hurt: int
    modes_unchanged: int
    avg_ground_truth_delta: float
    avg_speed_improvement: float
    avg_token_savings: float


class RagasEvaluationSummary(BaseModel):
    """Summary of RAGAS evaluation"""
    total_tests: int
    modes_tested: List[str]
    mode_scores: Dict[str, Dict[str, Any]]
    best_overall_mode: str
    best_overall_score: float


class RagasEvaluationResponse(BaseModel):
    """Full RAGAS evaluation response"""
    test_results: List[RagasTestResult]
    summary: RagasEvaluationSummary
    refrag_impact: Optional[RefragImpactAnalysis] = None
    timestamp: str
    duration_ms: int
