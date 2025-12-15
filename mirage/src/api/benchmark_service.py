"""
Benchmark Service API
Compare different RAG configurations (RAG, GraphRAG, REFRAG combinations)
Measure speed, relevancy, and quality metrics
Enhanced with TTFT, streaming metrics, and detailed retrieval stats
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from loguru import logger
import time
import httpx
from datetime import datetime

from ..core.orchestration import RAGWorkflow
from ..core.retrieval import get_retrieval_engine, RetrievalMode
from ..core.refrag.compressor import REFRAGCompressor
from ..core.generation import get_prompt_manager

router = APIRouter()

# Initialize components
rag_workflow = RAGWorkflow()
retrieval_engine = get_retrieval_engine()
prompt_manager = get_prompt_manager()
refrag_compressor = REFRAGCompressor(strategy="hybrid")


# Models
class BenchmarkQuery(BaseModel):
    query: str
    ground_truth_answer: Optional[str] = None  # For measuring accuracy
    relevant_entities: Optional[List[str]] = None  # Expected entities


class BenchmarkConfig(BaseModel):
    use_graph: bool
    use_refrag: bool
    name: str  # "Normal RAG", "GraphRAG", etc.


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

    # Quality Metrics (if ground truth provided)
    contains_answer: Optional[bool] = None
    entity_precision: Optional[float] = None
    entity_recall: Optional[float] = None


class ComparisonReport(BaseModel):
    query: str
    timestamp: datetime
    configurations: List[BenchmarkResult]

    # Comparison Summary
    fastest_config: str
    most_compressed_config: str
    most_entities_config: str

    # Recommendations
    recommendations: Dict[str, str]


# Endpoints
@router.post("/single", response_model=BenchmarkResult)
async def benchmark_single_config(
    query: BenchmarkQuery,
    config: BenchmarkConfig,
):
    """
    Benchmark a single RAG configuration

    Example:
        {
            "query": {"query": "من أسس مايكروسوفت؟"},
            "config": {"use_graph": true, "use_refrag": true, "name": "GraphRAG+REFRAG"}
        }
    """
    logger.info(f"Benchmarking '{config.name}': query='{query.query[:50]}...'")

    start_time = time.time()

    try:
        # Run RAG workflow with specified configuration
        result = rag_workflow.invoke(
            query=query.query,
            use_graph=config.use_graph,
            use_refrag=config.use_refrag,
        )

        total_time = (time.time() - start_time) * 1000

        # Extract metrics from result
        metadata = result.get("metadata", {})
        latency = metadata.get("latency_ms", {})
        compression_stats = metadata.get("compression_stats", {})

        # Build response
        benchmark_result = BenchmarkResult(
            config_name=config.name,
            query=query.query,

            # Performance
            total_time_ms=int(total_time),
            retrieval_time_ms=int(latency.get("graph_retrieval", 0)),
            compression_time_ms=int(latency.get("compression", 0)) if config.use_refrag else None,
            generation_time_ms=int(latency.get("generation", 0)),

            # Context
            original_context_length=compression_stats.get("original_length", 0),
            compressed_context_length=compression_stats.get("compressed_length") if config.use_refrag else None,
            compression_ratio=compression_stats.get("compression_ratio") if config.use_refrag else None,
            speedup_factor=compression_stats.get("speedup_factor") if config.use_refrag else None,

            # Retrieval
            nodes_retrieved=metadata.get("nodes_retrieved", 0),
            chunks_retrieved=len(result.get("retrieved_chunks", [])),
            entities_found=[e.get("name", "") for e in result.get("entities_found", [])[:10]],

            # Response
            response=result.get("response", ""),
            response_length=len(result.get("response", "")),
            citations_count=len(result.get("citations", [])),
        )

        # Calculate quality metrics if ground truth provided
        if query.ground_truth_answer:
            benchmark_result.contains_answer = (
                query.ground_truth_answer.lower() in result.get("response", "").lower()
            )

        if query.relevant_entities:
            found_entities = set(e.lower() for e in benchmark_result.entities_found or [])
            relevant_entities = set(e.lower() for e in query.relevant_entities)

            if found_entities:
                precision = len(found_entities & relevant_entities) / len(found_entities)
                recall = len(found_entities & relevant_entities) / len(relevant_entities) if relevant_entities else 0

                benchmark_result.entity_precision = round(precision, 3)
                benchmark_result.entity_recall = round(recall, 3)

        logger.info(
            f"Benchmark complete: {config.name} took {total_time:.0f}ms, "
            f"retrieved {benchmark_result.nodes_retrieved} nodes"
        )

        return benchmark_result

    except Exception as e:
        logger.error(f"Benchmark failed for {config.name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=ComparisonReport)
async def benchmark_compare_all(query: BenchmarkQuery):
    """
    Compare all RAG configurations for a single query

    Tests 4 configurations:
    1. Normal RAG (no graph, no compression)
    2. RAG + REFRAG (no graph, with compression)
    3. GraphRAG (with graph, no compression)
    4. GraphRAG + REFRAG (with graph, with compression)

    Example:
        {
            "query": "Who founded Microsoft?",
            "ground_truth_answer": "Bill Gates and Paul Allen",
            "relevant_entities": ["Microsoft", "Bill Gates", "Paul Allen"]
        }
    """
    logger.info(f"Running full comparison benchmark for: {query.query[:50]}...")

    # Define configurations to test
    configs = [
        BenchmarkConfig(use_graph=False, use_refrag=False, name="Normal RAG"),
        BenchmarkConfig(use_graph=False, use_refrag=True, name="RAG + REFRAG"),
        BenchmarkConfig(use_graph=True, use_refrag=False, name="GraphRAG"),
        BenchmarkConfig(use_graph=True, use_refrag=True, name="GraphRAG + REFRAG"),
    ]

    # Run benchmarks for all configurations
    results = []
    for config in configs:
        try:
            result = await benchmark_single_config(query, config)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to benchmark {config.name}: {e}")
            continue

    if not results:
        raise HTTPException(status_code=500, detail="All benchmarks failed")

    # Analyze results
    fastest = min(results, key=lambda r: r.total_time_ms)
    most_compressed = min(
        [r for r in results if r.compression_ratio is not None],
        key=lambda r: r.compression_ratio,
        default=None
    )
    most_entities = max(results, key=lambda r: r.nodes_retrieved)

    # Generate recommendations
    recommendations = {}

    # Speed recommendation
    if fastest.config_name == "Normal RAG":
        recommendations["speed"] = "Normal RAG is fastest but may have lower accuracy"
    elif fastest.config_name == "GraphRAG + REFRAG":
        recommendations["speed"] = "GraphRAG + REFRAG provides best overall speed with high quality"
    else:
        recommendations["speed"] = f"{fastest.config_name} is fastest at {fastest.total_time_ms}ms"

    # Quality recommendation
    if most_entities.config_name.startswith("GraphRAG"):
        recommendations["quality"] = "GraphRAG configurations retrieve more structured knowledge"

    # Efficiency recommendation
    if most_compressed:
        speedup = most_compressed.speedup_factor or 1.0
        recommendations["efficiency"] = (
            f"{most_compressed.config_name} achieves {speedup:.1f}x context compression, "
            f"allowing more information within token limits"
        )

    # Overall recommendation
    graphrag_refrag = next((r for r in results if r.config_name == "GraphRAG + REFRAG"), None)
    if graphrag_refrag:
        recommendations["overall"] = (
            f"GraphRAG + REFRAG recommended for production: "
            f"{graphrag_refrag.total_time_ms}ms response time, "
            f"{graphrag_refrag.nodes_retrieved} entities, "
            f"{graphrag_refrag.compression_ratio:.2f} compression ratio"
        )

    return ComparisonReport(
        query=query.query,
        timestamp=datetime.now(),
        configurations=results,
        fastest_config=fastest.config_name,
        most_compressed_config=most_compressed.config_name if most_compressed else "N/A",
        most_entities_config=most_entities.config_name,
        recommendations=recommendations,
    )


@router.post("/batch", response_model=List[ComparisonReport])
async def benchmark_batch(queries: List[BenchmarkQuery], max_queries: int = 10):
    """
    Run comparison benchmarks for multiple queries

    Useful for:
    - Testing system across different query types
    - A/B testing configurations
    - Performance regression testing

    Example:
        [
            {"query": "Who founded Microsoft?", "relevant_entities": ["Microsoft", "Bill Gates"]},
            {"query": "What is Python?", "relevant_entities": ["Python", "Programming"]},
            {"query": "من هو مؤسس جوجل؟", "relevant_entities": ["Google", "Larry Page"]}
        ]
    """
    if len(queries) > max_queries:
        raise HTTPException(
            status_code=400,
            detail=f"Too many queries. Maximum is {max_queries}"
        )

    logger.info(f"Running batch benchmark for {len(queries)} queries")

    reports = []
    for i, query in enumerate(queries, 1):
        logger.info(f"Processing query {i}/{len(queries)}: {query.query[:50]}...")

        try:
            report = await benchmark_compare_all(query)
            reports.append(report)
        except Exception as e:
            logger.error(f"Failed to benchmark query {i}: {e}")
            continue

    logger.info(f"Batch benchmark complete: {len(reports)}/{len(queries)} succeeded")

    return reports


@router.get("/summary")
async def get_benchmark_summary():
    """
    Get summary of available benchmark configurations and metrics
    """
    return {
        "available_configurations": [
            {
                "name": "Normal RAG",
                "use_graph": False,
                "use_refrag": False,
                "description": "Vector search only, no compression"
            },
            {
                "name": "RAG + REFRAG",
                "use_graph": False,
                "use_refrag": True,
                "description": "Vector search with REFRAG compression"
            },
            {
                "name": "GraphRAG",
                "use_graph": True,
                "use_refrag": False,
                "description": "Graph-based retrieval, no compression"
            },
            {
                "name": "GraphRAG + REFRAG",
                "use_graph": True,
                "use_refrag": True,
                "description": "Graph-based retrieval with REFRAG compression (recommended)"
            },
        ],
        "metrics_measured": {
            "performance": [
                "total_time_ms",
                "retrieval_time_ms",
                "compression_time_ms",
                "generation_time_ms",
                "ttft_ms"  # Time To First Token
            ],
            "context": [
                "original_context_length",
                "compressed_context_length",
                "compression_ratio",
                "speedup_factor"
            ],
            "retrieval": [
                "nodes_retrieved",
                "chunks_retrieved",
                "entities_found",
                "vector_chunks",
                "graph_chunks",
                "graph_1hop_chunks",
                "graph_2hop_chunks"
            ],
            "quality": [
                "contains_answer",
                "entity_precision",
                "entity_recall",
                "citations_count"
            ]
        },
        "endpoints": {
            "/benchmark/single": "Benchmark a single configuration",
            "/benchmark/compare": "Compare all 4 configurations for one query",
            "/benchmark/batch": "Compare configurations across multiple queries",
            "/benchmark/summary": "This endpoint - shows available options",
            "/benchmark/detailed": "Detailed benchmark with TTFT, retrieval stats, RefRAG comparison"
        }
    }


# =============================================================================
# NEW: Detailed Benchmark with TTFT, Streaming Metrics, RefRAG Comparison
# =============================================================================

class DetailedBenchmarkRequest(BaseModel):
    """Request for detailed benchmark"""
    query: str
    retrieval_modes: Optional[List[str]] = None  # Modes to test (default: all)
    compare_refrag: bool = True  # Run with and without RefRAG
    measure_ttft: bool = True  # Measure time to first token
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
    ttft_ms: Optional[float] = None  # Time to first token


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

    # Timing
    timing: TimingMetrics

    # Retrieval stats
    retrieval_stats: RetrievalStats

    # Compression stats
    compression: CompressionMetrics

    # Response
    response: str
    response_length: int

    # Quality (if ground truth)
    contains_keywords: Optional[List[str]] = None


class DetailedComparisonReport(BaseModel):
    """Full comparison report"""
    query: str
    timestamp: datetime
    results: List[DetailedBenchmarkResult]

    # Summary
    fastest_config: str
    best_compression_config: Optional[str] = None
    most_graph_usage_config: str

    # Recommendations
    recommendations: Dict[str, str]


@router.post("/detailed", response_model=DetailedComparisonReport)
async def benchmark_detailed(request: DetailedBenchmarkRequest):
    """
    Comprehensive benchmark with detailed metrics:

    - TTFT (Time To First Token) for streaming response latency
    - Per-chunk source breakdown (vector vs graph, hop distance)
    - RefRAG compression comparison (with vs without)
    - Multiple retrieval mode comparison

    This is the recommended endpoint for thorough performance analysis.
    """
    logger.info(f"Detailed benchmark: {request.query[:50]}...")

    # Default to key modes if not specified
    modes_to_test = request.retrieval_modes or ["naive", "local", "hybrid"]

    results = []

    for mode_str in modes_to_test:
        # Run with RefRAG
        if request.compare_refrag:
            result_with_refrag = await _run_detailed_benchmark(
                query=request.query,
                mode_str=mode_str,
                use_refrag=True,
                measure_ttft=request.measure_ttft,
                top_k=request.top_k
            )
            results.append(result_with_refrag)

            # Run without RefRAG
            result_without_refrag = await _run_detailed_benchmark(
                query=request.query,
                mode_str=mode_str,
                use_refrag=False,
                measure_ttft=request.measure_ttft,
                top_k=request.top_k
            )
            results.append(result_without_refrag)
        else:
            result = await _run_detailed_benchmark(
                query=request.query,
                mode_str=mode_str,
                use_refrag=False,
                measure_ttft=request.measure_ttft,
                top_k=request.top_k
            )
            results.append(result)

    # Analyze results
    fastest = min(results, key=lambda r: r.timing.total_ms)
    most_graph = max(results, key=lambda r: r.retrieval_stats.graph_total)

    best_compression = None
    compressed_results = [r for r in results if r.compression.enabled]
    if compressed_results:
        best_compression = min(compressed_results, key=lambda r: r.compression.compression_ratio)

    # Generate recommendations
    recommendations = {}

    # Speed recommendation
    recommendations["speed"] = f"{fastest.config_name} ({fastest.retrieval_mode}) is fastest at {fastest.timing.total_ms:.0f}ms"

    # Graph usage recommendation
    if most_graph.retrieval_stats.graph_total > 0:
        recommendations["graph_usage"] = (
            f"{most_graph.config_name} uses most graph data: "
            f"{most_graph.retrieval_stats.graph_total} graph chunks "
            f"({most_graph.retrieval_stats.graph_1hop_chunks} 1-hop, {most_graph.retrieval_stats.graph_2hop_chunks} 2-hop)"
        )

    # RefRAG recommendation
    if best_compression:
        savings = (1 - best_compression.compression.compression_ratio) * 100
        recommendations["compression"] = (
            f"RefRAG achieves {savings:.1f}% compression "
            f"({best_compression.compression.original_length} → {best_compression.compression.compressed_length} chars)"
        )

    # TTFT recommendation
    ttft_results = [r for r in results if r.timing.ttft_ms is not None]
    if ttft_results:
        fastest_ttft = min(ttft_results, key=lambda r: r.timing.ttft_ms)
        recommendations["ttft"] = f"Fastest TTFT: {fastest_ttft.config_name} at {fastest_ttft.timing.ttft_ms:.0f}ms"

    return DetailedComparisonReport(
        query=request.query,
        timestamp=datetime.now(),
        results=results,
        fastest_config=fastest.config_name,
        best_compression_config=best_compression.config_name if best_compression else None,
        most_graph_usage_config=most_graph.config_name,
        recommendations=recommendations
    )


async def _run_detailed_benchmark(
    query: str,
    mode_str: str,
    use_refrag: bool,
    measure_ttft: bool,
    top_k: int
) -> DetailedBenchmarkResult:
    """Run a single detailed benchmark"""
    start_time = time.time()

    config_name = f"{mode_str.upper()}" + (" + RefRAG" if use_refrag else "")

    try:
        # Parse mode
        mode = None
        if mode_str and mode_str != "auto":
            try:
                mode = RetrievalMode(mode_str)
            except ValueError:
                mode = RetrievalMode.NAIVE

        # 1. Retrieval
        retrieval_start = time.time()
        response = retrieval_engine.retrieve(query, mode=mode, top_k=top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000

        # 2. Calculate retrieval stats
        vector_count = 0
        graph_1hop = 0
        graph_2hop = 0
        entities_used = set()

        for r in response.results:
            hop = r.hop_distance if hasattr(r, 'hop_distance') else 0
            if hop == 0:
                vector_count += 1
            elif hop == 1:
                graph_1hop += 1
            else:
                graph_2hop += 1

            if hasattr(r, 'via_entity') and r.via_entity:
                entities_used.add(r.via_entity)

        retrieval_stats = RetrievalStats(
            total_chunks=len(response.results),
            vector_chunks=vector_count,
            graph_1hop_chunks=graph_1hop,
            graph_2hop_chunks=graph_2hop,
            graph_total=graph_1hop + graph_2hop,
            entities_used=list(entities_used)[:10]
        )

        # 3. Optional RefRAG compression
        compression_time = 0
        if use_refrag and response.results:
            compression_start = time.time()
            compression_input = [{"text": r.text, "chunk_id": r.chunk_id} for r in response.results]
            compression_result = refrag_compressor.compress(compression_input, query_context=query)
            compression_time = (time.time() - compression_start) * 1000

            compression = CompressionMetrics(
                enabled=True,
                original_length=compression_result["original_length"],
                compressed_length=compression_result["compressed_length"],
                compression_ratio=round(compression_result["compression_ratio"], 3),
                chunks_compressed=compression_result["chunks_compressed"],
                strategy=compression_result.get("strategy", "hybrid")
            )
        else:
            total_len = sum(len(r.text) for r in response.results) if response.results else 0
            compression = CompressionMetrics(
                enabled=False,
                original_length=total_len,
                compressed_length=total_len,
                compression_ratio=1.0
            )

        # 4. Build context and prompt
        context = [
            {"text": r.text, "document_id": r.document_id, "chunk_id": r.chunk_id}
            for r in response.results
        ]

        prompt = prompt_manager.create_qa_prompt(question=query, context=context)

        # 5. LLM generation with optional TTFT measurement
        generation_start = time.time()
        ttft_ms = None
        answer = ""

        try:
            timeout_config = httpx.Timeout(timeout=60.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                if measure_ttft:
                    # Use streaming to measure TTFT
                    async with client.stream(
                        "POST",
                        "http://tgi:80/generate_stream",
                        json={
                            "inputs": f"{prompt.system_message}\n\n{prompt.user_message}",
                            "parameters": {
                                "max_new_tokens": 200,
                                "temperature": 0.3,
                                "do_sample": True,
                                "top_p": 0.9,
                            }
                        }
                    ) as stream_response:
                        first_token_received = False
                        async for line in stream_response.aiter_lines():
                            if line.startswith("data:"):
                                if not first_token_received:
                                    ttft_ms = (time.time() - generation_start) * 1000
                                    first_token_received = True
                                try:
                                    import json
                                    data = json.loads(line[5:])
                                    token = data.get("token", {}).get("text", "")
                                    answer += token
                                except:
                                    pass
                else:
                    # Non-streaming for faster benchmark
                    llm_response = await client.post(
                        "http://tgi:80/generate",
                        json={
                            "inputs": f"{prompt.system_message}\n\n{prompt.user_message}",
                            "parameters": {
                                "max_new_tokens": 200,
                                "temperature": 0.3,
                                "do_sample": True,
                                "top_p": 0.9,
                                "return_full_text": False
                            }
                        }
                    )
                    llm_result = llm_response.json()
                    answer = llm_result.get("generated_text", "").strip()

        except Exception as e:
            logger.warning(f"LLM generation error in benchmark: {e}")
            answer = f"[Generation error: {str(e)[:50]}]"

        generation_time = (time.time() - generation_start) * 1000
        total_time = (time.time() - start_time) * 1000

        return DetailedBenchmarkResult(
            config_name=config_name,
            retrieval_mode=mode_str,
            use_refrag=use_refrag,
            timing=TimingMetrics(
                total_ms=round(total_time, 1),
                retrieval_ms=round(retrieval_time, 1),
                compression_ms=round(compression_time, 1) if use_refrag else None,
                generation_ms=round(generation_time, 1),
                ttft_ms=round(ttft_ms, 1) if ttft_ms else None
            ),
            retrieval_stats=retrieval_stats,
            compression=compression,
            response=answer[:500],  # Truncate for response size
            response_length=len(answer)
        )

    except Exception as e:
        logger.error(f"Detailed benchmark failed for {config_name}: {e}")
        # Return error result
        return DetailedBenchmarkResult(
            config_name=config_name,
            retrieval_mode=mode_str,
            use_refrag=use_refrag,
            timing=TimingMetrics(
                total_ms=(time.time() - start_time) * 1000,
                retrieval_ms=0,
                generation_ms=0
            ),
            retrieval_stats=RetrievalStats(
                total_chunks=0, vector_chunks=0, graph_1hop_chunks=0,
                graph_2hop_chunks=0, graph_total=0, entities_used=[]
            ),
            compression=CompressionMetrics(
                enabled=False, original_length=0, compressed_length=0, compression_ratio=1.0
            ),
            response=f"[Error: {str(e)[:100]}]",
            response_length=0
        )


@router.post("/refrag-comparison")
async def benchmark_refrag_comparison(query: str, retrieval_mode: str = "hybrid", top_k: int = 5):
    """
    Quick RefRAG comparison: same query with and without compression

    Returns side-by-side comparison showing compression benefits
    """
    logger.info(f"RefRAG comparison: {query[:50]}...")

    # Run without RefRAG
    without_refrag = await _run_detailed_benchmark(
        query=query,
        mode_str=retrieval_mode,
        use_refrag=False,
        measure_ttft=False,
        top_k=top_k
    )

    # Run with RefRAG
    with_refrag = await _run_detailed_benchmark(
        query=query,
        mode_str=retrieval_mode,
        use_refrag=True,
        measure_ttft=False,
        top_k=top_k
    )

    # Calculate improvements
    time_diff = without_refrag.timing.total_ms - with_refrag.timing.total_ms
    compression_savings = (1 - with_refrag.compression.compression_ratio) * 100 if with_refrag.compression.enabled else 0

    return {
        "query": query,
        "retrieval_mode": retrieval_mode,
        "without_refrag": {
            "total_time_ms": without_refrag.timing.total_ms,
            "context_length": without_refrag.compression.original_length,
            "response_preview": without_refrag.response[:200]
        },
        "with_refrag": {
            "total_time_ms": with_refrag.timing.total_ms,
            "original_length": with_refrag.compression.original_length,
            "compressed_length": with_refrag.compression.compressed_length,
            "compression_ratio": with_refrag.compression.compression_ratio,
            "response_preview": with_refrag.response[:200]
        },
        "comparison": {
            "time_saved_ms": round(time_diff, 1),
            "compression_savings_pct": round(compression_savings, 1),
            "recommendation": "RefRAG recommended" if compression_savings > 20 else "Minimal benefit from RefRAG"
        }
    }


# =============================================================================
# RAGAS EVALUATION SYSTEM
# Ground-truth based evaluation with semantic similarity scoring
# =============================================================================

import re
from typing import Set

# Import test cases from evaluation module
import sys
sys.path.insert(0, "/app/tools")
try:
    from evaluation_test_cases import TEST_CASES, TestCase, compute_semantic_overlap, tokenize_multilingual
    RAGAS_AVAILABLE = True
except ImportError:
    TEST_CASES = []
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS evaluation not available - evaluation_test_cases.py not found")


class RagasEvaluationRequest(BaseModel):
    """Request for RAGAS evaluation"""
    test_case_ids: List[str] = []  # Empty = all test cases
    modes: List[str] = ["naive", "local", "global", "hybrid", "mix"]
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
    # REFRAG comparison (optional)
    refrag_scores: Optional[RagasScores] = None
    refrag_answer: Optional[str] = None
    refrag_timing_ms: Optional[float] = None
    ground_truth_delta: Optional[float] = None  # Negative = REFRAG hurt quality
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
    mode_scores: Dict[str, Dict[str, Any]]  # mode -> {avg_score, passed, failed}
    best_overall_mode: str
    best_overall_score: float


class RagasEvaluationResponse(BaseModel):
    """Full RAGAS evaluation response"""
    test_results: List[RagasTestResult]
    summary: RagasEvaluationSummary
    refrag_impact: Optional[RefragImpactAnalysis] = None
    timestamp: str
    duration_ms: int


@router.get("/ragas/test-cases")
async def get_ragas_test_cases():
    """
    Get available RAGAS test cases with metadata.

    Returns list of test cases that can be used for evaluation.
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS evaluation not available", "test_cases": []}

    test_cases = []
    for tc in TEST_CASES:
        test_cases.append({
            "id": tc.id,
            "query": tc.query,
            "query_type": tc.query_type.value,
            "description": tc.description,
            "expected_entities": tc.expected_entities,
            "expected_answer_preview": tc.expected_answer[:200] + "..." if len(tc.expected_answer) > 200 else tc.expected_answer,
            "best_modes": tc.best_modes,
            "difficulty": "hard" if "MULTIHOP" in tc.id or "OVERVIEW" in tc.id else "medium" if "ENTITY" in tc.id else "easy"
        })

    return {
        "test_cases": test_cases,
        "total": len(test_cases)
    }


@router.post("/ragas", response_model=RagasEvaluationResponse)
async def run_ragas_evaluation(request: RagasEvaluationRequest):
    """
    Run RAGAS-style evaluation with ground truth comparison.

    This endpoint:
    1. Tests specified queries against ground truth answers
    2. Runs each query across all specified modes
    3. Optionally compares with/without REFRAG
    4. Returns detailed scoring metrics including semantic similarity

    RAGAS Metrics:
    - Ground Truth Similarity: Token overlap with expected answer (35% weight)
    - Answer Keywords: Key terms from expected answer (20% weight)
    - Chunk Content: How well chunks match query (20% weight)
    - Entity Match: Entities mentioned in answer (15% weight)
    - Chunk Count: Number of chunks retrieved (10% weight)
    """
    if not RAGAS_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAGAS evaluation not available")

    start_time = time.time()
    logger.info(f"Starting RAGAS evaluation: {len(request.test_case_ids) or 'all'} tests, modes={request.modes}")

    # Filter test cases
    if request.test_case_ids:
        test_cases = [tc for tc in TEST_CASES if tc.id in request.test_case_ids]
    else:
        test_cases = TEST_CASES

    if not test_cases:
        raise HTTPException(status_code=400, detail="No test cases found")

    test_results = []
    mode_scores = {mode: {"total_score": 0, "passed": 0, "failed": 0, "count": 0} for mode in request.modes}

    # REFRAG impact tracking
    refrag_deltas = []
    refrag_speedups = []
    refrag_savings = []
    refrag_improved = 0
    refrag_hurt = 0
    refrag_unchanged = 0

    for tc in test_cases:
        logger.info(f"Evaluating test case: {tc.id}")
        mode_results = []
        best_mode = None
        best_score = 0

        for mode in request.modes:
            try:
                # Run without REFRAG first
                result_no_refrag = await _run_ragas_single(
                    tc, mode, use_refrag=False, top_k=request.top_k
                )

                mode_result = RagasModeResult(
                    mode=mode,
                    answer=result_no_refrag["answer"],
                    scores=RagasScores(**result_no_refrag["scores"]),
                    timing_ms=result_no_refrag["timing_ms"],
                    chunks_used=result_no_refrag["chunks_used"],
                    entities_found=result_no_refrag["entities_found"]
                )

                # Run with REFRAG if requested
                if request.compare_refrag:
                    result_with_refrag = await _run_ragas_single(
                        tc, mode, use_refrag=True, top_k=request.top_k
                    )

                    mode_result.refrag_scores = RagasScores(**result_with_refrag["scores"])
                    mode_result.refrag_answer = result_with_refrag["answer"]
                    mode_result.refrag_timing_ms = result_with_refrag["timing_ms"]

                    # Calculate deltas
                    gt_delta = (result_with_refrag["scores"]["ground_truth_similarity"] -
                               result_no_refrag["scores"]["ground_truth_similarity"])
                    mode_result.ground_truth_delta = round(gt_delta, 3)

                    if result_with_refrag["timing_ms"] > 0:
                        mode_result.speed_improvement = round(
                            result_no_refrag["timing_ms"] / result_with_refrag["timing_ms"], 2
                        )

                    mode_result.token_savings = result_with_refrag.get("token_savings", 0)

                    # Track REFRAG impact
                    refrag_deltas.append(gt_delta)
                    if mode_result.speed_improvement:
                        refrag_speedups.append(mode_result.speed_improvement)
                    if mode_result.token_savings:
                        refrag_savings.append(mode_result.token_savings)

                    if gt_delta > 0.02:
                        refrag_improved += 1
                    elif gt_delta < -0.02:
                        refrag_hurt += 1
                    else:
                        refrag_unchanged += 1

                # Track mode scores
                score = result_no_refrag["scores"]["weighted_score"]
                mode_scores[mode]["total_score"] += score
                mode_scores[mode]["count"] += 1
                if result_no_refrag["scores"]["passed"]:
                    mode_scores[mode]["passed"] += 1
                else:
                    mode_scores[mode]["failed"] += 1

                if score > best_score:
                    best_score = score
                    best_mode = mode

                mode_results.append(mode_result)

            except Exception as e:
                logger.error(f"Error evaluating {tc.id} with mode {mode}: {e}")
                continue

        # Determine difficulty
        difficulty = "hard" if "MULTIHOP" in tc.id or "OVERVIEW" in tc.id else "medium" if "ENTITY" in tc.id else "easy"

        test_results.append(RagasTestResult(
            test_id=tc.id,
            query=tc.query,
            expected_answer=tc.expected_answer[:500],
            difficulty=difficulty,
            mode_results=mode_results,
            best_mode=best_mode or "unknown",
            best_score=round(best_score, 3)
        ))

    # Calculate summary
    for mode in request.modes:
        if mode_scores[mode]["count"] > 0:
            mode_scores[mode]["avg_score"] = round(
                mode_scores[mode]["total_score"] / mode_scores[mode]["count"], 3
            )
        else:
            mode_scores[mode]["avg_score"] = 0
        del mode_scores[mode]["total_score"]
        del mode_scores[mode]["count"]

    # Find best overall mode
    best_overall_mode = max(mode_scores.keys(), key=lambda m: mode_scores[m]["avg_score"])
    best_overall_score = mode_scores[best_overall_mode]["avg_score"]

    # Build REFRAG impact analysis
    refrag_impact = None
    if request.compare_refrag and refrag_deltas:
        refrag_impact = RefragImpactAnalysis(
            modes_improved=refrag_improved,
            modes_hurt=refrag_hurt,
            modes_unchanged=refrag_unchanged,
            avg_ground_truth_delta=round(sum(refrag_deltas) / len(refrag_deltas), 3) if refrag_deltas else 0,
            avg_speed_improvement=round(sum(refrag_speedups) / len(refrag_speedups), 2) if refrag_speedups else 1.0,
            avg_token_savings=round(sum(refrag_savings) / len(refrag_savings), 1) if refrag_savings else 0
        )

    duration_ms = int((time.time() - start_time) * 1000)

    return RagasEvaluationResponse(
        test_results=test_results,
        summary=RagasEvaluationSummary(
            total_tests=len(test_results),
            modes_tested=request.modes,
            mode_scores=mode_scores,
            best_overall_mode=best_overall_mode,
            best_overall_score=best_overall_score
        ),
        refrag_impact=refrag_impact,
        timestamp=datetime.now().isoformat(),
        duration_ms=duration_ms
    )


async def _run_ragas_single(
    test_case: "TestCase",
    mode_str: str,
    use_refrag: bool,
    top_k: int
) -> Dict[str, Any]:
    """
    Run a single RAGAS evaluation for a test case and mode.

    Returns dict with answer, scores, timing, etc.
    """
    import httpx

    start_time = time.time()

    try:
        # Parse mode
        mode = None
        if mode_str and mode_str != "auto":
            try:
                mode = RetrievalMode(mode_str)
            except ValueError:
                mode = RetrievalMode.NAIVE

        # 1. Retrieval
        retrieval_start = time.time()
        response = retrieval_engine.retrieve(test_case.query, mode=mode, top_k=top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000

        chunks = response.results

        # Extract entities
        entities_found = set()
        for r in chunks:
            if hasattr(r, 'via_entity') and r.via_entity:
                entities_found.add(r.via_entity)

        # 2. Optional REFRAG compression
        token_savings = 0
        if use_refrag and chunks:
            compression_input = [{"text": r.text, "chunk_id": r.chunk_id} for r in chunks]
            compression_result = refrag_compressor.compress(compression_input, query_context=test_case.query)
            original_len = compression_result["original_length"]
            compressed_len = compression_result["compressed_length"]
            if original_len > 0:
                token_savings = round((1 - compressed_len / original_len) * 100, 1)

        # 3. Build context and prompt
        context = [
            {"text": r.text, "document_id": r.document_id, "chunk_id": r.chunk_id}
            for r in chunks
        ]
        prompt = prompt_manager.create_qa_prompt(question=test_case.query, context=context)

        # 4. LLM generation
        generation_start = time.time()
        answer = ""

        try:
            timeout_config = httpx.Timeout(timeout=60.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                llm_response = await client.post(
                    "http://tgi:80/generate",
                    json={
                        "inputs": f"{prompt.system_message}\n\n{prompt.user_message}",
                        "parameters": {
                            "max_new_tokens": 300,
                            "temperature": 0.3,
                            "do_sample": True,
                            "top_p": 0.9,
                            "return_full_text": False
                        }
                    }
                )
                llm_result = llm_response.json()
                answer = llm_result.get("generated_text", "").strip()
        except Exception as e:
            logger.warning(f"LLM generation error: {e}")
            answer = f"[Generation error]"

        generation_time = (time.time() - generation_start) * 1000
        total_time = (time.time() - start_time) * 1000

        # 5. Calculate RAGAS scores
        scores = _calculate_ragas_scores(test_case, chunks, answer, list(entities_found))

        return {
            "answer": answer[:1000],
            "scores": scores,
            "timing_ms": round(total_time, 1),
            "chunks_used": len(chunks),
            "entities_found": list(entities_found)[:10],
            "token_savings": token_savings
        }

    except Exception as e:
        logger.error(f"RAGAS single evaluation error: {e}")
        return {
            "answer": f"[Error: {str(e)[:100]}]",
            "scores": {
                "chunk_count": 0,
                "entity_match": 0,
                "chunk_content": 0,
                "answer_keywords": 0,
                "ground_truth_similarity": 0,
                "weighted_score": 0,
                "passed": False
            },
            "timing_ms": 0,
            "chunks_used": 0,
            "entities_found": [],
            "token_savings": 0
        }


def _calculate_ragas_scores(test_case: "TestCase", chunks: List, answer: str, entities_found: List[str]) -> Dict[str, Any]:
    """
    Calculate RAGAS evaluation scores.

    Scoring weights:
    - ground_truth_similarity: 35% (most important - actual answer quality)
    - answer_keywords: 20%
    - chunk_content: 20%
    - entity_match: 15%
    - chunk_count: 10%
    """
    # Check 1: Chunk count
    chunk_count = len(chunks)
    chunk_count_score = 1.0 if chunk_count >= test_case.min_chunk_count else 0.0

    # Check 2: Entity match
    entity_match_score = 0.0
    if test_case.expected_entities:
        found_entities = set()
        entity_names_lower = [e.lower() for e in entities_found]
        for expected in test_case.expected_entities:
            expected_lower = expected.lower()
            if any(expected_lower in name or name in expected_lower for name in entity_names_lower):
                found_entities.add(expected)
        entity_match_score = len(found_entities) / len(test_case.expected_entities)
    else:
        entity_match_score = 1.0

    # Check 3: Chunk content match
    chunk_content_score = 0.0
    if test_case.expected_chunks:
        all_chunk_text = " ".join([getattr(c, 'text', '') for c in chunks]).lower()
        matches = 0
        for exp_chunk in test_case.expected_chunks:
            if all(pattern.lower() in all_chunk_text for pattern in exp_chunk.content_contains):
                matches += 1
        chunk_content_score = matches / len(test_case.expected_chunks)
    else:
        chunk_content_score = 1.0

    # Check 4: Answer keywords
    answer_keywords_score = 0.0
    if test_case.expected_answer_contains:
        answer_lower = answer.lower()
        keywords_found = sum(1 for kw in test_case.expected_answer_contains if kw.lower() in answer_lower)
        answer_keywords_score = keywords_found / len(test_case.expected_answer_contains)
    else:
        answer_keywords_score = 1.0

    # Check 5: Ground truth semantic similarity (most important!)
    ground_truth_score = 0.0
    if test_case.expected_answer and RAGAS_AVAILABLE:
        ground_truth_score = compute_semantic_overlap(answer, test_case.expected_answer)

    # Calculate weighted score
    weights = {
        "chunk_count": 0.10,
        "entity_match": 0.15,
        "chunk_content": 0.20,
        "answer_keywords": 0.20,
        "ground_truth_similarity": 0.35
    }

    weighted_score = (
        weights["chunk_count"] * chunk_count_score +
        weights["entity_match"] * entity_match_score +
        weights["chunk_content"] * chunk_content_score +
        weights["answer_keywords"] * answer_keywords_score +
        weights["ground_truth_similarity"] * (1.0 if ground_truth_score >= 0.3 else 0.0)
    )

    passed = weighted_score >= 0.6

    return {
        "chunk_count": round(chunk_count_score, 3),
        "entity_match": round(entity_match_score, 3),
        "chunk_content": round(chunk_content_score, 3),
        "answer_keywords": round(answer_keywords_score, 3),
        "ground_truth_similarity": round(ground_truth_score, 3),
        "weighted_score": round(weighted_score, 3),
        "passed": passed
    }
