"""
Benchmark Service API
Compare different RAG configurations (RAG, GraphRAG, REFRAG combinations)
Measure speed, relevancy, and quality metrics
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from loguru import logger
import time
from datetime import datetime

from ..core.orchestration import RAGWorkflow

router = APIRouter()

# Initialize RAG workflow
rag_workflow = RAGWorkflow()


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
                "generation_time_ms"
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
                "entities_found"
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
            "/benchmark/summary": "This endpoint - shows available options"
        }
    }
