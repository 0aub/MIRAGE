"""
Benchmark Service - API Routes
FastAPI endpoints for benchmarking RAG configurations
"""

from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime
from loguru import logger
import time

from ...core.orchestration import RAGWorkflow

from .models import (
    BenchmarkQuery,
    BenchmarkConfig,
    BenchmarkResult,
    ComparisonReport,
    DetailedBenchmarkRequest,
    DetailedComparisonReport,
    RagasEvaluationRequest,
    RagasEvaluationResponse,
    RagasTestResult,
    RagasModeResult,
    RagasScores,
    RagasEvaluationSummary,
    RefragImpactAnalysis,
)
from .runners import run_detailed_benchmark
from . import ragas

router = APIRouter()

# Initialize workflow
rag_workflow = RAGWorkflow()


@router.post("/single", response_model=BenchmarkResult)
async def benchmark_single_config(query: BenchmarkQuery, config: BenchmarkConfig):
    """Benchmark a single RAG configuration"""
    logger.info(f"Benchmarking '{config.name}': query='{query.query[:50]}...'")

    start_time = time.time()

    try:
        result = rag_workflow.invoke(
            query=query.query,
            use_graph=config.use_graph,
            use_refrag=config.use_refrag,
        )

        total_time = (time.time() - start_time) * 1000
        metadata = result.get("metadata", {})
        latency = metadata.get("latency_ms", {})
        compression_stats = metadata.get("compression_stats", {})

        benchmark_result = BenchmarkResult(
            config_name=config.name,
            query=query.query,
            total_time_ms=int(total_time),
            retrieval_time_ms=int(latency.get("graph_retrieval", 0)),
            compression_time_ms=int(latency.get("compression", 0)) if config.use_refrag else None,
            generation_time_ms=int(latency.get("generation", 0)),
            original_context_length=compression_stats.get("original_length", 0),
            compressed_context_length=compression_stats.get("compressed_length") if config.use_refrag else None,
            compression_ratio=compression_stats.get("compression_ratio") if config.use_refrag else None,
            speedup_factor=compression_stats.get("speedup_factor") if config.use_refrag else None,
            nodes_retrieved=metadata.get("nodes_retrieved", 0),
            chunks_retrieved=len(result.get("retrieved_chunks", [])),
            entities_found=[e.get("name", "") for e in result.get("entities_found", [])[:10]],
            response=result.get("response", ""),
            response_length=len(result.get("response", "")),
            citations_count=len(result.get("citations", [])),
        )

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

        logger.info(f"Benchmark complete: {config.name} took {total_time:.0f}ms")
        return benchmark_result

    except Exception as e:
        logger.error(f"Benchmark failed for {config.name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=ComparisonReport)
async def benchmark_compare_all(query: BenchmarkQuery):
    """Compare all RAG configurations for a single query"""
    logger.info(f"Running full comparison benchmark for: {query.query[:50]}...")

    configs = [
        BenchmarkConfig(use_graph=False, use_refrag=False, name="Normal RAG"),
        BenchmarkConfig(use_graph=False, use_refrag=True, name="RAG + REFRAG"),
        BenchmarkConfig(use_graph=True, use_refrag=False, name="GraphRAG"),
        BenchmarkConfig(use_graph=True, use_refrag=True, name="GraphRAG + REFRAG"),
    ]

    results = []
    for config in configs:
        try:
            result = await benchmark_single_config(query, config)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to benchmark {config.name}: {e}")

    if not results:
        raise HTTPException(status_code=500, detail="All benchmarks failed")

    fastest = min(results, key=lambda r: r.total_time_ms)
    most_compressed = min(
        [r for r in results if r.compression_ratio is not None],
        key=lambda r: r.compression_ratio,
        default=None
    )
    most_entities = max(results, key=lambda r: r.nodes_retrieved)

    recommendations = {}
    if fastest.config_name == "Normal RAG":
        recommendations["speed"] = "Normal RAG is fastest but may have lower accuracy"
    elif fastest.config_name == "GraphRAG + REFRAG":
        recommendations["speed"] = "GraphRAG + REFRAG provides best overall speed with high quality"
    else:
        recommendations["speed"] = f"{fastest.config_name} is fastest at {fastest.total_time_ms}ms"

    if most_entities.config_name.startswith("GraphRAG"):
        recommendations["quality"] = "GraphRAG configurations retrieve more structured knowledge"

    if most_compressed:
        speedup = most_compressed.speedup_factor or 1.0
        recommendations["efficiency"] = (
            f"{most_compressed.config_name} achieves {speedup:.1f}x context compression"
        )

    graphrag_refrag = next((r for r in results if r.config_name == "GraphRAG + REFRAG"), None)
    if graphrag_refrag:
        recommendations["overall"] = (
            f"GraphRAG + REFRAG recommended: {graphrag_refrag.total_time_ms}ms, "
            f"{graphrag_refrag.nodes_retrieved} entities"
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
    """Run comparison benchmarks for multiple queries"""
    if len(queries) > max_queries:
        raise HTTPException(status_code=400, detail=f"Too many queries. Maximum is {max_queries}")

    logger.info(f"Running batch benchmark for {len(queries)} queries")

    reports = []
    for i, query in enumerate(queries, 1):
        logger.info(f"Processing query {i}/{len(queries)}")
        try:
            report = await benchmark_compare_all(query)
            reports.append(report)
        except Exception as e:
            logger.error(f"Failed to benchmark query {i}: {e}")

    return reports


@router.get("/summary")
async def get_benchmark_summary():
    """Get summary of available benchmark configurations and metrics"""
    return {
        "available_configurations": [
            {"name": "Normal RAG", "use_graph": False, "use_refrag": False},
            {"name": "RAG + REFRAG", "use_graph": False, "use_refrag": True},
            {"name": "GraphRAG", "use_graph": True, "use_refrag": False},
            {"name": "GraphRAG + REFRAG", "use_graph": True, "use_refrag": True},
        ],
        "endpoints": {
            "/benchmark/single": "Benchmark a single configuration",
            "/benchmark/compare": "Compare all 4 configurations",
            "/benchmark/batch": "Compare across multiple queries",
            "/benchmark/detailed": "Detailed benchmark with TTFT",
            "/benchmark/ragas": "RAGAS evaluation system",
        }
    }


@router.post("/detailed", response_model=DetailedComparisonReport)
async def benchmark_detailed(request: DetailedBenchmarkRequest):
    """Comprehensive benchmark with TTFT, streaming metrics, RefRAG comparison"""
    logger.info(f"Detailed benchmark: {request.query[:50]}...")

    modes_to_test = request.retrieval_modes or ["vector", "local", "hybrid"]
    results = []

    for mode_str in modes_to_test:
        if request.compare_refrag:
            result_with = await run_detailed_benchmark(
                request.query, mode_str, True, request.measure_ttft, request.top_k
            )
            results.append(result_with)

            result_without = await run_detailed_benchmark(
                request.query, mode_str, False, request.measure_ttft, request.top_k
            )
            results.append(result_without)
        else:
            result = await run_detailed_benchmark(
                request.query, mode_str, False, request.measure_ttft, request.top_k
            )
            results.append(result)

    fastest = min(results, key=lambda r: r.timing.total_ms)
    most_graph = max(results, key=lambda r: r.retrieval_stats.graph_total)

    best_compression = None
    compressed = [r for r in results if r.compression.enabled]
    if compressed:
        best_compression = min(compressed, key=lambda r: r.compression.compression_ratio)

    recommendations = {
        "speed": f"{fastest.config_name} is fastest at {fastest.timing.total_ms:.0f}ms"
    }

    if most_graph.retrieval_stats.graph_total > 0:
        recommendations["graph_usage"] = (
            f"{most_graph.config_name} uses {most_graph.retrieval_stats.graph_total} graph chunks"
        )

    if best_compression:
        savings = (1 - best_compression.compression.compression_ratio) * 100
        recommendations["compression"] = f"RefRAG achieves {savings:.1f}% compression"

    return DetailedComparisonReport(
        query=request.query,
        timestamp=datetime.now(),
        results=results,
        fastest_config=fastest.config_name,
        best_compression_config=best_compression.config_name if best_compression else None,
        most_graph_usage_config=most_graph.config_name,
        recommendations=recommendations
    )


@router.post("/refrag-comparison")
async def benchmark_refrag_comparison(query: str, retrieval_mode: str = "hybrid", top_k: int = 5):
    """Quick RefRAG comparison: same query with and without compression"""
    logger.info(f"RefRAG comparison: {query[:50]}...")

    without = await run_detailed_benchmark(query, retrieval_mode, False, False, top_k)
    with_refrag = await run_detailed_benchmark(query, retrieval_mode, True, False, top_k)

    time_diff = without.timing.total_ms - with_refrag.timing.total_ms
    savings = (1 - with_refrag.compression.compression_ratio) * 100 if with_refrag.compression.enabled else 0

    return {
        "query": query,
        "retrieval_mode": retrieval_mode,
        "without_refrag": {
            "total_time_ms": without.timing.total_ms,
            "context_length": without.compression.original_length,
        },
        "with_refrag": {
            "total_time_ms": with_refrag.timing.total_ms,
            "compression_ratio": with_refrag.compression.compression_ratio,
        },
        "comparison": {
            "time_saved_ms": round(time_diff, 1),
            "compression_savings_pct": round(savings, 1),
        }
    }


# =============================================================================
# RAGAS ENDPOINTS
# =============================================================================

@router.get("/ragas/test-cases")
async def get_ragas_test_cases():
    """Get available RAGAS test cases"""
    if not ragas.is_available():
        return {"error": "RAGAS evaluation not available", "test_cases": []}

    test_cases = []
    for tc in ragas.get_test_cases():
        test_cases.append({
            "id": tc.id,
            "query": tc.query,
            "query_type": tc.query_type.value,
            "description": tc.description,
            "expected_entities": tc.expected_entities,
            "best_modes": tc.best_modes,
            "difficulty": "hard" if "MULTIHOP" in tc.id else "medium" if "ENTITY" in tc.id else "easy"
        })

    return {"test_cases": test_cases, "total": len(test_cases)}


@router.post("/ragas", response_model=RagasEvaluationResponse)
async def run_ragas_evaluation(request: RagasEvaluationRequest):
    """Run RAGAS-style evaluation with ground truth comparison"""
    if not ragas.is_available():
        raise HTTPException(status_code=503, detail="RAGAS evaluation not available")

    start_time = time.time()
    logger.info(f"Starting RAGAS evaluation: modes={request.modes}")

    all_test_cases = ragas.get_test_cases()
    if request.test_case_ids:
        test_cases = [tc for tc in all_test_cases if tc.id in request.test_case_ids]
    else:
        test_cases = all_test_cases

    if not test_cases:
        raise HTTPException(status_code=400, detail="No test cases found")

    test_results = []
    mode_scores = {mode: {"total_score": 0, "passed": 0, "failed": 0, "count": 0} for mode in request.modes}

    refrag_deltas, refrag_speedups, refrag_savings = [], [], []
    refrag_improved, refrag_hurt, refrag_unchanged = 0, 0, 0

    for tc in test_cases:
        logger.info(f"Evaluating: {tc.id}")
        mode_results = []
        best_mode, best_score = None, 0

        for mode in request.modes:
            try:
                result_no_refrag = await ragas.run_ragas_single(tc, mode, False, request.top_k)

                mode_result = RagasModeResult(
                    mode=mode,
                    answer=result_no_refrag["answer"],
                    scores=RagasScores(**result_no_refrag["scores"]),
                    timing_ms=result_no_refrag["timing_ms"],
                    chunks_used=result_no_refrag["chunks_used"],
                    entities_found=result_no_refrag["entities_found"]
                )

                if request.compare_refrag:
                    result_with_refrag = await ragas.run_ragas_single(tc, mode, True, request.top_k)
                    mode_result.refrag_scores = RagasScores(**result_with_refrag["scores"])
                    mode_result.refrag_answer = result_with_refrag["answer"]
                    mode_result.refrag_timing_ms = result_with_refrag["timing_ms"]

                    gt_delta = (result_with_refrag["scores"]["ground_truth_similarity"] -
                               result_no_refrag["scores"]["ground_truth_similarity"])
                    mode_result.ground_truth_delta = round(gt_delta, 3)

                    if result_with_refrag["timing_ms"] > 0:
                        mode_result.speed_improvement = round(
                            result_no_refrag["timing_ms"] / result_with_refrag["timing_ms"], 2
                        )
                    mode_result.token_savings = result_with_refrag.get("token_savings", 0)

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
                logger.error(f"Error evaluating {tc.id} with {mode}: {e}")

        difficulty = "hard" if "MULTIHOP" in tc.id else "medium" if "ENTITY" in tc.id else "easy"

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

    best_mode = max(mode_scores.keys(), key=lambda m: mode_scores[m]["avg_score"])
    best_score = mode_scores[best_mode]["avg_score"]

    refrag_impact = None
    if request.compare_refrag and refrag_deltas:
        refrag_impact = RefragImpactAnalysis(
            modes_improved=refrag_improved,
            modes_hurt=refrag_hurt,
            modes_unchanged=refrag_unchanged,
            avg_ground_truth_delta=round(sum(refrag_deltas) / len(refrag_deltas), 3),
            avg_speed_improvement=round(sum(refrag_speedups) / len(refrag_speedups), 2) if refrag_speedups else 1.0,
            avg_token_savings=round(sum(refrag_savings) / len(refrag_savings), 1) if refrag_savings else 0
        )

    return RagasEvaluationResponse(
        test_results=test_results,
        summary=RagasEvaluationSummary(
            total_tests=len(test_results),
            modes_tested=request.modes,
            mode_scores=mode_scores,
            best_overall_mode=best_mode,
            best_overall_score=best_score
        ),
        refrag_impact=refrag_impact,
        timestamp=datetime.now().isoformat(),
        duration_ms=int((time.time() - start_time) * 1000)
    )
