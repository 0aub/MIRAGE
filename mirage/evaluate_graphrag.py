#!/usr/bin/env python3
"""
GraphRAG Enhancement Evaluation Script

Tests the following enhancements:
1. Local Search with semantic entity embeddings
2. Global Search with JSON mode map phase
3. DRIFT Search with iterative follow-up questions
4. Vector RAG (renamed from Naive)

Usage:
    # Inside Docker container:
    python evaluate_graphrag.py --api http://localhost:8000

    # With full test suite:
    python evaluate_graphrag.py --api http://localhost:8000 --full
"""

import argparse
import json
import time
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import requests

# Try to import internal modules (for Docker environment)
try:
    from src.core.retrieval.base_retriever import RetrievalMode
    from src.evaluation.rag_metrics import RAGEvaluator
    INTERNAL_AVAILABLE = True
except ImportError:
    INTERNAL_AVAILABLE = False

@dataclass
class TestCase:
    id: str
    query: str
    query_type: str
    language: str
    ground_truth: str
    difficulty: str
    expected_mode: str = "auto"

@dataclass
class EvalResult:
    test_id: str
    query: str
    mode: str
    success: bool
    latency_ms: float
    answer: str = ""
    chunks_count: int = 0
    entities_found: int = 0
    semantic_entities: int = 0
    faithfulness: float = 0.0
    relevancy: float = 0.0
    error: str = ""
    metadata: Dict = field(default_factory=dict)


def get_graphrag_test_cases() -> List[TestCase]:
    """Test cases specifically designed to evaluate GraphRAG enhancements."""
    return [
        # === Local Search Tests (Entity-focused) ===
        TestCase(
            id="LOCAL_001",
            query="من هي الجهة المسؤولة عن تنظيم البيانات في المملكة؟",
            query_type="entity",
            language="ar",
            ground_truth="مكتب إدارة البيانات الوطنية (NDMO) أو الهيئة السعودية للبيانات والذكاء الاصطناعي (SDAIA)",
            difficulty="L1",
            expected_mode="local"
        ),
        TestCase(
            id="LOCAL_002",
            query="ما هي سدايا؟",
            query_type="entity_acronym",
            language="ar",
            ground_truth="سدايا هي الهيئة السعودية للبيانات والذكاء الاصطناعي (SDAIA)",
            difficulty="L1",
            expected_mode="local"
        ),
        TestCase(
            id="LOCAL_003",
            query="Who is responsible for data classification in Saudi Arabia?",
            query_type="entity",
            language="en",
            ground_truth="NDMO (National Data Management Office) or SDAIA",
            difficulty="L1",
            expected_mode="local"
        ),

        # === Global Search Tests (Thematic/Holistic) ===
        TestCase(
            id="GLOBAL_001",
            query="ما هي المبادئ الرئيسية لحماية الخصوصية في السياسات السعودية؟",
            query_type="thematic",
            language="ar",
            ground_truth="المبادئ تشمل الشفافية، تقليل البيانات، قصر الغرض، الأمن، وحقوق صاحب البيانات",
            difficulty="L3",
            expected_mode="global"
        ),
        TestCase(
            id="GLOBAL_002",
            query="Summarize the key themes across all data governance policies.",
            query_type="thematic",
            language="en",
            ground_truth="Key themes include data protection, privacy, classification, and governance frameworks",
            difficulty="L3",
            expected_mode="global"
        ),

        # === DRIFT Search Tests (Complex multi-hop) ===
        TestCase(
            id="DRIFT_001",
            query="كيف تؤثر تصنيفات البيانات على شروط مشاركتها وما هي الجهات المسؤولة عن التنظيم؟",
            query_type="multi-hop",
            language="ar",
            ground_truth="البيانات العامة يمكن مشاركتها بحرية، بينما السرية تتطلب ضوابط. NDMO وSDAIA مسؤولان عن التنظيم",
            difficulty="L4",
            expected_mode="drift"
        ),
        TestCase(
            id="DRIFT_002",
            query="Explain the relationship between NDMO and government entities regarding data classification and how it affects citizen access.",
            query_type="multi-hop",
            language="en",
            ground_truth="NDMO sets classification standards, entities implement them, and citizens can access public data through FOI policy",
            difficulty="L4",
            expected_mode="drift"
        ),

        # === Vector/Hybrid Tests (General retrieval) ===
        TestCase(
            id="VECTOR_001",
            query="ما هو تعريف البيانات الشخصية؟",
            query_type="factual",
            language="ar",
            ground_truth="البيانات الشخصية هي أي معلومات تتعلق بشخص طبيعي محدد أو يمكن تحديده",
            difficulty="L1",
            expected_mode="vector"
        ),
        TestCase(
            id="HYBRID_001",
            query="ما الفرق بين البيانات الشخصية والبيانات الحساسة؟",
            query_type="comparative",
            language="ar",
            ground_truth="البيانات الشخصية هي أي معلومة عن الفرد، بينما البيانات الحساسة تتطلب حماية أعلى",
            difficulty="L2",
            expected_mode="hybrid"
        ),
    ]


def evaluate_query(
    api_base: str,
    test_case: TestCase,
    mode: str
) -> EvalResult:
    """Evaluate a single query against the API."""

    start_time = time.time()

    try:
        response = requests.post(
            f"{api_base}/chat/ask",
            json={
                "message": test_case.query,
                "retrieval_mode": mode,
                "top_k": 10
            },
            timeout=120
        )

        latency = (time.time() - start_time) * 1000

        if response.status_code != 200:
            return EvalResult(
                test_id=test_case.id,
                query=test_case.query,
                mode=mode,
                success=False,
                latency_ms=latency,
                error=f"HTTP {response.status_code}"
            )

        data = response.json()
        answer = data.get("answer", "")
        chunks = data.get("chunks", [])
        metadata = data.get("metadata", {})

        # Extract GraphRAG-specific metrics from metadata
        entities_found = metadata.get("entities_found", 0)
        semantic_entities = metadata.get("semantic_entities_found", 0)
        entity_descriptions = metadata.get("entity_descriptions", {})

        # Calculate simple metrics
        faithfulness = calculate_faithfulness(answer, chunks)
        relevancy = calculate_relevancy(test_case.query, answer, test_case.ground_truth)

        return EvalResult(
            test_id=test_case.id,
            query=test_case.query,
            mode=mode,
            success=True,
            latency_ms=round(latency, 1),
            answer=answer[:500] if answer else "",
            chunks_count=len(chunks),
            entities_found=entities_found,
            semantic_entities=semantic_entities,
            faithfulness=round(faithfulness, 3),
            relevancy=round(relevancy, 3),
            metadata={
                "entity_descriptions_count": len(entity_descriptions),
                "global_answer": metadata.get("global_answer", "")[:200] if metadata.get("global_answer") else "",
                "communities_searched": metadata.get("communities_searched", 0),
            }
        )

    except Exception as e:
        return EvalResult(
            test_id=test_case.id,
            query=test_case.query,
            mode=mode,
            success=False,
            latency_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )


def calculate_faithfulness(answer: str, chunks: List[Dict]) -> float:
    """Simple faithfulness: overlap between answer and context."""
    if not answer or not chunks:
        return 0.0

    answer_tokens = set(answer.lower().split())
    context_tokens = set()
    for chunk in chunks:
        text = chunk.get("text", "")
        context_tokens.update(text.lower().split())

    if not answer_tokens:
        return 0.0

    overlap = len(answer_tokens & context_tokens)
    return min(1.0, overlap / len(answer_tokens))


def calculate_relevancy(query: str, answer: str, ground_truth: str) -> float:
    """Simple relevancy: overlap between answer and ground truth."""
    if not answer or not ground_truth:
        return 0.0

    answer_tokens = set(answer.lower().split())
    gt_tokens = set(ground_truth.lower().split())

    if not gt_tokens:
        return 0.0

    overlap = len(answer_tokens & gt_tokens)
    return min(1.0, overlap / len(gt_tokens) * 1.5)  # Boost


def run_evaluation(api_base: str, full: bool = False) -> Dict:
    """Run GraphRAG evaluation."""

    test_cases = get_graphrag_test_cases()

    # Modes to test
    modes = ["vector", "local", "global", "hybrid"]
    if full:
        modes.append("drift")

    results = {
        "timestamp": datetime.now().isoformat(),
        "api_base": api_base,
        "graphrag_version": "2.0",
        "modes_tested": modes,
        "summary": {},
        "detailed_results": []
    }

    print(f"\n{'='*80}")
    print("MIRAGE GraphRAG EVALUATION")
    print(f"{'='*80}")
    print(f"API: {api_base}")
    print(f"Test Cases: {len(test_cases)}")
    print(f"Modes: {', '.join(modes)}")
    print(f"{'='*80}\n")

    # Aggregate stats per mode
    mode_stats = {m: {
        "total_score": 0.0,
        "latency_sum": 0.0,
        "success": 0,
        "entities_sum": 0,
        "semantic_entities_sum": 0
    } for m in modes}

    for tc in test_cases:
        print(f"\n[{tc.id}] {tc.query[:60]}...")
        print(f"  Expected mode: {tc.expected_mode} | Difficulty: {tc.difficulty}")

        tc_results = {"test_id": tc.id, "modes": {}}

        for mode in modes:
            result = evaluate_query(api_base, tc, mode)
            tc_results["modes"][mode] = {
                "success": result.success,
                "latency_ms": result.latency_ms,
                "faithfulness": result.faithfulness,
                "relevancy": result.relevancy,
                "entities_found": result.entities_found,
                "semantic_entities": result.semantic_entities,
                "chunks_count": result.chunks_count,
                "answer_preview": result.answer[:100] if result.answer else "",
                "error": result.error
            }

            if result.success:
                score = (result.faithfulness + result.relevancy) / 2
                mode_stats[mode]["total_score"] += score
                mode_stats[mode]["latency_sum"] += result.latency_ms
                mode_stats[mode]["success"] += 1
                mode_stats[mode]["entities_sum"] += result.entities_found
                mode_stats[mode]["semantic_entities_sum"] += result.semantic_entities

                status = "✓"
                print(f"  {mode.upper():<8} | Score: {score:.2f} | Latency: {result.latency_ms:.0f}ms | Entities: {result.entities_found} (semantic: {result.semantic_entities})")
            else:
                status = "✗"
                print(f"  {mode.upper():<8} | FAILED: {result.error[:50]}")

        results["detailed_results"].append(tc_results)

    # Calculate summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Mode':<10} {'Avg Score':<12} {'Avg Latency':<12} {'Entities':<12} {'Semantic':<12} {'Success'}")
    print("-" * 70)

    for mode in modes:
        stats = mode_stats[mode]
        count = stats["success"]
        if count > 0:
            avg_score = stats["total_score"] / count
            avg_latency = stats["latency_sum"] / count
            avg_entities = stats["entities_sum"] / count
            avg_semantic = stats["semantic_entities_sum"] / count
        else:
            avg_score = avg_latency = avg_entities = avg_semantic = 0

        print(f"{mode.upper():<10} {avg_score:.3f}        {avg_latency:.0f}ms         {avg_entities:.1f}          {avg_semantic:.1f}          {count}/{len(test_cases)}")

        results["summary"][mode] = {
            "avg_score": round(avg_score, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_entities": round(avg_entities, 1),
            "avg_semantic_entities": round(avg_semantic, 1),
            "success_rate": f"{count}/{len(test_cases)}"
        }

    # GraphRAG enhancement metrics
    print(f"\n{'='*80}")
    print("GRAPHRAG ENHANCEMENT METRICS")
    print(f"{'='*80}")

    local_stats = mode_stats.get("local", {})
    if local_stats.get("success", 0) > 0:
        sem_ratio = local_stats["semantic_entities_sum"] / max(1, local_stats["entities_sum"])
        print(f"Local Search Semantic Entity Ratio: {sem_ratio:.2%}")

    global_stats = mode_stats.get("global", {})
    if global_stats.get("success", 0) > 0:
        print(f"Global Search Avg Score: {global_stats['total_score'] / global_stats['success']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GraphRAG Enhancement Evaluation")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--full", action="store_true", help="Include DRIFT search tests")
    parser.add_argument("--output", default="graphrag_eval_results.json", help="Output file")
    args = parser.parse_args()

    results = run_evaluation(args.api, args.full)

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
