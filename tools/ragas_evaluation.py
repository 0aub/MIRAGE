#!/usr/bin/env python3
"""
RAGAS-style Evaluation for MIRAGE GraphRAG System

This script evaluates retrieval and generation quality across different:
- Query complexity levels (L1-L4)
- Retrieval modes (naive, local, global, hybrid, mix)

RAGAS Metrics Implemented:
1. Answer Relevancy - Does the answer address the question?
2. Faithfulness - Is the answer grounded in the context?
3. Context Precision - Is retrieved context relevant?
4. Answer Correctness - Compared to ground truth (semantic similarity)

Query Complexity Levels:
- L1: Direct Factual (simple lookup)
- L2: Entity-Specific (about a specific entity)
- L3: Multi-Hop (requires graph traversal)
- L4: Overview/Aggregation (requires summarization)
"""

import json
import re
import time
import requests
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set
from datetime import datetime


# =============================================================================
# TOKENIZATION & SIMILARITY
# =============================================================================

def tokenize_multilingual(text: str) -> Set[str]:
    """Tokenize Arabic/English text for comparison."""
    if not text:
        return set()

    # Remove Arabic diacritics
    arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    text = arabic_diacritics.sub('', text)

    # Split on whitespace and punctuation
    tokens = re.split(r'[\s\.,،:;!?\-\(\)\[\]{}\"\']+', text.lower())

    # Stopwords
    stopwords = {
        'من', 'في', 'على', 'إلى', 'عن', 'هو', 'هي', 'هذا', 'هذه', 'التي', 'الذي',
        'ما', 'هل', 'كان', 'كانت', 'أن', 'إن', 'لا', 'لم', 'قد', 'و', 'أو',
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'and', 'or', 'but',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as'
    }

    return {t for t in tokens if len(t) >= 2 and t not in stopwords}


def compute_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity using token overlap."""
    tokens1 = tokenize_multilingual(text1)
    tokens2 = tokenize_multilingual(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    jaccard = intersection / union if union > 0 else 0.0
    recall = intersection / len(tokens2) if tokens2 else 0.0

    return round(0.4 * jaccard + 0.6 * recall, 3)


# =============================================================================
# TEST CASES - SDAIA Data Governance Policies
# =============================================================================

@dataclass
class TestCase:
    id: str
    query: str
    complexity: str  # L1, L2, L3, L4
    category: str    # factual, entity, multi-hop, overview
    language: str    # en, ar
    ground_truth: str
    best_modes: List[str]


TEST_CASES = [
    # =========================================================================
    # L1: DIRECT FACTUAL (Simple Lookup)
    # =========================================================================
    TestCase(
        id="L1_01",
        query="What is data classification?",
        complexity="L1",
        category="factual",
        language="en",
        ground_truth="""Data classification is the process of organizing data into categories based on its sensitivity level and the impact of unauthorized disclosure. According to NDMO policies, data is classified into four levels: Top Secret (highest sensitivity), Secret, Confidential, and Public (lowest sensitivity). Classification determines the required security controls and handling procedures.""",
        best_modes=["naive", "local"]
    ),
    TestCase(
        id="L1_02",
        query="ما هو تصنيف البيانات؟",
        complexity="L1",
        category="factual",
        language="ar",
        ground_truth="""تصنيف البيانات هو عملية تنظيم البيانات في فئات بناءً على مستوى حساسيتها وتأثير الإفصاح غير المصرح به. وفقاً لسياسات مكتب إدارة البيانات الوطنية، تُصنف البيانات إلى أربعة مستويات: سري للغاية (أعلى حساسية)، سري، محدود، وعام (أدنى حساسية). يحدد التصنيف ضوابط الأمان المطلوبة وإجراءات التعامل.""",
        best_modes=["naive", "local"]
    ),
    TestCase(
        id="L1_03",
        query="What are the data classification levels?",
        complexity="L1",
        category="factual",
        language="en",
        ground_truth="""The data classification levels according to Saudi national data governance policies are: 1) Top Secret - highest sensitivity, unauthorized disclosure could cause exceptionally grave damage; 2) Secret - disclosure could cause serious damage; 3) Confidential - disclosure could cause damage; 4) Public - no restrictions on disclosure. Each level has specific handling, storage, and transmission requirements.""",
        best_modes=["naive"]
    ),

    # =========================================================================
    # L2: ENTITY-SPECIFIC (About a Specific Entity)
    # =========================================================================
    TestCase(
        id="L2_01",
        query="What are the responsibilities of a Data Owner?",
        complexity="L2",
        category="entity",
        language="en",
        ground_truth="""A Data Owner is accountable for the data within their domain. Key responsibilities include: 1) Classifying data according to sensitivity levels; 2) Defining access controls and permissions; 3) Ensuring data quality and integrity; 4) Approving data sharing requests; 5) Ensuring compliance with data governance policies; 6) Managing the data lifecycle from creation to disposal; 7) Reporting data breaches and incidents.""",
        best_modes=["local", "hybrid"]
    ),
    TestCase(
        id="L2_02",
        query="ما هي مسؤوليات مالك البيانات؟",
        complexity="L2",
        category="entity",
        language="ar",
        ground_truth="""مالك البيانات مسؤول عن البيانات ضمن نطاقه. تشمل المسؤوليات الرئيسية: 1) تصنيف البيانات وفقاً لمستويات الحساسية؛ 2) تحديد ضوابط الوصول والصلاحيات؛ 3) ضمان جودة البيانات وسلامتها؛ 4) الموافقة على طلبات مشاركة البيانات؛ 5) ضمان الامتثال لسياسات حوكمة البيانات؛ 6) إدارة دورة حياة البيانات من الإنشاء إلى التخلص؛ 7) الإبلاغ عن انتهاكات وحوادث البيانات.""",
        best_modes=["local", "hybrid"]
    ),
    TestCase(
        id="L2_03",
        query="What is the role of the National Data Management Office?",
        complexity="L2",
        category="entity",
        language="en",
        ground_truth="""The National Data Management Office (NDMO) is Saudi Arabia's national regulator for data governance. Its roles include: 1) Developing national data policies and standards; 2) Overseeing data governance implementation across government; 3) Managing the national open data platform; 4) Ensuring data quality and interoperability; 5) Promoting data sharing between government entities; 6) Building data capabilities and awareness; 7) Monitoring compliance with data regulations.""",
        best_modes=["local", "naive"]
    ),

    # =========================================================================
    # L3: MULTI-HOP (Requires Graph Traversal)
    # =========================================================================
    TestCase(
        id="L3_01",
        query="How does data classification affect data sharing between government entities?",
        complexity="L3",
        category="multi-hop",
        language="en",
        ground_truth="""Data classification directly impacts data sharing: 1) Public data can be shared openly without restrictions; 2) Confidential data requires formal data sharing agreements; 3) Secret data requires approval from the Data Owner and security clearance; 4) Top Secret data has strict sharing limitations and requires highest-level authorization. All sharing must follow the principle of 'need to know' and comply with NDMO data sharing policies. Data sharing agreements must specify purpose, recipients, security measures, and retention period.""",
        best_modes=["local", "hybrid", "global"]
    ),
    TestCase(
        id="L3_02",
        query="ما هي العلاقة بين تصنيف البيانات وحماية البيانات الشخصية؟",
        complexity="L3",
        category="multi-hop",
        language="ar",
        ground_truth="""تصنيف البيانات وحماية البيانات الشخصية مترابطان: 1) البيانات الشخصية تُصنف عادةً كـ'محدود' أو أعلى؛ 2) البيانات الشخصية الحساسة (صحية، مالية، بيومترية) تتطلب تصنيفاً أعلى؛ 3) يجب الحصول على موافقة صاحب البيانات قبل المعالجة؛ 4) تطبق ضوابط أمنية مشددة على البيانات الشخصية؛ 5) يجب الالتزام بنظام حماية البيانات الشخصية السعودي؛ 6) حقوق الوصول والتصحيح والحذف مكفولة لأصحاب البيانات.""",
        best_modes=["local", "hybrid", "global"]
    ),
    TestCase(
        id="L3_03",
        query="What security controls are required for each data classification level?",
        complexity="L3",
        category="multi-hop",
        language="en",
        ground_truth="""Security controls vary by classification: Public: basic access logging, no encryption required; Confidential: access controls, encryption in transit, audit logs; Secret: strong encryption at rest and in transit, multi-factor authentication, physical security, detailed audit trails; Top Secret: highest encryption standards, isolated systems, biometric access, continuous monitoring, secure facilities. All levels require: data inventory, classification marking, incident reporting, and regular security assessments.""",
        best_modes=["local", "hybrid"]
    ),

    # =========================================================================
    # L4: OVERVIEW/AGGREGATION (Requires Summarization)
    # =========================================================================
    TestCase(
        id="L4_01",
        query="What are the main principles of the National Data Governance Framework?",
        complexity="L4",
        category="overview",
        language="en",
        ground_truth="""The National Data Governance Framework is built on these principles: 1) Open by Default - government data should be accessible unless restricted; 2) Timely Classification - data must be classified promptly; 3) Least Privilege - minimum access needed for tasks; 4) Data Quality - accuracy, completeness, consistency; 5) Accountability - clear ownership and responsibility; 6) Interoperability - standardized formats for sharing; 7) Privacy by Design - protection built into systems; 8) Transparency - clear data handling practices.""",
        best_modes=["global", "hybrid"]
    ),
    TestCase(
        id="L4_02",
        query="ما هي أهم مبادئ حوكمة البيانات الوطنية؟",
        complexity="L4",
        category="overview",
        language="ar",
        ground_truth="""يرتكز إطار حوكمة البيانات الوطنية على المبادئ التالية: 1) الانفتاح الافتراضي - البيانات الحكومية متاحة ما لم تكن مقيدة؛ 2) التصنيف في الوقت المناسب - يجب تصنيف البيانات فوراً؛ 3) الحد الأدنى من الصلاحيات - الوصول بقدر الحاجة فقط؛ 4) جودة البيانات - الدقة والاكتمال والاتساق؛ 5) المساءلة - ملكية ومسؤولية واضحة؛ 6) قابلية التشغيل البيني - تنسيقات موحدة للمشاركة؛ 7) الخصوصية بالتصميم - الحماية مدمجة في الأنظمة؛ 8) الشفافية - ممارسات واضحة للتعامل مع البيانات.""",
        best_modes=["global", "hybrid"]
    ),
    TestCase(
        id="L4_03",
        query="Summarize the key requirements for open data publishing in Saudi Arabia.",
        complexity="L4",
        category="overview",
        language="en",
        ground_truth="""Open data publishing requirements: 1) Data must be classified as 'Public'; 2) Published in machine-readable formats (CSV, JSON, XML); 3) Released under open license allowing reuse; 4) Accompanied by metadata describing content and quality; 5) Updated regularly to maintain currency; 6) Accessible via the national open data platform; 7) No personally identifiable information; 8) Free of charge for public access; 9) Entity must designate Open Data Coordinator; 10) Annual reporting on open data inventory.""",
        best_modes=["global", "hybrid", "naive"]
    ),
]


# =============================================================================
# RAGAS METRICS
# =============================================================================

def compute_answer_relevancy(question: str, answer: str) -> float:
    """Does the answer address the question?"""
    q_tokens = tokenize_multilingual(question)
    a_tokens = tokenize_multilingual(answer)

    if not q_tokens or not a_tokens:
        return 0.0

    # Check how many question keywords appear in answer
    overlap = len(q_tokens & a_tokens)
    return min(1.0, overlap / len(q_tokens))


def compute_faithfulness(answer: str, context: str) -> float:
    """Is the answer grounded in the retrieved context?"""
    if not answer or not context:
        return 0.0

    a_tokens = tokenize_multilingual(answer)
    c_tokens = tokenize_multilingual(context)

    if not a_tokens:
        return 0.0

    # How much of answer is supported by context
    grounded = len(a_tokens & c_tokens)
    return grounded / len(a_tokens)


def compute_context_precision(question: str, context: str) -> float:
    """Is the retrieved context relevant to the question?"""
    return compute_similarity(question, context)


def compute_answer_correctness(answer: str, ground_truth: str) -> float:
    """How close is the answer to ground truth?"""
    return compute_similarity(answer, ground_truth)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_query(test_case: TestCase, mode: str, api_base: str = "http://localhost:8000") -> Dict:
    """Run a single query and compute metrics."""

    start_time = time.time()

    try:
        response = requests.post(
            f"{api_base}/chat/ask",
            json={
                "message": test_case.query,
                "retrieval_mode": mode,
                "top_k": 5
            },
            timeout=120
        )

        latency = time.time() - start_time

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "latency_ms": latency * 1000
            }

        data = response.json()
        answer = data.get("answer", "")

        # Get context from chunks
        chunks = data.get("chunks", [])
        context = " ".join([c.get("text", "") for c in chunks]) if chunks else ""

        # Compute RAGAS metrics
        metrics = {
            "success": True,
            "latency_ms": round(latency * 1000, 1),
            "answer_length": len(answer),
            "context_length": len(context),
            "chunks_retrieved": len(chunks),

            # RAGAS Metrics
            "answer_relevancy": round(compute_answer_relevancy(test_case.query, answer), 3),
            "faithfulness": round(compute_faithfulness(answer, context), 3),
            "context_precision": round(compute_context_precision(test_case.query, context), 3),
            "answer_correctness": round(compute_answer_correctness(answer, test_case.ground_truth), 3),

            # Combined score
            "ragas_score": 0.0,

            # Sample data
            "answer_preview": answer[:200] if answer else ""
        }

        # Weighted RAGAS score
        metrics["ragas_score"] = round(
            0.25 * metrics["answer_relevancy"] +
            0.25 * metrics["faithfulness"] +
            0.20 * metrics["context_precision"] +
            0.30 * metrics["answer_correctness"],
            3
        )

        return metrics

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency_ms": (time.time() - start_time) * 1000
        }


def run_evaluation(api_base: str = "http://localhost:8000") -> Dict:
    """Run full RAGAS evaluation."""

    modes = ["naive", "local", "global", "hybrid", "mix"]

    results = {
        "timestamp": datetime.now().isoformat(),
        "test_cases": [],
        "by_complexity": {},
        "by_mode": {},
        "by_language": {},
        "summary": {}
    }

    print("=" * 70)
    print("RAGAS EVALUATION FOR MIRAGE")
    print("=" * 70)

    for tc in TEST_CASES:
        print(f"\n[{tc.id}] {tc.query[:50]}...")

        tc_result = {
            "id": tc.id,
            "query": tc.query,
            "complexity": tc.complexity,
            "category": tc.category,
            "language": tc.language,
            "best_modes": tc.best_modes,
            "modes": {}
        }

        for mode in modes:
            print(f"  Mode: {mode}...", end=" ", flush=True)
            metrics = evaluate_query(tc, mode, api_base)
            tc_result["modes"][mode] = metrics

            if metrics["success"]:
                print(f"RAGAS={metrics['ragas_score']:.2f} ({metrics['latency_ms']:.0f}ms)")
            else:
                print(f"FAILED: {metrics.get('error', 'unknown')}")

        results["test_cases"].append(tc_result)

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS...")
    print("=" * 70)

    # By complexity level
    for level in ["L1", "L2", "L3", "L4"]:
        level_results = [tc for tc in results["test_cases"] if tc["complexity"] == level]
        results["by_complexity"][level] = aggregate_metrics(level_results, modes)

    # By mode
    for mode in modes:
        mode_scores = []
        for tc in results["test_cases"]:
            if tc["modes"][mode]["success"]:
                mode_scores.append(tc["modes"][mode]["ragas_score"])

        results["by_mode"][mode] = {
            "avg_ragas_score": round(sum(mode_scores) / len(mode_scores), 3) if mode_scores else 0,
            "success_rate": len(mode_scores) / len(results["test_cases"]),
            "test_count": len(mode_scores)
        }

    # By language
    for lang in ["en", "ar"]:
        lang_results = [tc for tc in results["test_cases"] if tc["language"] == lang]
        results["by_language"][lang] = aggregate_metrics(lang_results, modes)

    # Overall summary
    all_scores = []
    for tc in results["test_cases"]:
        for mode, metrics in tc["modes"].items():
            if metrics["success"]:
                all_scores.append(metrics["ragas_score"])

    results["summary"] = {
        "total_tests": len(TEST_CASES),
        "total_evaluations": len(TEST_CASES) * len(modes),
        "overall_ragas_score": round(sum(all_scores) / len(all_scores), 3) if all_scores else 0,
        "best_mode": max(results["by_mode"].items(), key=lambda x: x[1]["avg_ragas_score"])[0],
        "best_complexity": max(results["by_complexity"].items(), key=lambda x: x[1].get("avg_ragas", 0))[0]
    }

    return results


def aggregate_metrics(test_cases: List[Dict], modes: List[str]) -> Dict:
    """Aggregate metrics for a group of test cases."""

    result = {
        "test_count": len(test_cases),
        "by_mode": {}
    }

    for mode in modes:
        scores = []
        latencies = []

        for tc in test_cases:
            if tc["modes"][mode]["success"]:
                scores.append(tc["modes"][mode]["ragas_score"])
                latencies.append(tc["modes"][mode]["latency_ms"])

        result["by_mode"][mode] = {
            "avg_ragas": round(sum(scores) / len(scores), 3) if scores else 0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "success_count": len(scores)
        }

    # Overall average for this group
    all_scores = []
    for mode in modes:
        if result["by_mode"][mode]["success_count"] > 0:
            all_scores.append(result["by_mode"][mode]["avg_ragas"])

    result["avg_ragas"] = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0

    return result


def print_results(results: Dict):
    """Print formatted results."""

    print("\n" + "=" * 70)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 70)

    # Summary
    print(f"\nOverall RAGAS Score: {results['summary']['overall_ragas_score']:.3f}")
    print(f"Best Mode: {results['summary']['best_mode']}")
    print(f"Best at Complexity: {results['summary']['best_complexity']}")

    # By Complexity
    print("\n" + "-" * 50)
    print("BY QUERY COMPLEXITY:")
    print("-" * 50)
    print(f"{'Level':<8} {'Naive':<10} {'Local':<10} {'Global':<10} {'Hybrid':<10} {'Mix':<10}")

    for level in ["L1", "L2", "L3", "L4"]:
        data = results["by_complexity"][level]
        row = f"{level:<8}"
        for mode in ["naive", "local", "global", "hybrid", "mix"]:
            score = data["by_mode"][mode]["avg_ragas"]
            row += f"{score:.3f}     "
        print(row)

    # By Mode
    print("\n" + "-" * 50)
    print("BY RETRIEVAL MODE:")
    print("-" * 50)

    for mode, data in results["by_mode"].items():
        print(f"{mode.upper():<10}: RAGAS={data['avg_ragas_score']:.3f}, Success={data['success_rate']*100:.0f}%")

    # By Language
    print("\n" + "-" * 50)
    print("BY LANGUAGE:")
    print("-" * 50)

    for lang, data in results["by_language"].items():
        lang_name = "English" if lang == "en" else "Arabic"
        print(f"{lang_name:<10}: Avg RAGAS={data['avg_ragas']:.3f} ({data['test_count']} tests)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS Evaluation for MIRAGE")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", default="/app/benchmark_results/ragas_evaluation.json", help="Output file")
    args = parser.parse_args()

    results = run_evaluation(args.api)
    print_results(results)

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")
