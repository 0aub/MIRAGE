#!/usr/bin/env python3
"""
MIRAGE Comprehensive Evaluation Suite
Tests all retrieval modes with scoring and comparison analysis.
"""

import urllib.request
import urllib.error
import json
import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE = "http://127.0.0.1:8000"
MODES = ["naive", "local", "global", "hybrid", "semantic", "mix", "drift"]
TIMEOUT = 120  # seconds


# =============================================================================
# TEST CASES - Multi-level evaluation
# =============================================================================

TEST_CASES = [
    # L1: Direct Factual (Simple entity lookup)
    {
        "query": "ما هي رؤية 2030؟",
        "level": "L1_factual",
        "expected_keywords": ["رؤية", "2030", "السعودية"],
        "chunk_keywords": ["رؤية", "2030"],
        "description": "Vision 2030 definition"
    },
    {
        "query": "ما هي جائزة الحكومة الرقمية؟",
        "level": "L1_factual",
        "expected_keywords": ["جائزة", "رقمية", "حكومة"],
        "chunk_keywords": ["جائزة", "رقمية"],
        "description": "Digital Government Award"
    },

    # L1: Entity Queries
    {
        "query": "من هي شركة علم؟",
        "level": "L1_entity",
        "expected_keywords": ["علم", "شركة"],
        "chunk_keywords": ["علم"],
        "description": "Elm company info"
    },
    {
        "query": "من هي شركة تيتو؟",
        "level": "L1_entity",
        "expected_keywords": ["تيتو"],
        "chunk_keywords": ["تيتو"],
        "description": "Teto company info"
    },
    {
        "query": "ما هي هيئة الزكاة والضريبة والجمارك؟",
        "level": "L1_entity",
        "expected_keywords": ["زكاة", "ضريبة", "جمارك"],
        "chunk_keywords": ["زكاة"],
        "description": "ZATCA entity info"
    },

    # L2: Relationship Queries
    {
        "query": "من هم شركاء جائزة الحكومة الرقمية؟",
        "level": "L2_relationship",
        "expected_keywords": ["شركاء", "جائزة"],
        "chunk_keywords": ["شريك", "جائزة"],
        "description": "Award partners relationship"
    },
    {
        "query": "ما هي الجهات المرشحة للجائزة؟",
        "level": "L2_relationship",
        "expected_keywords": ["جهات", "مرشح"],
        "chunk_keywords": ["مرشح", "جائزة"],
        "description": "Nominated entities"
    },

    # L2: Causal/Reasoning
    {
        "query": "لماذا تم إنشاء جائزة الحكومة الرقمية؟",
        "level": "L2_causal",
        "expected_keywords": ["جائزة", "هدف", "رقمية"],
        "chunk_keywords": ["جائزة", "رقمية"],
        "description": "Award purpose/reasoning"
    },

    # L3: Thematic/Holistic
    {
        "query": "ما هي أهم المواضيع المتعلقة بالتحول الرقمي؟",
        "level": "L3_thematic",
        "expected_keywords": ["تحول", "رقمي"],
        "chunk_keywords": ["رقمي", "تحول"],
        "description": "Digital transformation themes"
    },

    # Edge Cases
    {
        "query": "منصة يمامة",
        "level": "edge_short",
        "expected_keywords": ["يمامة"],
        "chunk_keywords": ["يمامة"],
        "description": "Short query - Yamamah platform"
    },
    {
        "query": "ما هي شركة ABC123؟",
        "level": "edge_unknown",
        "expected_keywords": [],  # Unknown entity - should handle gracefully
        "chunk_keywords": [],
        "description": "Unknown entity handling"
    },

    # Arabic Normalization Tests
    {
        "query": "ما هي هيئه الزكاه؟",  # Using ه instead of ة
        "level": "arabic_norm",
        "expected_keywords": ["زكاة", "هيئة"],
        "chunk_keywords": ["زكاة"],
        "description": "Arabic normalization (ة↔ه)"
    },

    # L4: Multi-hop queries (DRIFT should excel)
    {
        "query": "ما هي علاقة منتدى الحكومة الرقمية بشركائه وما أثرها على رؤية 2030؟",
        "level": "L4_multihop",
        "expected_keywords": ["منتدى", "شركاء", "رؤية", "2030"],
        "chunk_keywords": ["منتدى", "شريك"],
        "description": "Multi-hop: Forum -> Partners -> Vision 2030"
    },
    {
        "query": "كيف تساهم جائزة التميز في تحقيق أهداف التحول الرقمي؟",
        "level": "L4_multihop",
        "expected_keywords": ["جائزة", "تساهم", "تحول", "أهداف"],
        "chunk_keywords": ["جائزة", "تحول"],
        "description": "Multi-hop: Award -> Goals -> Transformation"
    },
    {
        "query": "ما هي الخدمات الرقمية التي تقدمها الجهات الحكومية للمواطنين؟",
        "level": "L4_multihop",
        "expected_keywords": ["خدمات", "رقمية", "حكومية", "مواطنين"],
        "chunk_keywords": ["خدمات", "رقمية"],
        "description": "Multi-hop: Entities -> Digital Services -> Citizens"
    },
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for comparison"""
    if not text:
        return text

    # Remove diacritics
    diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    text = diacritics.sub('', text)

    # Normalize variants
    text = text.replace('ة', 'ه')
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ى', 'ي')

    return text


def check_keywords(text: str, keywords: List[str]) -> tuple:
    """Check how many keywords are present in text"""
    if not text or not keywords:
        return 0, len(keywords) if keywords else 0

    text_normalized = normalize_arabic(text.lower())
    found = 0

    for kw in keywords:
        kw_normalized = normalize_arabic(kw.lower())
        if kw_normalized in text_normalized:
            found += 1

    return found, len(keywords)


def find_relevant_chunk_position(chunks: List[Dict], keywords: List[str]) -> int:
    """Find position of first chunk containing keywords (1-indexed, 0 = not found)"""
    if not chunks or not keywords:
        return 0

    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        found, total = check_keywords(text, keywords)
        if found > 0:
            return idx + 1

    return 0


@dataclass
class TestResult:
    """Result of a single test"""
    query: str
    level: str
    mode: str
    success: bool
    answer: str
    answer_keyword_score: float  # % of expected keywords in answer
    chunk_keyword_score: float   # % of chunk keywords in top chunks
    relevant_chunk_position: int # Position of first relevant chunk (0 = not found)
    latency_ms: float
    num_chunks: int
    error: Optional[str] = None


@dataclass
class ModeResults:
    """Aggregated results for a mode"""
    mode: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    avg_latency_ms: float = 0
    avg_answer_keyword_score: float = 0
    avg_chunk_keyword_score: float = 0
    avg_relevant_chunk_position: float = 0  # Lower is better
    results: List[TestResult] = field(default_factory=list)


# =============================================================================
# API CALLS
# =============================================================================

def call_api(query: str, mode: str, top_k: int = 5) -> Dict[str, Any]:
    """Call the chat API"""
    try:
        data = json.dumps({
            "message": query,
            "retrieval_mode": mode,
            "top_k": top_k
        }).encode('utf-8')

        req = urllib.request.Request(
            f"{API_BASE}/chat/ask",
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            return json.loads(response.read().decode('utf-8'))

    except urllib.error.URLError as e:
        return {"error": f"TIMEOUT/Network: {e}", "answer": "", "chunks": []}
    except Exception as e:
        return {"error": str(e), "answer": "", "chunks": []}


def run_test(test_case: Dict, mode: str) -> TestResult:
    """Run a single test case"""
    start = time.time()

    result = call_api(test_case["query"], mode)

    latency_ms = (time.time() - start) * 1000

    # Check for errors
    if "error" in result and result["error"]:
        return TestResult(
            query=test_case["query"],
            level=test_case["level"],
            mode=mode,
            success=False,
            answer="",
            answer_keyword_score=0,
            chunk_keyword_score=0,
            relevant_chunk_position=0,
            latency_ms=latency_ms,
            num_chunks=0,
            error=result["error"]
        )

    answer = result.get("answer", "")
    chunks = result.get("chunks", [])

    # Calculate answer keyword score
    answer_found, answer_total = check_keywords(
        answer, test_case["expected_keywords"]
    )
    answer_score = answer_found / answer_total if answer_total > 0 else 1.0

    # Calculate chunk keyword score
    all_chunk_text = " ".join([c.get("text", "") for c in chunks[:5]])
    chunk_found, chunk_total = check_keywords(
        all_chunk_text, test_case["chunk_keywords"]
    )
    chunk_score = chunk_found / chunk_total if chunk_total > 0 else 1.0

    # Find relevant chunk position
    relevant_pos = find_relevant_chunk_position(
        chunks, test_case["chunk_keywords"]
    )

    # Determine success
    # Success = answer has some keywords AND chunks have keywords at reasonable position
    has_answer = len(answer) > 30 and answer_score >= 0.25
    has_chunks = chunk_score >= 0.5 or relevant_pos > 0
    is_unknown_query = test_case["level"] == "edge_unknown"

    success = (has_answer and has_chunks) or is_unknown_query

    return TestResult(
        query=test_case["query"],
        level=test_case["level"],
        mode=mode,
        success=success,
        answer=answer[:200] + "..." if len(answer) > 200 else answer,
        answer_keyword_score=answer_score,
        chunk_keyword_score=chunk_score,
        relevant_chunk_position=relevant_pos,
        latency_ms=latency_ms,
        num_chunks=len(chunks)
    )


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation():
    """Run full evaluation suite"""
    print("=" * 80)
    print("MIRAGE COMPREHENSIVE EVALUATION SUITE")
    print("=" * 80)
    print(f"Target: {API_BASE}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Modes: {', '.join(MODES)}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()

    # Check API health
    try:
        req = urllib.request.Request(f"{API_BASE}/health")
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status != 200:
                print("ERROR: API not healthy!")
                return
    except Exception as e:
        print(f"ERROR: Cannot reach API: {e}")
        return

    # Run tests for each mode
    all_results: Dict[str, ModeResults] = {}

    for mode in MODES:
        print(f"\nTesting {mode.upper()}...")
        mode_results = ModeResults(mode=mode)

        for i, test_case in enumerate(TEST_CASES):
            result = run_test(test_case, mode)
            mode_results.results.append(result)
            mode_results.total_tests += 1

            if result.success:
                mode_results.passed += 1
            else:
                mode_results.failed += 1

            # Progress indicator
            status = "✓" if result.success else "✗"
            print(f"  {status} [{i+1}/{len(TEST_CASES)}] {test_case['description'][:40]}")

        # Calculate averages
        latencies = [r.latency_ms for r in mode_results.results]
        answer_scores = [r.answer_keyword_score for r in mode_results.results]
        chunk_scores = [r.chunk_keyword_score for r in mode_results.results]
        positions = [r.relevant_chunk_position for r in mode_results.results if r.relevant_chunk_position > 0]

        mode_results.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
        mode_results.avg_answer_keyword_score = sum(answer_scores) / len(answer_scores) if answer_scores else 0
        mode_results.avg_chunk_keyword_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0
        mode_results.avg_relevant_chunk_position = sum(positions) / len(positions) if positions else 0

        all_results[mode] = mode_results

    # Print summary
    print_summary(all_results)
    print_detailed_analysis(all_results)
    print_mode_comparison(all_results)
    print_final_scores(all_results)


def print_summary(results: Dict[str, ModeResults]):
    """Print summary table"""
    print("\n" + "=" * 80)
    print("SUMMARY BY MODE")
    print("=" * 80)
    print()

    print(f"{'Mode':<12} {'Pass Rate':>12} {'Avg Latency':>12} {'Answer KW':>12} {'Chunk KW':>12} {'Rank@1':>8}")
    print("-" * 70)

    for mode, mr in results.items():
        pass_rate = mr.passed / mr.total_tests * 100 if mr.total_tests > 0 else 0
        rank_at_1 = sum(1 for r in mr.results if r.relevant_chunk_position == 1) / mr.total_tests * 100

        print(f"{mode:<12} {pass_rate:>10.1f}% {mr.avg_latency_ms:>10.0f}ms "
              f"{mr.avg_answer_keyword_score:>11.1%} {mr.avg_chunk_keyword_score:>11.1%} "
              f"{rank_at_1:>6.1f}%")


def print_detailed_analysis(results: Dict[str, ModeResults]):
    """Print detailed analysis by query level"""
    print("\n" + "=" * 80)
    print("PERFORMANCE BY QUERY LEVEL")
    print("=" * 80)
    print()

    levels = ["L1_factual", "L1_entity", "L2_relationship", "L2_causal",
              "L3_thematic", "L4_multihop", "edge_short", "edge_unknown", "arabic_norm"]

    for level in levels:
        print(f"\n{level}:")
        print(f"  {'Mode':<12} {'Pass':>6} {'Latency':>10} {'Answer%':>10} {'Chunk@1':>10}")
        print("  " + "-" * 50)

        for mode, mr in results.items():
            level_results = [r for r in mr.results if r.level == level]
            if not level_results:
                continue

            passed = sum(1 for r in level_results if r.success)
            total = len(level_results)
            avg_latency = sum(r.latency_ms for r in level_results) / total
            avg_answer = sum(r.answer_keyword_score for r in level_results) / total
            rank1 = sum(1 for r in level_results if r.relevant_chunk_position == 1) / total * 100

            print(f"  {mode:<12} {passed}/{total:<4} {avg_latency:>8.0f}ms {avg_answer:>9.0%} {rank1:>9.0f}%")


def print_mode_comparison(results: Dict[str, ModeResults]):
    """Print mode comparison analysis"""
    print("\n" + "=" * 80)
    print("MODE COMBINATION ANALYSIS")
    print("=" * 80)
    print()

    print("How modes build on each other:")
    print()
    print("  NAIVE  → Base vector search + keyword fallback")
    print("    ↓")
    print("  LOCAL  → NAIVE + Entity disambiguation + Graph traversal")
    print("    ↓")
    print("  GLOBAL → NAIVE + Relationship traversal")
    print("    ↓")
    print("  HYBRID → NAIVE + LOCAL + GLOBAL (RRF fusion)")
    print("    ↓")
    print("  SEMANTIC → NAIVE + Cross-encoder re-ranking")
    print("    ↓")
    print("  MIX    → ALL modes (RRF fusion)")
    print("    ↓")
    print("  DRIFT  → GraphRAG: Global (community summaries) → Local → Claims")
    print()

    # Compare effectiveness
    print("Effectiveness Comparison:")
    print()

    if "naive" in results and "local" in results:
        naive_pass = results["naive"].passed
        local_pass = results["local"].passed
        diff = local_pass - naive_pass
        print(f"  LOCAL vs NAIVE: {'+' if diff >= 0 else ''}{diff} tests "
              f"({local_pass}/{results['local'].total_tests} vs {naive_pass}/{results['naive'].total_tests})")

    if "local" in results and "hybrid" in results:
        local_pass = results["local"].passed
        hybrid_pass = results["hybrid"].passed
        diff = hybrid_pass - local_pass
        print(f"  HYBRID vs LOCAL: {'+' if diff >= 0 else ''}{diff} tests")

    if "hybrid" in results and "mix" in results:
        hybrid_pass = results["hybrid"].passed
        mix_pass = results["mix"].passed
        diff = mix_pass - hybrid_pass
        print(f"  MIX vs HYBRID: {'+' if diff >= 0 else ''}{diff} tests")


def print_final_scores(results: Dict[str, ModeResults]):
    """Calculate and print final scores"""
    print("\n" + "=" * 80)
    print("FINAL SCORES (out of 100)")
    print("=" * 80)
    print()

    scores = {}

    for mode, mr in results.items():
        # Score components (weights)
        pass_rate = mr.passed / mr.total_tests if mr.total_tests > 0 else 0  # 40%
        answer_quality = mr.avg_answer_keyword_score  # 25%
        chunk_quality = mr.avg_chunk_keyword_score  # 20%

        # Latency score (lower is better, 0-1 scale, <1s = 1.0, >5s = 0.0)
        latency_score = max(0, 1 - (mr.avg_latency_ms - 1000) / 4000)  # 15%

        # Combined score
        score = (
            pass_rate * 40 +
            answer_quality * 25 +
            chunk_quality * 20 +
            latency_score * 15
        )

        scores[mode] = score

        # Print breakdown
        print(f"{mode.upper()}:")
        print(f"  Pass Rate (40%):    {pass_rate:.1%} → {pass_rate * 40:.1f}")
        print(f"  Answer Quality (25%): {answer_quality:.1%} → {answer_quality * 25:.1f}")
        print(f"  Chunk Quality (20%): {chunk_quality:.1%} → {chunk_quality * 20:.1f}")
        print(f"  Latency Score (15%): {latency_score:.1%} → {latency_score * 15:.1f}")
        print(f"  TOTAL: {score:.1f}/100")
        print()

    # Best mode
    best_mode = max(scores, key=scores.get)
    print(f"Best Overall Mode: {best_mode.upper()} ({scores[best_mode]:.1f}/100)")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    fastest = min(results.items(), key=lambda x: x[1].avg_latency_ms)
    most_accurate = max(results.items(), key=lambda x: x[1].passed)
    best_answer = max(results.items(), key=lambda x: x[1].avg_answer_keyword_score)

    print(f"  Fastest mode: {fastest[0].upper()} ({fastest[1].avg_latency_ms:.0f}ms avg)")
    print(f"  Most accurate: {most_accurate[0].upper()} ({most_accurate[1].passed}/{most_accurate[1].total_tests} passed)")
    print(f"  Best answers: {best_answer[0].upper()} ({best_answer[1].avg_answer_keyword_score:.0%} keyword coverage)")
    print()
    print("  Use NAIVE for: Simple factual queries, speed-critical applications")
    print("  Use LOCAL for: Entity-specific queries, name lookups")
    print("  Use HYBRID for: General-purpose queries, unknown query types")
    print("  Use SEMANTIC for: Complex reasoning, precise matching needed")
    print("  Use MIX for: Maximum accuracy, latency not critical")
    print("  Use DRIFT for: Multi-hop queries, thematic questions, relationship queries")


if __name__ == "__main__":
    run_evaluation()
