#!/usr/bin/env python3
"""
MIRAGE Strict Evaluation Suite V2
With proper Arabic normalization for keyword matching
"""

import requests
import time
import json
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

BASE_URL = "http://localhost:8000"

def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for comparison."""
    if not text:
        return ""
    # Normalize ة to ه
    text = text.replace('ة', 'ه')
    # Normalize alef variants
    text = re.sub(r'[أإآ]', 'ا', text)
    # Remove tashkeel
    text = re.sub(r'[\u064B-\u0652]', '', text)
    return text

def keyword_match(text: str, keywords: List[str]) -> int:
    """Count keyword matches with normalization."""
    normalized_text = normalize_arabic(text)
    count = 0
    for kw in keywords:
        normalized_kw = normalize_arabic(kw)
        if normalized_kw in normalized_text:
            count += 1
    return count

@dataclass
class TestCase:
    query: str
    query_type: str
    expected_keywords: List[str]
    expected_entity_type: str = ""
    min_chunks: int = 3
    max_latency_ms: int = 15000  # 15s threshold (LLM generation can be slow)
    should_find_info: bool = True
    description: str = ""

@dataclass
class TestResult:
    test_case: TestCase
    mode: str
    passed: bool
    latency_ms: float
    chunks_returned: int
    answer_length: int
    keywords_in_answer: int
    keywords_in_chunks: int
    issues: List[str] = field(default_factory=list)
    answer_preview: str = ""
    relevant_chunk_rank: int = -1  # Position of first relevant chunk (1-indexed)

def run_test(tc: TestCase, mode: str) -> TestResult:
    """Run a single test case with proper evaluation."""
    issues = []

    start = time.time()
    try:
        resp = requests.post(
            f"{BASE_URL}/chat/ask",
            json={"message": tc.query, "retrieval_mode": mode, "top_k": 5},
            timeout=120
        )
        latency = (time.time() - start) * 1000

        if resp.status_code != 200:
            return TestResult(
                test_case=tc, mode=mode, passed=False,
                latency_ms=latency, chunks_returned=0, answer_length=0,
                keywords_in_answer=0, keywords_in_chunks=0,
                issues=[f"HTTP {resp.status_code}"]
            )

        data = resp.json()
        answer = data.get("answer", "")
        chunks = data.get("chunks", [])

        # Find first relevant chunk position
        relevant_chunk_rank = -1
        for i, c in enumerate(chunks):
            chunk_text = c.get("text", "")
            if tc.expected_keywords and keyword_match(chunk_text, tc.expected_keywords) > 0:
                relevant_chunk_rank = i + 1
                break

        # Critical checks with Arabic normalization

        # 1. Latency check
        if latency > tc.max_latency_ms:
            issues.append(f"SLOW: {latency:.0f}ms > {tc.max_latency_ms}ms")

        # 2. Chunk count check
        if len(chunks) < tc.min_chunks:
            issues.append(f"LOW_CHUNKS: {len(chunks)} < {tc.min_chunks}")

        # 3. Empty/short answer check
        if not answer or len(answer) < 30:
            issues.append("EMPTY_ANSWER")

        # 4. "No info found" check - IMPROVED: Only flag if answer is PURELY negative (short)
        # Don't flag if answer has useful content (>100 chars) even if it mentions "no direct info"
        no_info_phrases = ["لا أجد أي", "لم يتم العثور على أي", "لا توجد معلومات متاحة", "لا أستطيع الإجابة"]
        is_pure_no_info = len(answer) < 100 and any(p in answer for p in no_info_phrases)
        if tc.should_find_info and is_pure_no_info:
            issues.append("NO_INFO: LLM couldn't synthesize answer")

        # 5. Keyword check in ANSWER (with normalization)
        keywords_in_answer = keyword_match(answer, tc.expected_keywords)
        keyword_ratio = keywords_in_answer / len(tc.expected_keywords) if tc.expected_keywords else 1.0
        if keyword_ratio < 0.3 and tc.expected_keywords and tc.should_find_info:
            issues.append(f"ANSWER_MISSING_KEYWORDS: {keywords_in_answer}/{len(tc.expected_keywords)}")

        # 6. Check if ANY chunk contains relevant content (with normalization)
        chunk_texts = " ".join([c.get("text", "") for c in chunks])
        keywords_in_chunks = keyword_match(chunk_texts, tc.expected_keywords)
        if keywords_in_chunks == 0 and tc.expected_keywords and tc.should_find_info:
            issues.append("NO_RELEVANT_CHUNKS")

        # 7. Check chunk ranking (relevant chunk should be in top 2)
        if relevant_chunk_rank > 2 and tc.expected_keywords and tc.should_find_info:
            issues.append(f"POOR_RANKING: relevant at position {relevant_chunk_rank}")

        passed = len(issues) == 0

        return TestResult(
            test_case=tc, mode=mode, passed=passed,
            latency_ms=latency, chunks_returned=len(chunks),
            answer_length=len(answer),
            keywords_in_answer=keywords_in_answer,
            keywords_in_chunks=keywords_in_chunks,
            issues=issues,
            answer_preview=answer[:150],
            relevant_chunk_rank=relevant_chunk_rank
        )

    except Exception as e:
        return TestResult(
            test_case=tc, mode=mode, passed=False,
            latency_ms=(time.time() - start) * 1000,
            chunks_returned=0, answer_length=0,
            keywords_in_answer=0, keywords_in_chunks=0,
            issues=[f"EXCEPTION: {str(e)[:80]}"]
        )


def get_test_cases() -> List[TestCase]:
    """Define comprehensive test cases."""
    return [
        # === L1: Direct Factual ===
        TestCase(
            query="ما هي جائزة الحكومة الرقمية؟",
            query_type="L1_factual",
            expected_keywords=["جائزة", "رقمية", "حكومة", "شريك"],
            description="Basic factual about digital government award"
        ),
        TestCase(
            query="ما هي رؤية 2030؟",
            query_type="L1_factual",
            expected_keywords=["رؤية", "2030", "سعودية", "تحول"],
            description="Saudi Vision 2030"
        ),

        # === L1: Entity Queries ===
        TestCase(
            query="من هي شركة تيتو؟",
            query_type="L1_entity",
            expected_keywords=["تيتو", "شركة"],
            expected_entity_type="Organization",
            description="Teto company entity"
        ),
        TestCase(
            query="ما هي هيئة الزكاة؟",
            query_type="L1_entity",
            expected_keywords=["زكاة", "هيئة", "ضريبة"],
            expected_entity_type="Organization",
            description="Zakat authority entity"
        ),
        TestCase(
            query="من هي شركة علم؟",
            query_type="L1_entity",
            expected_keywords=["علم", "شركة", "تقنية"],
            expected_entity_type="Organization",
            description="Elm company entity"
        ),

        # === L2: Multi-hop / Relationship ===
        TestCase(
            query="ما علاقة التحول الرقمي بالخدمات الحكومية؟",
            query_type="L2_relationship",
            expected_keywords=["تحول", "رقمي", "خدمات", "حكومية"],
            description="Relationship query"
        ),
        TestCase(
            query="كيف تساهم التقنية في تطوير الحكومة؟",
            query_type="L2_causal",
            expected_keywords=["تقنية", "تطوير", "حكومة", "رقمي"],
            description="Causal query"
        ),

        # === L3: Thematic / Holistic ===
        TestCase(
            query="ما هي أهم إنجازات الحكومة الرقمية في السعودية؟",
            query_type="L3_thematic",
            expected_keywords=["إنجاز", "حكومة", "رقمية", "سعودية"],
            description="Thematic query about achievements"
        ),
        TestCase(
            query="ما هي التحديات التي تواجه التحول الرقمي؟",
            query_type="L3_thematic",
            expected_keywords=["تحديات", "تحول", "رقمي"],
            description="Thematic query about challenges"
        ),

        # === Edge Cases ===
        TestCase(
            query="شركة علم",
            query_type="edge_short",
            expected_keywords=["علم"],
            min_chunks=1,
            description="Very short query"
        ),
        TestCase(
            query="ما هي شركة غوغل؟",
            query_type="edge_unknown",
            expected_keywords=[],
            should_find_info=False,
            description="Unknown entity (should gracefully handle)"
        ),
        TestCase(
            query="منصة يمامة",
            query_type="edge_platform",
            expected_keywords=["يمامة", "منصة"],
            description="Platform name query"
        ),

        # === Arabic Normalization Tests ===
        TestCase(
            query="وزارة الاتصالات",
            query_type="arabic_norm",
            expected_keywords=["وزارة", "اتصالات"],
            description="Ministry query (formal)"
        ),
        TestCase(
            query="الهيئة السعودية للبيانات",
            query_type="arabic_norm",
            expected_keywords=["هيئة", "سعودية", "بيانات"],
            description="Saudi Data Authority"
        ),
    ]


def run_mode_evaluation(test_cases: List[TestCase], mode: str) -> List[TestResult]:
    """Run all test cases for a mode."""
    results = []
    for tc in test_cases:
        result = run_test(tc, mode)
        results.append(result)
    return results


def print_results(all_results: Dict[str, List[TestResult]]):
    """Print detailed results."""
    print("\n" + "=" * 80)
    print("STRICT EVALUATION RESULTS (V2 - With Arabic Normalization)")
    print("=" * 80)

    # Per-mode summary
    mode_stats = {}
    for mode, results in all_results.items():
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        avg_latency = sum(r.latency_ms for r in results) / total
        mode_stats[mode] = {
            "passed": passed,
            "total": total,
            "rate": passed / total * 100,
            "avg_latency": avg_latency
        }

    print("\nMODE SUMMARY:")
    print("-" * 60)
    for mode, stats in sorted(mode_stats.items(), key=lambda x: -x[1]["rate"]):
        bar = "█" * int(stats["rate"] / 5) + "░" * (20 - int(stats["rate"] / 5))
        print(f"  {mode:12} [{bar}] {stats['rate']:5.1f}% ({stats['passed']}/{stats['total']}) | avg {stats['avg_latency']:.0f}ms")

    # Detailed failures
    print("\n" + "=" * 80)
    print("FAILURES BY MODE (REAL ISSUES)")
    print("=" * 80)

    for mode, results in all_results.items():
        failures = [r for r in results if not r.passed]
        if failures:
            print(f"\n{mode.upper()} FAILURES ({len(failures)}):")
            print("-" * 60)
            for r in failures:
                print(f"  Query: {r.test_case.query}")
                print(f"    Type: {r.test_case.query_type}")
                print(f"    Latency: {r.latency_ms:.0f}ms | Chunks: {r.chunks_returned} | Relevant@: {r.relevant_chunk_rank}")
                print(f"    Keywords: answer={r.keywords_in_answer}/{len(r.test_case.expected_keywords)}, chunks={r.keywords_in_chunks}")
                for issue in r.issues:
                    print(f"    ✗ {issue}")
                print()

    # Issue breakdown
    print("\n" + "=" * 80)
    print("ISSUE BREAKDOWN (REAL ISSUES)")
    print("=" * 80)

    issue_counts = defaultdict(int)
    for mode, results in all_results.items():
        for r in results:
            for issue in r.issues:
                issue_type = issue.split(":")[0]
                issue_counts[issue_type] += 1

    for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {issue}: {count}")

    # Query type performance
    print("\n" + "=" * 80)
    print("PERFORMANCE BY QUERY TYPE")
    print("=" * 80)

    query_type_stats = defaultdict(lambda: {"passed": 0, "total": 0, "latencies": [], "ranking_issues": 0})
    for mode, results in all_results.items():
        for r in results:
            qt = r.test_case.query_type
            query_type_stats[qt]["total"] += 1
            query_type_stats[qt]["latencies"].append(r.latency_ms)
            if r.passed:
                query_type_stats[qt]["passed"] += 1
            if r.relevant_chunk_rank > 2:
                query_type_stats[qt]["ranking_issues"] += 1

    print(f"\n{'Query Type':<20} {'Pass Rate':>10} {'Avg Latency':>12} {'Rank Issues':>12}")
    print("-" * 60)
    for qt, stats in sorted(query_type_stats.items()):
        rate = stats["passed"] / stats["total"] * 100
        avg_lat = sum(stats["latencies"]) / len(stats["latencies"])
        print(f"  {qt:<18} {rate:>8.1f}% {avg_lat:>10.0f}ms {stats['ranking_issues']:>10}")


def run_ranking_analysis(modes: List[str]):
    """Analyze chunk ranking quality."""
    print("\n" + "=" * 80)
    print("CHUNK RANKING ANALYSIS")
    print("=" * 80)

    test_queries = [
        ("ما هي جائزة الحكومة الرقمية؟", ["جائزة", "رقمية", "شريك"]),
        ("من هي شركة علم؟", ["علم", "شركة"]),
        ("ما هي رؤية 2030؟", ["رؤية", "2030"]),
        ("من هي شركة تيتو؟", ["تيتو"]),
        ("منصة يمامة", ["يمامة"]),
    ]

    for query, keywords in test_queries:
        print(f"\nQuery: {query}")
        print(f"Keywords: {keywords}")

        for mode in modes:
            resp = requests.post(
                f"{BASE_URL}/chat/ask",
                json={"message": query, "retrieval_mode": mode, "top_k": 5},
                timeout=60
            )
            data = resp.json()
            chunks = data.get("chunks", [])

            # Find first relevant chunk
            relevant_pos = -1
            for i, c in enumerate(chunks):
                if keyword_match(c.get("text", ""), keywords) > 0:
                    relevant_pos = i + 1
                    break

            status = "✓" if relevant_pos <= 2 else ("⚠" if relevant_pos > 0 else "✗")
            pos_str = f"@{relevant_pos}" if relevant_pos > 0 else "NOT FOUND"
            print(f"  {mode:12} {status} Relevant chunk: {pos_str}")


def run_latency_analysis():
    """Latency stress test."""
    print("\n" + "=" * 80)
    print("LATENCY STRESS TEST (10 iterations)")
    print("=" * 80)

    query = "ما هي جائزة الحكومة الرقمية؟"
    modes = ["naive", "local", "hybrid", "semantic", "global_search"]

    for mode in modes:
        latencies = []
        for i in range(10):
            start = time.time()
            try:
                resp = requests.post(
                    f"{BASE_URL}/chat/ask",
                    json={"message": query, "retrieval_mode": mode, "top_k": 5},
                    timeout=120
                )
                latency = (time.time() - start) * 1000
                latencies.append(latency)
            except:
                latencies.append(120000)

        avg = sum(latencies) / len(latencies)
        min_l = min(latencies)
        max_l = max(latencies)
        p90 = sorted(latencies)[9]

        status = "✓" if avg < 5000 else "⚠" if avg < 10000 else "✗"
        print(f"  {mode:12} {status} | avg={avg:5.0f}ms | min={min_l:5.0f} | max={max_l:5.0f} | p90={p90:5.0f}")


def check_corpus_coverage():
    """Check what content exists in the corpus."""
    print("\n" + "=" * 80)
    print("CORPUS COVERAGE CHECK")
    print("=" * 80)

    entities_to_check = [
        "جائزة", "جائزه",  # award (formal/colloquial)
        "رؤية", "2030",
        "تيتو",
        "علم",
        "يمامة",
        "زكاة", "زكاه",
        "وزارة", "وزاره",
        "هيئة", "هيئه",
    ]

    print("\nSearching for entities in corpus via keyword search...")
    for entity in entities_to_check:
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/ask",
                json={"message": entity, "retrieval_mode": "naive", "top_k": 10},
                timeout=30
            )
            data = resp.json()
            chunks = data.get("chunks", [])

            # Count chunks containing this keyword
            matches = sum(1 for c in chunks if entity in c.get("text", ""))
            print(f"  '{entity}': {matches}/{len(chunks)} chunks contain this keyword")
        except Exception as e:
            print(f"  '{entity}': ERROR - {e}")


def main():
    print("=" * 80)
    print("MIRAGE STRICT EVALUATION SUITE V2")
    print("With Arabic Normalization & Ranking Analysis")
    print("=" * 80)
    print(f"Target: {BASE_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Get test cases
    test_cases = get_test_cases()
    print(f"\nTest cases: {len(test_cases)}")

    # 2. Run tests for each mode
    modes = ["naive", "local", "hybrid", "semantic"]
    all_results = {}

    for mode in modes:
        print(f"\nTesting {mode}...")
        results = run_mode_evaluation(test_cases, mode)
        all_results[mode] = results
        passed = sum(1 for r in results if r.passed)
        print(f"  {passed}/{len(results)} passed")

    # 3. Print detailed results
    print_results(all_results)

    # 4. Ranking analysis
    run_ranking_analysis(modes)

    # 5. Latency analysis
    run_latency_analysis()

    # 6. Corpus coverage
    check_corpus_coverage()

    # 7. Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    total_tests = sum(len(r) for r in all_results.values())
    total_passed = sum(sum(1 for r in results if r.passed) for results in all_results.values())
    overall_rate = total_passed / total_tests * 100

    # Critical issues
    critical_issues = []
    warnings = []

    # Check for major failures
    for mode, results in all_results.items():
        mode_rate = sum(1 for r in results if r.passed) / len(results) * 100
        if mode_rate < 50:
            critical_issues.append(f"{mode} mode: {mode_rate:.0f}% pass rate")
        elif mode_rate < 70:
            warnings.append(f"{mode} mode: {mode_rate:.0f}% pass rate")

    # Issue counts
    issue_counts = defaultdict(int)
    for mode, results in all_results.items():
        for r in results:
            for issue in r.issues:
                issue_type = issue.split(":")[0]
                issue_counts[issue_type] += 1

    print(f"\nOverall Pass Rate: {overall_rate:.1f}% ({total_passed}/{total_tests})")

    if critical_issues:
        print(f"\nCritical Issues ({len(critical_issues)}):")
        for issue in critical_issues:
            print(f"  ✗ {issue}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")

    print(f"\nIssue Distribution:")
    for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        pct = count / total_tests * 100
        print(f"  {issue}: {count} ({pct:.1f}%)")

    # Verdict
    if overall_rate >= 75 and len(critical_issues) == 0:
        print("\n✓ PRODUCTION READY")
    elif overall_rate >= 60:
        print("\n⚠ NEEDS IMPROVEMENT - Address ranking and corpus gaps")
    else:
        print("\n✗ NOT READY - Major issues found")


if __name__ == "__main__":
    main()
