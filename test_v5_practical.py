#!/usr/bin/env python3
"""
MIRAGE V5 Practical Test Suite
Tests via API and direct imports with correct path.
"""

import sys
import os
import time
import json
import requests
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test results
RESULTS = {
    "tests": [],
    "passed": 0,
    "failed": 0,
    "warnings": [],
    "critical_issues": []
}


def log_test(name: str, passed: bool, details: str = "", duration_ms: float = 0):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {name} ({duration_ms:.1f}ms)")
    if details:
        print(f"      └─ {details}")

    RESULTS["tests"].append({
        "name": name,
        "passed": passed,
        "details": details,
        "duration_ms": duration_ms
    })

    if passed:
        RESULTS["passed"] += 1
    else:
        RESULTS["failed"] += 1


def log_warning(msg: str):
    print(f"⚠️  WARNING: {msg}")
    RESULTS["warnings"].append(msg)


def log_critical(msg: str):
    print(f"🚨 CRITICAL: {msg}")
    RESULTS["critical_issues"].append(msg)


# =============================================================================
# TEST 1: API Health and Basic Functionality
# =============================================================================
def test_api_health():
    """Test API is healthy and responding."""
    print("\n" + "="*60)
    print("TEST 1: API Health Check")
    print("="*60)

    try:
        start = time.time()
        response = requests.get("http://localhost:8000/health", timeout=10)
        duration = (time.time() - start) * 1000

        log_test("API health endpoint", response.status_code == 200,
                 f"Status: {response.status_code}", duration)

        # Check documents endpoint
        start = time.time()
        response = requests.get("http://localhost:8000/documents", timeout=10)
        duration = (time.time() - start) * 1000

        if response.status_code == 200:
            docs = response.json()
            doc_count = len(docs.get("documents", []))
            log_test("Documents endpoint", True, f"Found {doc_count} documents", duration)
        else:
            log_test("Documents endpoint", False, f"Status: {response.status_code}", duration)

    except Exception as e:
        log_test("API health", False, str(e), 0)
        log_critical(f"API not accessible: {e}")


# =============================================================================
# TEST 2: Component Imports
# =============================================================================
def test_component_imports():
    """Test all V5 components can be imported."""
    print("\n" + "="*60)
    print("TEST 2: Component Imports")
    print("="*60)

    components = [
        ("core.retrieval.community_selector", "DynamicCommunitySelector"),
        ("core.retrieval.hyde", "HyDEQueryEnhancer"),
        ("core.retrieval.dual_level_retrieval", "DualLevelRetriever"),
        ("core.retrieval.hippocampal_retrieval", "HippocampalRetriever"),
        ("core.retrieval.observability", "RAGObservability"),
        ("core.retrieval.v5_engine", "MIRAGEV5Engine"),
        ("core.graph_builder.incremental_updater", "IncrementalGraphUpdater"),
        ("core.graph_builder.coreference_resolver", "CoreferenceResolver"),
    ]

    for module_name, class_name in components:
        try:
            start = time.time()
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            duration = (time.time() - start) * 1000
            log_test(f"Import {class_name}", True, f"From {module_name}", duration)
        except Exception as e:
            log_test(f"Import {class_name}", False, str(e), 0)
            log_critical(f"Cannot import {class_name}: {e}")


# =============================================================================
# TEST 3: PPR Algorithm (Pure Python)
# =============================================================================
def test_ppr_algorithm():
    """Test Personalized PageRank algorithm."""
    print("\n" + "="*60)
    print("TEST 3: PPR Algorithm")
    print("="*60)

    try:
        from core.retrieval.hippocampal_retrieval import PersonalizedPageRank

        ppr = PersonalizedPageRank(damping_factor=0.85, max_iterations=50)

        # Test graph: A -> B -> C -> D
        test_graph = {
            "A": ["B", "C"],
            "B": ["A", "C", "D"],
            "C": ["A", "B", "D"],
            "D": ["B", "C"]
        }

        start = time.time()
        result = ppr.compute(test_graph, seed_nodes=["A"])
        duration = (time.time() - start) * 1000

        log_test(
            "PPR computation",
            result.converged,
            f"Converged in {result.iterations} iterations",
            duration
        )

        # Check scores distribution
        scores = result.node_scores
        if scores:
            seed_score = scores.get("A", 0)
            max_score = max(scores.values())
            min_score = min(scores.values())

            log_test(
                "PPR score distribution",
                seed_score > 0,
                f"Seed(A): {seed_score:.4f}, Max: {max_score:.4f}, Min: {min_score:.4f}",
                0
            )

            # Seed should be among top scores due to teleportation
            if seed_score < max_score * 0.3:
                log_warning(f"Seed score unexpectedly low: {seed_score:.4f}")

        # Test multi-seed
        result2 = ppr.compute(test_graph, seed_nodes=["A", "D"])
        log_test(
            "PPR multi-seed",
            result2.converged,
            f"Nodes scored: {len(result2.node_scores)}",
            0
        )

    except Exception as e:
        log_test("PPR Algorithm", False, str(e), 0)
        log_critical(f"PPR failed: {traceback.format_exc()}")


# =============================================================================
# TEST 4: Coreference Resolution (Pure Python)
# =============================================================================
def test_coreference_resolution():
    """Test entity coreference resolution."""
    print("\n" + "="*60)
    print("TEST 4: Coreference Resolution")
    print("="*60)

    try:
        from core.graph_builder.coreference_resolver import CoreferenceResolver

        resolver = CoreferenceResolver(use_llm=False)
        log_test("Initialize resolver", True, "Created without LLM")

        # Test built-in aliases
        test_cases = [
            ("السعودية", "المملكة العربية السعودية", "Arabic alias"),
            ("KSA", "المملكة العربية السعودية", "English acronym"),
            ("AI", "الذكاء الاصطناعي", "Tech term"),
            ("UAE", "الإمارات العربية المتحدة", "Country"),
        ]

        for alias, expected, desc in test_cases:
            start = time.time()
            result = resolver.resolve(alias)
            duration = (time.time() - start) * 1000

            passed = result.canonical == expected
            log_test(
                f"Resolve '{alias}' ({desc})",
                passed,
                f"→ '{result.canonical}' (expected: '{expected}')",
                duration
            )

        # Test title removal
        title_cases = [
            "الدكتور أحمد",
            "Dr. John",
            "المهندس محمد"
        ]

        for name in title_cases:
            result = resolver.resolve(name)
            has_title = any(t in result.canonical for t in ["الدكتور", "Dr.", "المهندس"])
            log_test(
                f"Title removal: '{name[:15]}'",
                not has_title or result.method == "normalized",
                f"→ '{result.canonical}'",
                0
            )

        # Test custom alias addition
        resolver.add_alias("وزارة التعليم", "MOE")
        result = resolver.resolve("MOE")
        log_test(
            "Custom alias",
            result.canonical == "وزارة التعليم",
            f"Added MOE → {result.canonical}",
            0
        )

    except Exception as e:
        log_test("Coreference Resolution", False, str(e), 0)
        log_critical(f"Coreference failed: {traceback.format_exc()}")


# =============================================================================
# TEST 5: Observability (Pure Python)
# =============================================================================
def test_observability():
    """Test production observability."""
    print("\n" + "="*60)
    print("TEST 5: Observability")
    print("="*60)

    try:
        from core.retrieval.observability import (
            RAGObservability,
            CostTracker,
            QueryTrace
        )

        obs = RAGObservability()
        log_test("Initialize observability", True, "Created")

        # Test tracing
        trace = obs.start_trace("test query")
        time.sleep(0.01)

        with obs.record_step(trace, "step1") as step:
            step.metadata["test"] = True
            time.sleep(0.005)

        with obs.record_step(trace, "step2") as step:
            step.metadata["chunks"] = 10

        obs.end_trace(trace, success=True)

        log_test(
            "Query tracing",
            trace.total_duration_ms > 0,
            f"Duration: {trace.total_duration_ms:.1f}ms, Steps: {len(trace.steps)}",
            0
        )

        # Test step timing
        step_timings = trace.get_step_timings()
        log_test(
            "Step timing",
            "step1" in step_timings and "step2" in step_timings,
            f"Steps: {list(step_timings.keys())}",
            0
        )

        # Test metrics
        obs.record_mode("test_mode")
        obs.record_tokens(100, 50)
        obs.record_cache_hit(True)
        obs.record_cache_hit(False)

        metrics = obs.get_metrics()
        log_test(
            "Metrics collection",
            metrics.total_queries > 0,
            f"Queries: {metrics.total_queries}, Cache hit: {metrics.cache_hit_rate:.1%}",
            0
        )

        # Test cost tracker
        cost = CostTracker()
        cost.record_usage("embedding", 1000)
        cost.record_usage("generation", 500)

        total = cost.get_total_cost()
        log_test(
            "Cost tracking",
            total > 0,
            f"Total: ${total:.6f}",
            0
        )

        # Test dashboard
        dashboard = obs.get_dashboard_data()
        required_keys = ["metrics", "recent_traces"]
        has_keys = all(k in dashboard for k in required_keys)
        log_test(
            "Dashboard data",
            has_keys,
            f"Keys: {list(dashboard.keys())}",
            0
        )

    except Exception as e:
        log_test("Observability", False, str(e), 0)
        log_critical(f"Observability failed: {traceback.format_exc()}")


# =============================================================================
# TEST 6: HyDE Enhancement
# =============================================================================
def test_hyde_enhancement():
    """Test HyDE query enhancement."""
    print("\n" + "="*60)
    print("TEST 6: HyDE Enhancement")
    print("="*60)

    try:
        from core.retrieval.hyde import HyDEQueryEnhancer
        from core.models.embedding_manager import get_embedding_manager

        embedder = get_embedding_manager()
        log_test("Get embedding manager", embedder is not None, "Loaded")

        enhancer = HyDEQueryEnhancer(
            embedder=embedder,
            llm_endpoint="http://tgi:80",
            alpha=0.5
        )
        log_test("Initialize HyDE", True, "Created")

        # Test enhancement
        test_queries = [
            ("ما هي جائزة الحكومة الرقمية؟", "Arabic"),
            ("What is digital transformation?", "English"),
        ]

        for query, lang in test_queries:
            start = time.time()
            try:
                result = enhancer.enhance(query, mode="combined")
                duration = (time.time() - start) * 1000

                has_hypothetical = len(result.hypothetical_answer or "") > 10
                has_embedding = result.combined_embedding is not None

                if not has_hypothetical:
                    log_warning(f"No/short hypothetical for: {query[:30]}")

                log_test(
                    f"HyDE ({lang})",
                    has_embedding,
                    f"Hypothetical: '{(result.hypothetical_answer or '')[:40]}...' "
                    f"(len: {len(result.hypothetical_answer or '')})",
                    duration
                )
            except Exception as e:
                log_test(f"HyDE ({lang})", False, str(e), 0)

    except Exception as e:
        log_test("HyDE Enhancement", False, str(e), 0)
        log_critical(f"HyDE failed: {traceback.format_exc()}")


# =============================================================================
# TEST 7: API-based Retrieval Comparison
# =============================================================================
def test_api_retrieval():
    """Test retrieval via API."""
    print("\n" + "="*60)
    print("TEST 7: API Retrieval")
    print("="*60)

    test_queries = [
        ("ما هي جائزة الحكومة الرقمية؟", "naive"),
        ("ما هي جائزة الحكومة الرقمية؟", "local"),
        ("ما هي جائزة الحكومة الرقمية؟", "hybrid"),
        ("ما هي جائزة الحكومة الرقمية؟", "semantic"),
    ]

    for query, mode in test_queries:
        try:
            start = time.time()
            response = requests.post(
                "http://localhost:8000/chat/ask",
                json={
                    "message": query,
                    "retrieval_mode": mode,
                    "top_k": 5
                },
                timeout=60
            )
            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                sources = len(data.get("sources", []))

                has_answer = len(answer) > 20
                log_test(
                    f"Retrieve ({mode})",
                    has_answer,
                    f"Answer: {len(answer)} chars, Sources: {sources}",
                    duration
                )
            else:
                log_test(f"Retrieve ({mode})", False, f"Status: {response.status_code}", duration)

        except Exception as e:
            log_test(f"Retrieve ({mode})", False, str(e), 0)


# =============================================================================
# TEST 8: V5 Components Integration
# =============================================================================
def test_v5_integration():
    """Test V5 component integration."""
    print("\n" + "="*60)
    print("TEST 8: V5 Integration")
    print("="*60)

    try:
        from core.retrieval.v5_engine import MIRAGEV5Engine, V5Config

        # Test with minimal config (no external dependencies)
        config = V5Config(
            use_hyde=False,  # Skip LLM-dependent features first
            use_dual_level=True,
            use_hippocampal=False,  # Skip graph-dependent features
            use_dynamic_community_selection=False,
            enable_tracing=True,
            enable_cost_tracking=True
        )

        engine = MIRAGEV5Engine(config=config)
        log_test("V5 Engine init (minimal)", True, "Created")

        # Test config options
        log_test(
            "V5 config",
            True,
            f"hyde={config.use_hyde}, dual={config.use_dual_level}, "
            f"ppr={config.use_hippocampal}",
            0
        )

        # Test metrics method
        try:
            metrics = engine.get_metrics()
            log_test("V5 metrics", True, f"Keys: {list(metrics.keys())[:3]}...", 0)
        except Exception as e:
            log_test("V5 metrics", False, str(e), 0)

        # Test dashboard method
        try:
            dashboard = engine.get_dashboard_data()
            log_test("V5 dashboard", True, f"Keys: {list(dashboard.keys())}", 0)
        except Exception as e:
            log_test("V5 dashboard", False, str(e), 0)

        # Test full config
        full_config = V5Config(
            use_hyde=True,
            use_dual_level=True,
            use_hippocampal=True,
            use_dynamic_community_selection=True,
            enable_tracing=True
        )

        full_engine = MIRAGEV5Engine(config=full_config)
        log_test("V5 Engine init (full)", True, "Created with all features")

    except Exception as e:
        log_test("V5 Integration", False, str(e), 0)
        log_critical(f"V5 integration failed: {traceback.format_exc()}")


# =============================================================================
# TEST 9: Performance Benchmarks
# =============================================================================
def test_performance():
    """Test performance benchmarks."""
    print("\n" + "="*60)
    print("TEST 9: Performance Benchmarks")
    print("="*60)

    # API response times
    query = "ما هي جائزة الحكومة الرقمية؟"
    modes = ["naive", "local", "hybrid"]

    for mode in modes:
        times = []
        for _ in range(3):
            try:
                start = time.time()
                response = requests.post(
                    "http://localhost:8000/chat/ask",
                    json={"message": query, "retrieval_mode": mode, "top_k": 5},
                    timeout=60
                )
                times.append((time.time() - start) * 1000)
            except:
                pass

        if times:
            avg = sum(times) / len(times)
            min_t = min(times)
            max_t = max(times)

            # Acceptable: < 5 seconds for any mode
            passed = avg < 5000
            log_test(
                f"Latency ({mode})",
                passed,
                f"Avg: {avg:.0f}ms, Min: {min_t:.0f}ms, Max: {max_t:.0f}ms",
                avg
            )

            if avg > 3000:
                log_warning(f"{mode} mode is slow: {avg:.0f}ms average")
        else:
            log_test(f"Latency ({mode})", False, "No successful requests", 0)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*60)
    print("MIRAGE V5 PRACTICAL TEST SUITE")
    print("Critical Evaluation")
    print("="*60)

    test_api_health()
    test_component_imports()
    test_ppr_algorithm()
    test_coreference_resolution()
    test_observability()
    test_hyde_enhancement()
    test_api_retrieval()
    test_v5_integration()
    test_performance()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total = RESULTS["passed"] + RESULTS["failed"]
    pass_rate = RESULTS["passed"] / total * 100 if total > 0 else 0

    print(f"\nTotal: {total}")
    print(f"Passed: {RESULTS['passed']} ({pass_rate:.1f}%)")
    print(f"Failed: {RESULTS['failed']}")

    if RESULTS["warnings"]:
        print(f"\n⚠️  Warnings ({len(RESULTS['warnings'])}):")
        for w in RESULTS["warnings"][:10]:
            print(f"   - {w[:80]}")

    if RESULTS["critical_issues"]:
        print(f"\n🚨 Critical ({len(RESULTS['critical_issues'])}):")
        for c in RESULTS["critical_issues"][:5]:
            print(f"   - {c[:100]}")

    # Score
    print("\n" + "="*60)
    print("PRACTICAL SCORE")
    print("="*60)

    # Components
    component_scores = {
        "Pass Rate": pass_rate / 10,
        "No Critical Issues": 10 if not RESULTS["critical_issues"] else max(0, 10 - len(RESULTS["critical_issues"]) * 2),
        "Low Warnings": max(0, 10 - len(RESULTS["warnings"])),
    }

    print("\nScores:")
    for name, score in component_scores.items():
        print(f"  {name}: {score:.1f}/10")

    final_score = sum(component_scores.values()) / len(component_scores)
    print(f"\n>>> PRACTICAL SCORE: {final_score:.1f}/10 <<<")

    return final_score


if __name__ == "__main__":
    main()
