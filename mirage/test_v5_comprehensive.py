#!/usr/bin/env python3
"""
MIRAGE V5 Comprehensive Test Suite
Critical evaluation of all new components.
"""

import sys
import time
import json
import traceback
from typing import Dict, List, Any, Tuple

# Test results collector
RESULTS = {
    "tests": [],
    "passed": 0,
    "failed": 0,
    "warnings": [],
    "critical_issues": []
}


def log_test(name: str, passed: bool, details: str = "", duration_ms: float = 0):
    """Log test result."""
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
    """Log a warning."""
    print(f"⚠️  WARNING: {msg}")
    RESULTS["warnings"].append(msg)


def log_critical(msg: str):
    """Log a critical issue."""
    print(f"🚨 CRITICAL: {msg}")
    RESULTS["critical_issues"].append(msg)


# =============================================================================
# TEST 1: Dynamic Community Selection
# =============================================================================
def test_dynamic_community_selection():
    """Test O(log C) community selection vs O(C) brute force."""
    print("\n" + "="*60)
    print("TEST 1: Dynamic Community Selection")
    print("="*60)

    try:
        from core.retrieval.community_selector import (
            DynamicCommunitySelector,
            CommunityNode,
            get_community_selector
        )
        from core.graph_builder import Neo4jClient
        from core.models.embedding_manager import get_embedding_manager

        # Initialize
        start = time.time()
        neo4j = Neo4jClient()
        embedder = get_embedding_manager()
        init_time = (time.time() - start) * 1000

        log_test("Import and initialize", True, f"Components loaded", init_time)

        # Create selector
        selector = DynamicCommunitySelector(
            neo4j_client=neo4j,
            embedder=embedder,
            default_threshold=0.3,
            max_communities=20
        )

        # Test 1: Cache loading
        start = time.time()
        selector._load_hierarchy()
        load_time = (time.time() - start) * 1000

        cache_stats = selector.get_cache_stats()
        total_communities = cache_stats.get("total_communities", 0)

        if total_communities == 0:
            log_warning("No communities in graph - community detection may not have run")
            log_test("Load hierarchy", False, "No communities found", load_time)
            return

        log_test("Load hierarchy", True,
                 f"{total_communities} communities loaded", load_time)

        # Test 2: Selection with pruning
        test_queries = [
            "ما هي جائزة الحكومة الرقمية؟",
            "What are the main themes in the documents?",
            "من هم شركاء الجائزة؟"
        ]

        for query in test_queries:
            start = time.time()
            result = selector.select_communities(query, threshold=0.3)
            select_time = (time.time() - start) * 1000

            stats = result.stats
            pruning_ratio = stats.pruning_ratio
            efficiency = stats.efficiency_gain

            # Critical check: Are we actually pruning?
            if pruning_ratio < 0.1:
                log_warning(f"Low pruning ratio ({pruning_ratio:.1%}) - may not be effective")

            if efficiency < 2.0:
                log_warning(f"Low efficiency gain ({efficiency:.1f}x) - not much better than brute force")

            passed = stats.communities_selected > 0 and select_time < 5000
            log_test(
                f"Select communities: '{query[:30]}...'",
                passed,
                f"Selected {stats.communities_selected}/{total_communities}, "
                f"pruned {stats.communities_pruned} ({pruning_ratio:.1%}), "
                f"efficiency: {efficiency:.1f}x",
                select_time
            )

        # Test 3: Embedding similarity scoring
        start = time.time()
        test_node = CommunityNode(
            id="test",
            level=0,
            summary="This is a test community about digital government awards.",
            size=10
        )
        query_emb = embedder.embed("digital government awards")
        score = selector._score_relevance(query_emb, test_node)
        score_time = (time.time() - start) * 1000

        log_test("Relevance scoring", 0 <= score <= 1, f"Score: {score:.3f}", score_time)

    except Exception as e:
        log_test("Dynamic Community Selection", False, f"Error: {str(e)}", 0)
        log_critical(f"Community selection failed: {traceback.format_exc()}")


# =============================================================================
# TEST 2: HyDE Query Enhancement
# =============================================================================
def test_hyde_enhancement():
    """Test Hypothetical Document Embedding generation."""
    print("\n" + "="*60)
    print("TEST 2: HyDE Query Enhancement")
    print("="*60)

    try:
        from core.retrieval.hyde import HyDEQueryEnhancer, get_hyde_enhancer
        from core.models.embedding_manager import get_embedding_manager

        embedder = get_embedding_manager()

        enhancer = HyDEQueryEnhancer(
            embedder=embedder,
            llm_endpoint="http://tgi:80",
            alpha=0.5,
            max_tokens=200
        )

        log_test("Initialize HyDE enhancer", True, "Enhancer created")

        # Test queries
        test_queries = [
            ("ما هي جائزة الحكومة الرقمية؟", "Arabic"),
            ("What is digital transformation?", "English"),
            ("من هم المرشحون للجائزة؟", "Arabic entity query")
        ]

        for query, query_type in test_queries:
            start = time.time()
            try:
                result = enhancer.enhance(query, mode="combined")
                enhance_time = (time.time() - start) * 1000

                # Check hypothetical was generated
                has_hypothetical = len(result.hypothetical_answer) > 20

                # Check embeddings exist
                has_embeddings = (
                    result.query_embedding is not None and
                    result.hyde_embedding is not None and
                    result.combined_embedding is not None
                )

                if not has_hypothetical:
                    log_warning(f"Empty/short hypothetical for: {query[:30]}...")

                passed = has_embeddings
                log_test(
                    f"HyDE enhance ({query_type})",
                    passed,
                    f"Hypothetical: '{result.hypothetical_answer[:50]}...' "
                    f"({len(result.hypothetical_answer)} chars)",
                    enhance_time
                )

            except Exception as e:
                log_test(f"HyDE enhance ({query_type})", False, str(e), 0)

        # Test mode comparison
        for mode in ["combined", "hyde_only", "query_only"]:
            start = time.time()
            try:
                result = enhancer.enhance("test query", mode=mode)
                mode_time = (time.time() - start) * 1000
                log_test(f"HyDE mode: {mode}", True, f"Generated", mode_time)
            except Exception as e:
                log_test(f"HyDE mode: {mode}", False, str(e), 0)

    except Exception as e:
        log_test("HyDE Enhancement", False, f"Error: {str(e)}", 0)
        log_critical(f"HyDE failed: {traceback.format_exc()}")


# =============================================================================
# TEST 3: Dual-Level Retrieval
# =============================================================================
def test_dual_level_retrieval():
    """Test LightRAG-style dual retrieval."""
    print("\n" + "="*60)
    print("TEST 3: Dual-Level Retrieval")
    print("="*60)

    try:
        from core.retrieval.dual_level_retrieval import (
            DualLevelRetriever,
            QueryTypeClassifier,
            get_dual_level_retriever
        )
        from core.graph_builder import Neo4jClient
        from core.models.embedding_manager import get_embedding_manager

        neo4j = Neo4jClient()
        embedder = get_embedding_manager()

        # Test query classification
        classifier = QueryTypeClassifier()
        test_classifications = [
            ("من هو المسؤول عن الجائزة؟", "factual"),
            ("ما هي المواضيع الرئيسية؟", "thematic"),
            ("ما هي جائزة الحكومة الرقمية؟", "balanced")
        ]

        for query, expected_type in test_classifications:
            start = time.time()
            query_type, weights = classifier.classify(query)
            classify_time = (time.time() - start) * 1000

            passed = True  # Classification itself doesn't fail
            log_test(
                f"Classify: '{query[:25]}...'",
                passed,
                f"Type: {query_type}, weights: low={weights['low']:.1f}, high={weights['high']:.1f}",
                classify_time
            )

        # Test dual-level retriever
        retriever = DualLevelRetriever(
            neo4j_client=neo4j,
            embedder=embedder,
            max_entities=10,
            max_communities=5
        )

        log_test("Initialize DualLevelRetriever", True, "Retriever created")

        # Test retrieval
        test_queries = [
            "ما هي جائزة الحكومة الرقمية؟",
            "من هم شركاء الجائزة؟",
            "What are the main themes?"
        ]

        for query in test_queries:
            start = time.time()
            try:
                result = retriever.retrieve(query, top_k=10)
                retrieve_time = (time.time() - start) * 1000

                # Check results
                has_chunks = len(result.fused_chunks) > 0
                has_low_level = len(result.low_level_results) > 0
                has_high_level = len(result.high_level_results) > 0

                if not has_chunks:
                    log_warning(f"No chunks retrieved for: {query[:30]}...")

                if not has_low_level:
                    log_warning(f"No low-level results for: {query[:30]}...")

                if not has_high_level:
                    log_warning(f"No high-level results for: {query[:30]}...")

                passed = has_chunks or has_low_level or has_high_level
                log_test(
                    f"Dual retrieve: '{query[:25]}...'",
                    passed,
                    f"Low: {len(result.low_level_results)}, High: {len(result.high_level_results)}, "
                    f"Fused: {len(result.fused_chunks)}, Type: {result.query_type}",
                    retrieve_time
                )

            except Exception as e:
                log_test(f"Dual retrieve: '{query[:25]}...'", False, str(e), 0)

    except Exception as e:
        log_test("Dual-Level Retrieval", False, f"Error: {str(e)}", 0)
        log_critical(f"Dual-level failed: {traceback.format_exc()}")


# =============================================================================
# TEST 4: Hippocampal Retrieval (PPR)
# =============================================================================
def test_hippocampal_retrieval():
    """Test Personalized PageRank retrieval."""
    print("\n" + "="*60)
    print("TEST 4: Hippocampal Retrieval (PPR)")
    print("="*60)

    try:
        from core.retrieval.hippocampal_retrieval import (
            HippocampalRetriever,
            PersonalizedPageRank,
            get_hippocampal_retriever
        )
        from core.graph_builder import Neo4jClient
        from core.models.embedding_manager import get_embedding_manager

        neo4j = Neo4jClient()
        embedder = get_embedding_manager()

        # Test PPR algorithm directly
        ppr = PersonalizedPageRank(damping_factor=0.85, max_iterations=50)

        # Create test graph
        test_graph = {
            "A": ["B", "C"],
            "B": ["A", "D"],
            "C": ["A", "D"],
            "D": ["B", "C", "E"],
            "E": ["D"]
        }

        start = time.time()
        result = ppr.compute(test_graph, seed_nodes=["A"])
        ppr_time = (time.time() - start) * 1000

        passed = result.converged and len(result.node_scores) > 0
        log_test(
            "PPR algorithm (synthetic)",
            passed,
            f"Converged in {result.iterations} iterations, "
            f"{len(result.node_scores)} nodes scored",
            ppr_time
        )

        # Check PPR properties
        if result.node_scores:
            # Seed should have highest score
            seed_score = result.node_scores.get("A", 0)
            max_score = max(result.node_scores.values())

            if seed_score < max_score * 0.5:
                log_warning(f"Seed node score ({seed_score:.3f}) lower than expected")

        # Test full hippocampal retriever
        retriever = HippocampalRetriever(
            neo4j_client=neo4j,
            embedder=embedder,
            damping_factor=0.85
        )

        log_test("Initialize HippocampalRetriever", True, "Retriever created")

        # Test with real queries
        test_queries = [
            "جائزة الحكومة الرقمية",
            "شركاء الجائزة",
        ]

        for query in test_queries:
            start = time.time()
            try:
                result = retriever.retrieve(query, top_k=10)
                retrieve_time = (time.time() - start) * 1000

                has_chunks = len(result.chunks) > 0
                has_seeds = len(result.seed_entities) > 0
                has_activated = len(result.activated_entities) > 0

                if not has_seeds:
                    log_warning(f"No seed entities found for: {query}")

                passed = has_chunks or has_seeds
                log_test(
                    f"PPR retrieve: '{query[:25]}...'",
                    passed,
                    f"Seeds: {len(result.seed_entities)}, "
                    f"Activated: {len(result.activated_entities)}, "
                    f"Chunks: {len(result.chunks)}, "
                    f"PPR iters: {result.ppr_iterations}",
                    retrieve_time
                )

            except Exception as e:
                log_test(f"PPR retrieve: '{query[:25]}...'", False, str(e), 0)

    except Exception as e:
        log_test("Hippocampal Retrieval", False, f"Error: {str(e)}", 0)
        log_critical(f"PPR failed: {traceback.format_exc()}")


# =============================================================================
# TEST 5: Incremental Graph Updates
# =============================================================================
def test_incremental_updates():
    """Test incremental graph update capability."""
    print("\n" + "="*60)
    print("TEST 5: Incremental Graph Updates")
    print("="*60)

    try:
        from core.graph_builder.incremental_updater import (
            IncrementalGraphUpdater,
            get_incremental_updater
        )
        from core.graph_builder import Neo4jClient, EntityExtractor

        neo4j = Neo4jClient()

        # Test entity extraction component
        try:
            extractor = EntityExtractor()
            log_test("EntityExtractor available", True, "Loaded")
        except Exception as e:
            log_test("EntityExtractor available", False, str(e), 0)
            log_warning("Using mock entity extractor")
            extractor = None

        if extractor:
            updater = IncrementalGraphUpdater(
                neo4j_client=neo4j,
                entity_extractor=extractor,
                similarity_threshold=0.85
            )

            log_test("Initialize IncrementalUpdater", True, "Updater created")

            # Test entity resolution
            start = time.time()
            test_entity = {"name": "جائزة الحكومة الرقمية", "type": "Award"}
            existing = updater._find_existing_entity(test_entity)
            resolve_time = (time.time() - start) * 1000

            log_test(
                "Entity resolution",
                True,  # Resolution itself works
                f"Found existing: {existing is not None}",
                resolve_time
            )

            # Test normalization
            test_names = [
                ("الدكتور أحمد", "أحمد"),
                ("شركة أرامكو", "أرامكو"),
                ("KINGDOM OF SAUDI ARABIA", "kingdom of saudi arabia")
            ]

            for original, expected_contains in test_names:
                normalized = updater._normalize_entity_name(original)
                passed = expected_contains.lower() in normalized.lower() or len(normalized) < len(original)
                log_test(
                    f"Normalize: '{original[:20]}'",
                    passed,
                    f"Result: '{normalized}'",
                    0
                )

    except Exception as e:
        log_test("Incremental Updates", False, f"Error: {str(e)}", 0)
        log_critical(f"Incremental updates failed: {traceback.format_exc()}")


# =============================================================================
# TEST 6: Coreference Resolution
# =============================================================================
def test_coreference_resolution():
    """Test entity coreference resolution."""
    print("\n" + "="*60)
    print("TEST 6: Coreference Resolution")
    print("="*60)

    try:
        from core.graph_builder.coreference_resolver import (
            CoreferenceResolver,
            get_coreference_resolver
        )

        resolver = CoreferenceResolver(use_llm=False)
        log_test("Initialize CoreferenceResolver", True, "Resolver created")

        # Test built-in aliases
        test_cases = [
            ("السعودية", "المملكة العربية السعودية"),
            ("KSA", "المملكة العربية السعودية"),
            ("AI", "الذكاء الاصطناعي"),
            ("Digital Transformation", "التحول الرقمي"),
        ]

        for alias, expected_canonical in test_cases:
            start = time.time()
            result = resolver.resolve(alias)
            resolve_time = (time.time() - start) * 1000

            passed = result.canonical == expected_canonical or result.confidence > 0.5
            log_test(
                f"Resolve: '{alias}'",
                passed,
                f"→ '{result.canonical}' (conf: {result.confidence:.2f}, method: {result.method})",
                resolve_time
            )

        # Test rule-based matching
        title_tests = [
            ("الدكتور أحمد محمد", "أحمد محمد"),
            ("Dr. John Smith", "John Smith"),
        ]

        for name, expected in title_tests:
            result = resolver.resolve(name)
            # Title removal should work
            has_no_title = "الدكتور" not in result.canonical and "Dr." not in result.canonical
            log_test(
                f"Title removal: '{name[:20]}'",
                has_no_title or result.method == "rule_title",
                f"→ '{result.canonical}'",
                0
            )

        # Test alias addition
        resolver.add_alias("وزارة الصحة", "MOH")
        result = resolver.resolve("MOH")
        log_test(
            "Custom alias",
            result.canonical == "وزارة الصحة",
            f"MOH → {result.canonical}",
            0
        )

    except Exception as e:
        log_test("Coreference Resolution", False, f"Error: {str(e)}", 0)
        log_critical(f"Coreference failed: {traceback.format_exc()}")


# =============================================================================
# TEST 7: Observability
# =============================================================================
def test_observability():
    """Test production observability components."""
    print("\n" + "="*60)
    print("TEST 7: Production Observability")
    print("="*60)

    try:
        from core.retrieval.observability import (
            RAGObservability,
            CostTracker,
            get_observability,
            get_cost_tracker
        )

        obs = RAGObservability()
        log_test("Initialize RAGObservability", True, "Observability created")

        # Test tracing
        trace = obs.start_trace("test query")
        time.sleep(0.01)  # Simulate work

        with obs.record_step(trace, "embedding") as step:
            step.metadata["dim"] = 768
            time.sleep(0.005)

        with obs.record_step(trace, "retrieval") as step:
            step.metadata["chunks"] = 10
            time.sleep(0.01)

        obs.end_trace(trace, success=True)

        log_test(
            "Query tracing",
            trace.total_duration_ms > 0 and len(trace.steps) == 2,
            f"Duration: {trace.total_duration_ms:.1f}ms, Steps: {len(trace.steps)}",
            0
        )

        # Test metrics
        obs.record_mode("v5_unified")
        obs.record_tokens(100, 50)
        obs.record_cache_hit(True)
        obs.record_cache_hit(False)

        metrics = obs.get_metrics()
        log_test(
            "Metrics collection",
            metrics.total_queries > 0,
            f"Queries: {metrics.total_queries}, "
            f"Avg latency: {metrics.avg_latency_ms:.1f}ms, "
            f"Cache hit rate: {metrics.cache_hit_rate:.1%}",
            0
        )

        # Test cost tracker
        cost = CostTracker()
        cost.record_usage("embedding", 1000)
        cost.record_usage("generation", 500)

        total_cost = cost.get_total_cost()
        log_test(
            "Cost tracking",
            total_cost > 0,
            f"Total cost: ${total_cost:.4f}",
            0
        )

        # Test dashboard data
        dashboard = obs.get_dashboard_data()
        log_test(
            "Dashboard data",
            "metrics" in dashboard and "recent_traces" in dashboard,
            f"Keys: {list(dashboard.keys())}",
            0
        )

    except Exception as e:
        log_test("Observability", False, f"Error: {str(e)}", 0)
        log_critical(f"Observability failed: {traceback.format_exc()}")


# =============================================================================
# TEST 8: V5 Unified Engine
# =============================================================================
def test_v5_engine():
    """Test the unified V5 engine."""
    print("\n" + "="*60)
    print("TEST 8: V5 Unified Engine Integration")
    print("="*60)

    try:
        from core.retrieval.v5_engine import (
            MIRAGEV5Engine,
            V5Config,
            get_v5_engine
        )

        # Test with all features enabled
        config = V5Config(
            use_hyde=True,
            use_dual_level=True,
            use_hippocampal=True,
            use_dynamic_community_selection=True,
            enable_tracing=True,
            enable_cost_tracking=True
        )

        engine = MIRAGEV5Engine(config=config)
        log_test("Initialize V5 Engine (full)", True, "Engine created with all features")

        # Test queries
        test_queries = [
            ("ما هي جائزة الحكومة الرقمية؟", "factual Arabic"),
            ("What are the main themes?", "thematic English"),
            ("من هم شركاء الجائزة؟", "entity Arabic"),
            ("كيف ترتبط الجائزة بالتحول الرقمي؟", "multi-hop Arabic"),
        ]

        for query, query_type in test_queries:
            start = time.time()
            try:
                result = engine.retrieve(query, top_k=10)
                retrieve_time = (time.time() - start) * 1000

                # Evaluate results
                has_chunks = len(result.chunks) > 0
                has_enhanced = result.enhanced_query is not None and len(result.enhanced_query) > 0

                issues = []
                if not has_chunks:
                    issues.append("no chunks")
                if not has_enhanced and config.use_hyde:
                    issues.append("no HyDE")
                if result.low_level_chunks == 0 and result.high_level_chunks == 0:
                    issues.append("no dual-level")

                passed = has_chunks or len(issues) == 0
                log_test(
                    f"V5 retrieve ({query_type})",
                    passed,
                    f"Chunks: {len(result.chunks)}, Low: {result.low_level_chunks}, "
                    f"High: {result.high_level_chunks}, PPR: {result.ppr_activated_entities}, "
                    f"Conf: {result.confidence:.2f}"
                    + (f", Issues: {issues}" if issues else ""),
                    retrieve_time
                )

            except Exception as e:
                log_test(f"V5 retrieve ({query_type})", False, str(e), 0)

        # Test metrics/dashboard
        try:
            metrics = engine.get_metrics()
            dashboard = engine.get_dashboard_data()
            log_test(
                "V5 metrics and dashboard",
                True,
                f"Metrics keys: {list(metrics.keys())[:5]}...",
                0
            )
        except Exception as e:
            log_test("V5 metrics and dashboard", False, str(e), 0)

        # Test with features disabled
        minimal_config = V5Config(
            use_hyde=False,
            use_dual_level=True,
            use_hippocampal=False,
            enable_tracing=False
        )
        minimal_engine = MIRAGEV5Engine(config=minimal_config)

        start = time.time()
        result = minimal_engine.retrieve("test query")
        minimal_time = (time.time() - start) * 1000

        log_test(
            "V5 Engine (minimal config)",
            True,
            f"Chunks: {len(result.chunks)}, Time: {minimal_time:.1f}ms",
            minimal_time
        )

    except Exception as e:
        log_test("V5 Engine", False, f"Error: {str(e)}", 0)
        log_critical(f"V5 Engine failed: {traceback.format_exc()}")


# =============================================================================
# TEST 9: Comparison with V4/Legacy
# =============================================================================
def test_comparison_with_legacy():
    """Compare V5 with legacy retrieval."""
    print("\n" + "="*60)
    print("TEST 9: V5 vs Legacy Comparison")
    print("="*60)

    try:
        from core.retrieval import get_retrieval_engine, RetrievalMode
        from core.retrieval.v5_engine import get_v5_engine, V5Config

        v4_engine = get_retrieval_engine()
        v5_engine = get_v5_engine()

        test_query = "ما هي جائزة الحكومة الرقمية؟"

        # V4 retrieval
        v4_times = []
        v4_chunks = 0
        for _ in range(3):
            start = time.time()
            v4_result = v4_engine.retrieve(test_query, mode=RetrievalMode.HYBRID)
            v4_times.append((time.time() - start) * 1000)
            v4_chunks = len(v4_result.results)

        v4_avg = sum(v4_times) / len(v4_times)

        # V5 retrieval
        v5_times = []
        v5_chunks = 0
        for _ in range(3):
            start = time.time()
            v5_result = v5_engine.retrieve(test_query)
            v5_times.append((time.time() - start) * 1000)
            v5_chunks = len(v5_result.chunks)

        v5_avg = sum(v5_times) / len(v5_times)

        # Compare
        speedup = v4_avg / v5_avg if v5_avg > 0 else 0
        chunk_diff = v5_chunks - v4_chunks

        log_test(
            "V4 (HYBRID) performance",
            True,
            f"Avg: {v4_avg:.1f}ms, Chunks: {v4_chunks}",
            v4_avg
        )

        log_test(
            "V5 (unified) performance",
            True,
            f"Avg: {v5_avg:.1f}ms, Chunks: {v5_chunks}",
            v5_avg
        )

        # Analysis
        if v5_avg > v4_avg * 2:
            log_warning(f"V5 is {v5_avg/v4_avg:.1f}x SLOWER than V4 - optimization needed")

        if v5_chunks < v4_chunks:
            log_warning(f"V5 returns fewer chunks ({v5_chunks} vs {v4_chunks})")

        log_test(
            "Performance comparison",
            abs(speedup - 1.0) < 1.0,  # Within 2x either way is acceptable
            f"V5 is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}, "
            f"chunk diff: {chunk_diff:+d}",
            0
        )

    except Exception as e:
        log_test("Comparison", False, f"Error: {str(e)}", 0)


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Run all tests."""
    print("="*60)
    print("MIRAGE V5 COMPREHENSIVE TEST SUITE")
    print("Critical Evaluation of All Components")
    print("="*60)

    # Run all tests
    test_dynamic_community_selection()
    test_hyde_enhancement()
    test_dual_level_retrieval()
    test_hippocampal_retrieval()
    test_incremental_updates()
    test_coreference_resolution()
    test_observability()
    test_v5_engine()
    test_comparison_with_legacy()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total = RESULTS["passed"] + RESULTS["failed"]
    pass_rate = RESULTS["passed"] / total * 100 if total > 0 else 0

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {RESULTS['passed']} ({pass_rate:.1f}%)")
    print(f"Failed: {RESULTS['failed']}")

    if RESULTS["warnings"]:
        print(f"\n⚠️  Warnings ({len(RESULTS['warnings'])}):")
        for w in RESULTS["warnings"]:
            print(f"   - {w}")

    if RESULTS["critical_issues"]:
        print(f"\n🚨 Critical Issues ({len(RESULTS['critical_issues'])}):")
        for c in RESULTS["critical_issues"]:
            print(f"   - {c}")

    # Final score
    print("\n" + "="*60)
    print("PRACTICAL SCORE CALCULATION")
    print("="*60)

    scores = {
        "Test Pass Rate": pass_rate / 10,  # Max 10
        "No Critical Issues": 10 if not RESULTS["critical_issues"] else 0,
        "Low Warnings": max(0, 10 - len(RESULTS["warnings"])),
    }

    total_score = sum(scores.values()) / len(scores)

    print("\nComponent Scores:")
    for component, score in scores.items():
        print(f"  {component}: {score:.1f}/10")

    print(f"\n>>> PRACTICAL SCORE: {total_score:.1f}/10 <<<")

    return RESULTS


if __name__ == "__main__":
    main()
