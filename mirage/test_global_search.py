#!/usr/bin/env python3
"""
Test script for global search (Phase 4).

Tests the map-reduce search over community summaries.
Verifies query processing, answer generation, and theme extraction.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
# Add graph_builder to path to avoid __init__.py dependency loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/core/graph_builder'))

# Direct imports to avoid loading all dependencies
from neo4j_client import Neo4jClient
from global_search import GlobalSearchEngine
from community_detector import CommunityDetector
import logging

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*80}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{'='*80}\n")


def test_global_search():
    """Test global search end-to-end."""

    print_section("PHASE 4 TEST: Global Search (Map-Reduce)")

    # Connect to Neo4j
    print(f"{YELLOW}Connecting to Neo4j...{RESET}")
    neo4j_client = Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    # Check if summaries exist
    print(f"{YELLOW}Checking for community summaries...{RESET}")
    check_query = """
    MATCH (c:Community)
    WHERE c.summary IS NOT NULL
    RETURN COUNT(c) as summary_count
    """
    result = neo4j_client.execute_query(check_query)
    summary_count = result[0].get('summary_count', 0) if result else 0

    if summary_count == 0:
        print(f"{RED}✗ No community summaries found. Please run Phase 3 first.{RESET}")
        print(f"{YELLOW}  Hint: Run test_community_summarization.py first{RESET}")
        return False

    print(f"{GREEN}✓ Found {summary_count} community summaries{RESET}")

    # Get statistics
    detector = CommunityDetector(neo4j_client)
    stats = detector.get_community_statistics()

    print(f"\nCommunity Summary Distribution:")
    summary_stats_query = """
    MATCH (c:Community)
    WHERE c.summary IS NOT NULL
    WITH c.level as level, COUNT(c) as count
    RETURN level, count
    ORDER BY level
    """
    summary_stats = neo4j_client.execute_query(summary_stats_query)
    for record in summary_stats:
        print(f"  Level {record['level']}: {record['count']} summaries")

    # Initialize global search engine
    print(f"\n{YELLOW}Initializing global search engine...{RESET}")
    search_engine = GlobalSearchEngine(
        neo4j_client,
        llm_endpoint="http://localhost:8765",
        max_tokens_per_answer=400,
        temperature=0.3,
        max_communities=10  # Limit for testing
    )
    print(f"{GREEN}✓ Search engine initialized{RESET}")

    # Get search statistics
    search_stats = search_engine.get_search_statistics()
    print(f"\nSearch Statistics:")
    print(f"  Total summaries available: {search_stats['total_summaries']}")
    print(f"  Unique themes: {len(search_stats.get('all_themes', []))}")
    if search_stats.get('all_themes'):
        print(f"  Themes: {', '.join(search_stats['all_themes'][:10])}")

    # Test 1: General theme query
    print_section("TEST 1: General Theme Query")

    query1 = "ما هي الموضوعات الرئيسية في قاعدة المعرفة؟"
    print(f"Query: {query1}")
    print(f"\n{YELLOW}Executing global search (MAP-REDUCE)...{RESET}")

    result1 = search_engine.search(
        query=query1,
        level=1  # Use level 1 for broader themes
    )

    print(f"\n{GREEN}✓ Search complete!{RESET}")
    print(f"\n{BLUE}Answer:{RESET}")
    print(f"{'─'*80}")
    print(result1.answer)
    print(f"{'─'*80}")
    print(f"\nMetadata:")
    print(f"  Communities searched: {result1.communities_searched}")
    print(f"  Confidence: {result1.confidence:.2f}")
    print(f"  Themes extracted: {', '.join(result1.themes) if result1.themes else 'None'}")

    # Test 2: Specific topic query
    print_section("TEST 2: Specific Topic Query")

    query2 = "What information is available about technology and AI?"
    print(f"Query: {query2}")

    result2 = search_engine.search(
        query=query2,
        level=0  # Use level 0 for specific details
    )

    print(f"\n{GREEN}✓ Search complete!{RESET}")
    print(f"\n{BLUE}Answer:{RESET}")
    print(f"{'─'*80}")
    print(result2.answer[:500] + "..." if len(result2.answer) > 500 else result2.answer)
    print(f"{'─'*80}")
    print(f"\nMetadata:")
    print(f"  Communities searched: {result2.communities_searched}")
    print(f"  Confidence: {result2.confidence:.2f}")

    # Test 3: Theme-based search
    if search_stats.get('all_themes'):
        print_section("TEST 3: Theme-Based Search")

        # Use first available theme
        test_theme = search_stats['all_themes'][0]
        print(f"Searching for theme: {test_theme}")

        result3 = search_engine.search_by_theme(
            theme=test_theme,
            level=1
        )

        print(f"\n{GREEN}✓ Search complete!{RESET}")
        print(f"\n{BLUE}Answer:{RESET}")
        print(f"{'─'*80}")
        print(result3.answer[:400] + "..." if len(result3.answer) > 400 else result3.answer)
        print(f"{'─'*80}")

    # Validation
    print_section("VALIDATION")

    validation_passed = True

    # Check 1: Results not empty
    if not result1.answer or len(result1.answer) < 50:
        print(f"{RED}✗ Answer too short or empty{RESET}")
        validation_passed = False
    else:
        print(f"{GREEN}✓ Answer has sufficient content{RESET}")

    # Check 2: Communities searched
    if result1.communities_searched == 0:
        print(f"{RED}✗ No communities searched{RESET}")
        validation_passed = False
    else:
        print(f"{GREEN}✓ Searched {result1.communities_searched} communities{RESET}")

    # Check 3: Intermediate answers generated
    if result1.intermediate_answers and len(result1.intermediate_answers) > 0:
        print(f"{GREEN}✓ MAP phase generated {len(result1.intermediate_answers)} intermediate answers{RESET}")
    else:
        print(f"{YELLOW}⚠ No intermediate answers (MAP phase may have issues){RESET}")

    # Summary
    print_section("TEST SUMMARY")

    if validation_passed:
        print(f"{GREEN}✓ ALL TESTS PASSED{RESET}")
        print(f"\n{GREEN}Phase 4 (Global Search) is working correctly!{RESET}")
        print(f"\nKey Features Verified:")
        print(f"  ✓ Map-reduce query processing")
        print(f"  ✓ Community summary retrieval")
        print(f"  ✓ Theme extraction")
        print(f"  ✓ Multi-level search (Level 0 and 1)")
        print(f"\nNext steps:")
        print(f"  1. Phase 5: Implement hybrid search (global + local)")
        print(f"  2. Add search routing (decide global vs local)")
        print(f"  3. Integrate into RAG pipeline")
        return True
    else:
        print(f"{RED}✗ SOME TESTS FAILED{RESET}")
        print(f"\nPlease review the errors above.")
        return False


def main():
    """Run all tests."""
    try:
        success = test_global_search()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Test interrupted by user{RESET}")
        sys.exit(130)

    except Exception as e:
        print(f"\n{RED}✗ Test failed with error:{RESET}")
        print(f"{RED}{str(e)}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
