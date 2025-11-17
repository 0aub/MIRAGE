#!/usr/bin/env python3
"""
Test script for community summarization (Phase 3).

Tests the hierarchical summary generation using Allam LLM.
Verifies token limits, summary quality, and storage.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.graph_builder.neo4j_client import Neo4jClient
from core.graph_builder.community_summarizer import CommunitySummarizer
from core.graph_builder.community_detector import CommunityDetector
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


def test_summarization():
    """Test community summarization end-to-end."""

    print_section("PHASE 3 TEST: Community Summarization")

    # Connect to Neo4j
    print(f"{YELLOW}Connecting to Neo4j...{RESET}")
    neo4j_client = Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    # Check if communities exist
    print(f"{YELLOW}Checking for communities...{RESET}")
    check_query = """
    MATCH (c:Community)
    RETURN COUNT(c) as community_count
    """
    result = neo4j_client.execute_query(check_query)
    community_count = result[0].get('community_count', 0) if result else 0

    if community_count == 0:
        print(f"{RED}✗ No communities in graph. Please run Phase 2 first.{RESET}")
        print(f"{YELLOW}  Hint: Run test_community_detection.py first{RESET}")
        return False

    print(f"{GREEN}✓ Found {community_count} communities{RESET}")

    # Get community statistics
    detector = CommunityDetector(neo4j_client)
    stats = detector.get_community_statistics()

    print(f"\nCommunity Structure:")
    for level_stat in stats['levels']:
        print(
            f"  Level {level_stat['level']}: "
            f"{level_stat['community_count']} communities "
            f"(avg size: {level_stat['avg_size']:.1f})"
        )

    # Initialize summarizer
    print(f"\n{YELLOW}Initializing community summarizer...{RESET}")
    summarizer = CommunitySummarizer(
        neo4j_client,
        llm_endpoint="http://localhost:8765",  # TGI endpoint
        max_tokens_per_summary=500,
        temperature=0.3
    )
    print(f"{GREEN}✓ Summarizer initialized{RESET}")

    # Test single community first
    print_section("TEST: Single Community Summary")

    # Get first Level 0 community
    communities_l0 = detector.get_all_communities_at_level(level=0)

    if not communities_l0:
        print(f"{RED}✗ No Level 0 communities found{RESET}")
        return False

    test_community = communities_l0[0]
    test_id = test_community['id']

    print(f"Testing with community: {test_id}")
    print(f"  Entities: {test_community['entity_count']}")

    # Generate single summary
    print(f"\n{YELLOW}Generating summary...{RESET}")
    summary = summarizer.generate_community_summary(test_id, level=0)

    if summary:
        print(f"{GREEN}✓ Summary generated successfully!{RESET}")
        print(f"\n{BLUE}Summary:{RESET}")
        print(f"{'─'*80}")
        print(summary.summary)
        print(f"{'─'*80}")
        print(f"\nMetadata:")
        print(f"  Tokens: ~{summary.token_count}")
        print(f"  Themes: {', '.join(summary.themes) if summary.themes else 'None'}")
        print(f"  Key entities: {', '.join(summary.key_entities[:5])}")
    else:
        print(f"{RED}✗ Failed to generate summary{RESET}")
        return False

    # Test batch generation
    print_section("TEST: Batch Summary Generation")

    max_test_level = min(1, max(s['level'] for s in stats['levels']))
    print(f"Generating summaries for levels 0-{max_test_level}...")
    print(f"{YELLOW}Note: This will call Allam multiple times{RESET}")

    # Limit to small batch for testing
    print(f"\n{YELLOW}Generating summaries (max 5 communities per level)...{RESET}")

    summaries = []
    for level in range(max_test_level + 1):
        communities = detector.get_all_communities_at_level(level)[:5]  # Limit to 5
        print(f"\nLevel {level}: Processing {len(communities)} communities...")

        for i, comm in enumerate(communities, 1):
            summary = summarizer.generate_community_summary(comm['id'], level)
            if summary:
                summaries.append(summary)
                print(f"  {i}/{len(communities)}: {comm['id']} ✓")
            else:
                print(f"  {i}/{len(communities)}: {comm['id']} ✗")

    print(f"\n{GREEN}✓ Generated {len(summaries)} summaries{RESET}")

    # Store summaries
    print_section("TEST: Summary Storage")

    print(f"{YELLOW}Storing summaries in Neo4j...{RESET}")
    storage_stats = summarizer.store_summaries_in_neo4j(summaries, clear_existing=False)

    print(f"{GREEN}✓ Storage complete!{RESET}")
    print(f"  Stored: {storage_stats['summaries_stored']}")
    print(f"  Failed: {storage_stats['summaries_failed']}")

    # Verify storage
    print_section("TEST: Summary Retrieval")

    # Retrieve a stored summary
    test_stored = summarizer.get_summary(summaries[0].community_id)

    if test_stored:
        print(f"{GREEN}✓ Successfully retrieved summary for {summaries[0].community_id}{RESET}")
        print(f"\n{BLUE}Stored Summary:{RESET}")
        print(f"{'─'*80}")
        print(test_stored['summary'][:200] + "...")
        print(f"{'─'*80}")
    else:
        print(f"{RED}✗ Failed to retrieve stored summary{RESET}")
        return False

    # Get summary statistics
    summary_stats = summarizer.get_summary_statistics()

    print(f"\n{BLUE}Summary Statistics:{RESET}")
    print(f"Total summaries: {summary_stats['total_summaries']}")
    for level_stat in summary_stats['levels']:
        print(
            f"  Level {level_stat['level']}: "
            f"{level_stat['summary_count']} summaries "
            f"(avg {level_stat['avg_tokens']:.0f} tokens)"
        )

    # Validation
    print_section("VALIDATION")

    validation_passed = True

    # Check 1: All summaries stored
    if storage_stats['summaries_failed'] > 0:
        print(f"{YELLOW}⚠ {storage_stats['summaries_failed']} summaries failed to store{RESET}")
    else:
        print(f"{GREEN}✓ All summaries stored successfully{RESET}")

    # Check 2: Token limits respected
    max_tokens = max(s.token_count for s in summaries)
    if max_tokens > 600:  # 500 max + 100 buffer
        print(f"{RED}✗ Token limit exceeded: {max_tokens} tokens{RESET}")
        validation_passed = False
    else:
        print(f"{GREEN}✓ Token limits respected (max: {max_tokens} tokens){RESET}")

    # Check 3: Summaries have content
    avg_length = sum(len(s.summary) for s in summaries) / len(summaries)
    if avg_length < 50:
        print(f"{RED}✗ Summaries too short (avg: {avg_length:.0f} chars){RESET}")
        validation_passed = False
    else:
        print(f"{GREEN}✓ Summaries have sufficient content (avg: {avg_length:.0f} chars){RESET}")

    # Summary
    print_section("TEST SUMMARY")

    if validation_passed:
        print(f"{GREEN}✓ ALL TESTS PASSED{RESET}")
        print(f"\n{GREEN}Phase 3 (Community Summarization) is working correctly!{RESET}")
        print(f"\nNext steps:")
        print(f"  1. Phase 4: Implement global search over summaries")
        print(f"  2. Test map-reduce query answering")
        return True
    else:
        print(f"{RED}✗ SOME TESTS FAILED{RESET}")
        print(f"\nPlease review the errors above.")
        return False


def main():
    """Run all tests."""
    try:
        success = test_summarization()
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
