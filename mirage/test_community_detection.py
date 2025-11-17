#!/usr/bin/env python3
"""
Test script for community detection implementation.

Tests the Phase 2 community detection on existing graph data.
Verifies hierarchical structure, storage, and visualization.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.database.neo4j_client import Neo4jClient
from core.graph_builder.community_detector import CommunityDetector
from core.graph_builder.community_visualizer import CommunityVisualizer, print_community_tree
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


def test_community_detection():
    """Test community detection end-to-end."""

    print_section("PHASE 2 TEST: Community Detection")

    # Connect to Neo4j
    print(f"{YELLOW}Connecting to Neo4j...{RESET}")
    neo4j_client = Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    # Check if graph has data
    print(f"{YELLOW}Checking graph data...{RESET}")
    count_query = """
    MATCH (e:Entity)
    RETURN COUNT(e) as entity_count
    """
    result = neo4j_client.execute_query(count_query)
    entity_count = result[0].get('entity_count', 0) if result else 0

    if entity_count == 0:
        print(f"{RED}✗ No entities in graph. Please run ingestion first.{RESET}")
        print(f"{YELLOW}  Hint: Upload documents through the UI or run ingestion script{RESET}")
        return False

    print(f"{GREEN}✓ Found {entity_count} entities in graph{RESET}")

    # Count relationships
    rel_query = """
    MATCH ()-[r]->()
    RETURN COUNT(r) as rel_count
    """
    result = neo4j_client.execute_query(rel_query)
    rel_count = result[0].get('rel_count', 0) if result else 0
    print(f"{GREEN}✓ Found {rel_count} relationships{RESET}")

    # Initialize detector
    print(f"\n{YELLOW}Initializing community detector...{RESET}")
    detector = CommunityDetector(neo4j_client)

    # Detect communities
    print(f"\n{YELLOW}Running community detection...{RESET}")
    print(f"  Parameters:")
    print(f"    - Resolution: 1.0")
    print(f"    - Hierarchy levels: 3")
    print(f"    - Min community size: 3")

    result = detector.detect_communities(
        resolution=1.0,
        levels=3,
        min_community_size=3
    )

    print(f"\n{GREEN}✓ Community detection complete!{RESET}")
    print(f"  - Total communities: {result.total_communities}")
    print(f"  - Hierarchy levels: {result.hierarchy_levels}")
    print(f"  - Modularity: {result.modularity:.3f}")

    if result.modularity < 0.3:
        print(f"  {YELLOW}⚠ Low modularity - graph may not have strong community structure{RESET}")
    else:
        print(f"  {GREEN}✓ Good modularity score{RESET}")

    # Store in Neo4j
    print(f"\n{YELLOW}Storing communities in Neo4j...{RESET}")
    stats = detector.store_communities_in_neo4j(result, clear_existing=True)

    print(f"{GREEN}✓ Communities stored successfully!{RESET}")
    print(f"  - Community nodes created: {stats['communities_created']}")
    print(f"  - BELONGS_TO relationships: {stats['belongs_to_rels']}")
    print(f"  - PARENT_OF relationships: {stats['parent_of_rels']}")

    # Visualize results
    print_section("VISUALIZATION: Community Structure")

    visualizer = CommunityVisualizer(neo4j_client)

    # Print statistics
    visualizer.print_statistics()

    # Print hierarchy
    visualizer.print_hierarchy(max_entities_shown=5)

    # Print tree view
    print_community_tree(neo4j_client, max_depth=3)

    # Test specific queries
    print_section("TEST: Community Queries")

    # Get communities at finest level
    communities_l0 = detector.get_all_communities_at_level(level=0)
    print(f"Level 0 (finest): {len(communities_l0)} communities")

    if communities_l0:
        # Show first community
        first_comm = communities_l0[0]
        print(f"\nExample Community: {first_comm['id']}")
        print(f"  Entities ({first_comm['entity_count']}): {', '.join(first_comm['entities'][:5])}")
        if first_comm['entity_count'] > 5:
            print(f"  ... and {first_comm['entity_count'] - 5} more")

        # Get detailed info
        info = detector.get_community_info(first_comm['id'])
        if info:
            print(f"\n  Parent community: {info.get('parent', 'None')}")
            print(f"  Child communities: {len(info.get('children', []))}")

    # Test entity lookup
    if communities_l0 and communities_l0[0]['entities']:
        test_entity = communities_l0[0]['entities'][0]
        print(f"\n{YELLOW}Testing entity community lookup: '{test_entity}'{RESET}")

        entity_communities = detector.get_entity_communities(test_entity)
        print(f"Entity belongs to {len(entity_communities)} communities across hierarchy:")
        for comm in entity_communities:
            print(f"  Level {comm['level']}: {comm['id']} ({comm['size']} entities)")

    # Final validation
    print_section("VALIDATION")

    validation_passed = True

    # Check 1: All communities stored
    if stats['communities_created'] != result.total_communities:
        print(f"{RED}✗ Community count mismatch{RESET}")
        validation_passed = False
    else:
        print(f"{GREEN}✓ All communities stored correctly{RESET}")

    # Check 2: Hierarchy structure exists
    if result.hierarchy_levels > 1 and stats['parent_of_rels'] == 0:
        print(f"{RED}✗ No hierarchy relationships found{RESET}")
        validation_passed = False
    else:
        print(f"{GREEN}✓ Hierarchy structure created{RESET}")

    # Check 3: All entities assigned to communities
    check_query = """
    MATCH (e:Entity)
    OPTIONAL MATCH (e)-[:BELONGS_TO]->(c:Community {level: 0})
    RETURN
        COUNT(e) as total_entities,
        COUNT(c) as entities_in_communities
    """
    result = neo4j_client.execute_query(check_query)
    if result:
        total = result[0].get('total_entities', 0)
        assigned = result[0].get('entities_in_communities', 0)

        if assigned < total * 0.9:  # At least 90% should be in communities
            print(f"{YELLOW}⚠ Only {assigned}/{total} entities assigned to communities{RESET}")
        else:
            print(f"{GREEN}✓ {assigned}/{total} entities assigned to communities{RESET}")

    # Summary
    print_section("TEST SUMMARY")

    if validation_passed:
        print(f"{GREEN}✓ ALL TESTS PASSED{RESET}")
        print(f"\n{GREEN}Phase 2 (Community Detection) is working correctly!{RESET}")
        print(f"\nNext steps:")
        print(f"  1. Phase 3: Generate community summaries using Allam")
        print(f"  2. Test global search over summaries")
        return True
    else:
        print(f"{RED}✗ SOME TESTS FAILED{RESET}")
        print(f"\nPlease review the errors above.")
        return False


def main():
    """Run all tests."""
    try:
        success = test_community_detection()
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
