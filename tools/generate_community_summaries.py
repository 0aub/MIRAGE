#!/usr/bin/env python3
"""
Generate Community Summaries for GraphRAG Global Search

This script generates summaries for all communities in the graph,
enabling the global_search retrieval mode to work properly.
"""

import sys
sys.path.insert(0, '/app')

from src.core.graph_builder import Neo4jClient
from src.core.graph_builder.community_summarizer import CommunitySummarizer

def main():
    print("=" * 60)
    print("MIRAGE Community Summary Generator")
    print("=" * 60)

    # Connect to Neo4j
    print("\n[1/4] Connecting to Neo4j...")
    client = Neo4jClient()
    client.connect()
    print("  Connected!")

    # Check current community structure
    print("\n[2/4] Checking community structure...")
    levels = client.execute_query('''
        MATCH (c:Community)
        RETURN c.level as level, count(*) as count
        ORDER BY c.level
    ''')

    total_communities = 0
    for r in levels:
        level = r.get('level', 'unknown')
        count = r.get('count', 0)
        print(f"  Level {level}: {count} communities")
        total_communities += count

    if total_communities == 0:
        print("\n  ERROR: No communities found! Run community detection first.")
        return

    # Check existing summaries
    existing = client.execute_query('''
        MATCH (c:Community) WHERE c.summary IS NOT NULL
        RETURN count(c) as count
    ''')
    existing_count = existing[0].get('count', 0) if existing else 0
    print(f"  Existing summaries: {existing_count}")

    # Initialize summarizer
    print("\n[3/4] Initializing CommunitySummarizer...")
    summarizer = CommunitySummarizer(
        neo4j_client=client,
        llm_endpoint="http://mirage-tgi:80",  # TGI endpoint for Allam (port 80)
        max_tokens_per_summary=200,
        temperature=0.3
    )
    print("  Initialized!")

    # Generate summaries
    print("\n[4/4] Generating summaries...")
    print("  This may take a few minutes depending on community count...")

    summaries = summarizer.generate_all_summaries(
        max_level=2,  # Summarize L0, L1, L2
        batch_size=5
    )

    if not summaries:
        print("  WARNING: No summaries generated!")
        return

    print(f"  Generated {len(summaries)} summaries")

    # Store summaries in Neo4j
    print("\n  Storing summaries in Neo4j...")
    stats = summarizer.store_summaries_in_neo4j(summaries, clear_existing=True)

    print(f"  Stored: {stats.get('summaries_stored', 0)}")
    print(f"  Failed: {stats.get('summaries_failed', 0)}")

    # Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    final_stats = summarizer.get_summary_statistics()
    print(f"Total summaries in Neo4j: {final_stats.get('total_summaries', 0)}")

    for level_stat in final_stats.get('levels', []):
        print(f"  Level {level_stat.get('level')}: {level_stat.get('summary_count')} summaries, avg {level_stat.get('avg_tokens', 0):.0f} tokens")

    # Sample summary
    if summaries:
        print("\n" + "-" * 60)
        print("SAMPLE SUMMARY (first community)")
        print("-" * 60)
        sample = summaries[0]
        print(f"Community: {sample.community_id}")
        print(f"Level: {sample.level}")
        print(f"Entities: {sample.entity_count}")
        print(f"Summary: {sample.summary[:300]}...")
        print(f"Themes: {sample.themes}")

    print("\n" + "=" * 60)
    print("DONE! global_search mode should now work.")
    print("=" * 60)


if __name__ == "__main__":
    main()
