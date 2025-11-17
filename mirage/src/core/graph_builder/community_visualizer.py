"""
Community Visualization Utilities

Simple utilities to visualize and analyze community detection results.
Helps understand the hierarchical structure and validate implementation.
"""

from typing import Dict, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class CommunityVisualizer:
    """
    Visualize and analyze community detection results.

    Features:
    - Print hierarchical community structure
    - Generate community statistics
    - Export community data for analysis

    Usage:
        visualizer = CommunityVisualizer(neo4j_client)
        visualizer.print_hierarchy()
        visualizer.print_statistics()
    """

    def __init__(self, neo4j_client):
        """
        Initialize visualizer.

        Args:
            neo4j_client: Neo4j client instance
        """
        self.neo4j_client = neo4j_client

    def print_hierarchy(self, max_entities_shown: int = 5):
        """
        Print the complete community hierarchy.

        Shows communities at each level with sample entities.

        Args:
            max_entities_shown: Maximum entities to show per community
        """
        print("\n" + "="*80)
        print("COMMUNITY HIERARCHY")
        print("="*80)

        # Get statistics first
        stats = self._get_statistics()

        if stats['total_communities'] == 0:
            print("No communities found in the graph.")
            return

        print(f"\nTotal Communities: {stats['total_communities']}")
        print(f"Hierarchy Levels: {len(stats['levels'])}")
        print()

        # Print each level
        for level_stats in stats['levels']:
            level = level_stats['level']
            communities = self._get_communities_at_level(level)

            print(f"\n{'─'*80}")
            print(f"LEVEL {level} - {level_stats['community_count']} communities")
            print(f"Average size: {level_stats['avg_size']:.1f} entities")
            print(f"Size range: {level_stats['min_size']} - {level_stats['max_size']}")
            print(f"{'─'*80}")

            for i, comm in enumerate(communities, 1):
                self._print_community(comm, max_entities_shown, indent=2)

                # Don't print too many
                if i >= 20:
                    remaining = len(communities) - i
                    if remaining > 0:
                        print(f"  ... and {remaining} more communities")
                    break

        print("\n" + "="*80 + "\n")

    def _print_community(
        self,
        community: Dict,
        max_entities: int,
        indent: int = 0
    ):
        """Print a single community."""
        prefix = " " * indent

        comm_id = community.get('id', 'Unknown')
        size = community.get('size', 0)
        entities = community.get('entities', [])[:max_entities]
        entity_count = community.get('entity_count', 0)

        print(f"{prefix}📦 {comm_id} ({entity_count} entities)")

        if entities:
            entities_str = ", ".join(entities)
            if len(community.get('entities', [])) > max_entities:
                entities_str += f", ... (+{len(community.get('entities', [])) - max_entities} more)"
            print(f"{prefix}   Entities: {entities_str}")

    def print_statistics(self):
        """Print detailed community statistics."""
        print("\n" + "="*80)
        print("COMMUNITY STATISTICS")
        print("="*80)

        stats = self._get_statistics()

        if stats['total_communities'] == 0:
            print("No communities found.")
            return

        print(f"\nTotal Communities: {stats['total_communities']}")
        print("\nPer-Level Breakdown:")
        print(f"{'Level':<8} {'Count':<10} {'Avg Size':<12} {'Min Size':<10} {'Max Size':<10}")
        print("─" * 60)

        for level_stats in stats['levels']:
            print(
                f"{level_stats['level']:<8} "
                f"{level_stats['community_count']:<10} "
                f"{level_stats['avg_size']:<12.1f} "
                f"{level_stats['min_size']:<10} "
                f"{level_stats['max_size']:<10}"
            )

        print("="*80 + "\n")

    def print_entity_communities(self, entity_name: str):
        """
        Print all communities an entity belongs to.

        Args:
            entity_name: Entity name
        """
        print(f"\n{'='*80}")
        print(f"Communities for Entity: {entity_name}")
        print(f"{'='*80}\n")

        communities = self._get_entity_communities(entity_name)

        if not communities:
            print(f"Entity '{entity_name}' not found in any community.")
            return

        for comm in communities:
            print(f"Level {comm['level']}: {comm['id']} ({comm['size']} entities)")

        print()

    def export_community_data(self, level: int = 0) -> List[Dict]:
        """
        Export community data for external analysis.

        Args:
            level: Which level to export (default: 0 = finest)

        Returns:
            List of community dicts with full data
        """
        communities = self._get_communities_at_level(level)

        logger.info(f"Exported {len(communities)} communities from level {level}")

        return communities

    def find_entity_community(
        self,
        entity_name: str,
        level: int = 0
    ) -> Optional[str]:
        """
        Find which community an entity belongs to at a specific level.

        Args:
            entity_name: Entity name
            level: Hierarchy level

        Returns:
            Community ID or None
        """
        query = """
        MATCH (e:Entity {name: $entity_name})-[:BELONGS_TO]->(c:Community {level: $level})
        RETURN c.id as community_id
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'entity_name': entity_name,
                'level': level
            })

            if results:
                return results[0].get('community_id')
            return None

        except Exception as e:
            logger.error(f"Error finding community for {entity_name}: {e}")
            return None

    def get_community_overlap(
        self,
        community_id1: str,
        community_id2: str
    ) -> Dict:
        """
        Get overlap between two communities.

        Args:
            community_id1: First community ID
            community_id2: Second community ID

        Returns:
            Dict with overlap statistics
        """
        query = """
        MATCH (c1:Community {id: $comm_id1})
        MATCH (c2:Community {id: $comm_id2})
        MATCH (e1:Entity)-[:BELONGS_TO]->(c1)
        MATCH (e2:Entity)-[:BELONGS_TO]->(c2)

        WITH c1, c2,
             COLLECT(DISTINCT e1.name) as entities1,
             COLLECT(DISTINCT e2.name) as entities2

        WITH c1, c2, entities1, entities2,
             [e IN entities1 WHERE e IN entities2] as overlap

        RETURN
            SIZE(entities1) as size1,
            SIZE(entities2) as size2,
            SIZE(overlap) as overlap_count,
            overlap as shared_entities
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'comm_id1': community_id1,
                'comm_id2': community_id2
            })

            if results:
                record = results[0]
                size1 = record.get('size1', 0)
                size2 = record.get('size2', 0)
                overlap_count = record.get('overlap_count', 0)

                jaccard_similarity = 0.0
                if size1 + size2 - overlap_count > 0:
                    jaccard_similarity = overlap_count / (size1 + size2 - overlap_count)

                return {
                    'community1_size': size1,
                    'community2_size': size2,
                    'overlap_count': overlap_count,
                    'shared_entities': record.get('shared_entities', []),
                    'jaccard_similarity': jaccard_similarity
                }

            return {
                'community1_size': 0,
                'community2_size': 0,
                'overlap_count': 0,
                'shared_entities': [],
                'jaccard_similarity': 0.0
            }

        except Exception as e:
            logger.error(f"Error calculating overlap: {e}")
            return {
                'community1_size': 0,
                'community2_size': 0,
                'overlap_count': 0,
                'shared_entities': [],
                'jaccard_similarity': 0.0
            }

    def _get_statistics(self) -> Dict:
        """Get community statistics from Neo4j."""
        query = """
        MATCH (c:Community)
        WITH c.level as level,
             COUNT(c) as community_count,
             AVG(c.size) as avg_size,
             MIN(c.size) as min_size,
             MAX(c.size) as max_size
        RETURN
            level,
            community_count,
            avg_size,
            min_size,
            max_size
        ORDER BY level
        """

        try:
            results = self.neo4j_client.execute_query(query)

            stats = {
                'levels': [],
                'total_communities': 0
            }

            for record in results:
                level_stats = {
                    'level': record.get('level'),
                    'community_count': record.get('community_count', 0),
                    'avg_size': record.get('avg_size', 0),
                    'min_size': record.get('min_size', 0),
                    'max_size': record.get('max_size', 0)
                }
                stats['levels'].append(level_stats)
                stats['total_communities'] += level_stats['community_count']

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'levels': [], 'total_communities': 0}

    def _get_communities_at_level(self, level: int) -> List[Dict]:
        """Get all communities at a specific level."""
        query = """
        MATCH (c:Community {level: $level})
        OPTIONAL MATCH (e:Entity)-[:BELONGS_TO]->(c)
        RETURN
            c.id as id,
            c.level as level,
            c.size as size,
            COUNT(e) as entity_count,
            COLLECT(e.name) as entities
        ORDER BY c.id
        """

        try:
            results = self.neo4j_client.execute_query(query, {'level': level})

            communities = []
            for record in results:
                communities.append({
                    'id': record.get('id'),
                    'level': record.get('level'),
                    'size': record.get('size'),
                    'entity_count': record.get('entity_count', 0),
                    'entities': record.get('entities', [])
                })

            return communities

        except Exception as e:
            logger.error(f"Error getting communities at level {level}: {e}")
            return []

    def _get_entity_communities(self, entity_name: str) -> List[Dict]:
        """Get all communities an entity belongs to."""
        query = """
        MATCH (e:Entity {name: $entity_name})-[:BELONGS_TO]->(c:Community)
        RETURN
            c.id as id,
            c.level as level,
            c.size as size
        ORDER BY c.level
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'entity_name': entity_name
            })

            communities = []
            for record in results:
                communities.append({
                    'id': record.get('id'),
                    'level': record.get('level'),
                    'size': record.get('size')
                })

            return communities

        except Exception as e:
            logger.error(f"Error getting communities for {entity_name}: {e}")
            return []


def print_community_tree(
    neo4j_client,
    root_community_id: Optional[str] = None,
    max_depth: int = 3
):
    """
    Print a tree view of the community hierarchy.

    Args:
        neo4j_client: Neo4j client
        root_community_id: Start from this community (None = start from top level)
        max_depth: Maximum depth to print
    """
    print("\n" + "="*80)
    print("COMMUNITY TREE")
    print("="*80 + "\n")

    if root_community_id:
        # Start from specific community
        query = """
        MATCH path = (root:Community {id: $root_id})-[:PARENT_OF*0..%d]->(child:Community)
        RETURN child.id as id, child.level as level, LENGTH(path) as depth
        ORDER BY depth, id
        """ % max_depth

        results = neo4j_client.execute_query(query, {'root_id': root_community_id})
    else:
        # Start from top level (highest level number)
        query = """
        MATCH (c:Community)
        WITH MAX(c.level) as max_level
        MATCH path = (root:Community {level: max_level})-[:PARENT_OF*0..%d]->(child:Community)
        RETURN child.id as id, child.level as level, LENGTH(path) as depth
        ORDER BY root.id, depth, child.id
        """ % max_depth

        results = neo4j_client.execute_query(query)

    # Group by depth
    by_depth = defaultdict(list)
    for record in results:
        by_depth[record.get('depth', 0)].append(record.get('id'))

    # Print tree
    for depth in sorted(by_depth.keys()):
        indent = "  " * depth
        communities = by_depth[depth]

        for comm_id in communities:
            print(f"{indent}{'└─ ' if depth > 0 else ''}📦 {comm_id}")

    print("\n" + "="*80 + "\n")
