"""
Community Detection Neo4j Operations

All Neo4j storage, retrieval, and query operations for communities.
"""

import networkx as nx
from typing import List, Dict, Optional
import logging

from .models import Community, CommunityDetectionResult

logger = logging.getLogger(__name__)


class CommunityNeo4jOps:
    """Neo4j operations mixin for community detection."""

    def __init__(self, neo4j_client):
        """Initialize with Neo4j client."""
        self.neo4j_client = neo4j_client

    def load_graph_from_neo4j(self) -> nx.Graph:
        """
        Load the knowledge graph from Neo4j into NetworkX.

        Returns:
            NetworkX Graph with entities as nodes and relationships as edges
        """
        query = """
        // Get all entities and relationships
        MATCH (e1:Entity)-[r]->(e2:Entity)
        RETURN
            e1.name as source,
            e2.name as target,
            type(r) as rel_type,
            COALESCE(r.weight, 1.0) as weight
        """

        try:
            results = self.neo4j_client.execute_query(query)

            # Build NetworkX graph
            graph = nx.Graph()

            for record in results:
                source = record.get('source')
                target = record.get('target')
                weight = record.get('weight', 1.0)

                if source and target:
                    # Add edge with weight
                    graph.add_edge(source, target, weight=weight)

            return graph

        except Exception as e:
            logger.error(f"Error loading graph from Neo4j: {e}")
            return nx.Graph()  # Return empty graph

    def store_communities_in_neo4j(
        self,
        result: CommunityDetectionResult,
        clear_existing: bool = True
    ) -> Dict:
        """
        Store detected communities in Neo4j.

        Creates:
        - Community nodes with properties (id, level, size)
        - BELONGS_TO relationships from entities to communities
        - PARENT_OF relationships for hierarchy

        Args:
            result: CommunityDetectionResult to store
            clear_existing: If True, delete existing communities first

        Returns:
            Dict with storage statistics
        """
        if clear_existing:
            logger.info("Clearing existing communities from Neo4j...")
            self._clear_communities()

        logger.info(f"Storing {result.total_communities} communities in Neo4j...")

        stats = {
            'communities_created': 0,
            'belongs_to_rels': 0,
            'parent_of_rels': 0
        }

        # Pass 1: Create all community nodes and BELONGS_TO relationships
        for community in result.communities:
            self._create_community_node(community)
            stats['communities_created'] += 1

            # Create BELONGS_TO relationships
            for entity_name in community.entities:
                self._create_belongs_to_relationship(entity_name, community)
                stats['belongs_to_rels'] += 1

        # Pass 2: Create PARENT_OF relationships (after all nodes exist)
        for community in result.communities:
            if community.parent_community:
                self._create_parent_relationship(community)
                stats['parent_of_rels'] += 1

        logger.info(
            f"Stored communities: {stats['communities_created']} nodes, "
            f"{stats['belongs_to_rels']} BELONGS_TO, "
            f"{stats['parent_of_rels']} PARENT_OF"
        )

        return stats

    def _clear_communities(self):
        """Delete all existing community nodes and relationships."""
        query = """
        // Delete all Community nodes and their relationships
        MATCH (c:Community)
        DETACH DELETE c
        """

        try:
            self.neo4j_client.execute_query(query)
            logger.info("Existing communities cleared")
        except Exception as e:
            logger.error(f"Error clearing communities: {e}")

    def _create_community_node(self, community: Community):
        """Create a Community node in Neo4j."""
        query = """
        CREATE (c:Community {
            id: $id,
            level: $level,
            size: $size,
            entity_count: $entity_count
        })
        """

        try:
            self.neo4j_client.execute_query(query, {
                'id': community.id,
                'level': community.level,
                'size': community.size,
                'entity_count': len(community.entities)
            })
        except Exception as e:
            logger.error(f"Error creating community node {community.id}: {e}")

    def _create_belongs_to_relationship(
        self,
        entity_name: str,
        community: Community
    ):
        """Create BELONGS_TO relationship from entity to community."""
        query = """
        MATCH (e:Entity {name: $entity_name})
        MATCH (c:Community {id: $community_id})
        MERGE (e)-[:BELONGS_TO {level: $level}]->(c)
        """

        try:
            self.neo4j_client.execute_query(query, {
                'entity_name': entity_name,
                'community_id': community.id,
                'level': community.level
            })
        except Exception as e:
            logger.error(
                f"Error creating BELONGS_TO relationship "
                f"for {entity_name} -> {community.id}: {e}"
            )

    def _create_parent_relationship(self, community: Community):
        """Create PARENT_OF relationship in hierarchy."""
        query = """
        MATCH (parent:Community {id: $parent_id})
        MATCH (child:Community {id: $child_id})
        MERGE (parent)-[:PARENT_OF]->(child)
        """

        try:
            self.neo4j_client.execute_query(query, {
                'parent_id': community.parent_community,
                'child_id': community.id
            })
        except Exception as e:
            logger.error(
                f"Error creating PARENT_OF relationship "
                f"{community.parent_community} -> {community.id}: {e}"
            )

    def get_community_info(self, community_id: str) -> Optional[Dict]:
        """
        Get information about a specific community.

        Args:
            community_id: Community identifier (e.g., "L0_C1")

        Returns:
            Dict with community info or None
        """
        query = """
        MATCH (c:Community {id: $community_id})
        OPTIONAL MATCH (c)-[:PARENT_OF]->(child:Community)
        OPTIONAL MATCH (parent:Community)-[:PARENT_OF]->(c)
        OPTIONAL MATCH (e:Entity)-[:BELONGS_TO]->(c)
        RETURN
            c.id as id,
            c.level as level,
            c.size as size,
            COLLECT(DISTINCT e.name) as entities,
            COLLECT(DISTINCT child.id) as children,
            parent.id as parent
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'community_id': community_id
            })

            if results:
                record = results[0]
                return {
                    'id': record.get('id'),
                    'level': record.get('level'),
                    'size': record.get('size'),
                    'entities': record.get('entities', []),
                    'children': [c for c in record.get('children', []) if c],
                    'parent': record.get('parent')
                }
            return None

        except Exception as e:
            logger.error(f"Error getting community info for {community_id}: {e}")
            return None

    def get_all_communities_at_level(self, level: int) -> List[Dict]:
        """
        Get all communities at a specific hierarchy level.

        Args:
            level: Hierarchy level (0 = finest granularity)

        Returns:
            List of community info dicts
        """
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

    def get_entity_communities(
        self,
        entity_name: str
    ) -> List[Dict]:
        """
        Get all communities an entity belongs to (across all levels).

        Args:
            entity_name: Entity name

        Returns:
            List of communities (one per level)
        """
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
            logger.error(f"Error getting communities for entity {entity_name}: {e}")
            return []

    def get_community_statistics(self) -> Dict:
        """
        Get overall statistics about communities in the graph.

        Returns:
            Dict with statistics
        """
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
            logger.error(f"Error getting community statistics: {e}")
            return {'levels': [], 'total_communities': 0}

    def get_key_entities(self, entity_names: List[str], limit: int = 5) -> List[str]:
        """Get key entities by importance and confidence from Neo4j."""
        if not entity_names:
            return []

        try:
            query = """
            MATCH (e:Entity)
            WHERE e.name IN $names
            RETURN e.name AS name
            ORDER BY
                CASE e.importance WHEN 'high' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END DESC,
                e.confidence DESC
            LIMIT $limit
            """
            results = self.neo4j_client.execute_query(query, {
                "names": entity_names,
                "limit": limit
            })
            return [r["name"] for r in results if r.get("name")]
        except Exception as e:
            logger.warning(f"Could not get key entities: {e}")
            return entity_names[:limit]

    def get_entity_descriptions(self, entity_names: List[str]) -> Dict[str, str]:
        """Get descriptions for entities from Neo4j (GraphRAG requirement)."""
        if not entity_names:
            return {}

        try:
            query = """
            MATCH (e:Entity)
            WHERE e.name IN $names AND e.description IS NOT NULL AND e.description <> ''
            RETURN e.name AS name, e.description AS description
            """
            results = self.neo4j_client.execute_query(query, {"names": entity_names})
            return {r["name"]: r["description"] for r in results if r.get("description")}
        except Exception as e:
            logger.warning(f"Could not get entity descriptions: {e}")
            return {}

    def get_community_relationships(self, entity_names: List[str], limit: int = 30) -> List[Dict]:
        """Get relationships between entities in a community."""
        if not entity_names or len(entity_names) < 2:
            return []

        try:
            query = """
            MATCH (e1:Entity)-[r]->(e2:Entity)
            WHERE e1.name IN $names AND e2.name IN $names
              AND type(r) <> 'BELONGS_TO'
            RETURN e1.name AS source, e2.name AS target, type(r) AS type,
                   r.description AS description
            LIMIT $limit
            """
            results = self.neo4j_client.execute_query(query, {
                "names": entity_names,
                "limit": limit
            })
            return [
                {
                    "source": r["source"],
                    "target": r["target"],
                    "type": r["type"],
                    "description": r.get("description", "")
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Could not get community relationships: {e}")
            return []

    def update_community_summary(self, community_id: str, summary: str) -> bool:
        """
        Update a community's summary in Neo4j.

        Args:
            community_id: Community identifier
            summary: Generated summary text

        Returns:
            True if successful
        """
        try:
            query = """
            MATCH (c:Community {id: $community_id})
            SET c.summary = $summary, c.summary_updated_at = timestamp()
            RETURN c.id
            """
            result = self.neo4j_client.execute_query(query, {
                "community_id": community_id,
                "summary": summary
            })
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to update community summary: {e}")
            return False

    def get_communities_needing_summaries(self, level: Optional[int] = None) -> List[str]:
        """
        Get community IDs that don't have summaries yet.

        Args:
            level: Optional filter by hierarchy level

        Returns:
            List of community IDs needing summaries
        """
        try:
            if level is not None:
                query = """
                MATCH (c:Community {level: $level})
                WHERE c.summary IS NULL OR c.summary = ''
                RETURN c.id AS id
                ORDER BY c.size DESC
                """
                results = self.neo4j_client.execute_query(query, {"level": level})
            else:
                query = """
                MATCH (c:Community)
                WHERE c.summary IS NULL OR c.summary = ''
                RETURN c.id AS id
                ORDER BY c.level, c.size DESC
                """
                results = self.neo4j_client.execute_query(query, {})

            return [r["id"] for r in results]
        except Exception as e:
            logger.error(f"Failed to get communities needing summaries: {e}")
            return []

    def get_communities_with_summaries(self, level: Optional[int] = None) -> List[Dict]:
        """
        Get all communities that have summaries (for global search).

        Args:
            level: Optional filter by hierarchy level

        Returns:
            List of community dicts with summaries
        """
        try:
            if level is not None:
                query = """
                MATCH (c:Community {level: $level})
                WHERE c.summary IS NOT NULL AND c.summary <> ''
                RETURN c.id AS id, c.level AS level, c.size AS size,
                       c.summary AS summary, c.key_entities AS key_entities
                ORDER BY c.size DESC
                """
                results = self.neo4j_client.execute_query(query, {"level": level})
            else:
                query = """
                MATCH (c:Community)
                WHERE c.summary IS NOT NULL AND c.summary <> ''
                RETURN c.id AS id, c.level AS level, c.size AS size,
                       c.summary AS summary, c.key_entities AS key_entities
                ORDER BY c.level, c.size DESC
                """
                results = self.neo4j_client.execute_query(query, {})

            return [
                {
                    "id": r["id"],
                    "level": r["level"],
                    "size": r["size"],
                    "summary": r["summary"],
                    "key_entities": r.get("key_entities", [])
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Failed to get communities with summaries: {e}")
            return []
