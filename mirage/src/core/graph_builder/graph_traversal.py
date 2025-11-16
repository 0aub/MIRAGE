"""
Multi-Hop Graph Traversal for GraphRAG

Enables entity-centric retrieval by traversing the knowledge graph
to find related entities, relationships, and contextual information.

This is a core component of Local Search in GraphRAG.
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TraversalResult:
    """Result from graph traversal."""
    seed_entities: List[str]
    discovered_entities: List[Dict]
    relationships: List[Dict]
    paths: List[List[str]] = field(default_factory=list)
    depth: int = 0
    total_nodes: int = 0
    total_edges: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "seed_entities": self.seed_entities,
            "discovered_entities": self.discovered_entities,
            "relationships": self.relationships,
            "paths": self.paths,
            "depth": self.depth,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges
        }


class GraphTraversal:
    """
    Multi-hop graph traversal for entity-centric retrieval.

    Features:
    - Multi-hop traversal (1-3 hops)
    - Relationship type filtering
    - Path finding between entities
    - Neighbor discovery
    - Community-aware traversal

    Usage:
        traversal = GraphTraversal(neo4j_client)
        result = traversal.traverse_from_entities(
            ["Ahmed Hassan", "IBM"],
            max_hops=2
        )
    """

    def __init__(self, neo4j_client):
        """
        Initialize graph traversal.

        Args:
            neo4j_client: Neo4j client instance
        """
        self.neo4j_client = neo4j_client

    def traverse_from_entities(
        self,
        entity_names: List[str],
        max_hops: int = 2,
        relationship_types: Optional[List[str]] = None,
        limit_per_hop: int = 50
    ) -> TraversalResult:
        """
        Traverse graph starting from seed entities.

        Args:
            entity_names: Starting entities
            max_hops: Maximum traversal depth (1-3 recommended)
            relationship_types: Optional filter for relationship types
            limit_per_hop: Maximum entities to discover per hop

        Returns:
            TraversalResult with all discovered entities and relationships

        Example:
            >>> result = traversal.traverse_from_entities(
            ...     ["Ahmed Hassan", "IBM"],
            ...     max_hops=2
            ... )
            >>> print(f"Found {result.total_nodes} entities")
        """
        if not entity_names:
            logger.warning("No seed entities provided for traversal")
            return TraversalResult(
                seed_entities=[],
                discovered_entities=[],
                relationships=[],
                depth=0,
                total_nodes=0,
                total_edges=0
            )

        # Build relationship filter for Cypher query
        relationship_filter = ""
        if relationship_types:
            types_str = "|".join(f"`{rt}`" for rt in relationship_types)
            relationship_filter = f":{types_str}"

        # Cypher query for multi-hop traversal
        query = f"""
        // Start from seed entities
        MATCH (start:Entity)
        WHERE start.name IN $entity_names

        // Traverse up to max_hops (using variable-length pattern)
        MATCH path = (start)-[{relationship_filter}*1..{max_hops}]-(discovered:Entity)

        // Collect all nodes and relationships in the path
        WITH start, discovered, path,
             nodes(path) as path_nodes,
             relationships(path) as path_rels
        LIMIT $limit_per_hop

        // Return discovered entities and their properties
        RETURN DISTINCT
            discovered.name as name,
            discovered.type as type,
            properties(discovered) as properties,
            labels(discovered) as labels,
            length(path) as distance
        ORDER BY distance, name
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'entity_names': entity_names,
                'limit_per_hop': limit_per_hop
            })

            # Process discovered entities
            discovered_entities = []
            seen_entities = set(entity_names)  # Don't include seed entities in discovered

            for record in results:
                name = record.get('name')
                if name and name not in seen_entities:
                    discovered_entities.append({
                        'name': name,
                        'type': record.get('type', 'Unknown'),
                        'properties': record.get('properties', {}),
                        'labels': record.get('labels', []),
                        'distance': record.get('distance', 0)
                    })
                    seen_entities.add(name)

            # Get all relationships between seed and discovered entities
            all_entity_names = list(seen_entities)
            relationships = self._get_relationships_between(all_entity_names)

            # Find paths between seed entities
            paths = self._find_paths_between_seeds(entity_names, max_hops)

            logger.info(
                f"Traversal from {len(entity_names)} seed entities: "
                f"discovered {len(discovered_entities)} entities, "
                f"{len(relationships)} relationships "
                f"(depth: {max_hops})"
            )

            return TraversalResult(
                seed_entities=entity_names,
                discovered_entities=discovered_entities,
                relationships=relationships,
                paths=paths,
                depth=max_hops,
                total_nodes=len(all_entity_names),
                total_edges=len(relationships)
            )

        except Exception as e:
            logger.error(f"Error during graph traversal: {e}")
            return TraversalResult(
                seed_entities=entity_names,
                discovered_entities=[],
                relationships=[],
                depth=max_hops,
                total_nodes=len(entity_names),
                total_edges=0
            )

    def _get_relationships_between(
        self,
        entity_names: List[str]
    ) -> List[Dict]:
        """
        Get all relationships between a set of entities.

        Args:
            entity_names: List of entity names

        Returns:
            List of relationship dicts
        """
        query = """
        MATCH (e1:Entity)-[r]->(e2:Entity)
        WHERE e1.name IN $entity_names
          AND e2.name IN $entity_names
        RETURN
            e1.name as source,
            e2.name as target,
            type(r) as relationship_type,
            properties(r) as properties
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'entity_names': entity_names
            })

            relationships = []
            for record in results:
                relationships.append({
                    'source': record.get('source'),
                    'target': record.get('target'),
                    'type': record.get('relationship_type', 'RELATED_TO'),
                    'properties': record.get('properties', {})
                })

            return relationships

        except Exception as e:
            logger.error(f"Error getting relationships: {e}")
            return []

    def _find_paths_between_seeds(
        self,
        seed_entities: List[str],
        max_length: int = 3
    ) -> List[List[str]]:
        """
        Find all paths between seed entities.

        Args:
            seed_entities: Seed entity names
            max_length: Maximum path length

        Returns:
            List of paths (each path is a list of entity names)
        """
        if len(seed_entities) < 2:
            return []

        query = """
        MATCH (start:Entity), (end:Entity)
        WHERE start.name IN $seed_entities
          AND end.name IN $seed_entities
          AND start.name < end.name  // Avoid duplicate paths

        MATCH path = shortestPath((start)-[*1..%d]-(end))

        RETURN [node in nodes(path) | node.name] as path
        LIMIT 20
        """ % max_length

        try:
            results = self.neo4j_client.execute_query(query, {
                'seed_entities': seed_entities
            })

            paths = [record.get('path', []) for record in results]
            return paths

        except Exception as e:
            logger.error(f"Error finding paths: {e}")
            return []

    def get_entity_neighbors(
        self,
        entity_name: str,
        hops: int = 1,
        direction: str = "both"
    ) -> List[Dict]:
        """
        Get direct neighbors of an entity.

        Args:
            entity_name: Entity to start from
            hops: Number of hops (default 1 for direct neighbors)
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of neighbor entities with their relationships

        Example:
            >>> neighbors = traversal.get_entity_neighbors("Ahmed Hassan", hops=1)
            >>> print(f"Ahmed Hassan has {len(neighbors)} direct neighbors")
        """
        # Build direction pattern
        if direction == "outgoing":
            pattern = f"-[*1..{hops}]->"
        elif direction == "incoming":
            pattern = f"<-[*1..{hops}]-"
        else:  # both
            pattern = f"-[*1..{hops}]-"

        query = f"""
        MATCH (start:Entity {{name: $entity_name}}){pattern}(neighbor:Entity)
        WHERE start <> neighbor
        RETURN DISTINCT
            neighbor.name as name,
            neighbor.type as type,
            properties(neighbor) as properties
        LIMIT 100
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'entity_name': entity_name
            })

            neighbors = []
            for record in results:
                neighbors.append({
                    'name': record.get('name'),
                    'type': record.get('type', 'Unknown'),
                    'properties': record.get('properties', {})
                })

            logger.debug(f"Found {len(neighbors)} neighbors for '{entity_name}' ({hops} hops)")
            return neighbors

        except Exception as e:
            logger.error(f"Error getting neighbors for '{entity_name}': {e}")
            return []

    def get_entity_community(
        self,
        entity_name: str,
        level: int = 0
    ) -> Optional[str]:
        """
        Get the community that an entity belongs to.

        Args:
            entity_name: Entity name
            level: Community hierarchy level (0 = finest granularity)

        Returns:
            Community ID or None if no community found

        Example:
            >>> community_id = traversal.get_entity_community("Ahmed Hassan", level=0)
            >>> if community_id:
            ...     print(f"Ahmed Hassan belongs to community: {community_id}")
        """
        query = """
        MATCH (e:Entity {name: $entity_name})-[:BELONGS_TO {level: $level}]->(c:Community)
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
            logger.error(f"Error getting community for '{entity_name}': {e}")
            return None

    def get_community_entities(
        self,
        community_id: str
    ) -> List[Dict]:
        """
        Get all entities in a community.

        Args:
            community_id: Community identifier

        Returns:
            List of entity dicts

        Example:
            >>> entities = traversal.get_community_entities("L0_C1")
            >>> print(f"Community L0_C1 has {len(entities)} entities")
        """
        query = """
        MATCH (e:Entity)-[:BELONGS_TO]->(c:Community {id: $community_id})
        RETURN
            e.name as name,
            e.type as type,
            properties(e) as properties
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'community_id': community_id
            })

            entities = []
            for record in results:
                entities.append({
                    'name': record.get('name'),
                    'type': record.get('type', 'Unknown'),
                    'properties': record.get('properties', {})
                })

            return entities

        except Exception as e:
            logger.error(f"Error getting community entities for '{community_id}': {e}")
            return []

    def expand_entities_with_context(
        self,
        entity_names: List[str],
        max_hops: int = 2,
        include_communities: bool = True
    ) -> Dict:
        """
        Expand entities with full context for retrieval.

        Combines multi-hop traversal with community information
        to provide rich context for query answering.

        Args:
            entity_names: Starting entities
            max_hops: Traversal depth
            include_communities: Whether to include community summaries

        Returns:
            Dict with entities, relationships, communities, and summaries

        Example:
            >>> context = traversal.expand_entities_with_context(
            ...     ["Ahmed Hassan"],
            ...     max_hops=2,
            ...     include_communities=True
            ... )
            >>> print(f"Context includes {len(context['entities'])} entities")
        """
        # Perform multi-hop traversal
        traversal_result = self.traverse_from_entities(
            entity_names,
            max_hops=max_hops
        )

        context = {
            'seed_entities': entity_names,
            'entities': traversal_result.discovered_entities,
            'relationships': traversal_result.relationships,
            'paths': traversal_result.paths,
            'communities': [],
            'community_summaries': []
        }

        # Add community information if requested
        if include_communities:
            community_ids = set()

            # Get communities for seed entities
            for entity_name in entity_names:
                comm_id = self.get_entity_community(entity_name, level=0)
                if comm_id:
                    community_ids.add(comm_id)

            # Get community summaries
            for comm_id in community_ids:
                summary = self._get_community_summary(comm_id)
                if summary:
                    context['communities'].append(comm_id)
                    context['community_summaries'].append(summary)

        return context

    def _get_community_summary(self, community_id: str) -> Optional[str]:
        """
        Get community summary.

        Args:
            community_id: Community identifier

        Returns:
            Summary text or None
        """
        query = """
        MATCH (c:Community {id: $community_id})
        RETURN c.summary as summary
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'community_id': community_id
            })

            if results:
                return results[0].get('summary')
            return None

        except Exception as e:
            logger.error(f"Error getting summary for community '{community_id}': {e}")
            return None

    def find_connection_between_entities(
        self,
        entity1: str,
        entity2: str,
        max_path_length: int = 4
    ) -> Optional[List[str]]:
        """
        Find shortest path between two entities.

        Args:
            entity1: First entity name
            entity2: Second entity name
            max_path_length: Maximum path length to search

        Returns:
            List of entity names forming the path, or None if no path found

        Example:
            >>> path = traversal.find_connection_between_entities(
            ...     "Ahmed Hassan",
            ...     "IBM"
            ... )
            >>> if path:
            ...     print(f"Connection: {' -> '.join(path)}")
        """
        query = """
        MATCH (e1:Entity {name: $entity1}), (e2:Entity {name: $entity2})
        MATCH path = shortestPath((e1)-[*1..%d]-(e2))
        RETURN [node in nodes(path) | node.name] as path
        """ % max_path_length

        try:
            results = self.neo4j_client.execute_query(query, {
                'entity1': entity1,
                'entity2': entity2
            })

            if results:
                return results[0].get('path')
            return None

        except Exception as e:
            logger.error(f"Error finding connection between '{entity1}' and '{entity2}': {e}")
            return None

    def get_traversal_statistics(self, entity_names: List[str]) -> Dict:
        """
        Get statistics about graph structure around entities.

        Args:
            entity_names: Entities to analyze

        Returns:
            Dict with various statistics

        Example:
            >>> stats = traversal.get_traversal_statistics(["Ahmed Hassan"])
            >>> print(f"Degree: {stats['avg_degree']}")
        """
        query = """
        MATCH (e:Entity)
        WHERE e.name IN $entity_names

        WITH e, size((e)-[]-()) as degree
        RETURN
            count(e) as entity_count,
            avg(degree) as avg_degree,
            max(degree) as max_degree,
            min(degree) as min_degree
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'entity_names': entity_names
            })

            if results:
                record = results[0]
                return {
                    'entity_count': record.get('entity_count', 0),
                    'avg_degree': record.get('avg_degree', 0),
                    'max_degree': record.get('max_degree', 0),
                    'min_degree': record.get('min_degree', 0)
                }

            return {
                'entity_count': 0,
                'avg_degree': 0,
                'max_degree': 0,
                'min_degree': 0
            }

        except Exception as e:
            logger.error(f"Error getting traversal statistics: {e}")
            return {}
