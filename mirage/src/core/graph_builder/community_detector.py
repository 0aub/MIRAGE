"""
Community Detection Module for GraphRAG

Implements hierarchical community detection using Louvain algorithm.
This is a core component of GraphRAG that enables:
- Global search over community summaries
- Theme identification
- Entity clustering

0 LLM calls needed - pure graph algorithm.
"""

import networkx as nx
from community import community_louvain
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """Represents a community at a specific hierarchy level."""
    id: str
    level: int
    entities: List[str]
    size: int
    parent_community: Optional[str] = None
    child_communities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "level": self.level,
            "entities": self.entities,
            "size": self.size,
            "parent_community": self.parent_community,
            "child_communities": self.child_communities
        }


@dataclass
class CommunityDetectionResult:
    """Result from community detection."""
    communities: List[Community]
    hierarchy_levels: int
    total_communities: int
    modularity: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "communities": [c.to_dict() for c in self.communities],
            "hierarchy_levels": self.hierarchy_levels,
            "total_communities": self.total_communities,
            "modularity": self.modularity
        }


class CommunityDetector:
    """
    Hierarchical community detection using Louvain algorithm.

    Features:
    - Detects communities at multiple hierarchy levels (3-5 levels)
    - Uses Louvain algorithm (similar to Leiden, more widely supported)
    - No LLM calls needed - pure graph algorithm
    - Stores communities in Neo4j

    Usage:
        detector = CommunityDetector(neo4j_client)
        result = detector.detect_communities(resolution=1.0, levels=3)
        detector.store_communities_in_neo4j(result)
    """

    def __init__(self, neo4j_client):
        """
        Initialize community detector.

        Args:
            neo4j_client: Neo4j client instance
        """
        self.neo4j_client = neo4j_client

    def detect_communities(
        self,
        resolution: float = 1.0,
        levels: int = 3,
        min_community_size: int = 3
    ) -> CommunityDetectionResult:
        """
        Detect hierarchical communities in the knowledge graph.

        Args:
            resolution: Resolution parameter for Louvain (higher = more communities)
            levels: Number of hierarchy levels to detect (3-5 recommended)
            min_community_size: Minimum number of entities per community

        Returns:
            CommunityDetectionResult with all detected communities

        Example:
            >>> detector = CommunityDetector(neo4j_client)
            >>> result = detector.detect_communities(resolution=1.0, levels=3)
            >>> print(f"Detected {result.total_communities} communities")
        """
        logger.info(f"Starting community detection (resolution={resolution}, levels={levels})")

        # Step 1: Load graph from Neo4j into NetworkX
        graph = self._load_graph_from_neo4j()

        if graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, no communities to detect")
            return CommunityDetectionResult(
                communities=[],
                hierarchy_levels=0,
                total_communities=0,
                modularity=0.0
            )

        logger.info(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        # Step 2: Run hierarchical Louvain algorithm
        communities = self._detect_hierarchical_communities(
            graph,
            resolution=resolution,
            levels=levels,
            min_size=min_community_size
        )

        # Step 3: Calculate modularity for finest level
        level_0_partition = {
            entity: comm.id
            for comm in communities
            if comm.level == 0
            for entity in comm.entities
        }
        modularity = community_louvain.modularity(level_0_partition, graph)

        logger.info(
            f"Community detection complete: "
            f"{len(communities)} communities across {levels} levels, "
            f"modularity: {modularity:.3f}"
        )

        return CommunityDetectionResult(
            communities=communities,
            hierarchy_levels=levels,
            total_communities=len(communities),
            modularity=modularity
        )

    def _load_graph_from_neo4j(self) -> nx.Graph:
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

    def _detect_hierarchical_communities(
        self,
        graph: nx.Graph,
        resolution: float,
        levels: int,
        min_size: int
    ) -> List[Community]:
        """
        Detect communities at multiple hierarchy levels.

        Strategy:
        - Level 0: Finest granularity (most communities)
        - Level 1: Merge small communities from Level 0
        - Level 2+: Progressive coarsening

        Args:
            graph: NetworkX graph
            resolution: Resolution parameter
            levels: Number of hierarchy levels
            min_size: Minimum community size

        Returns:
            List of Community objects
        """
        all_communities = []
        current_graph = graph.copy()

        # Track mapping from super-node IDs to original entities
        # At level 0, nodes are entity names; at level 1+, nodes are community IDs
        node_to_entities: Dict[str, List[str]] = {}
        # Initialize: each entity maps to itself
        for node in graph.nodes():
            node_to_entities[str(node)] = [str(node)]

        for level in range(levels):
            logger.info(f"Detecting communities at level {level}...")

            # Run Louvain algorithm
            # Adjust resolution: higher for finer granularity at lower levels
            level_resolution = resolution * (1.5 ** (levels - level - 1))
            partition = community_louvain.best_partition(
                current_graph,
                resolution=level_resolution,
                random_state=42
            )

            # Group nodes by community
            communities_dict = defaultdict(list)
            for node, comm_id in partition.items():
                communities_dict[comm_id].append(str(node))

            # Filter out small communities
            communities_dict = {
                comm_id: nodes
                for comm_id, nodes in communities_dict.items()
                if len(nodes) >= min_size or level == 0  # Keep all at level 0
            }

            # Create Community objects with ORIGINAL entities (not super-node IDs)
            level_communities = []
            new_node_to_entities: Dict[str, List[str]] = {}

            for comm_id, nodes in communities_dict.items():
                # Collect original entities from all nodes in this community
                original_entities = []
                for node in nodes:
                    original_entities.extend(node_to_entities.get(str(node), [str(node)]))
                original_entities = list(set(original_entities))  # Remove duplicates

                community = Community(
                    id=f"L{level}_C{comm_id}",
                    level=level,
                    entities=original_entities,
                    size=len(original_entities)
                )
                level_communities.append(community)

                # Update mapping for next level
                new_node_to_entities[str(comm_id)] = original_entities

            all_communities.extend(level_communities)
            node_to_entities = new_node_to_entities

            logger.info(f"Level {level}: Detected {len(level_communities)} communities")

            # Prepare for next level: create super-graph
            if level < levels - 1:
                current_graph = self._create_super_graph(current_graph, partition)

        # Build parent-child relationships
        all_communities = self._build_hierarchy_relationships(all_communities)

        return all_communities

    def _create_super_graph(
        self,
        graph: nx.Graph,
        partition: Dict[str, int]
    ) -> nx.Graph:
        """
        Create a super-graph where each community becomes a super-node.

        Args:
            graph: Original graph
            partition: Community assignments {node: community_id}

        Returns:
            Super-graph with communities as nodes
        """
        super_graph = nx.Graph()

        # Group nodes by community
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)

        # Add edges between communities based on inter-community edges
        for comm_id1, nodes1 in communities.items():
            for comm_id2, nodes2 in communities.items():
                if comm_id1 >= comm_id2:
                    continue

                # Count edges between communities
                edge_count = 0
                for node1 in nodes1:
                    for node2 in nodes2:
                        if graph.has_edge(node1, node2):
                            edge_count += 1

                # Add edge if communities are connected
                if edge_count > 0:
                    super_graph.add_edge(
                        comm_id1,
                        comm_id2,
                        weight=edge_count
                    )

        return super_graph

    def _build_hierarchy_relationships(
        self,
        communities: List[Community]
    ) -> List[Community]:
        """
        Build parent-child relationships between communities.

        Strategy:
        - A Level N community is the parent of Level N-1 communities
          if they share entities

        Args:
            communities: List of communities

        Returns:
            Communities with parent/child relationships set
        """
        # Group by level
        by_level = defaultdict(list)
        for comm in communities:
            by_level[comm.level].append(comm)

        # Build relationships bottom-up
        max_level = max(by_level.keys())

        for level in range(max_level):
            child_level = level
            parent_level = level + 1

            children = by_level[child_level]
            parents = by_level[parent_level]

            # For each child, find parent with most entity overlap
            for child in children:
                child_entities = set(child.entities)

                best_parent = None
                best_overlap = 0

                for parent in parents:
                    parent_entities = set(parent.entities)
                    overlap = len(child_entities & parent_entities)

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_parent = parent

                # Set relationships
                if best_parent:
                    child.parent_community = best_parent.id
                    if child.id not in best_parent.child_communities:
                        best_parent.child_communities.append(child.id)

        return communities

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

        Example:
            >>> detector.store_communities_in_neo4j(result, clear_existing=True)
            {'communities_created': 42, 'belongs_to_rels': 1250, 'parent_of_rels': 15}
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
