"""
Community Detection Module for GraphRAG

Implements hierarchical community detection using Louvain algorithm.
This is a core component of GraphRAG that enables:
- Global search over community summaries
- Theme identification
- Entity clustering

0 LLM calls needed - pure graph algorithm.
"""

from community import community_louvain
from typing import List, Dict, Optional
from collections import defaultdict
import logging

from .models import Community, CommunityDetectionResult
from .algorithms import detect_hierarchical_communities, build_hierarchy_relationships
from .neo4j_ops import CommunityNeo4jOps

logger = logging.getLogger(__name__)

# Resolution levels for GraphRAG hierarchical communities
RESOLUTION_LEVELS = [
    (0, 2.0, "subtopic"),   # Level 0: Fine-grained (many small communities)
    (1, 1.0, "topic"),      # Level 1: Topic-level (default granularity)
    (2, 0.5, "domain"),     # Level 2: Domain-level groupings
    (3, 0.1, "macro"),      # Level 3: Very broad themes
]


class CommunityDetector(CommunityNeo4jOps):
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
        super().__init__(neo4j_client)

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
        graph = self.load_graph_from_neo4j()

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
        communities = detect_hierarchical_communities(
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

    def detect_multi_resolution_communities(
        self,
        min_community_size: int = 2
    ) -> CommunityDetectionResult:
        """
        GraphRAG Enhancement: Detect communities at multiple resolution levels.

        This creates a hierarchical community structure similar to Microsoft GraphRAG:
        - Level 0 (resolution=2.0): Fine-grained subtopics
        - Level 1 (resolution=1.0): Standard topics
        - Level 2 (resolution=0.5): Broader domains
        - Level 3 (resolution=0.1): High-level themes

        Args:
            min_community_size: Minimum entities per community

        Returns:
            CommunityDetectionResult with hierarchical communities
        """
        logger.info("Starting multi-resolution community detection (GraphRAG style)...")

        # Load graph from Neo4j
        graph = self.load_graph_from_neo4j()

        if graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, no communities to detect")
            return CommunityDetectionResult(
                communities=[],
                hierarchy_levels=0,
                total_communities=0,
                modularity=0.0
            )

        logger.info(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        all_communities = []

        for level, resolution, level_name in RESOLUTION_LEVELS:
            logger.info(f"Detecting level {level} ({level_name}) with resolution {resolution}...")

            # Run Louvain with specific resolution
            try:
                partition = community_louvain.best_partition(
                    graph,
                    resolution=resolution,
                    random_state=42
                )

                # Group nodes by community
                communities_dict = defaultdict(list)
                for node, comm_id in partition.items():
                    communities_dict[comm_id].append(str(node))

                # Filter small communities (except at finest level)
                filtered = {}
                for comm_id, entities in communities_dict.items():
                    if len(entities) >= min_community_size or (level == 0 and len(entities) >= 1):
                        filtered[comm_id] = entities

                logger.info(f"Level {level}: {len(filtered)} communities after filtering")

                # Create Community objects with entity descriptions
                for comm_id, entities in filtered.items():
                    # Get key entities (top 5 by importance/confidence)
                    key_entities = self.get_key_entities(entities, limit=5)

                    # Get entity descriptions for context
                    entity_descriptions = self.get_entity_descriptions(entities)

                    community = Community(
                        id=f"L{level}_C{comm_id}",
                        level=level,
                        entities=entities,
                        size=len(entities),
                        resolution=resolution,
                        key_entities=key_entities,
                        entity_descriptions=entity_descriptions
                    )
                    all_communities.append(community)

            except Exception as e:
                logger.error(f"Error detecting communities at level {level}: {e}")

        # Build hierarchy relationships
        all_communities = build_hierarchy_relationships(all_communities)

        # Calculate modularity at default level (level 1)
        level_1_communities = [c for c in all_communities if c.level == 1]
        modularity = 0.0
        if level_1_communities:
            partition = {}
            for comm in level_1_communities:
                for entity in comm.entities:
                    partition[entity] = comm.id
            try:
                modularity = community_louvain.modularity(partition, graph)
            except:
                pass

        logger.info(
            f"Multi-resolution detection complete: {len(all_communities)} communities "
            f"across {len(RESOLUTION_LEVELS)} levels"
        )

        return CommunityDetectionResult(
            communities=all_communities,
            hierarchy_levels=len(RESOLUTION_LEVELS),
            total_communities=len(all_communities),
            modularity=modularity
        )

    def get_community_for_summarization(self, community_id: str) -> Optional[Dict]:
        """
        Get community data formatted for LLM summarization.

        Returns entity names, descriptions, and relationships for summary generation.

        Args:
            community_id: Community identifier

        Returns:
            Dict with entities, descriptions, relationships for summarization
        """
        # Get community info
        community_info = self.get_community_info(community_id)
        if not community_info:
            return None

        entities = community_info.get('entities', [])

        # Get entity descriptions
        descriptions = self.get_entity_descriptions(entities)

        # Get relationships within community
        relationships = self.get_community_relationships(entities)

        return {
            "community_id": community_id,
            "level": community_info.get('level'),
            "entity_count": len(entities),
            "entities": entities,
            "entity_descriptions": descriptions,
            "relationships": relationships,
            "key_entities": entities[:5]  # Top entities
        }
