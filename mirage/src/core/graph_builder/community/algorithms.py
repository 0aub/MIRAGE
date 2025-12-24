"""
Community Detection Algorithms

Pure graph algorithms for hierarchical community detection using Louvain.
"""

import networkx as nx
from community import community_louvain
from typing import List, Dict
from collections import defaultdict
import logging

from .models import Community

logger = logging.getLogger(__name__)


def detect_hierarchical_communities(
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
            current_graph = create_super_graph(current_graph, partition)

    # Build parent-child relationships
    all_communities = build_hierarchy_relationships(all_communities)

    return all_communities


def create_super_graph(
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


def build_hierarchy_relationships(
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

    if not by_level:
        return communities

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
