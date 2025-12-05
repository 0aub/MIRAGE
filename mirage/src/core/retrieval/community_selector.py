"""
Dynamic Community Selection - O(log C) Hierarchical Pruning

Key Innovation from Microsoft GraphRAG:
- Instead of querying ALL communities (expensive O(C))
- Traverse hierarchy and prune irrelevant subtrees early
- Result: O(log C) complexity with minimal quality loss

Algorithm:
1. Start at root level communities
2. Score relevance using embedding similarity (cheap, no LLM)
3. If relevant (> threshold): traverse children
4. If irrelevant: prune entire subtree (HUGE savings)
5. Return only relevant leaf communities for map-reduce

This is THE efficiency innovation that makes GraphRAG production-ready.
"""

import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


@dataclass
class CommunityNode:
    """Represents a community in the hierarchy."""
    id: str
    level: int
    summary: str
    summary_embedding: Optional[np.ndarray] = None
    size: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    key_entities: List[str] = field(default_factory=list)
    is_leaf: bool = False


@dataclass
class SelectionStats:
    """Statistics from community selection process."""
    total_communities: int = 0
    communities_scored: int = 0
    communities_pruned: int = 0
    communities_selected: int = 0
    pruned_at_depth: List[int] = field(default_factory=list)
    selection_time_ms: float = 0.0

    @property
    def pruning_ratio(self) -> float:
        """Fraction of communities pruned (higher = more efficient)."""
        if self.total_communities == 0:
            return 0.0
        return self.communities_pruned / self.total_communities

    @property
    def efficiency_gain(self) -> float:
        """How many fewer communities queried vs brute force."""
        if self.communities_selected == 0:
            return 0.0
        return self.total_communities / self.communities_selected


@dataclass
class SelectionResult:
    """Result of community selection."""
    selected_communities: List[CommunityNode]
    stats: SelectionStats
    query: str


class DynamicCommunitySelector:
    """
    Hierarchical community selection with early pruning.

    Key Insight:
    - Community summaries have embeddings
    - If a parent community's summary is irrelevant to query,
      ALL its children will also be irrelevant
    - By pruning early, we avoid scoring/querying entire subtrees

    Complexity:
    - Brute force: O(C) where C = total communities
    - With pruning: O(log C) to O(√C) depending on query specificity

    Example:
        selector = DynamicCommunitySelector(neo4j_client, embedder)
        result = selector.select_communities(query, threshold=0.3)
        # Now only query result.selected_communities instead of all
    """

    def __init__(
        self,
        neo4j_client,
        embedder,
        default_threshold: float = 0.3,
        min_communities: int = 3,
        max_communities: int = 20
    ):
        """
        Initialize community selector.

        Args:
            neo4j_client: Neo4j client for graph queries
            embedder: Embedding manager for query/summary embeddings
            default_threshold: Relevance threshold for pruning (0-1)
            min_communities: Minimum communities to select
            max_communities: Maximum communities to return
        """
        self.neo4j_client = neo4j_client
        self.embedder = embedder
        self.default_threshold = default_threshold
        self.min_communities = min_communities
        self.max_communities = max_communities

        # Cache for community hierarchy
        self._hierarchy_cache: Dict[str, CommunityNode] = {}
        self._root_communities: List[str] = []
        self._cache_valid = False

        logger.info(
            f"DynamicCommunitySelector initialized: "
            f"threshold={default_threshold}, min={min_communities}, max={max_communities}"
        )

    def select_communities(
        self,
        query: str,
        threshold: Optional[float] = None,
        force_refresh: bool = False
    ) -> SelectionResult:
        """
        Select relevant communities using hierarchical pruning.

        Algorithm:
        1. Get query embedding
        2. Load community hierarchy (cached)
        3. Start at root level
        4. Score each community's relevance (embedding similarity)
        5. Traverse children of relevant communities
        6. Prune subtrees of irrelevant communities
        7. Return leaf communities for map-reduce

        Args:
            query: User query string
            threshold: Override default relevance threshold
            force_refresh: Force reload of hierarchy cache

        Returns:
            SelectionResult with selected communities and stats
        """
        import time
        start_time = time.time()

        threshold = threshold or self.default_threshold
        stats = SelectionStats()

        # Step 1: Get query embedding
        query_embedding = self._get_query_embedding(query)
        if query_embedding is None:
            logger.warning("Failed to get query embedding, returning all communities")
            return self._fallback_all_communities(query, stats)

        # Step 2: Load hierarchy
        if not self._cache_valid or force_refresh:
            self._load_hierarchy()

        stats.total_communities = len(self._hierarchy_cache)

        if stats.total_communities == 0:
            logger.warning("No communities in hierarchy")
            return SelectionResult(
                selected_communities=[],
                stats=stats,
                query=query
            )

        # Step 3: Hierarchical traversal with pruning
        selected = []
        visited = set()

        def traverse(community_id: str, depth: int = 0):
            """Recursively traverse and prune."""
            if community_id in visited:
                return
            visited.add(community_id)

            community = self._hierarchy_cache.get(community_id)
            if not community:
                return

            # Score relevance
            relevance = self._score_relevance(query_embedding, community)
            stats.communities_scored += 1

            if relevance < threshold:
                # PRUNE: Skip this community AND all children
                stats.communities_pruned += 1
                stats.pruned_at_depth.append(depth)

                # Count pruned children (they won't be visited)
                pruned_children = self._count_descendants(community_id)
                stats.communities_pruned += pruned_children

                logger.debug(
                    f"Pruned community {community_id} at depth {depth} "
                    f"(relevance={relevance:.3f} < {threshold}), "
                    f"skipping {pruned_children} children"
                )
                return

            # Community is relevant
            if community.is_leaf or not community.children_ids:
                # Leaf node - add to selection
                selected.append(community)
                stats.communities_selected += 1
            else:
                # Internal node - traverse children
                for child_id in community.children_ids:
                    traverse(child_id, depth + 1)

        # Start traversal from roots
        for root_id in self._root_communities:
            traverse(root_id, depth=0)

        # Ensure minimum communities
        if len(selected) < self.min_communities:
            selected = self._expand_selection(
                selected,
                query_embedding,
                self.min_communities - len(selected)
            )
            stats.communities_selected = len(selected)

        # Limit to max communities
        if len(selected) > self.max_communities:
            # Sort by relevance and take top
            selected = sorted(
                selected,
                key=lambda c: self._score_relevance(query_embedding, c),
                reverse=True
            )[:self.max_communities]
            stats.communities_selected = len(selected)

        stats.selection_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Community selection: {stats.communities_selected}/{stats.total_communities} selected, "
            f"{stats.communities_pruned} pruned ({stats.pruning_ratio:.1%}), "
            f"efficiency gain: {stats.efficiency_gain:.1f}x, "
            f"time: {stats.selection_time_ms:.1f}ms"
        )

        return SelectionResult(
            selected_communities=selected,
            stats=stats,
            query=query
        )

    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for query."""
        try:
            embedding = self.embedder.embed(query)
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return None

    def _score_relevance(
        self,
        query_embedding: np.ndarray,
        community: CommunityNode
    ) -> float:
        """
        Score relevance using cosine similarity.

        Fast operation - no LLM call, just vector math.
        """
        if community.summary_embedding is None:
            # Fallback: embed summary on-the-fly (cached next time)
            if community.summary:
                try:
                    embedding = self.embedder.embed(community.summary)
                    if isinstance(embedding, list):
                        embedding = np.array(embedding)
                    community.summary_embedding = embedding
                except Exception:
                    return 0.5  # Default moderate relevance
            else:
                return 0.3  # Low relevance for empty summary

        # Cosine similarity
        similarity = np.dot(query_embedding, community.summary_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_summary = np.linalg.norm(community.summary_embedding)

        if norm_query > 0 and norm_summary > 0:
            similarity = similarity / (norm_query * norm_summary)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, (similarity + 1) / 2))

    def _load_hierarchy(self):
        """Load community hierarchy from Neo4j."""
        logger.info("Loading community hierarchy from Neo4j...")

        # Get all communities with their hierarchy info
        query = """
        MATCH (c:Community)
        OPTIONAL MATCH (parent:Community)-[:PARENT_OF]->(c)
        OPTIONAL MATCH (c)-[:PARENT_OF]->(child:Community)
        RETURN
            c.id as id,
            c.level as level,
            c.summary as summary,
            c.summary_embedding as summary_embedding,
            c.size as size,
            c.key_entities as key_entities,
            parent.id as parent_id,
            COLLECT(DISTINCT child.id) as children_ids
        ORDER BY c.level DESC
        """

        try:
            results = self.neo4j_client.execute_query(query)

            self._hierarchy_cache.clear()
            self._root_communities.clear()
            max_level = 0

            for record in results:
                community_id = record.get('id')
                if not community_id:
                    continue

                level = record.get('level', 0)
                max_level = max(max_level, level)

                # Parse embedding if stored
                summary_embedding = None
                raw_embedding = record.get('summary_embedding')
                if raw_embedding:
                    try:
                        if isinstance(raw_embedding, list):
                            summary_embedding = np.array(raw_embedding)
                        elif isinstance(raw_embedding, str):
                            import json
                            summary_embedding = np.array(json.loads(raw_embedding))
                    except Exception:
                        pass

                children_ids = [c for c in record.get('children_ids', []) if c]

                community = CommunityNode(
                    id=community_id,
                    level=level,
                    summary=record.get('summary', ''),
                    summary_embedding=summary_embedding,
                    size=record.get('size', 0),
                    parent_id=record.get('parent_id'),
                    children_ids=children_ids,
                    key_entities=record.get('key_entities', []) or [],
                    is_leaf=len(children_ids) == 0
                )

                self._hierarchy_cache[community_id] = community

            # Identify root communities (highest level, no parent)
            self._root_communities = [
                cid for cid, c in self._hierarchy_cache.items()
                if c.level == max_level or c.parent_id is None
            ]

            self._cache_valid = True

            logger.info(
                f"Loaded {len(self._hierarchy_cache)} communities, "
                f"{len(self._root_communities)} roots, "
                f"max level: {max_level}"
            )

        except Exception as e:
            logger.error(f"Failed to load hierarchy: {e}")
            self._cache_valid = False

    def _count_descendants(self, community_id: str) -> int:
        """Count all descendants of a community."""
        count = 0
        community = self._hierarchy_cache.get(community_id)
        if not community:
            return 0

        for child_id in community.children_ids:
            count += 1
            count += self._count_descendants(child_id)

        return count

    def _expand_selection(
        self,
        selected: List[CommunityNode],
        query_embedding: np.ndarray,
        needed: int
    ) -> List[CommunityNode]:
        """Expand selection to meet minimum requirement."""
        selected_ids = {c.id for c in selected}

        # Get all leaf communities not yet selected
        candidates = [
            c for c in self._hierarchy_cache.values()
            if c.is_leaf and c.id not in selected_ids
        ]

        # Sort by relevance
        candidates_scored = [
            (c, self._score_relevance(query_embedding, c))
            for c in candidates
        ]
        candidates_scored.sort(key=lambda x: x[1], reverse=True)

        # Add top candidates
        for candidate, score in candidates_scored[:needed]:
            selected.append(candidate)

        return selected

    def _fallback_all_communities(
        self,
        query: str,
        stats: SelectionStats
    ) -> SelectionResult:
        """Fallback: return all leaf communities."""
        if not self._cache_valid:
            self._load_hierarchy()

        all_leaves = [
            c for c in self._hierarchy_cache.values()
            if c.is_leaf
        ][:self.max_communities]

        stats.communities_selected = len(all_leaves)
        stats.total_communities = len(self._hierarchy_cache)

        return SelectionResult(
            selected_communities=all_leaves,
            stats=stats,
            query=query
        )

    def invalidate_cache(self):
        """Invalidate hierarchy cache (call after graph updates)."""
        self._cache_valid = False
        self._hierarchy_cache.clear()
        self._root_communities.clear()
        logger.info("Community hierarchy cache invalidated")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cache_valid": self._cache_valid,
            "total_communities": len(self._hierarchy_cache),
            "root_communities": len(self._root_communities),
            "leaf_communities": sum(
                1 for c in self._hierarchy_cache.values() if c.is_leaf
            )
        }


# Singleton instance
_community_selector: Optional[DynamicCommunitySelector] = None


def get_community_selector(
    neo4j_client=None,
    embedder=None,
    **kwargs
) -> DynamicCommunitySelector:
    """Get or create community selector singleton."""
    global _community_selector

    if _community_selector is None:
        if neo4j_client is None:
            from ..graph_builder import Neo4jClient
            neo4j_client = Neo4jClient()

        if embedder is None:
            from ..models.embedding_manager import get_embedding_manager
            embedder = get_embedding_manager()

        _community_selector = DynamicCommunitySelector(
            neo4j_client=neo4j_client,
            embedder=embedder,
            **kwargs
        )

    return _community_selector
