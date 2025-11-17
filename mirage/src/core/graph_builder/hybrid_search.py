"""
Hybrid Search Module for GraphRAG

Combines Global Search (community summaries) and Local Search (entity traversal)
with intelligent routing to select the best search mode for each query.

This is the COMPLETE GraphRAG implementation:
- Global Search: Holistic questions, themes, summaries
- Local Search: Specific facts, entity details, relationships
- Router: Automatically selects best mode

Usage:
    hybrid = HybridSearchEngine(neo4j_client, graph_traversal, llm_endpoint)
    result = hybrid.search("What are the main themes?")  # Auto-routes to Global
    result = hybrid.search("Who works at IBM?")  # Auto-routes to Local
"""

import re
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""
    query: str
    answer: str
    search_mode: str  # "global", "local", or "hybrid"
    confidence: float
    metadata: dict

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "search_mode": self.search_mode,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class QueryRouter:
    """
    Routes queries to appropriate search mode (global vs local).

    Uses rule-based heuristics to classify queries.

    Rules:
    - Global: "what themes", "summarize", "overview", "main topics"
    - Local: "who", "where", "when", specific entity names
    - Hybrid: Complex questions that need both
    """

    def __init__(self):
        # Keywords indicating global search
        self.global_keywords = [
            # English
            'theme', 'themes', 'topic', 'topics', 'overview', 'summary', 'summarize',
            'main', 'primary', 'key', 'important', 'significant', 'overall',
            'general', 'broad', 'entire', 'whole', 'all', 'everything',
            # Arabic
            'موضوع', 'موضوعات', 'ملخص', 'نظرة عامة', 'رئيسي', 'رئيسية',
            'أساسي', 'أساسية', 'مهم', 'مهمة', 'عام', 'عامة', 'كل', 'جميع'
        ]

        # Keywords indicating local search
        self.local_keywords = [
            # English
            'who', 'where', 'when', 'which', 'specific', 'detail', 'exactly',
            'person', 'people', 'organization', 'company', 'location', 'place',
            # Arabic
            'من', 'أين', 'متى', 'محدد', 'تحديدا', 'شخص', 'أشخاص',
            'منظمة', 'شركة', 'مكان', 'موقع'
        ]

    def route(self, query: str, entity_names: list = None) -> str:
        """
        Route query to appropriate search mode.

        Args:
            query: User question
            entity_names: Optional list of entities found in query

        Returns:
            "global", "local", or "hybrid"
        """
        query_lower = query.lower()

        # Count keyword matches
        global_score = sum(1 for kw in self.global_keywords if kw in query_lower)
        local_score = sum(1 for kw in self.local_keywords if kw in query_lower)

        # If specific entities mentioned, boost local score
        if entity_names and len(entity_names) > 0:
            local_score += len(entity_names)

        # Decision logic
        if global_score > local_score:
            return "global"
        elif local_score > global_score:
            return "local"
        else:
            # Tie or both zero - use hybrid
            return "hybrid"


class HybridSearchEngine:
    """
    Hybrid search engine combining global and local search.

    Features:
    - Automatic query routing (global vs local vs hybrid)
    - Fallback mechanisms (if one mode fails, try other)
    - Result merging (for hybrid mode)

    Usage:
        engine = HybridSearchEngine(neo4j_client, graph_traversal, llm_endpoint)

        # Auto-routing
        result = engine.search("What are the main themes?")  # Routes to global
        result = engine.search("Who works at IBM?")  # Routes to local

        # Manual mode selection
        result = engine.search("...", mode="global")
        result = engine.search("...", mode="local")
    """

    def __init__(
        self,
        neo4j_client,
        graph_traversal,
        llm_endpoint: str = "http://tgi:8765",
        auto_route: bool = True
    ):
        """
        Initialize hybrid search engine.

        Args:
            neo4j_client: Neo4j client instance
            graph_traversal: GraphTraversal instance
            llm_endpoint: TGI endpoint for Allam
            auto_route: If True, automatically route queries
        """
        self.neo4j_client = neo4j_client
        self.graph_traversal = graph_traversal
        self.llm_endpoint = llm_endpoint
        self.auto_route = auto_route

        # Initialize sub-engines (lazy loading to avoid circular imports)
        self._global_engine = None
        self._local_engine = None
        self.router = QueryRouter()

    @property
    def global_engine(self):
        """Lazy load global search engine."""
        if self._global_engine is None:
            from .global_search import GlobalSearchEngine
            self._global_engine = GlobalSearchEngine(
                self.neo4j_client,
                self.llm_endpoint
            )
        return self._global_engine

    @property
    def local_engine(self):
        """Lazy load local search engine."""
        if self._local_engine is None:
            from .local_search import LocalSearchEngine
            self._local_engine = LocalSearchEngine(
                self.neo4j_client,
                self.graph_traversal,
                self.llm_endpoint
            )
        return self._local_engine

    def search(
        self,
        query: str,
        mode: Optional[str] = None
    ) -> HybridSearchResult:
        """
        Execute hybrid search with automatic or manual routing.

        Args:
            query: User question
            mode: Optional manual mode selection ("global", "local", "hybrid")
                  If None, uses auto-routing

        Returns:
            HybridSearchResult with answer and metadata

        Example:
            >>> engine = HybridSearchEngine(neo4j_client, graph_traversal)
            >>> # Auto-routing
            >>> result = engine.search("What are the main themes?")
            >>> # Manual mode
            >>> result = engine.search("Who works at IBM?", mode="local")
        """
        logger.info(f"Hybrid search: '{query}' (mode: {mode or 'auto'})")

        # Determine search mode
        if mode is None and self.auto_route:
            mode = self.router.route(query)
            logger.info(f"Auto-routed to: {mode}")

        # Execute search based on mode
        if mode == "global":
            return self._global_search(query)
        elif mode == "local":
            return self._local_search(query)
        elif mode == "hybrid":
            return self._hybrid_search(query)
        else:
            # Default to hybrid if unknown mode
            logger.warning(f"Unknown mode '{mode}', using hybrid")
            return self._hybrid_search(query)

    def _global_search(self, query: str) -> HybridSearchResult:
        """Execute global search only."""
        try:
            result = self.global_engine.search(query, level=1)

            return HybridSearchResult(
                query=query,
                answer=result.answer,
                search_mode="global",
                confidence=result.confidence,
                metadata={
                    "communities_searched": result.communities_searched,
                    "themes": result.themes
                }
            )

        except Exception as e:
            logger.error(f"Global search failed: {e}")
            # Fallback to local search
            logger.info("Falling back to local search...")
            return self._local_search(query)

    def _local_search(self, query: str) -> HybridSearchResult:
        """Execute local search only."""
        try:
            result = self.local_engine.search(query)

            return HybridSearchResult(
                query=query,
                answer=result.answer,
                search_mode="local",
                confidence=result.confidence,
                metadata={
                    "seed_entities": result.seed_entities,
                    "discovered_entities": result.discovered_entities,
                    "relationships": result.relationships
                }
            )

        except Exception as e:
            logger.error(f"Local search failed: {e}")
            # Fallback to global search
            logger.info("Falling back to global search...")
            return self._global_search(query)

    def _hybrid_search(self, query: str) -> HybridSearchResult:
        """
        Execute both global and local search, merge results.

        Strategy:
        1. Run both searches in parallel (or sequential for simplicity)
        2. Merge answers
        3. Return combined result with higher confidence
        """
        logger.info("Executing hybrid search (global + local)")

        try:
            # Execute both searches
            global_result = self.global_engine.search(query, level=1)
            local_result = self.local_engine.search(query)

            # Merge answers
            merged_answer = self._merge_answers(
                global_answer=global_result.answer,
                local_answer=local_result.answer,
                query=query
            )

            # Calculate combined confidence (average)
            combined_confidence = (global_result.confidence + local_result.confidence) / 2.0

            return HybridSearchResult(
                query=query,
                answer=merged_answer,
                search_mode="hybrid",
                confidence=combined_confidence,
                metadata={
                    "global": {
                        "communities_searched": global_result.communities_searched,
                        "themes": global_result.themes
                    },
                    "local": {
                        "seed_entities": local_result.seed_entities,
                        "discovered_entities": local_result.discovered_entities,
                        "relationships": local_result.relationships
                    }
                }
            )

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Try fallback to global only
            try:
                return self._global_search(query)
            except:
                # Last resort: local only
                return self._local_search(query)

    def _merge_answers(
        self,
        global_answer: str,
        local_answer: str,
        query: str
    ) -> str:
        """
        Merge global and local search answers.

        Simple strategy: Combine both answers with labels.
        Advanced strategy: Could use LLM to merge intelligently.

        Args:
            global_answer: Answer from global search
            local_answer: Answer from local search
            query: Original question

        Returns:
            Merged answer
        """
        # Simple merge: labeled sections
        merged = f"""**نظرة عامة (Global):**
{global_answer}

**تفاصيل محددة (Local):**
{local_answer}
"""

        # TODO: Future enhancement - use LLM to merge intelligently
        return merged

    def get_statistics(self) -> dict:
        """
        Get statistics about search capabilities.

        Returns:
            Dict with statistics from both engines
        """
        stats = {}

        # Global search statistics
        try:
            global_stats = self.global_engine.get_search_statistics()
            stats['global'] = global_stats
        except Exception as e:
            logger.error(f"Error getting global stats: {e}")
            stats['global'] = {}

        # Local search statistics (entity count)
        try:
            entity_query = "MATCH (e:Entity) RETURN COUNT(e) as count"
            result = self.neo4j_client.execute_query(entity_query)
            entity_count = result[0].get('count', 0) if result else 0

            rel_query = "MATCH ()-[r]->() RETURN COUNT(r) as count"
            result = self.neo4j_client.execute_query(rel_query)
            rel_count = result[0].get('count', 0) if result else 0

            stats['local'] = {
                'total_entities': entity_count,
                'total_relationships': rel_count
            }
        except Exception as e:
            logger.error(f"Error getting local stats: {e}")
            stats['local'] = {}

        return stats
