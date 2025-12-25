"""
MIRAGE V2 Unified Retrieval Engine

Orchestrates all retrieval modes with automatic routing.
"""

from typing import List, Dict, Any, Optional
import time
from loguru import logger

from ..base_retriever import RetrievalMode, RetrievalResult, RetrievalResponse
from ..query_router import QueryRouter, get_query_router
from ..fusion import ResultFusion, FusionConfig

from ...evaluation import get_metrics_tracker
from ...graph_builder import get_entity_disambiguator

from .config import RetrievalEngineConfig
from .vector_mode import VectorModeMixin
from .local_mode import LocalModeMixin
from .global_mode import GlobalModeMixin
from .hybrid_mode import HybridModeMixin


class RetrievalEngine(
    VectorModeMixin,
    LocalModeMixin,
    GlobalModeMixin,
    HybridModeMixin
):
    """
    Unified retrieval engine supporting 8 modes.

    Modes:
    1. VECTOR - Vector similarity search (baseline RAG)
    2. LOCAL - Entity-focused retrieval (GraphRAG local search)
    3. GLOBAL - Relationship-focused retrieval
    4. GLOBAL_SEARCH - Map-reduce over community summaries (GraphRAG global search)
    5. HYBRID - Combines local + global
    6. MIX - All modes with RRF fusion
    7. SEMANTIC - Deep semantic matching with cross-encoder
    8. BYPASS - No retrieval (returns empty)
    9. DRIFT - GraphRAG DRIFT search (dynamic global+local)

    Features:
    - Automatic query routing
    - Multiple fusion strategies
    - Fallback on errors
    - Detailed retrieval explanations
    - Global search for holistic queries (GraphRAG map-reduce)
    """

    def __init__(
        self,
        config: Optional[RetrievalEngineConfig] = None,
        router: Optional[QueryRouter] = None,
        embedder=None,
        index_manager=None,
        graph_client=None
    ):
        """
        Initialize retrieval engine.

        Args:
            config: Engine configuration
            router: Query router for auto-routing
            embedder: Embedding manager for queries
            index_manager: Vector index manager
            graph_client: Neo4j client for graph operations
        """
        self.config = config or RetrievalEngineConfig()
        self.router = router or get_query_router(default_mode=self.config.default_mode)
        self.fusion = ResultFusion(FusionConfig())

        # Components (lazy-loaded)
        self._embedder = embedder
        self._index_manager = index_manager
        self._graph_client = graph_client

        # Metrics tracker
        self._metrics_tracker = None
        if self.config.track_metrics:
            self._metrics_tracker = get_metrics_tracker()
            logger.info("Metrics tracking enabled")

        # Entity disambiguator - lazy initialized
        self._entity_disambiguator = None

        # Entity chunks cache (LRU)
        self._entity_chunks_cache: Dict[str, List[Dict]] = {}
        self._cache_max_size = 500
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info(
            f"RetrievalEngine initialized: default_mode={self.config.default_mode.value}, "
            f"auto_route={self.config.auto_route}, track_metrics={self._metrics_tracker is not None}"
        )

    @property
    def embedder(self):
        """Lazy load embedder"""
        if self._embedder is None:
            from ...models.embedding_manager import get_embedding_manager
            self._embedder = get_embedding_manager()
        return self._embedder

    @property
    def index_manager(self):
        """Lazy load index manager"""
        if self._index_manager is None:
            from ...indexing import get_index_manager
            self._index_manager = get_index_manager()
        return self._index_manager

    @property
    def graph_client(self):
        """Lazy load graph client"""
        if self._graph_client is None:
            from ...graph_builder import Neo4jClient
            self._graph_client = Neo4jClient()
        return self._graph_client

    @property
    def entity_disambiguator(self):
        """Lazy load entity disambiguator (MIRAGE V4)"""
        if self._entity_disambiguator is None:
            if self.graph_client:
                self._entity_disambiguator = get_entity_disambiguator(self.graph_client)
                logger.info("Entity disambiguator initialized")
        return self._entity_disambiguator

    def _get_cached_entity_chunks(self, entity_name: str, limit: int = 3) -> List[Dict]:
        """Get entity chunks with LRU caching."""
        cache_key = f"{entity_name}:{limit}"

        if cache_key in self._entity_chunks_cache:
            self._cache_hits += 1
            return self._entity_chunks_cache[cache_key]

        self._cache_misses += 1

        chunks = []
        if self.graph_client:
            try:
                chunks = self.graph_client.get_entity_chunks(entity_name, limit=limit)
            except Exception as e:
                logger.debug(f"Entity chunks fetch failed for {entity_name}: {e}")

        if len(self._entity_chunks_cache) >= self._cache_max_size:
            first_key = next(iter(self._entity_chunks_cache))
            del self._entity_chunks_cache[first_key]

        self._entity_chunks_cache[cache_key] = chunks
        return chunks

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get entity cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._entity_chunks_cache),
            "max_size": self._cache_max_size
        }

    def retrieve(
        self,
        query: str,
        mode: Optional[RetrievalMode] = None,
        top_k: Optional[int] = None,
        auto_route: Optional[bool] = None,
        **kwargs
    ) -> RetrievalResponse:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User query string
            mode: Specific mode to use (overrides auto-routing)
            top_k: Number of results to return
            auto_route: Override config auto_route setting
            **kwargs: Additional parameters

        Returns:
            RetrievalResponse with results
        """
        start_time = time.time()
        top_k = min(top_k or self.config.default_top_k, self.config.max_top_k)

        # Determine mode
        if mode is None:
            should_route = auto_route if auto_route is not None else self.config.auto_route
            if should_route:
                routing = self.router.route(query)
                mode = routing.recommended_mode
                logger.info(f"Query routed to {mode.value}: {routing.reasoning}")
            else:
                mode = self.config.default_mode

        # Handle BYPASS mode
        if mode == RetrievalMode.BYPASS:
            return RetrievalResponse(
                results=[],
                query=query,
                mode=mode,
                metadata={"bypassed": True}
            )

        # Handle MIX mode
        if mode == RetrievalMode.MIX:
            return self._retrieve_mix(query, top_k, **kwargs)

        # Regular retrieval
        try:
            response = self._retrieve_with_mode(query, mode, top_k, **kwargs)
            response.retrieval_time_ms = (time.time() - start_time) * 1000
            return response
        except Exception as e:
            logger.error(f"Retrieval error with {mode.value}: {e}")
            if self.config.fallback_on_error and mode != self.config.fallback_mode:
                logger.info(f"Falling back to {self.config.fallback_mode.value}")
                return self._retrieve_with_mode(
                    query, self.config.fallback_mode, top_k, **kwargs
                )
            raise

    def _retrieve_with_mode(
        self,
        query: str,
        mode: RetrievalMode,
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """Execute retrieval with specific mode"""
        query_embedding = None
        if self.embedder:
            query_embedding = self.embedder.embed(query)

        if mode == RetrievalMode.VECTOR or mode == RetrievalMode.NAIVE:
            return self._vector_retrieve(query, query_embedding, top_k, **kwargs)
        elif mode == RetrievalMode.LOCAL:
            return self._local_retrieve(query, query_embedding, top_k, **kwargs)
        elif mode == RetrievalMode.GLOBAL:
            return self._global_retrieve(query, query_embedding, top_k, **kwargs)
        elif mode == RetrievalMode.GLOBAL_SEARCH:
            return self._global_search_retrieve(query, top_k, **kwargs)
        elif mode == RetrievalMode.HYBRID:
            return self._hybrid_retrieve(query, query_embedding, top_k, **kwargs)
        elif mode == RetrievalMode.SEMANTIC:
            return self._semantic_retrieve(query, query_embedding, top_k, **kwargs)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

    def get_metrics_stats(self) -> Dict[str, Any]:
        """Get aggregated metrics statistics."""
        if self._metrics_tracker:
            return self._metrics_tracker.get_stats()
        return {"tracking_enabled": False}

    def explain_retrieval(
        self,
        query: str,
        mode: Optional[RetrievalMode] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Get detailed explanation of retrieval process."""
        routing = self.router.route(query)
        response = self.retrieve(query, mode=mode, top_k=top_k)

        return {
            "query": query,
            "routing": {
                "query_type": routing.query_type.value,
                "recommended_mode": routing.recommended_mode.value,
                "confidence": routing.confidence,
                "reasoning": routing.reasoning,
                "alternatives": [
                    {"mode": m.value, "score": s}
                    for m, s in routing.alternative_modes
                ]
            },
            "actual_mode": response.mode.value if response.mode else "unknown",
            "results": {
                "count": len(response.results),
                "total_candidates": response.total_candidates,
                "time_ms": response.retrieval_time_ms,
                "top_results": [
                    {
                        "chunk_id": r.chunk_id,
                        "score": r.score,
                        "mode": r.retrieval_mode,
                        "via_entity": r.via_entity,
                        "via_relationship": r.via_relationship,
                        "hop_distance": r.hop_distance
                    }
                    for r in response.results[:5]
                ]
            },
            "metadata": response.metadata
        }


# Global instance
_retrieval_engine: Optional[RetrievalEngine] = None


def get_retrieval_engine(**kwargs) -> RetrievalEngine:
    """Get or create global RetrievalEngine"""
    global _retrieval_engine

    if _retrieval_engine is None:
        _retrieval_engine = RetrievalEngine(**kwargs)

    return _retrieval_engine
