"""
MIRAGE V2 Unified Retrieval Engine
Orchestrates all retrieval modes with automatic routing.
"""

from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, field
from enum import Enum
import time
from loguru import logger
import numpy as np

from .base_retriever import (
    BaseRetriever,
    RetrievalMode,
    RetrievalResult,
    RetrievalResponse
)
from .query_router import QueryRouter, RoutingDecision, get_query_router
from .fusion import ResultFusion, FusionConfig


@dataclass
class RetrievalEngineConfig:
    """Configuration for the retrieval engine"""
    # Default mode when no routing
    default_mode: RetrievalMode = RetrievalMode.HYBRID

    # Top-k settings
    default_top_k: int = 10
    max_top_k: int = 50

    # Score thresholds
    min_score: float = 0.0

    # Fusion settings
    fusion_method: str = "rrf"  # "rrf", "weighted", "interleave"
    mode_weights: Dict[str, float] = field(default_factory=lambda: {
        "naive": 0.6,
        "local": 0.8,
        "global": 0.9,
        "hybrid": 1.0,
        "semantic": 0.85
    })

    # Auto-routing
    auto_route: bool = True

    # Fallback behavior
    fallback_on_error: bool = True
    fallback_mode: RetrievalMode = RetrievalMode.NAIVE


class RetrievalEngine:
    """
    Unified retrieval engine supporting 7 modes.

    Modes:
    1. NAIVE - Simple vector search
    2. LOCAL - Entity-focused retrieval
    3. GLOBAL - Relationship-focused retrieval
    4. HYBRID - Combines local + global
    5. MIX - All modes with RRF fusion
    6. SEMANTIC - Deep semantic matching
    7. BYPASS - No retrieval (returns empty)

    Features:
    - Automatic query routing
    - Multiple fusion strategies
    - Fallback on errors
    - Detailed retrieval explanations
    """

    def __init__(
        self,
        config: Optional[RetrievalEngineConfig] = None,
        router: Optional[QueryRouter] = None,
        embedder = None,
        index_manager = None,
        graph_client = None
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

        # Retriever cache
        self._retrievers: Dict[RetrievalMode, BaseRetriever] = {}

        logger.info(
            f"RetrievalEngine initialized: default_mode={self.config.default_mode.value}, "
            f"auto_route={self.config.auto_route}"
        )

    @property
    def embedder(self):
        """Lazy load embedder"""
        if self._embedder is None:
            try:
                from ..models.embedding_manager import get_embedding_manager
                self._embedder = get_embedding_manager()
            except ImportError:
                logger.warning("Could not load embedding manager")
        return self._embedder

    @property
    def index_manager(self):
        """Lazy load index manager"""
        if self._index_manager is None:
            try:
                from ..indexing import get_index_manager
                self._index_manager = get_index_manager()
            except ImportError:
                logger.warning("Could not load index manager")
        return self._index_manager

    @property
    def graph_client(self):
        """Lazy load graph client"""
        if self._graph_client is None:
            try:
                from ..graph_builder import Neo4jClient
                self._graph_client = Neo4jClient()
            except ImportError:
                logger.warning("Could not load graph client")
        return self._graph_client

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
                logger.info(
                    f"Query routed to {mode.value}: {routing.reasoning}"
                )
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

        # Handle MIX mode (run all and fuse)
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

        # Get query embedding
        query_embedding = None
        if self.embedder:
            query_embedding = self.embedder.embed(query)

        if mode == RetrievalMode.NAIVE:
            return self._naive_retrieve(query, query_embedding, top_k, **kwargs)
        elif mode == RetrievalMode.LOCAL:
            return self._local_retrieve(query, query_embedding, top_k, **kwargs)
        elif mode == RetrievalMode.GLOBAL:
            return self._global_retrieve(query, query_embedding, top_k, **kwargs)
        elif mode == RetrievalMode.HYBRID:
            return self._hybrid_retrieve(query, query_embedding, top_k, **kwargs)
        elif mode == RetrievalMode.SEMANTIC:
            return self._semantic_retrieve(query, query_embedding, top_k, **kwargs)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

    def _naive_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """Simple vector search"""
        if query_embedding is None:
            return RetrievalResponse(
                results=[],
                query=query,
                mode=RetrievalMode.NAIVE,
                metadata={"error": "No embedding available"}
            )

        if self.index_manager is None:
            return RetrievalResponse(
                results=[],
                query=query,
                mode=RetrievalMode.NAIVE,
                metadata={"error": "Index manager not available"}
            )

        # Search chunks
        search_results = self.index_manager.search_chunks(
            query_embedding,
            top_k=top_k,
            score_threshold=self.config.min_score
        )

        # Convert to RetrievalResult
        results = [
            RetrievalResult(
                chunk_id=r.id,
                document_id=r.payload.get("document_id", ""),
                text=r.payload.get("text", ""),
                score=r.score,
                retrieval_mode="naive",
                metadata=r.payload
            )
            for r in search_results
        ]

        return RetrievalResponse(
            results=results,
            query=query,
            mode=RetrievalMode.NAIVE,
            total_candidates=len(search_results)
        )

    def _local_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """
        Entity-focused retrieval: query → chunks → entities → enriched context

        Improved L2 strategy:
        1. Get relevant chunks via vector search
        2. Extract entities from those chunks
        3. Filter entities by type based on query patterns
        4. Include entity names in retrieval metadata for better answers
        """
        if query_embedding is None or self.index_manager is None:
            return self._naive_retrieve(query, query_embedding, top_k, **kwargs)

        # 1. First get naive results as base
        naive_response = self._naive_retrieve(query, query_embedding, top_k * 2, **kwargs)

        # 2. Detect entity types from query patterns
        entity_types = self._detect_entity_types(query)

        # 3. Extract entities from retrieved chunks
        extracted_entities = []
        entity_chunks = []
        if self.graph_client and naive_response.results:
            try:
                # Get chunk IDs from naive results
                chunk_ids = [r.chunk_id for r in naive_response.results if r.chunk_id]

                # Get entities mentioned in these chunks
                if chunk_ids:
                    entities = self.graph_client.get_entities_from_chunks(
                        chunk_ids=chunk_ids,
                        entity_types=entity_types if entity_types else None,
                        limit=20
                    )
                    extracted_entities = entities

                    # For each entity, get its associated chunks
                    for entity in entities[:10]:
                        chunks = self.graph_client.get_entity_chunks(
                            entity.get("name", ""), limit=2
                        )
                        for chunk in chunks:
                            entity_chunks.append({
                                "chunk_id": chunk.get("chunk_id", ""),
                                "document_id": chunk.get("document_id", ""),
                                "text": chunk.get("text", ""),
                                "entity": entity.get("name", ""),
                                "entity_type": entity.get("type", "")
                            })

                # Also try direct term matching for backup
                query_terms = [t for t in query.split() if len(t) > 2]
                for term in query_terms[:3]:
                    entities = self.graph_client.search_entities_by_name(term, limit=3)
                    for entity in entities:
                        if entity not in extracted_entities:
                            extracted_entities.append(entity)
                        chunks = self.graph_client.get_entity_chunks(
                            entity.get("name", ""), limit=2
                        )
                        for chunk in chunks:
                            entity_chunks.append({
                                "chunk_id": chunk.get("chunk_id", ""),
                                "document_id": chunk.get("document_id", ""),
                                "text": chunk.get("text", ""),
                                "entity": entity.get("name", ""),
                                "entity_type": entity.get("type", "")
                            })

            except Exception as e:
                logger.warning(f"Neo4j entity search failed: {e}")

        # 4. Combine with naive results, prioritizing entity-connected chunks
        results = []
        seen_chunks = set()

        # First add entity-connected chunks
        for ec in entity_chunks[:top_k // 2]:
            if ec["chunk_id"] and ec["chunk_id"] not in seen_chunks:
                seen_chunks.add(ec["chunk_id"])
                results.append(RetrievalResult(
                    chunk_id=ec["chunk_id"],
                    document_id=ec["document_id"],
                    text=ec["text"],
                    score=0.85,  # High score for entity match
                    retrieval_mode="local",
                    via_entity=ec["entity"],
                    hop_distance=1
                ))

        # Then add naive results
        for r in naive_response.results:
            if r.chunk_id not in seen_chunks and len(results) < top_k:
                seen_chunks.add(r.chunk_id)
                r.retrieval_mode = "local"
                results.append(r)

        # Extract entity names for metadata
        entity_names = list({e.get("name", "") for e in extracted_entities if e.get("name")})

        return RetrievalResponse(
            results=results[:top_k],
            query=query,
            mode=RetrievalMode.LOCAL,
            total_candidates=len(entity_chunks) + len(naive_response.results),
            metadata={
                "entities_found": len(extracted_entities),
                "entity_names": entity_names[:20],  # Top 20 entity names
                "entity_types_filtered": entity_types or [],
            }
        )

    def _detect_entity_types(self, query: str) -> List[str]:
        """
        Detect which entity types to filter based on query patterns

        Args:
            query: User query string

        Returns:
            List of entity types to filter for (empty = no filter)
        """
        query_lower = query.lower()

        # Check for combined patterns first (who are the nominees/partners = typically orgs)
        if "من هم" in query_lower or "من هي" in query_lower:
            # If asking about nominees, partners, organizations - return both types
            if any(p in query_lower for p in ["مرشح", "شريك", "جهة", "هيئة", "وزارة", "جائزة"]):
                return ["Organization", "Person"]

        # Arabic patterns for organization queries (check first as more specific)
        org_patterns = [
            "الجهات", "المؤسسات", "الشركات",  # Organizations, institutions, companies
            "الوزارات", "الهيئات", "المرشحون",  # Ministries, authorities, nominees
            "شريك", "الشركاء", "جائزة",  # Partner(s), award
        ]

        # Check for organization queries
        if any(p in query_lower for p in org_patterns):
            return ["Organization"]

        # Arabic patterns for person queries
        person_patterns = [
            "من هو", "من هي",  # Who is (singular)
            "الأشخاص", "الشخص", "المسؤول",  # People, person, responsible
            "المدير", "الرئيس", "الوزير",  # Director, president, minister
        ]

        # Check for person queries
        if any(p in query_lower for p in person_patterns):
            return ["Person"]

        # Generic "who are" without specific context - include both
        if "من هم" in query_lower:
            return ["Organization", "Person"]

        return []  # No filter

    def _global_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """Relationship-focused retrieval: query → relationships → entities → chunks (via Neo4j)"""
        if query_embedding is None or self.index_manager is None:
            return self._naive_retrieve(query, query_embedding, top_k, **kwargs)

        # 1. Get naive results as base
        naive_response = self._naive_retrieve(query, query_embedding, top_k * 2, **kwargs)

        # 2. Try to find relationships in Neo4j related to the query
        relationship_chunks = []
        if self.graph_client:
            try:
                # Extract key terms from query for relationship matching
                query_terms = [t for t in query.split() if len(t) > 2]

                # Search for entities and get their relationships
                for term in query_terms[:3]:
                    entities = self.graph_client.search_entities_by_name(term, limit=3)
                    for entity in entities:
                        entity_name = entity.get("name", "")
                        # Get relationships where this entity is source or target
                        relationships = self.graph_client.get_entity_relationships(
                            entity_name, limit=5
                        )
                        for rel in relationships:
                            # Get chunks that mention both entities in the relationship
                            source_chunks = self.graph_client.get_entity_chunks(
                                rel.get("source", ""), limit=2
                            )
                            target_chunks = self.graph_client.get_entity_chunks(
                                rel.get("target", ""), limit=2
                            )
                            # Prefer chunks that mention both entities
                            source_ids = {c.get("chunk_id") for c in source_chunks}
                            for chunk in target_chunks:
                                if chunk.get("chunk_id") in source_ids:
                                    relationship_chunks.append({
                                        "chunk_id": chunk.get("chunk_id", ""),
                                        "document_id": chunk.get("document_id", ""),
                                        "text": chunk.get("text", ""),
                                        "relationship": rel.get("type", "RELATED_TO"),
                                        "source": rel.get("source", ""),
                                        "target": rel.get("target", "")
                                    })
                            # Also add individual chunks
                            for chunk in source_chunks + target_chunks:
                                relationship_chunks.append({
                                    "chunk_id": chunk.get("chunk_id", ""),
                                    "document_id": chunk.get("document_id", ""),
                                    "text": chunk.get("text", ""),
                                    "relationship": rel.get("type", "RELATED_TO"),
                                    "source": rel.get("source", ""),
                                    "target": rel.get("target", "")
                                })
            except Exception as e:
                logger.warning(f"Neo4j relationship search failed: {e}")

        # 3. Combine with naive results, prioritizing relationship-connected chunks
        results = []
        seen_chunks = set()

        # First add relationship-connected chunks
        for rc in relationship_chunks[:top_k // 2]:
            if rc["chunk_id"] and rc["chunk_id"] not in seen_chunks:
                seen_chunks.add(rc["chunk_id"])
                results.append(RetrievalResult(
                    chunk_id=rc["chunk_id"],
                    document_id=rc["document_id"],
                    text=rc["text"],
                    score=0.9,  # High score for relationship match
                    retrieval_mode="global",
                    via_relationship=rc["relationship"],
                    hop_distance=2
                ))

        # Then add naive results
        for r in naive_response.results:
            if r.chunk_id not in seen_chunks and len(results) < top_k:
                seen_chunks.add(r.chunk_id)
                r.retrieval_mode = "global"
                results.append(r)

        return RetrievalResponse(
            results=results[:top_k],
            query=query,
            mode=RetrievalMode.GLOBAL,
            total_candidates=len(relationship_chunks) + len(naive_response.results),
            metadata={"relationships_found": len(relationship_chunks)}
        )

    def _hybrid_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """Combines naive + local + global with RRF"""
        # Get results from each mode
        naive_response = self._naive_retrieve(query, query_embedding, top_k, **kwargs)
        local_response = self._local_retrieve(query, query_embedding, top_k, **kwargs)
        global_response = self._global_retrieve(query, query_embedding, top_k, **kwargs)

        # Fuse with weights
        weights = [
            self.config.mode_weights.get("naive", 0.6),
            self.config.mode_weights.get("local", 0.8),
            self.config.mode_weights.get("global", 0.9)
        ]

        fused = self.fusion.fuse_responses(
            [naive_response, local_response, global_response],
            method=self.config.fusion_method,
            weights=weights
        )

        # Update mode
        fused.mode = RetrievalMode.HYBRID
        fused.query = query
        fused.results = fused.results[:top_k]

        return fused

    def _semantic_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """Deep semantic matching with re-ranking"""
        # Start with naive retrieval but with more candidates
        naive_response = self._naive_retrieve(
            query, query_embedding, top_k * 3, **kwargs
        )

        # Would apply cross-encoder re-ranking here
        # For now, just return naive results
        naive_response.mode = RetrievalMode.SEMANTIC
        naive_response.results = naive_response.results[:top_k]

        return naive_response

    def _retrieve_mix(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """Run all modes and fuse with RRF"""
        # Get query embedding once
        query_embedding = None
        if self.embedder:
            query_embedding = self.embedder.embed(query)

        # Run all modes
        responses = []
        weights = []

        for mode in [RetrievalMode.NAIVE, RetrievalMode.LOCAL,
                     RetrievalMode.GLOBAL, RetrievalMode.SEMANTIC]:
            try:
                response = self._retrieve_with_mode(
                    query, mode, top_k, **kwargs
                )
                responses.append(response)
                weights.append(self.config.mode_weights.get(mode.value, 1.0))
            except Exception as e:
                logger.warning(f"Mode {mode.value} failed in mix: {e}")

        if not responses:
            return RetrievalResponse(
                results=[],
                query=query,
                mode=RetrievalMode.MIX,
                metadata={"error": "All modes failed"}
            )

        # Fuse results
        fused = self.fusion.fuse_responses(
            responses,
            method="rrf",
            weights=weights
        )

        fused.mode = RetrievalMode.MIX
        fused.query = query
        fused.results = fused.results[:top_k]

        return fused

    def explain_retrieval(
        self,
        query: str,
        mode: Optional[RetrievalMode] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Get detailed explanation of retrieval process.

        Returns dict with routing decision, mode used, and result breakdown.
        """
        # Get routing decision
        routing = self.router.route(query)

        # Get results
        response = self.retrieve(query, mode=mode, top_k=top_k)

        # Build explanation
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


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_retrieval_engine: Optional[RetrievalEngine] = None


def get_retrieval_engine(**kwargs) -> RetrievalEngine:
    """Get or create global RetrievalEngine"""
    global _retrieval_engine

    if _retrieval_engine is None:
        _retrieval_engine = RetrievalEngine(**kwargs)

    return _retrieval_engine
