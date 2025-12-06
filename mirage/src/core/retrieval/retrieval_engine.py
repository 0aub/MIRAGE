"""
MIRAGE V2 Unified Retrieval Engine
Orchestrates all retrieval modes with automatic routing.
"""

from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, field
from enum import Enum
import time
import re
from loguru import logger
import numpy as np


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for consistent matching.

    Handles common Arabic text variations:
    - ة ↔ ه (ta marbuta / ha) - critical for entity names
    - أ/إ/آ → ا (alef variants)
    - ى → ي (alef maqsura / ya)
    - Remove diacritics (tashkeel)

    Args:
        text: Arabic text to normalize

    Returns:
        Normalized text for matching
    """
    if not text:
        return text

    # Remove diacritics (tashkeel)
    diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    text = diacritics.sub('', text)

    # Normalize ta marbuta ة → ه (both directions for matching)
    # We convert to ه as it's more common in informal Arabic text
    text = text.replace('ة', 'ه')

    # Normalize alef variants → ا
    text = text.replace('أ', 'ا')
    text = text.replace('إ', 'ا')
    text = text.replace('آ', 'ا')

    # Normalize alef maqsura → ي
    text = text.replace('ى', 'ي')

    return text


def get_arabic_variants(text: str) -> List[str]:
    """
    Generate common Arabic text variants for a phrase.

    Returns both the original and normalized forms,
    plus common variants for better matching.

    Args:
        text: Original Arabic text

    Returns:
        List of text variants to search for
    """
    variants = [text]

    # Add normalized form
    normalized = normalize_arabic(text)
    if normalized != text:
        variants.append(normalized)

    # If text has ه, also try ة (reverse normalization)
    if 'ه' in text:
        reverse = text.replace('ه', 'ة')
        if reverse not in variants:
            variants.append(reverse)

    # If text has ة, also try ه
    if 'ة' in text:
        reverse = text.replace('ة', 'ه')
        if reverse not in variants:
            variants.append(reverse)

    return list(set(variants))

from .base_retriever import (
    BaseRetriever,
    RetrievalMode,
    RetrievalResult,
    RetrievalResponse
)
from .query_router import QueryRouter, RoutingDecision, get_query_router
from .fusion import ResultFusion, FusionConfig

# Required imports - no fallback (enforced in Docker)
from ..evaluation import get_metrics_tracker, MetricsTracker
from ..graph_builder import EntityDisambiguator, get_entity_disambiguator


@dataclass
class RetrievalEngineConfig:
    """Configuration for the retrieval engine"""
    # Default mode when no routing
    default_mode: RetrievalMode = RetrievalMode.LOCAL  # Phase 4: LOCAL is best performer (92.9%)

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

    # Metrics tracking (Phase 3)
    track_metrics: bool = True


class RetrievalEngine:
    """
    Unified retrieval engine supporting 8 modes.

    Modes:
    1. NAIVE - Simple vector search
    2. LOCAL - Entity-focused retrieval
    3. GLOBAL - Relationship-focused retrieval
    4. GLOBAL_SEARCH - Map-reduce over community summaries (TRUE GraphRAG)
    5. HYBRID - Combines local + global
    6. MIX - All modes with RRF fusion
    7. SEMANTIC - Deep semantic matching
    8. BYPASS - No retrieval (returns empty)

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

        # Metrics tracker (Phase 3) - required component
        self._metrics_tracker = None
        if self.config.track_metrics:
            self._metrics_tracker = get_metrics_tracker()
            logger.info("Metrics tracking enabled")

        # Entity disambiguator (MIRAGE V4) - lazy initialized
        self._entity_disambiguator = None

        logger.info(
            f"RetrievalEngine initialized: default_mode={self.config.default_mode.value}, "
            f"auto_route={self.config.auto_route}, track_metrics={self._metrics_tracker is not None}"
        )

    @property
    def embedder(self):
        """Lazy load embedder"""
        if self._embedder is None:
            from ..models.embedding_manager import get_embedding_manager
            self._embedder = get_embedding_manager()
        return self._embedder

    @property
    def index_manager(self):
        """Lazy load index manager"""
        if self._index_manager is None:
            from ..indexing import get_index_manager
            self._index_manager = get_index_manager()
        return self._index_manager

    @property
    def graph_client(self):
        """Lazy load graph client"""
        if self._graph_client is None:
            from ..graph_builder import Neo4jClient
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
        elif mode == RetrievalMode.GLOBAL_SEARCH:
            return self._global_search_retrieve(query, top_k, **kwargs)
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

        MIRAGE V4 Enhanced L2 strategy:
        1. Get relevant chunks via vector search
        2. Extract entities from those chunks
        3. Use entity disambiguation (cross-encoder) for better matching
        4. Filter entities by type based on query patterns
        5. Include entity names in retrieval metadata for better answers
        """
        if query_embedding is None or self.index_manager is None:
            return self._naive_retrieve(query, query_embedding, top_k, **kwargs)

        # 1. First get naive results as base
        naive_response = self._naive_retrieve(query, query_embedding, top_k * 2, **kwargs)

        # 1b. Keyword-based search fallback for Arabic entities
        # Vector search may miss chunks due to embedding model limitations
        # Use Arabic variants to handle ة↔ه normalization issues
        entity_phrases = self._extract_arabic_entity_phrases(query)
        keyword_chunks = []
        if entity_phrases and self.index_manager:
            try:
                seen_chunks = set()
                # Search for chunks containing key entity phrases directly
                for phrase in entity_phrases[:5]:
                    if len(phrase) >= 3:
                        # Get Arabic variants for this phrase (handles ة↔ه, أ/إ/آ→ا)
                        variants = get_arabic_variants(phrase)
                        for variant in variants:
                            keyword_results = self.index_manager.keyword_search(
                                variant, limit=3
                            )
                            for kr in keyword_results:
                                chunk_id = kr.get("chunk_id", kr.get("id", ""))
                                if chunk_id and chunk_id not in seen_chunks:
                                    seen_chunks.add(chunk_id)
                                    keyword_chunks.append(RetrievalResult(
                                        chunk_id=chunk_id,
                                        document_id=kr.get("document_id", ""),
                                        text=kr.get("text", ""),
                                        score=0.90,  # High score for keyword match
                                        retrieval_mode="keyword",
                                        metadata={"keyword_match": phrase, "matched_variant": variant}
                                    ))
            except Exception as e:
                logger.debug(f"Keyword search failed: {e}")

        # Merge keyword results into naive results (boost or add)
        result_map = {r.chunk_id: r for r in naive_response.results if r.chunk_id}
        added_count = 0
        boosted_count = 0

        for kr in keyword_chunks:
            if not kr.chunk_id:
                continue

            if kr.chunk_id in result_map:
                # Chunk already exists - boost its score for keyword match
                existing = result_map[kr.chunk_id]
                old_score = existing.score
                existing.score = max(existing.score, 0.92)  # Boost to high score
                existing.metadata = existing.metadata or {}
                existing.metadata["keyword_boost"] = True
                existing.metadata["keyword_match"] = kr.metadata.get("keyword_match", "")
                boosted_count += 1
                logger.debug(f"Keyword boost: {kr.chunk_id} {old_score:.2f} → {existing.score:.2f}")
            else:
                # New chunk - add with high score
                naive_response.results.insert(0, kr)
                result_map[kr.chunk_id] = kr
                added_count += 1

        # Re-sort by score after boosting
        if added_count > 0 or boosted_count > 0:
            naive_response.results.sort(key=lambda x: x.score, reverse=True)
            logger.debug(f"Keyword search: added={added_count}, boosted={boosted_count}")

        # 2. Detect entity types from query patterns
        entity_types = self._detect_entity_types(query)

        # 3. Extract entities from retrieved chunks
        extracted_entities = []
        entity_chunks = []
        disambiguated_entities = []  # MIRAGE V4: Track disambiguated entities

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

                # MIRAGE V4: Use entity disambiguator for semantic matching
                # Use Arabic entity phrase extraction for better compound entity matching
                if self.entity_disambiguator:
                    query_terms = self._extract_arabic_entity_phrases(query)
                    logger.debug(f"Arabic entity phrases extracted: {query_terms}")
                    for term in query_terms[:8]:  # Check more phrases
                        try:
                            result = self.entity_disambiguator.disambiguate(
                                query_entity=term,
                                entity_type=entity_types[0] if entity_types else None,
                                context=query
                            )
                            if result.matched_entity:
                                disambiguated_entities.append({
                                    "query_term": term,
                                    "matched": result.matched_entity,
                                    "score": result.similarity_score,
                                    "match_type": result.match_type
                                })
                                # Get chunks for disambiguated entity
                                chunks = self.graph_client.get_entity_chunks(
                                    result.matched_entity, limit=3
                                )
                                for chunk in chunks:
                                    # Higher score for semantically disambiguated matches
                                    entity_chunks.append({
                                        "chunk_id": chunk.get("chunk_id", ""),
                                        "document_id": chunk.get("document_id", ""),
                                        "text": chunk.get("text", ""),
                                        "entity": result.matched_entity,
                                        "entity_type": chunk.get("entity_type", ""),
                                        "disambiguated": True,
                                        "match_score": result.similarity_score
                                    })
                        except Exception as e:
                            logger.debug(f"Disambiguation failed for term '{term}': {e}")
                else:
                    # Fallback: direct term matching with Arabic phrase extraction
                    query_terms = self._extract_arabic_entity_phrases(query)
                    for term in query_terms[:5]:
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

        # 4. Text content fallback: Boost naive results containing entity phrases
        # This handles cases where entities weren't extracted during ingestion
        entity_phrases = self._extract_arabic_entity_phrases(query)
        for r in naive_response.results:
            text_lower = r.text.lower() if r.text else ""
            for phrase in entity_phrases:
                if phrase in r.text or phrase in text_lower:
                    # Found entity phrase in chunk text - boost score
                    entity_chunks.append({
                        "chunk_id": r.chunk_id,
                        "document_id": r.document_id,
                        "text": r.text,
                        "entity": phrase,
                        "entity_type": "TextMatch",
                        "text_match": True,
                        "match_score": 0.85
                    })
                    break  # Only add once per chunk

        # 5. Combine with naive results, prioritizing entity-connected chunks
        results = []
        seen_chunks = set()

        # Sort entity_chunks: disambiguated first (higher score), then by match_score
        entity_chunks_sorted = sorted(
            entity_chunks,
            key=lambda x: (x.get("disambiguated", False), x.get("match_score", 0.5)),
            reverse=True
        )

        # First add entity-connected chunks
        for ec in entity_chunks_sorted[:top_k // 2]:
            if ec["chunk_id"] and ec["chunk_id"] not in seen_chunks:
                seen_chunks.add(ec["chunk_id"])
                # MIRAGE V4: Higher score for disambiguated matches
                base_score = 0.90 if ec.get("disambiguated") else 0.85
                match_score = ec.get("match_score", 0.5)
                final_score = min(0.95, base_score * match_score + (0.1 if ec.get("disambiguated") else 0))

                results.append(RetrievalResult(
                    chunk_id=ec["chunk_id"],
                    document_id=ec["document_id"],
                    text=ec["text"],
                    score=final_score,
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
                # MIRAGE V4: Include disambiguation results
                "disambiguated_entities": disambiguated_entities[:10] if disambiguated_entities else [],
                "disambiguation_enabled": self.entity_disambiguator is not None,
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

    def _extract_arabic_entity_phrases(self, query: str) -> List[str]:
        """
        Extract compound Arabic entity phrases from query.

        Arabic entities often have prefixes like:
        - شركة (company) + name → "شركة تيتو"
        - هيئة (authority) + name → "هيئة الزكاة"
        - وزارة (ministry) + name → "وزارة الصحة"
        - جامعة (university) + name

        Returns:
            List of entity phrases to search for
        """
        import re

        phrases = []
        query_clean = query.replace("؟", " ").replace("،", " ").strip()

        # Pattern 1: شركة/هيئة/وزارة + following word(s)
        prefixes = ["شركة", "شركه", "هيئة", "هيئه", "وزارة", "وزاره", "جامعة", "جامعه", "مركز", "جائزة", "جائزه"]
        for prefix in prefixes:
            pattern = rf"{prefix}\s+([^\s]+(?:\s+[^\s]+)?)"
            matches = re.findall(pattern, query_clean)
            for match in matches:
                full_phrase = f"{prefix} {match}".strip()
                phrases.append(full_phrase)
                # Also add just the name part
                phrases.append(match.strip())

        # Pattern 2: Words that look like proper nouns (no common words)
        common_words = {"من", "هي", "هو", "ما", "هل", "في", "على", "إلى", "عن", "التي", "الذي", "هذا", "هذه", "تلك", "ذلك"}
        words = query_clean.split()
        for word in words:
            if len(word) > 2 and word not in common_words:
                # Check if it might be an entity name
                if not word.startswith("ال") or len(word) > 4:  # Skip generic articles
                    phrases.append(word)

        return list(set(phrases))  # Deduplicate

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

    def _global_search_retrieve(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """
        TRUE GraphRAG Global Search: Map-reduce over community summaries.

        This enables answering holistic queries like:
        - "What are the main themes across all documents?"
        - "Summarize the key topics in this knowledge base"
        - "What patterns emerge from the data?"

        Strategy:
        1. MAP: Query each community summary in parallel
        2. FILTER: Keep relevant partial answers
        3. REDUCE: Synthesize coherent final answer
        4. Get supporting chunks from relevant communities
        """
        try:
            from .global_search import GlobalSearchEngine

            # Initialize global search engine
            global_engine = GlobalSearchEngine(
                neo4j_client=self.graph_client,
                llm_endpoint="http://tgi:80",
                max_communities=kwargs.get('max_communities', 30),
                min_relevance=kwargs.get('min_relevance', 0.3),
                community_level=kwargs.get('community_level', 0)
            )

            # Execute global search
            result = global_engine.search(query)

            logger.info(
                f"Global search: {result.communities_searched} communities queried, "
                f"{len(result.partial_answers)} relevant answers"
            )

            # Get supporting chunks from relevant communities
            supporting_chunks = []
            if result.partial_answers and self.graph_client:
                community_ids = [pa.community_id for pa in result.partial_answers[:5]]
                supporting_chunks = self._get_chunks_from_communities(community_ids, top_k)

            # Build results
            results = [
                RetrievalResult(
                    chunk_id=chunk.get("chunk_id", ""),
                    document_id=chunk.get("document_id", ""),
                    text=chunk.get("text", ""),
                    score=0.85,
                    retrieval_mode="global_search",
                    metadata={
                        "community_id": chunk.get("community_id", ""),
                        "from_global_search": True
                    }
                )
                for chunk in supporting_chunks
            ]

            return RetrievalResponse(
                results=results[:top_k],
                query=query,
                mode=RetrievalMode.GLOBAL_SEARCH,
                total_candidates=result.communities_searched,
                metadata={
                    "global_answer": result.answer,
                    "communities_searched": result.communities_searched,
                    "total_communities": result.total_communities,
                    "partial_answers_count": len(result.partial_answers),
                    "themes": result.themes,
                    "confidence": result.confidence,
                    "is_global_search": True
                }
            )

        except Exception as e:
            logger.error(f"Global search failed: {e}")
            # Fallback to relationship-based global retrieval
            query_embedding = self.embedder.embed(query) if self.embedder else None
            return self._global_retrieve(query, query_embedding, top_k, **kwargs)

    def _get_chunks_from_communities(
        self,
        community_ids: List[str],
        limit: int = 10
    ) -> List[Dict]:
        """Get representative chunks from specified communities."""
        if not self.graph_client or not community_ids:
            return []

        try:
            # Note: Relationship is (Chunk)-[:MENTIONS]->(Entity)
            query = """
            MATCH (c:Community)
            WHERE c.id IN $community_ids
            MATCH (e:Entity)-[:BELONGS_TO]->(c)
            MATCH (chunk:Chunk)-[:MENTIONS]->(e)
            RETURN DISTINCT
                chunk.id as chunk_id,
                chunk.text as text,
                chunk.document_id as document_id,
                c.id as community_id
            LIMIT $limit
            """

            results = self.graph_client.execute_query(query, {
                'community_ids': community_ids,
                'limit': limit
            })

            return results

        except Exception as e:
            logger.warning(f"Error getting chunks from communities: {e}")
            return []

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
        """
        Deep semantic matching with cross-encoder re-ranking.

        Strategy:
        1. Retrieve top_k * 3 candidates via naive vector search
        2. Re-rank using cross-encoder for improved precision
        3. Return top_k highest scoring results
        """
        # Start with naive retrieval but with more candidates
        naive_response = self._naive_retrieve(
            query, query_embedding, top_k * 3, **kwargs
        )

        # Apply cross-encoder re-ranking
        try:
            from .reranker import get_reranker

            reranker = get_reranker()

            # Convert results to candidates for reranker
            candidates = [
                {
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "text": r.text,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in naive_response.results
            ]

            # Rerank
            reranked = reranker.rerank(query, candidates, top_k=top_k)

            # Convert back to RetrievalResult
            results = [
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    text=r.text,
                    score=r.rerank_score,  # Use rerank score
                    retrieval_mode="semantic",
                    metadata={
                        "original_score": r.original_score,
                        "rank_change": r.rank_change
                    }
                )
                for r in reranked
            ]

            logger.info(
                f"Semantic reranking: {len(candidates)} candidates -> {len(results)} results"
            )

            return RetrievalResponse(
                results=results,
                query=query,
                mode=RetrievalMode.SEMANTIC,
                total_candidates=len(naive_response.results),
                metadata={
                    "reranked": True,
                    "reranker_model": reranker.model_id
                }
            )

        except Exception as e:
            logger.warning(f"Reranking failed, falling back to naive: {e}")
            # Fallback to naive results
            naive_response.mode = RetrievalMode.SEMANTIC
            naive_response.results = naive_response.results[:top_k]
            naive_response.metadata = {"reranked": False, "fallback_reason": str(e)}
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

    def get_metrics_stats(self) -> Dict[str, Any]:
        """
        Get aggregated metrics statistics.

        Returns:
            Dict with retrieval metrics (MRR, NDCG, MAP) and query count
        """
        if self._metrics_tracker:
            return self._metrics_tracker.get_stats()
        return {"tracking_enabled": False}

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
