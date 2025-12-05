"""
Dual-Level Retrieval - LightRAG-Inspired Two-Level Search

Key Innovation from LightRAG (EMNLP 2025):
- Low-level: Precise entity and relationship retrieval
  - Answers: "Who founded X?" "When did Y happen?"
  - Direct entity matching, relationship traversal

- High-level: Broad topic and theme retrieval
  - Answers: "What are the main themes?" "Summarize X"
  - Community summaries, topic clusters

Running BOTH simultaneously and fusing results gives:
- Best of both worlds
- 6000x cost reduction vs pure GraphRAG
- Better coverage across query types

This replaces the need for mode selection - just run both and fuse.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import time


@dataclass
class LowLevelResult:
    """Result from low-level (entity-focused) retrieval."""
    entity_name: str
    entity_type: str
    relationships: List[Dict] = field(default_factory=list)
    chunks: List[Dict] = field(default_factory=list)
    score: float = 0.0
    match_type: str = "direct"  # direct, fuzzy, semantic


@dataclass
class HighLevelResult:
    """Result from high-level (community/theme) retrieval."""
    community_id: str
    community_summary: str
    relevance_score: float
    themes: List[str] = field(default_factory=list)
    chunks: List[Dict] = field(default_factory=list)


@dataclass
class DualLevelResult:
    """Combined result from dual-level retrieval."""
    query: str
    low_level_results: List[LowLevelResult] = field(default_factory=list)
    high_level_results: List[HighLevelResult] = field(default_factory=list)
    fused_chunks: List[Dict] = field(default_factory=list)
    query_type: str = "balanced"  # factual, thematic, balanced
    weights: Dict[str, float] = field(default_factory=dict)
    retrieval_time_ms: float = 0.0

    @property
    def total_entities(self) -> int:
        return len(self.low_level_results)

    @property
    def total_communities(self) -> int:
        return len(self.high_level_results)


class QueryTypeClassifier:
    """
    Classify query as factual (low-level) or thematic (high-level).

    Determines how to weight the fusion of dual-level results.
    """

    def classify(self, query: str) -> Tuple[str, Dict[str, float]]:
        """
        Classify query type and return weights.

        Returns:
            (query_type, weights_dict)
            - "factual": Favor low-level (entity matches)
            - "thematic": Favor high-level (community summaries)
            - "balanced": Equal weight
        """
        query_lower = query.lower()

        # Factual patterns (favor low-level)
        factual_patterns = [
            "من هو", "من هي",  # Who is
            "متى", "when",  # When
            "أين", "where",  # Where
            "كم", "how many", "how much",  # Quantity
            "ما اسم", "what is the name",  # Name
            "رقم", "number",  # Number
        ]

        # Thematic patterns (favor high-level)
        thematic_patterns = [
            "ما هي المواضيع", "what are the themes",  # Themes
            "لخص", "summarize",  # Summarize
            "ما هي الأنماط", "what patterns",  # Patterns
            "بشكل عام", "overall",  # Overall
            "الموضوعات الرئيسية", "main topics",  # Main topics
            "كيف يمكن وصف", "how would you describe",  # Describe
        ]

        # Check for factual
        if any(p in query_lower for p in factual_patterns):
            return "factual", {"low": 0.8, "high": 0.2}

        # Check for thematic
        if any(p in query_lower for p in thematic_patterns):
            return "thematic", {"low": 0.2, "high": 0.8}

        # Default: balanced
        return "balanced", {"low": 0.5, "high": 0.5}


class DualLevelRetriever:
    """
    LightRAG-style dual retrieval for comprehensive coverage.

    Executes low-level (entity) and high-level (community) retrieval
    simultaneously, then fuses results based on query type.

    This is more efficient than selecting a single mode because:
    1. Both retrievals run in parallel
    2. Results are complementary, not redundant
    3. Fusion produces better coverage than either alone

    Usage:
        retriever = DualLevelRetriever(neo4j_client, embedder)
        result = retriever.retrieve(query)
        # Use result.fused_chunks for generation
    """

    def __init__(
        self,
        neo4j_client,
        embedder,
        entity_extractor=None,
        community_selector=None,
        max_entities: int = 10,
        max_communities: int = 5,
        max_chunks_per_source: int = 3,
        parallel_workers: int = 2
    ):
        """
        Initialize dual-level retriever.

        Args:
            neo4j_client: Neo4j client for graph queries
            embedder: Embedding manager for similarity
            entity_extractor: Optional entity extractor for query entities
            community_selector: Optional dynamic community selector
            max_entities: Max entities to retrieve from low-level
            max_communities: Max communities to retrieve from high-level
            max_chunks_per_source: Max chunks per entity/community
            parallel_workers: Workers for parallel retrieval
        """
        self.neo4j_client = neo4j_client
        self.embedder = embedder
        self.entity_extractor = entity_extractor
        self.community_selector = community_selector
        self.max_entities = max_entities
        self.max_communities = max_communities
        self.max_chunks_per_source = max_chunks_per_source

        self.classifier = QueryTypeClassifier()
        self._executor = ThreadPoolExecutor(max_workers=parallel_workers)

        logger.info(
            f"DualLevelRetriever initialized: "
            f"max_entities={max_entities}, max_communities={max_communities}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        query_embedding: Optional[np.ndarray] = None
    ) -> DualLevelResult:
        """
        Execute dual-level retrieval.

        Algorithm:
        1. Classify query type (factual/thematic/balanced)
        2. Execute low-level and high-level retrieval in parallel
        3. Fuse results based on query type weights
        4. Deduplicate and rank fused chunks

        Args:
            query: User query string
            top_k: Number of final chunks to return
            query_embedding: Optional pre-computed query embedding

        Returns:
            DualLevelResult with fused chunks
        """
        start_time = time.time()

        # Step 1: Classify query
        query_type, weights = self.classifier.classify(query)
        logger.debug(f"Query type: {query_type}, weights: {weights}")

        # Get query embedding if not provided
        if query_embedding is None:
            query_embedding = self._get_embedding(query)

        # Step 2: Execute both levels in parallel
        low_future = self._executor.submit(
            self._low_level_retrieve, query, query_embedding
        )
        high_future = self._executor.submit(
            self._high_level_retrieve, query, query_embedding
        )

        # Wait for results
        low_results = low_future.result(timeout=30)
        high_results = high_future.result(timeout=30)

        # Step 3: Fuse results
        fused_chunks = self._fuse_results(
            low_results, high_results, weights, top_k
        )

        retrieval_time = (time.time() - start_time) * 1000

        logger.info(
            f"Dual-level retrieval: {len(low_results)} entities, "
            f"{len(high_results)} communities, "
            f"{len(fused_chunks)} fused chunks, "
            f"{retrieval_time:.1f}ms"
        )

        return DualLevelResult(
            query=query,
            low_level_results=low_results,
            high_level_results=high_results,
            fused_chunks=fused_chunks,
            query_type=query_type,
            weights=weights,
            retrieval_time_ms=retrieval_time
        )

    def _low_level_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray]
    ) -> List[LowLevelResult]:
        """
        Low-level retrieval: precise entity and relationship matching.

        Strategy:
        1. Extract entities from query
        2. Find matching entities in graph
        3. Get relationships for matched entities
        4. Get chunks mentioning these entities
        """
        results = []

        try:
            # Extract query entities
            query_entities = self._extract_query_entities(query)

            for query_entity in query_entities[:self.max_entities]:
                # Find matching entities in graph
                matches = self._find_entity_matches(
                    query_entity, query_embedding
                )

                for match in matches[:3]:  # Top 3 matches per query entity
                    entity_name = match.get("name", "")
                    entity_type = match.get("type", "")

                    # Get relationships
                    relationships = self._get_entity_relationships(entity_name)

                    # Get mentioning chunks
                    chunks = self._get_entity_chunks(entity_name)

                    results.append(LowLevelResult(
                        entity_name=entity_name,
                        entity_type=entity_type,
                        relationships=relationships[:5],
                        chunks=chunks[:self.max_chunks_per_source],
                        score=match.get("score", 0.5),
                        match_type=match.get("match_type", "direct")
                    ))

        except Exception as e:
            logger.warning(f"Low-level retrieval error: {e}")

        return results[:self.max_entities]

    def _high_level_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray]
    ) -> List[HighLevelResult]:
        """
        High-level retrieval: community and theme matching.

        Strategy:
        1. Get relevant communities (using dynamic selector if available)
        2. Score community summaries against query
        3. Get representative chunks from top communities
        """
        results = []

        try:
            # Get relevant communities
            if self.community_selector:
                selection = self.community_selector.select_communities(query)
                communities = [
                    {
                        "id": c.id,
                        "summary": c.summary,
                        "key_entities": c.key_entities
                    }
                    for c in selection.selected_communities
                ]
            else:
                # Fallback: get all communities with summaries
                communities = self._get_communities_with_summaries()

            # Score and filter communities
            for community in communities[:self.max_communities * 2]:
                community_id = community.get("id", "")
                summary = community.get("summary", "")

                # Score relevance
                relevance = self._score_relevance(query_embedding, summary)

                if relevance > 0.3:  # Threshold
                    # Get chunks from community
                    chunks = self._get_community_chunks(community_id)

                    results.append(HighLevelResult(
                        community_id=community_id,
                        community_summary=summary[:300],
                        relevance_score=relevance,
                        themes=community.get("themes", []) or [],
                        chunks=chunks[:self.max_chunks_per_source]
                    ))

            # Sort by relevance
            results.sort(key=lambda x: x.relevance_score, reverse=True)

        except Exception as e:
            logger.warning(f"High-level retrieval error: {e}")

        return results[:self.max_communities]

    def _fuse_results(
        self,
        low_results: List[LowLevelResult],
        high_results: List[HighLevelResult],
        weights: Dict[str, float],
        top_k: int
    ) -> List[Dict]:
        """
        Fuse low-level and high-level results.

        Uses weighted scoring based on query type.
        """
        fused = []
        seen_chunks = set()

        low_weight = weights.get("low", 0.5)
        high_weight = weights.get("high", 0.5)

        # Add low-level chunks
        for result in low_results:
            for chunk in result.chunks:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)

                    # Score: entity match score * low_weight
                    chunk_score = result.score * low_weight

                    fused.append({
                        **chunk,
                        "fused_score": chunk_score,
                        "source": "low_level",
                        "via_entity": result.entity_name,
                        "entity_type": result.entity_type
                    })

        # Add high-level chunks
        for result in high_results:
            for chunk in result.chunks:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)

                    # Score: community relevance * high_weight
                    chunk_score = result.relevance_score * high_weight

                    fused.append({
                        **chunk,
                        "fused_score": chunk_score,
                        "source": "high_level",
                        "via_community": result.community_id,
                        "themes": result.themes
                    })

        # Sort by fused score
        fused.sort(key=lambda x: x.get("fused_score", 0), reverse=True)

        return fused[:top_k]

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entity mentions from query."""
        if self.entity_extractor:
            try:
                entities = self.entity_extractor.extract(query)
                return [e.get("name", "") for e in entities if e.get("name")]
            except Exception:
                pass

        # Fallback: split on common patterns and extract noun phrases
        # Simple heuristic: take words > 2 chars, not stopwords
        stopwords = {"من", "في", "على", "إلى", "هل", "ما", "هو", "هي", "أن", "كان"}
        words = query.split()
        return [w for w in words if len(w) > 2 and w not in stopwords]

    def _find_entity_matches(
        self,
        query_entity: str,
        query_embedding: Optional[np.ndarray]
    ) -> List[Dict]:
        """Find entities in graph matching query entity."""
        matches = []

        try:
            # Direct name match
            direct_query = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $term OR e.name =~ $regex
            RETURN e.name as name, e.type as type, 1.0 as score, 'direct' as match_type
            LIMIT 5
            """

            results = self.neo4j_client.execute_query(direct_query, {
                'term': query_entity,
                'regex': f"(?i).*{query_entity}.*"
            })

            for r in results:
                matches.append({
                    "name": r.get("name"),
                    "type": r.get("type"),
                    "score": r.get("score", 1.0),
                    "match_type": r.get("match_type", "direct")
                })

            # If no direct matches and we have embedding, try semantic
            if not matches and query_embedding is not None:
                # Get entities with embeddings and compare
                # This is expensive, so limit the search
                fuzzy_query = """
                MATCH (e:Entity)
                WHERE e.embedding IS NOT NULL
                RETURN e.name as name, e.type as type, e.embedding as embedding
                LIMIT 50
                """

                fuzzy_results = self.neo4j_client.execute_query(fuzzy_query)

                for r in fuzzy_results:
                    emb = r.get("embedding")
                    if emb:
                        try:
                            if isinstance(emb, str):
                                import json
                                emb = np.array(json.loads(emb))
                            elif isinstance(emb, list):
                                emb = np.array(emb)

                            similarity = np.dot(query_embedding, emb) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                            )

                            if similarity > 0.5:
                                matches.append({
                                    "name": r.get("name"),
                                    "type": r.get("type"),
                                    "score": float(similarity),
                                    "match_type": "semantic"
                                })
                        except Exception:
                            pass

                # Sort by score
                matches.sort(key=lambda x: x.get("score", 0), reverse=True)

        except Exception as e:
            logger.debug(f"Entity match error for '{query_entity}': {e}")

        return matches

    def _get_entity_relationships(self, entity_name: str) -> List[Dict]:
        """Get relationships for an entity."""
        try:
            query = """
            MATCH (e:Entity {name: $name})-[r]->(target:Entity)
            RETURN type(r) as rel_type, target.name as target_name, target.type as target_type
            UNION
            MATCH (source:Entity)-[r]->(e:Entity {name: $name})
            RETURN type(r) as rel_type, source.name as target_name, source.type as target_type
            LIMIT 10
            """

            results = self.neo4j_client.execute_query(query, {'name': entity_name})

            return [
                {
                    "type": r.get("rel_type"),
                    "target": r.get("target_name"),
                    "target_type": r.get("target_type")
                }
                for r in results
            ]

        except Exception as e:
            logger.debug(f"Relationship query error for '{entity_name}': {e}")
            return []

    def _get_entity_chunks(self, entity_name: str) -> List[Dict]:
        """Get chunks that mention an entity."""
        try:
            query = """
            MATCH (chunk:Chunk)-[:MENTIONS]->(e:Entity {name: $name})
            RETURN
                chunk.id as chunk_id,
                chunk.text as text,
                chunk.document_id as document_id
            LIMIT $limit
            """

            results = self.neo4j_client.execute_query(query, {
                'name': entity_name,
                'limit': self.max_chunks_per_source
            })

            return [
                {
                    "chunk_id": r.get("chunk_id"),
                    "text": r.get("text"),
                    "document_id": r.get("document_id")
                }
                for r in results
            ]

        except Exception as e:
            logger.debug(f"Chunk query error for '{entity_name}': {e}")
            return []

    def _get_communities_with_summaries(self) -> List[Dict]:
        """Get communities that have summaries."""
        try:
            query = """
            MATCH (c:Community)
            WHERE c.summary IS NOT NULL AND c.summary <> ''
            RETURN c.id as id, c.summary as summary, c.key_entities as key_entities, c.themes as themes
            ORDER BY c.size DESC
            LIMIT $limit
            """

            results = self.neo4j_client.execute_query(query, {
                'limit': self.max_communities * 3
            })

            return results

        except Exception as e:
            logger.debug(f"Community query error: {e}")
            return []

    def _get_community_chunks(self, community_id: str) -> List[Dict]:
        """Get representative chunks from a community."""
        try:
            query = """
            MATCH (c:Community {id: $community_id})
            MATCH (e:Entity)-[:BELONGS_TO]->(c)
            MATCH (chunk:Chunk)-[:MENTIONS]->(e)
            RETURN DISTINCT
                chunk.id as chunk_id,
                chunk.text as text,
                chunk.document_id as document_id
            LIMIT $limit
            """

            results = self.neo4j_client.execute_query(query, {
                'community_id': community_id,
                'limit': self.max_chunks_per_source
            })

            return [
                {
                    "chunk_id": r.get("chunk_id"),
                    "text": r.get("text"),
                    "document_id": r.get("document_id")
                }
                for r in results
            ]

        except Exception as e:
            logger.debug(f"Community chunk query error: {e}")
            return []

    def _score_relevance(
        self,
        query_embedding: Optional[np.ndarray],
        text: str
    ) -> float:
        """Score relevance of text to query using embedding similarity."""
        if query_embedding is None or not text:
            return 0.5  # Default moderate score

        try:
            text_embedding = self._get_embedding(text)
            if text_embedding is None:
                return 0.5

            similarity = np.dot(query_embedding, text_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
            )

            # Normalize to 0-1
            return max(0.0, min(1.0, (similarity + 1) / 2))

        except Exception:
            return 0.5

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text."""
        try:
            embedding = self.embedder.embed(text)
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            return embedding
        except Exception:
            return None

    def __del__(self):
        """Cleanup executor."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# Singleton
_dual_level_retriever: Optional[DualLevelRetriever] = None


def get_dual_level_retriever(
    neo4j_client=None,
    embedder=None,
    **kwargs
) -> DualLevelRetriever:
    """Get or create dual-level retriever singleton."""
    global _dual_level_retriever

    if _dual_level_retriever is None:
        if neo4j_client is None:
            from ..graph_builder import Neo4jClient
            neo4j_client = Neo4jClient()

        if embedder is None:
            from ..models.embedding_manager import get_embedding_manager
            embedder = get_embedding_manager()

        _dual_level_retriever = DualLevelRetriever(
            neo4j_client=neo4j_client,
            embedder=embedder,
            **kwargs
        )

    return _dual_level_retriever
