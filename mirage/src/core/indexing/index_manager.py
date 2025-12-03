"""
MIRAGE V2 Triple-Level Index Manager
Manages three Qdrant collections: chunks, entities, relationships.

This is a key differentiator from basic RAG - we index not just text chunks,
but also entities and relationships with their own embeddings, enabling:
- Entity-focused retrieval (local search)
- Relationship-focused retrieval (global search)
- Combined retrieval (hybrid/mix search)
"""

import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams,
        Distance,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        Range,
        SearchParams,
        CollectionStatus,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed")

import numpy as np


@dataclass
class IndexedChunk:
    """Chunk with embedding and metadata"""
    chunk_id: str
    document_id: str
    text: str
    embedding: Optional[np.ndarray] = None
    index: int = 0
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexedEntity:
    """Entity with embedding and metadata"""
    entity_id: str
    name: str
    entity_type: str
    description: str = ""
    embedding: Optional[np.ndarray] = None
    # Graph properties
    document_ids: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    mention_count: int = 1
    confidence: float = 1.0
    # Temporal
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    # Community
    community_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexedRelationship:
    """Relationship with embedding and metadata"""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    description: str = ""
    embedding: Optional[np.ndarray] = None
    # Properties
    document_ids: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    weight: float = 1.0
    confidence: float = 1.0
    # Temporal
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result from vector search"""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[np.ndarray] = None


class IndexManager:
    """
    Manages triple-level indexing in Qdrant.

    Collections:
    1. chunks: Text chunks from documents
    2. entities: Extracted entities with embeddings
    3. relationships: Relationships between entities with embeddings

    This enables:
    - Naive search: chunks only
    - Local search: entities → chunks
    - Global search: relationships → entities → chunks
    - Hybrid: combination of above
    """

    def __init__(
        self,
        host: str = "qdrant",
        port: int = 6333,
        collection_prefix: str = "mirage",
        embedding_dim: int = 768,
        distance: str = "Cosine"
    ):
        """
        Initialize index manager.

        Args:
            host: Qdrant host
            port: Qdrant port
            collection_prefix: Prefix for collection names
            embedding_dim: Embedding dimension
            distance: Distance metric (Cosine, Dot, Euclidean)
        """
        if not QDRANT_AVAILABLE:
            raise RuntimeError("qdrant-client not installed")

        self.host = host
        self.port = port
        self.prefix = collection_prefix
        self.embedding_dim = embedding_dim
        self.distance = getattr(Distance, distance.upper(), Distance.COSINE)

        # Collection names
        self.chunks_collection = f"{collection_prefix}_chunks"
        self.entities_collection = f"{collection_prefix}_entities"
        self.relationships_collection = f"{collection_prefix}_relationships"

        # Client
        self._client: Optional[QdrantClient] = None

        logger.info(
            f"IndexManager initialized: host={host}:{port}, "
            f"prefix={collection_prefix}, dim={embedding_dim}"
        )

    def _get_client(self) -> QdrantClient:
        """Get or create Qdrant client"""
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    def _to_uuid(self, id_str: str) -> str:
        """
        Convert string ID to UUID format for Qdrant.

        Uses MD5 hash to generate deterministic UUIDs from any string.
        This ensures the same string always maps to the same UUID.
        """
        import hashlib
        hash_bytes = hashlib.md5(id_str.encode()).digest()
        return str(uuid.UUID(bytes=hash_bytes))

    def ensure_collections(self) -> Dict[str, bool]:
        """
        Ensure all three collections exist.

        Returns:
            Dict of collection_name -> created (True if newly created)
        """
        client = self._get_client()
        results = {}

        for collection_name in [
            self.chunks_collection,
            self.entities_collection,
            self.relationships_collection
        ]:
            try:
                # Check if exists
                collections = client.get_collections().collections
                exists = any(c.name == collection_name for c in collections)

                if not exists:
                    # Create collection
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=self.embedding_dim,
                            distance=self.distance
                        )
                    )
                    logger.info(f"Created collection: {collection_name}")
                    results[collection_name] = True
                else:
                    results[collection_name] = False

            except Exception as e:
                logger.error(f"Failed to ensure collection {collection_name}: {e}")
                raise

        return results

    # =========================================================================
    # CHUNK OPERATIONS
    # =========================================================================

    def index_chunk(self, chunk: IndexedChunk) -> str:
        """
        Index a single chunk.

        Args:
            chunk: Chunk to index

        Returns:
            Chunk ID
        """
        client = self._get_client()

        if chunk.embedding is None:
            raise ValueError("Chunk must have embedding")

        point = PointStruct(
            id=self._to_uuid(chunk.chunk_id),
            vector=chunk.embedding.tolist() if isinstance(chunk.embedding, np.ndarray) else chunk.embedding,
            payload={
                "chunk_id": chunk.chunk_id,  # Store original ID in payload
                "document_id": chunk.document_id,
                "text": chunk.text,
                "index": chunk.index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
                "created_at": chunk.created_at,
                **chunk.metadata
            }
        )

        client.upsert(
            collection_name=self.chunks_collection,
            points=[point]
        )

        return chunk.chunk_id

    def index_chunks(self, chunks: List[IndexedChunk], batch_size: int = 100) -> int:
        """
        Index multiple chunks.

        Args:
            chunks: List of chunks
            batch_size: Batch size for upserts

        Returns:
            Number of chunks indexed
        """
        client = self._get_client()
        total = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            points = []
            for chunk in batch:
                if chunk.embedding is None:
                    continue

                points.append(PointStruct(
                    id=self._to_uuid(chunk.chunk_id),
                    vector=chunk.embedding.tolist() if isinstance(chunk.embedding, np.ndarray) else chunk.embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,  # Store original ID in payload
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        "index": chunk.index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "token_count": chunk.token_count,
                        "created_at": chunk.created_at,
                        **chunk.metadata
                    }
                ))

            if points:
                client.upsert(
                    collection_name=self.chunks_collection,
                    points=points
                )
                total += len(points)

        logger.info(f"Indexed {total} chunks")
        return total

    def search_chunks(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        score_threshold: float = 0.0,
        document_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search chunks by embedding.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            score_threshold: Minimum score
            document_id: Filter by document

        Returns:
            List of SearchResult
        """
        client = self._get_client()

        # Build filter
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )

        results = client.search(
            collection_name=self.chunks_collection,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True
        )

        return [
            SearchResult(
                id=r.payload.get("chunk_id", str(r.id)),  # Use original ID from payload
                score=r.score,
                payload=r.payload
            )
            for r in results
        ]

    # =========================================================================
    # ENTITY OPERATIONS
    # =========================================================================

    def index_entity(self, entity: IndexedEntity) -> str:
        """
        Index a single entity.

        Args:
            entity: Entity to index

        Returns:
            Entity ID
        """
        client = self._get_client()

        if entity.embedding is None:
            raise ValueError("Entity must have embedding")

        point = PointStruct(
            id=self._to_uuid(entity.entity_id),
            vector=entity.embedding.tolist() if isinstance(entity.embedding, np.ndarray) else entity.embedding,
            payload={
                "entity_id": entity.entity_id,  # Store original ID in payload
                "name": entity.name,
                "entity_type": entity.entity_type,
                "description": entity.description,
                "document_ids": entity.document_ids,
                "chunk_ids": entity.chunk_ids,
                "mention_count": entity.mention_count,
                "confidence": entity.confidence,
                "first_seen": entity.first_seen,
                "last_seen": entity.last_seen,
                "community_id": entity.community_id,
                **entity.metadata
            }
        )

        client.upsert(
            collection_name=self.entities_collection,
            points=[point]
        )

        return entity.entity_id

    def index_entities(self, entities: List[IndexedEntity], batch_size: int = 100) -> int:
        """Index multiple entities"""
        client = self._get_client()
        total = 0

        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]

            points = []
            for entity in batch:
                if entity.embedding is None:
                    continue

                points.append(PointStruct(
                    id=self._to_uuid(entity.entity_id),
                    vector=entity.embedding.tolist() if isinstance(entity.embedding, np.ndarray) else entity.embedding,
                    payload={
                        "entity_id": entity.entity_id,  # Store original ID in payload
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "description": entity.description,
                        "document_ids": entity.document_ids,
                        "chunk_ids": entity.chunk_ids,
                        "mention_count": entity.mention_count,
                        "confidence": entity.confidence,
                        "first_seen": entity.first_seen,
                        "last_seen": entity.last_seen,
                        "community_id": entity.community_id,
                        **entity.metadata
                    }
                ))

            if points:
                client.upsert(
                    collection_name=self.entities_collection,
                    points=points
                )
                total += len(points)

        logger.info(f"Indexed {total} entities")
        return total

    def search_entities(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        score_threshold: float = 0.0,
        entity_type: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[SearchResult]:
        """
        Search entities by embedding.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            score_threshold: Minimum score
            entity_type: Filter by entity type
            min_confidence: Minimum confidence

        Returns:
            List of SearchResult
        """
        client = self._get_client()

        # Build filter
        conditions = []
        if entity_type:
            conditions.append(
                FieldCondition(
                    key="entity_type",
                    match=MatchValue(value=entity_type)
                )
            )
        if min_confidence > 0:
            conditions.append(
                FieldCondition(
                    key="confidence",
                    range=Range(gte=min_confidence)
                )
            )

        query_filter = Filter(must=conditions) if conditions else None

        results = client.search(
            collection_name=self.entities_collection,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True
        )

        return [
            SearchResult(
                id=r.payload.get("entity_id", str(r.id)),  # Use original ID from payload
                score=r.score,
                payload=r.payload
            )
            for r in results
        ]

    # =========================================================================
    # RELATIONSHIP OPERATIONS
    # =========================================================================

    def index_relationship(self, relationship: IndexedRelationship) -> str:
        """
        Index a single relationship.

        Args:
            relationship: Relationship to index

        Returns:
            Relationship ID
        """
        client = self._get_client()

        if relationship.embedding is None:
            raise ValueError("Relationship must have embedding")

        point = PointStruct(
            id=self._to_uuid(relationship.relationship_id),
            vector=relationship.embedding.tolist() if isinstance(relationship.embedding, np.ndarray) else relationship.embedding,
            payload={
                "relationship_id": relationship.relationship_id,  # Store original ID in payload
                "source_entity_id": relationship.source_entity_id,
                "target_entity_id": relationship.target_entity_id,
                "relationship_type": relationship.relationship_type,
                "description": relationship.description,
                "document_ids": relationship.document_ids,
                "chunk_ids": relationship.chunk_ids,
                "weight": relationship.weight,
                "confidence": relationship.confidence,
                "first_seen": relationship.first_seen,
                "last_seen": relationship.last_seen,
                **relationship.metadata
            }
        )

        client.upsert(
            collection_name=self.relationships_collection,
            points=[point]
        )

        return relationship.relationship_id

    def index_relationships(self, relationships: List[IndexedRelationship], batch_size: int = 100) -> int:
        """Index multiple relationships"""
        client = self._get_client()
        total = 0

        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]

            points = []
            for rel in batch:
                if rel.embedding is None:
                    continue

                points.append(PointStruct(
                    id=self._to_uuid(rel.relationship_id),
                    vector=rel.embedding.tolist() if isinstance(rel.embedding, np.ndarray) else rel.embedding,
                    payload={
                        "relationship_id": rel.relationship_id,  # Store original ID in payload
                        "source_entity_id": rel.source_entity_id,
                        "target_entity_id": rel.target_entity_id,
                        "relationship_type": rel.relationship_type,
                        "description": rel.description,
                        "document_ids": rel.document_ids,
                        "chunk_ids": rel.chunk_ids,
                        "weight": rel.weight,
                        "confidence": rel.confidence,
                        "first_seen": rel.first_seen,
                        "last_seen": rel.last_seen,
                        **rel.metadata
                    }
                ))

            if points:
                client.upsert(
                    collection_name=self.relationships_collection,
                    points=points
                )
                total += len(points)

        logger.info(f"Indexed {total} relationships")
        return total

    def search_relationships(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        score_threshold: float = 0.0,
        relationship_type: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[SearchResult]:
        """
        Search relationships by embedding.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            score_threshold: Minimum score
            relationship_type: Filter by type
            min_confidence: Minimum confidence

        Returns:
            List of SearchResult
        """
        client = self._get_client()

        # Build filter
        conditions = []
        if relationship_type:
            conditions.append(
                FieldCondition(
                    key="relationship_type",
                    match=MatchValue(value=relationship_type)
                )
            )
        if min_confidence > 0:
            conditions.append(
                FieldCondition(
                    key="confidence",
                    range=Range(gte=min_confidence)
                )
            )

        query_filter = Filter(must=conditions) if conditions else None

        results = client.search(
            collection_name=self.relationships_collection,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True
        )

        return [
            SearchResult(
                id=r.payload.get("relationship_id", str(r.id)),  # Use original ID from payload
                score=r.score,
                payload=r.payload
            )
            for r in results
        ]

    # =========================================================================
    # CROSS-COLLECTION OPERATIONS
    # =========================================================================

    def get_chunks_for_entity(self, entity_id: str) -> List[SearchResult]:
        """Get chunks associated with an entity"""
        client = self._get_client()

        # First get the entity to find its chunk_ids
        entity_result = client.retrieve(
            collection_name=self.entities_collection,
            ids=[self._to_uuid(entity_id)],
            with_payload=True
        )

        if not entity_result:
            return []

        chunk_ids = entity_result[0].payload.get("chunk_ids", [])
        if not chunk_ids:
            return []

        # Convert chunk_ids to UUIDs for retrieval
        chunk_uuids = [self._to_uuid(cid) for cid in chunk_ids]

        # Retrieve chunks
        chunks = client.retrieve(
            collection_name=self.chunks_collection,
            ids=chunk_uuids,
            with_payload=True
        )

        return [
            SearchResult(
                id=c.payload.get("chunk_id", str(c.id)),  # Use original ID from payload
                score=1.0,
                payload=c.payload
            )
            for c in chunks
        ]

    def get_entities_for_chunk(self, chunk_id: str) -> List[SearchResult]:
        """Get entities that appear in a chunk"""
        client = self._get_client()

        # Scroll through entities to find those with this chunk_id
        # Note: In production, you might want to maintain a reverse index
        results = []

        scroll_result = client.scroll(
            collection_name=self.entities_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="chunk_ids",
                        match=MatchValue(value=chunk_id)
                    )
                ]
            ),
            limit=100,
            with_payload=True
        )

        for point in scroll_result[0]:
            results.append(
                SearchResult(
                    id=point.payload.get("entity_id", str(point.id)),  # Use original ID from payload
                    score=1.0,
                    payload=point.payload
                )
            )

        return results

    # =========================================================================
    # STATISTICS AND MANAGEMENT
    # =========================================================================

    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all collections"""
        client = self._get_client()
        stats = {}

        for name in [self.chunks_collection, self.entities_collection, self.relationships_collection]:
            try:
                info = client.get_collection(name)
                stats[name] = {
                    "points_count": info.points_count,
                    "vectors_count": info.vectors_count,
                    "status": info.status.value,
                }
            except Exception as e:
                stats[name] = {"error": str(e)}

        return stats

    def delete_by_document(self, document_id: str) -> Dict[str, int]:
        """
        Delete all indexed data for a document.

        Args:
            document_id: Document ID

        Returns:
            Dict of collection -> deleted count
        """
        client = self._get_client()
        deleted = {}

        for collection_name in [self.chunks_collection, self.entities_collection, self.relationships_collection]:
            try:
                result = client.delete(
                    collection_name=collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="document_id" if collection_name == self.chunks_collection else "document_ids",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )
                # Note: Qdrant delete doesn't return count, so we track attempts
                deleted[collection_name] = 1
            except Exception as e:
                logger.warning(f"Delete from {collection_name} failed: {e}")
                deleted[collection_name] = 0

        return deleted

    def clear_all(self) -> bool:
        """Clear all collections (use with caution!)"""
        client = self._get_client()

        for name in [self.chunks_collection, self.entities_collection, self.relationships_collection]:
            try:
                client.delete_collection(name)
                logger.info(f"Deleted collection: {name}")
            except Exception as e:
                logger.warning(f"Failed to delete {name}: {e}")

        # Recreate
        self.ensure_collections()
        return True

    def close(self) -> None:
        """Close Qdrant client"""
        if self._client:
            self._client.close()
            self._client = None


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_index_manager: Optional[IndexManager] = None


def get_index_manager(
    host: Optional[str] = None,
    port: Optional[int] = None,
    **kwargs
) -> IndexManager:
    """Get or create global IndexManager"""
    global _index_manager

    if _index_manager is None:
        # Load from config
        try:
            from ...config.v2_config_loader import get_v2_config
            config = get_v2_config()
            qdrant_config = config.get_qdrant_config()

            _index_manager = IndexManager(
                host=host or qdrant_config.get("host", "qdrant"),
                port=port or qdrant_config.get("port", 6333),
                collection_prefix=qdrant_config.get("collections", {}).get("prefix", "mirage"),
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Failed to load Qdrant config: {e}")
            _index_manager = IndexManager(
                host=host or "qdrant",
                port=port or 6333,
                **kwargs
            )

    return _index_manager
