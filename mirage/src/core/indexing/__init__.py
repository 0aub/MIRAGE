"""
MIRAGE V2 Indexing Module
Provides triple-level indexing: chunks, entities, relationships.

Usage:
    from core.indexing import get_index_manager, IndexManager

    # Get index manager
    manager = get_index_manager()

    # Ensure collections exist
    manager.ensure_collections()

    # Index chunks
    from core.indexing import IndexedChunk
    chunk = IndexedChunk(
        chunk_id="chunk_001",
        document_id="doc_001",
        text="Sample text",
        embedding=embedding_vector
    )
    manager.index_chunk(chunk)

    # Search chunks
    results = manager.search_chunks(query_embedding, top_k=10)

    # Index entities (enables local search)
    from core.indexing import IndexedEntity
    entity = IndexedEntity(
        entity_id="ent_001",
        name="MIRAGE",
        entity_type="TECHNOLOGY",
        embedding=entity_embedding
    )
    manager.index_entity(entity)

    # Index relationships (enables global search)
    from core.indexing import IndexedRelationship
    rel = IndexedRelationship(
        relationship_id="rel_001",
        source_entity_id="ent_001",
        target_entity_id="ent_002",
        relationship_type="develops",
        embedding=rel_embedding
    )
    manager.index_relationship(rel)
"""

from .index_manager import (
    IndexManager,
    get_index_manager,
    IndexedChunk,
    IndexedEntity,
    IndexedRelationship,
    SearchResult,
)

__all__ = [
    "IndexManager",
    "get_index_manager",
    "IndexedChunk",
    "IndexedEntity",
    "IndexedRelationship",
    "SearchResult",
]
