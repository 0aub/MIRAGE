"""
Vector Store Module
Phase 5: Full Integration

Handles:
- Qdrant vector database integration
- Chunk embeddings storage
- Semantic search and retrieval
"""

from .qdrant_client import QdrantVectorStore

__all__ = [
    "QdrantVectorStore",
]
