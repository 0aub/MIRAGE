"""
Embeddings Module
Phase 3: REFRAG Integration

Handles:
- Jina v3 embeddings (API and local)
- Semantic similarity computation
- Embedding caching
"""

from .jina_embedder import JinaEmbedder

__all__ = [
    "JinaEmbedder",
]
