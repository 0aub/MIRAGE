"""
REFRAG Module - MIRAGE V4 Enhanced
Phase 3: True REFRAG Implementation

Two approaches available:

1. Text Compression (Legacy):
   - REFRAGCompressor: Token pruning, sentence extraction
   - CompressionPolicy: RL-based compression decisions
   - Achieves 2-3x compression

2. True REFRAG (New - Meta's Approach):
   - ChunkEmbedder: 16-token chunks → dense embeddings
   - REFRAGRetriever: Fast embedding-based retrieval
   - Achieves up to 30x speedup through:
     * Pre-computed chunk embeddings
     * O(N) embedding comparison vs O(N*T) attention
     * Selective chunk decoding

Usage (True REFRAG):
    from core.refrag import REFRAGRetriever, get_refrag_retriever

    retriever = get_refrag_retriever(embedding_model)
    retriever.index_document(text, document_id)
    result = retriever.retrieve(query)
    print(f"Speedup: {result.speedup_factor}x")
"""

# Legacy text compression
from .compressor import REFRAGCompressor
from .policy import CompressionPolicy
from .cache import CompressionCache

# True REFRAG (Meta's approach) - MIRAGE V4 (required, no fallback)
from .chunk_embedder import (
    ChunkEmbedder,
    REFRAGChunk,
    REFRAGDocument,
    get_chunk_embedder,
)
from .refrag_retriever import (
    REFRAGRetriever,
    REFRAGRetrievalResult,
    get_refrag_retriever,
)

# RL Expansion Policy (Meta's REINFORCE-based chunk selection) (required, no fallback)
from .rl_expansion_policy import (
    RLExpansionPolicy,
    ExpansionDecision,
    PolicyNetwork,
    get_rl_policy,
    train_expansion_policy,
)

__all__ = [
    # Legacy compression
    "REFRAGCompressor",
    "CompressionPolicy",
    "CompressionCache",
    # True REFRAG (MIRAGE V4)
    "ChunkEmbedder",
    "REFRAGChunk",
    "REFRAGDocument",
    "get_chunk_embedder",
    "REFRAGRetriever",
    "REFRAGRetrievalResult",
    "get_refrag_retriever",
    # RL Expansion Policy
    "RLExpansionPolicy",
    "ExpansionDecision",
    "PolicyNetwork",
    "get_rl_policy",
    "train_expansion_policy",
]
