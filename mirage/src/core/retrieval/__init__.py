"""
MIRAGE V2 Retrieval Module
Unified retrieval engine with 7 modes and automatic routing.

Usage:
    from core.retrieval import get_retrieval_engine, RetrievalMode

    # Get engine
    engine = get_retrieval_engine()

    # Auto-routed retrieval
    response = engine.retrieve("What is MIRAGE?")

    # Specific mode
    response = engine.retrieve("Compare X and Y", mode=RetrievalMode.MIX)

    # Get explanation
    explanation = engine.explain_retrieval("Who is Ahmad?")

Modes:
    - NAIVE: Simple vector search
    - LOCAL: Entity-focused (entity → chunks)
    - GLOBAL: Relationship-focused (relationship → entity → chunks)
    - HYBRID: Combines local + global
    - MIX: All modes with RRF fusion
    - SEMANTIC: Deep semantic matching
    - BYPASS: No retrieval (for testing)
"""

from .base_retriever import (
    BaseRetriever,
    RetrievalMode,
    RetrievalResult,
    RetrievalResponse,
)

from .query_router import (
    QueryRouter,
    QueryType,
    RoutingDecision,
    get_query_router,
)

from .fusion import (
    ResultFusion,
    FusionConfig,
    combine_with_rrf,
    rrf_score,
)

from .retrieval_engine import (
    RetrievalEngine,
    RetrievalEngineConfig,
    get_retrieval_engine,
)

# Keep backward compatibility with existing HybridRetriever
try:
    from .hybrid_retriever import HybridRetriever
except ImportError:
    HybridRetriever = None

# Cross-encoder reranker
try:
    from .reranker import (
        CrossEncoderReranker,
        RerankResult,
        get_reranker,
    )
except ImportError:
    CrossEncoderReranker = None
    RerankResult = None
    get_reranker = None

__all__ = [
    # Base classes
    "BaseRetriever",
    "RetrievalMode",
    "RetrievalResult",
    "RetrievalResponse",
    # Query routing
    "QueryRouter",
    "QueryType",
    "RoutingDecision",
    "get_query_router",
    # Fusion
    "ResultFusion",
    "FusionConfig",
    "combine_with_rrf",
    "rrf_score",
    # Unified engine
    "RetrievalEngine",
    "RetrievalEngineConfig",
    "get_retrieval_engine",
    # Reranking
    "CrossEncoderReranker",
    "RerankResult",
    "get_reranker",
    # Legacy
    "HybridRetriever",
]
