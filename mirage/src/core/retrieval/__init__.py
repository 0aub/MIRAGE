"""
MIRAGE V2 Retrieval Module
Unified retrieval engine with 8 modes and automatic routing.

Usage:
    from core.retrieval import get_retrieval_engine, RetrievalMode

    # Get engine
    engine = get_retrieval_engine()

    # Auto-routed retrieval
    response = engine.retrieve("What is MIRAGE?")

    # Specific mode
    response = engine.retrieve("Compare X and Y", mode=RetrievalMode.MIX)

    # Global search (GraphRAG map-reduce)
    response = engine.retrieve("What are the main themes?", mode=RetrievalMode.GLOBAL_SEARCH)

    # Get explanation
    explanation = engine.explain_retrieval("Who is Ahmad?")

Modes:
    - NAIVE: Simple vector search
    - LOCAL: Entity-focused (entity → chunks)
    - GLOBAL: Relationship-focused (relationship → entity → chunks)
    - GLOBAL_SEARCH: Map-reduce over community summaries (GraphRAG)
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

# Backward compatibility with existing HybridRetriever (required, no fallback)
from .hybrid_retriever import HybridRetriever

# Cross-encoder reranker (required, no fallback)
from .reranker import (
    CrossEncoderReranker,
    RerankResult,
    get_reranker,
)

# Global search (GraphRAG map-reduce) (required, no fallback)
from .global_search import (
    GlobalSearchEngine,
    GlobalSearchResult,
    PartialAnswer,
    get_global_search_engine,
)

# Retrieval validator (CRAG-style self-correction) (required, no fallback)
from .validator import (
    RetrievalValidator,
    ValidatedResult,
    ValidationStatus,
    get_validator,
)

# Query processor (expansion, decomposition) (required, no fallback)
from .query_processor import (
    QueryProcessor,
    ProcessedQuery,
    QueryIntent,
    QueryComplexity,
    get_query_processor,
)

# Result diversifier (MMR) (required, no fallback)
from .diversifier import (
    ResultDiversifier,
    DiversifiedResult,
    get_diversifier,
)

# MIRAGE V4: Answer synthesizer (required, no fallback)
from .answer_synthesizer import (
    AnswerSynthesizer,
    SynthesizedAnswer,
    SynthesisSource,
    SynthesisStrategy,
    get_answer_synthesizer,
    synthesize_answer,
)

# MIRAGE V4: Summary validator (required, no fallback)
from .summary_validator import (
    SummaryValidator,
    ValidationResult as SummaryValidationResult,
    ValidationIssue,
    ValidationLevel,
    get_summary_validator,
    validate_summary,
)

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
    # Global Search (GraphRAG)
    "GlobalSearchEngine",
    "GlobalSearchResult",
    "PartialAnswer",
    "get_global_search_engine",
    # Validation (CRAG-style)
    "RetrievalValidator",
    "ValidatedResult",
    "ValidationStatus",
    "get_validator",
    # Query processing
    "QueryProcessor",
    "ProcessedQuery",
    "QueryIntent",
    "QueryComplexity",
    "get_query_processor",
    # Result diversification (MMR)
    "ResultDiversifier",
    "DiversifiedResult",
    "get_diversifier",
    # MIRAGE V4: Answer synthesis
    "AnswerSynthesizer",
    "SynthesizedAnswer",
    "SynthesisSource",
    "SynthesisStrategy",
    "get_answer_synthesizer",
    "synthesize_answer",
    # MIRAGE V4: Summary validation
    "SummaryValidator",
    "SummaryValidationResult",
    "ValidationIssue",
    "ValidationLevel",
    "get_summary_validator",
    "validate_summary",
    # Legacy
    "HybridRetriever",
]
