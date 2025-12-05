"""
MIRAGE V5 Retrieval Module
Unified retrieval engine with SOTA innovations for 10/10 performance.

MIRAGE V5 Features:
    - Dynamic Community Selection: O(log C) hierarchical pruning
    - Dual-Level Retrieval: LightRAG-style low + high level
    - HyDE Query Enhancement: Hypothetical document embeddings
    - Personalized PageRank: HippoRAG-style associative retrieval
    - Production Observability: Full tracing and metrics

Usage:
    # V5 Unified Engine (recommended)
    from core.retrieval import get_v5_engine

    engine = get_v5_engine()
    result = engine.retrieve("What is MIRAGE?")
    # result includes: chunks, metadata, trace

    # Legacy V2 Engine (still supported)
    from core.retrieval import get_retrieval_engine, RetrievalMode

    engine = get_retrieval_engine()
    response = engine.retrieve("What is MIRAGE?")

Retrieval Modes (V2):
    - NAIVE: Simple vector search
    - LOCAL: Entity-focused (entity → chunks)
    - GLOBAL: Relationship-focused (relationship → entity → chunks)
    - GLOBAL_SEARCH: Map-reduce over community summaries (GraphRAG)
    - HYBRID: Combines local + global
    - MIX: All modes with RRF fusion
    - SEMANTIC: Deep semantic matching with cross-encoder
    - BYPASS: No retrieval (for testing)

V5 Components:
    - DynamicCommunitySelector: Hierarchical community pruning
    - HyDEQueryEnhancer: Hypothetical document generation
    - DualLevelRetriever: LightRAG-style retrieval
    - HippocampalRetriever: PPR-based graph retrieval
    - RAGObservability: Full tracing and metrics
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

# MIRAGE V5: Dynamic Community Selection (O(log C) pruning)
from .community_selector import (
    DynamicCommunitySelector,
    CommunityNode,
    SelectionResult,
    SelectionStats,
    get_community_selector,
)

# MIRAGE V5: HyDE Query Enhancement
from .hyde import (
    HyDEQueryEnhancer,
    HyPEQueryEnhancer,
    EnhancedQuery,
    get_hyde_enhancer,
)

# MIRAGE V5: Dual-Level Retrieval (LightRAG-style)
from .dual_level_retrieval import (
    DualLevelRetriever,
    DualLevelResult,
    LowLevelResult,
    HighLevelResult,
    QueryTypeClassifier,
    get_dual_level_retriever,
)

# MIRAGE V5: Hippocampal Retrieval (Personalized PageRank)
from .hippocampal_retrieval import (
    HippocampalRetriever,
    HippocampalRetrievalResult,
    PersonalizedPageRank,
    PPRResult,
    get_hippocampal_retriever,
)

# MIRAGE V5: Production Observability
from .observability import (
    RAGObservability,
    QueryTrace,
    TraceStep,
    MetricsSnapshot,
    CostTracker,
    get_observability,
    get_cost_tracker,
)

# MIRAGE V5: Unified Engine (combines all SOTA innovations)
from .v5_engine import (
    MIRAGEV5Engine,
    V5Config,
    V5RetrievalResult,
    get_v5_engine,
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
    # MIRAGE V5: Dynamic Community Selection
    "DynamicCommunitySelector",
    "CommunityNode",
    "SelectionResult",
    "SelectionStats",
    "get_community_selector",
    # MIRAGE V5: HyDE Query Enhancement
    "HyDEQueryEnhancer",
    "HyPEQueryEnhancer",
    "EnhancedQuery",
    "get_hyde_enhancer",
    # MIRAGE V5: Dual-Level Retrieval
    "DualLevelRetriever",
    "DualLevelResult",
    "LowLevelResult",
    "HighLevelResult",
    "QueryTypeClassifier",
    "get_dual_level_retriever",
    # MIRAGE V5: Hippocampal Retrieval
    "HippocampalRetriever",
    "HippocampalRetrievalResult",
    "PersonalizedPageRank",
    "PPRResult",
    "get_hippocampal_retriever",
    # MIRAGE V5: Observability
    "RAGObservability",
    "QueryTrace",
    "TraceStep",
    "MetricsSnapshot",
    "CostTracker",
    "get_observability",
    "get_cost_tracker",
    # MIRAGE V5: Unified Engine
    "MIRAGEV5Engine",
    "V5Config",
    "V5RetrievalResult",
    "get_v5_engine",
]
