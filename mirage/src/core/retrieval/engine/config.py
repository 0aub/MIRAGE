"""
Retrieval Engine Configuration

Configuration dataclass for the unified retrieval engine.
"""

from typing import Dict
from dataclasses import dataclass, field

from ..base_retriever import RetrievalMode


@dataclass
class RetrievalEngineConfig:
    """Configuration for the retrieval engine"""
    # Default mode when no routing
    # Updated 2024-12: GLOBAL is best performer (100%, avg 0.842)
    default_mode: RetrievalMode = RetrievalMode.GLOBAL

    # Top-k settings
    default_top_k: int = 10
    max_top_k: int = 50

    # Score thresholds
    min_score: float = 0.0

    # Fusion settings
    fusion_method: str = "rrf"  # "rrf", "weighted", "interleave"
    # Weights tuned based on evaluation results (2024-12):
    # global: 12/12 (0.842), local: 12/12 (0.833), hybrid: 11/12 (0.825), vector: 11/12 (0.817)
    mode_weights: Dict[str, float] = field(default_factory=lambda: {
        "vector": 0.7,
        "local": 0.95,
        "global": 1.0,    # Best performer - highest weight
        "hybrid": 0.85,
        "semantic": 0.9
    })

    # Auto-routing
    auto_route: bool = True

    # Fallback behavior
    fallback_on_error: bool = True
    fallback_mode: RetrievalMode = RetrievalMode.VECTOR

    # Metrics tracking (Phase 3)
    track_metrics: bool = True

    # ==========================================================================
    # Scoring constants (previously magic numbers)
    # ==========================================================================

    # Keyword search scoring
    keyword_match_score_vector: float = 0.88  # Score for keyword matches in vector mode
    keyword_match_score_local: float = 0.90   # Score for keyword matches in local mode
    keyword_boost_score: float = 0.92         # Boosted score for existing results with keyword match

    # Term boost (for results containing query terms)
    term_boost_factor: float = 0.05           # Boost per matching term
    term_boost_max: float = 0.15              # Maximum total boost from terms
    max_boosted_score: float = 0.98           # Maximum score after boosting

    # Entity/graph context scoring
    entity_chunk_base_score: float = 0.85     # Base score for entity-linked chunks
    disambiguated_chunk_score: float = 0.90   # Base score for disambiguated entity chunks
    entity_chunk_max_score: float = 0.95      # Maximum score for entity chunks
    graph_relationship_score: float = 0.85    # Score for graph relationship context
    community_context_score: float = 0.80     # Score for community summary context
    text_match_score: float = 0.85            # Score for text-based entity matching

    # ==========================================================================
    # Limit constants (previously magic numbers)
    # ==========================================================================

    # Keyword search limits
    max_entity_phrases: int = 5               # Max entity phrases to extract from query
    keyword_limit_per_variant: int = 3        # Max results per keyword variant
    keyword_batch_total_limit: int = 30       # Max total results from batch keyword search

    # Entity processing limits
    max_entities_to_process: int = 10         # Max entities to fetch chunks for
    chunks_per_entity: int = 2                # Chunks to fetch per entity
    max_disambiguate_terms: int = 8           # Max terms to disambiguate
    max_entity_search_results: int = 20       # Max entities from Neo4j search
    max_relationships: int = 15               # Max relationships to fetch
    max_communities: int = 5                  # Max communities to fetch
