"""
Graph Builder Module
Phase 2: Dynamic Graph Construction + GraphRAG Components

Handles:
- Entity extraction (CAMeLTools for Arabic, spaCy for English)
- Entity disambiguation (cross-encoder semantic matching)
- Relationship discovery (co-occurrence, dependency parsing)
- Neo4j graph storage
- Dynamic schema evolution

GraphRAG Components (COMPLETE - Phases 1-5):
- Entity normalization (prevent duplicates)
- Entity disambiguation (link query entities to graph)
- Multi-hop graph traversal
- Community detection (Louvain algorithm)
- Community visualization
- Community summarization (Allam LLM)
- Global search (map-reduce over summaries)
- Local search (entity-centric traversal)
- Hybrid search (global + local + routing)
"""

from .entity_extractor import EntityExtractor
from .relationship_extractor import RelationshipExtractor
from .neo4j_client import Neo4jClient
from .entity_normalizer import EntityNormalizer, normalize_entity

# Phase 2 Enhancement: Ensemble Entity Extraction (required, no fallback)
from .ensemble_extractor import (
    EnsembleEntityExtractor,
    ExtractedEntity,
    get_ensemble_extractor,
)

# MIRAGE V4: Entity Disambiguation (cross-encoder linking) (required, no fallback)
from .entity_disambiguator import (
    EntityDisambiguator,
    DisambiguationResult,
    get_entity_disambiguator,
)

from .graph_traversal import GraphTraversal, TraversalResult
from .community_detector import CommunityDetector, Community, CommunityDetectionResult
from .community_visualizer import CommunityVisualizer, print_community_tree
from .community_summarizer import CommunitySummarizer, CommunitySummary
from .global_search import GlobalSearchEngine, SearchResult
from .local_search import LocalSearchEngine, LocalSearchResult
from .hybrid_search import HybridSearchEngine, HybridSearchResult, QueryRouter
from .cooccurrence_extractor import CooccurrenceExtractor
from .semantic_similarity import SemanticSimilarityExtractor

# MIRAGE V4: Relationship type normalization (required, no fallback)
from .relationship_normalizer import (
    RelationshipTypeNormalizer,
    NormalizedRelationship,
    get_relationship_normalizer,
    normalize_relationship_type,
    to_cypher_relationship_type,
)

# MIRAGE V4: LLM-based relationship enrichment (required, no fallback)
from .relationship_enricher import (
    RelationshipEnricher,
    EnrichedRelationship,
    get_relationship_enricher,
    enrich_relationship,
)

# MIRAGE V5: Incremental Graph Updates
from .incremental_updater import (
    IncrementalGraphUpdater,
    UpdateResult,
    AffectedCommunity,
    get_incremental_updater,
)

# MIRAGE V5: Coreference Resolution
from .coreference_resolver import (
    CoreferenceResolver,
    ResolvedEntity,
    get_coreference_resolver,
)

__all__ = [
    # Original components
    "EntityExtractor",
    "RelationshipExtractor",
    "Neo4jClient",
    # Phase 1: Entity normalization and graph traversal
    "EntityNormalizer",
    "normalize_entity",
    "GraphTraversal",
    "TraversalResult",
    # Phase 2 Enhancement: Ensemble extraction
    "EnsembleEntityExtractor",
    "ExtractedEntity",
    "get_ensemble_extractor",
    # MIRAGE V4: Entity disambiguation
    "EntityDisambiguator",
    "DisambiguationResult",
    "get_entity_disambiguator",
    # Phase 2: Community detection
    "CommunityDetector",
    "Community",
    "CommunityDetectionResult",
    "CommunityVisualizer",
    "print_community_tree",
    # Phase 3: Community summarization
    "CommunitySummarizer",
    "CommunitySummary",
    # Phase 4: Global search (DEPRECATED - use core.retrieval)
    "GlobalSearchEngine",
    "SearchResult",
    # Phase 5: Local & hybrid search (DEPRECATED - use core.retrieval)
    "LocalSearchEngine",
    "LocalSearchResult",
    "HybridSearchEngine",
    "HybridSearchResult",
    "QueryRouter",
    # Relationship extractors
    "CooccurrenceExtractor",
    "SemanticSimilarityExtractor",
    # MIRAGE V4: Relationship normalization
    "RelationshipTypeNormalizer",
    "NormalizedRelationship",
    "get_relationship_normalizer",
    "normalize_relationship_type",
    "to_cypher_relationship_type",
    # MIRAGE V4: Relationship enrichment
    "RelationshipEnricher",
    "EnrichedRelationship",
    "get_relationship_enricher",
    "enrich_relationship",
    # MIRAGE V5: Incremental Graph Updates
    "IncrementalGraphUpdater",
    "UpdateResult",
    "AffectedCommunity",
    "get_incremental_updater",
    # MIRAGE V5: Coreference Resolution
    "CoreferenceResolver",
    "ResolvedEntity",
    "get_coreference_resolver",
]
