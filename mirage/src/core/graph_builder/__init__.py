"""
Graph Builder Module
Phase 2: Dynamic Graph Construction + GraphRAG Components

Handles:
- Entity extraction (CAMeLTools for Arabic, spaCy for English)
- Relationship discovery (co-occurrence, dependency parsing)
- Neo4j graph storage
- Dynamic schema evolution

GraphRAG Components (Phase 1, 2 & 3):
- Entity normalization (prevent duplicates)
- Multi-hop graph traversal
- Community detection (Louvain algorithm)
- Community visualization
- Community summarization (Allam LLM)
"""

from .entity_extractor import EntityExtractor
from .relationship_extractor import RelationshipExtractor
from .neo4j_client import Neo4jClient
from .entity_normalizer import EntityNormalizer, normalize_entity
from .graph_traversal import GraphTraversal, TraversalResult
from .community_detector import CommunityDetector, Community, CommunityDetectionResult
from .community_visualizer import CommunityVisualizer, print_community_tree
from .community_summarizer import CommunitySummarizer, CommunitySummary

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
    # Phase 2: Community detection
    "CommunityDetector",
    "Community",
    "CommunityDetectionResult",
    "CommunityVisualizer",
    "print_community_tree",
    # Phase 3: Community summarization
    "CommunitySummarizer",
    "CommunitySummary",
]
