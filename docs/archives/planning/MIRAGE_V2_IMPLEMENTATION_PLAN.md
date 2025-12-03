# MIRAGE v2: Comprehensive Implementation Plan

## Vision: A Research-Grade Hybrid RAG System

Build a RAG system that **exceeds LightRAG** in every dimension while maintaining Arabic support and enabling rigorous scientific evaluation.

---

## Executive Summary

| Dimension | LightRAG | MIRAGE v2 (Target) |
|-----------|----------|-------------------|
| Indexing | Dual-level (entity + relationship) | **Triple-level** (entity + relationship + chunk with cross-links) |
| Query Modes | 6 modes | **7 modes** (add "semantic" mode) |
| Multilingual | English only | **Arabic + English + extensible** |
| Chunking | Token-based only | **Multiple strategies** (token, semantic, hybrid) |
| Evaluation | None built-in | **Full experiment framework** |
| Versioning | None | **All parameters versioned** |
| Graph | Simple entity-relationship | **Enriched with communities + hierarchies** |
| Embeddings | Single model | **Multi-model support + evaluation** |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MIRAGE v2 ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CONFIGURATION LAYER                               │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │   │
│  │  │ Experiment  │ │   Prompt    │ │   Model     │ │  Retrieval  │   │   │
│  │  │   Config    │ │  Registry   │ │  Registry   │ │   Config    │   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    INGESTION PIPELINE                                │   │
│  │                                                                       │   │
│  │  Document → Chunker → Entity Extractor → Relationship Extractor     │   │
│  │     │         │              │                    │                  │   │
│  │     │    (configurable)  (configurable)     (configurable)          │   │
│  │     │         │              │                    │                  │   │
│  │     ▼         ▼              ▼                    ▼                  │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │              TRIPLE-LEVEL INDEX                              │    │   │
│  │  │  ┌───────────┐  ┌───────────────┐  ┌─────────────────────┐  │    │   │
│  │  │  │  Chunks   │  │   Entities    │  │   Relationships     │  │    │   │
│  │  │  │  (Qdrant) │  │   (Qdrant)    │  │     (Qdrant)        │  │    │   │
│  │  │  └───────────┘  └───────────────┘  └─────────────────────┘  │    │   │
│  │  │         │              │                    │                │    │   │
│  │  │         └──────────────┼────────────────────┘                │    │   │
│  │  │                        ▼                                     │    │   │
│  │  │              ┌─────────────────┐                             │    │   │
│  │  │              │   Knowledge     │                             │    │   │
│  │  │              │   Graph (Neo4j) │                             │    │   │
│  │  │              └─────────────────┘                             │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    RETRIEVAL ENGINE                                  │   │
│  │                                                                       │   │
│  │  Query → Router → Mode Selection → Multi-Index Search → Fusion      │   │
│  │                                                                       │   │
│  │  Modes: naive | local | global | hybrid | mix | semantic | bypass   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    GENERATION LAYER                                  │   │
│  │                                                                       │   │
│  │  Context Assembly → Prompt Template → LLM → Response + Citations    │   │
│  │                          │              │                            │   │
│  │                    (configurable)  (configurable)                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    EVALUATION FRAMEWORK                              │   │
│  │                                                                       │   │
│  │  Metrics Collection → Experiment Tracking → Comparison Reports      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Configuration Management System

### 1.1 Experiment Configuration

All parameters centralized in versioned YAML configs:

```yaml
# config/experiments/experiment_001.yaml
experiment:
  id: "exp_001"
  name: "Baseline LightRAG Comparison"
  version: "1.0.0"
  created_at: "2024-12-03"
  description: "Compare MIRAGE v2 against LightRAG baseline"

# Chunking Configuration
chunking:
  strategy: "token"  # Options: token | semantic | sentence | hybrid
  params:
    token:
      max_tokens: 600
      overlap_tokens: 50
      tokenizer: "cl100k_base"  # tiktoken encoding
    semantic:
      similarity_threshold: 0.7
      min_chunk_size: 200
      max_chunk_size: 1200
    hybrid:
      primary: "token"
      fallback: "semantic"
      switch_threshold: 0.5

# Embedding Configuration
embedding:
  model: "jina-embeddings-v3"  # Options: jina-v3 | jina-v4 | bge-m3 | multilingual-e5
  dimensions: 1024
  task_types:
    chunk: "retrieval.passage"
    query: "retrieval.query"
    entity: "classification"
    relationship: "classification"
  batch_size: 32
  normalize: true

# LLM Configuration
llm:
  provider: "tgi"  # Options: tgi | openai | anthropic | google | ollama
  model: "allam-7b"  # Model identifier
  params:
    temperature: 0.1
    max_tokens: 4096
    top_p: 0.95
  timeout: 60
  retry:
    max_attempts: 3
    backoff: exponential

# Entity Extraction Configuration
extraction:
  method: "llm"  # Options: llm | spacy | camel | hybrid
  llm_params:
    max_tokens_per_chunk: 600
    entity_types:
      - Person
      - Organization
      - Location
      - Event
      - Technology
      - Policy
      - Product
    relationship_types:
      - leads
      - manages
      - founded
      - located_in
      - works_for
      - collaborates_with
      - developed
      - participated_in
  quality_filters:
    min_entity_length: 2
    max_entity_length: 100
    blacklisted_entities:
      - "opportunity"
      - "sector"
      - "field"
      - "aspect"
    blacklisted_relationships:
      - "related_to"
      - "associated_with"
  gleaning:
    enabled: true
    max_retries: 2

# Retrieval Configuration
retrieval:
  default_mode: "hybrid"  # Options: naive | local | global | hybrid | mix | semantic | bypass
  modes:
    naive:
      top_k: 10
      score_threshold: 0.5
    local:
      entity_top_k: 5
      chunk_per_entity: 3
      graph_hops: 1
      score_threshold: 0.4
    global:
      relationship_top_k: 10
      entities_per_relationship: 2
      score_threshold: 0.3
    hybrid:
      local_weight: 0.5
      global_weight: 0.5
      fusion: "rrf"  # Options: rrf | weighted | max
    mix:
      naive_weight: 0.2
      local_weight: 0.4
      global_weight: 0.4
      fusion: "rrf"
    semantic:
      entity_similarity_threshold: 0.7
      relationship_similarity_threshold: 0.6
      use_graph_context: true
  reranking:
    enabled: true
    model: "cross-encoder"
    top_k: 5

# Graph Configuration
graph:
  entity_index:
    enabled: true
    collection: "mirage_entities"
  relationship_index:
    enabled: true
    collection: "mirage_relationships"
  community_detection:
    enabled: false  # Phase 2 feature
    algorithm: "leiden"
  traversal:
    max_hops: 2
    max_nodes_per_hop: 50
    relationship_decay: 0.7

# Generation Configuration
generation:
  prompt_template: "default_v1"  # References prompt registry
  context_window: 8192
  max_context_tokens: 6000
  citation_style: "inline"  # Options: inline | footnote | none
  language_detection: true
  response_language: "auto"  # Options: auto | en | ar

# Evaluation Configuration
evaluation:
  metrics:
    - retrieval_precision
    - retrieval_recall
    - retrieval_f1
    - answer_relevancy
    - answer_faithfulness
    - answer_completeness
    - latency_p50
    - latency_p95
    - token_usage
  ground_truth_dataset: "eval/ground_truth_v1.json"
  num_samples: 100
```

### 1.2 Directory Structure

```
mirage/
├── config/
│   ├── experiments/           # Versioned experiment configs
│   │   ├── exp_001_baseline.yaml
│   │   ├── exp_002_allam_vs_llama.yaml
│   │   └── exp_003_chunking_comparison.yaml
│   ├── prompts/               # Versioned prompt templates
│   │   ├── entity_extraction/
│   │   │   ├── v1_basic.yaml
│   │   │   ├── v2_detailed.yaml
│   │   │   └── v3_arabic_optimized.yaml
│   │   ├── relationship_extraction/
│   │   ├── query_analysis/
│   │   └── generation/
│   ├── models/                # Model registry
│   │   ├── llm_registry.yaml
│   │   └── embedding_registry.yaml
│   └── defaults.yaml          # Default configuration
├── src/
│   ├── config/                # Configuration management
│   │   ├── __init__.py
│   │   ├── experiment_config.py
│   │   ├── config_loader.py
│   │   └── config_validator.py
│   ├── core/
│   │   ├── chunking/          # Chunking strategies
│   │   │   ├── __init__.py
│   │   │   ├── base_chunker.py
│   │   │   ├── token_chunker.py
│   │   │   ├── semantic_chunker.py
│   │   │   ├── sentence_chunker.py
│   │   │   └── hybrid_chunker.py
│   │   ├── extraction/        # Entity/Relationship extraction
│   │   │   ├── __init__.py
│   │   │   ├── base_extractor.py
│   │   │   ├── llm_extractor.py
│   │   │   ├── spacy_extractor.py
│   │   │   ├── camel_extractor.py
│   │   │   └── hybrid_extractor.py
│   │   ├── indexing/          # Triple-level indexing
│   │   │   ├── __init__.py
│   │   │   ├── chunk_index.py
│   │   │   ├── entity_index.py
│   │   │   ├── relationship_index.py
│   │   │   └── index_manager.py
│   │   ├── retrieval/         # Multi-mode retrieval
│   │   │   ├── __init__.py
│   │   │   ├── query_router.py
│   │   │   ├── naive_retriever.py
│   │   │   ├── local_retriever.py
│   │   │   ├── global_retriever.py
│   │   │   ├── hybrid_retriever.py
│   │   │   ├── mix_retriever.py
│   │   │   ├── semantic_retriever.py
│   │   │   ├── fusion.py
│   │   │   └── reranker.py
│   │   ├── generation/        # Response generation
│   │   │   ├── __init__.py
│   │   │   ├── context_builder.py
│   │   │   ├── prompt_manager.py
│   │   │   └── generator.py
│   │   ├── llm/               # LLM abstraction
│   │   │   ├── __init__.py
│   │   │   ├── base_llm.py
│   │   │   ├── tgi_client.py
│   │   │   ├── openai_client.py
│   │   │   ├── anthropic_client.py
│   │   │   └── ollama_client.py
│   │   └── embeddings/        # Embedding abstraction
│   │       ├── __init__.py
│   │       ├── base_embedder.py
│   │       ├── jina_embedder.py
│   │       ├── bge_embedder.py
│   │       └── e5_embedder.py
│   └── evaluation/            # Evaluation framework
│       ├── __init__.py
│       ├── metrics/
│       │   ├── retrieval_metrics.py
│       │   ├── generation_metrics.py
│       │   └── latency_metrics.py
│       ├── datasets/
│       │   ├── ground_truth_loader.py
│       │   └── synthetic_generator.py
│       ├── experiment_runner.py
│       ├── comparison_report.py
│       └── visualization.py
└── eval/                      # Evaluation data
    ├── ground_truth/
    ├── results/
    └── reports/
```

---

## Part 2: Triple-Level Indexing

### 2.1 Index Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRIPLE-LEVEL INDEX                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  LEVEL 1: CHUNK INDEX (Qdrant: mirage_chunks)                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Vector: chunk_embedding (1024d)                              │   │
│  │ Payload: {                                                   │   │
│  │   chunk_id, document_id, text, char_count, token_count,     │   │
│  │   entity_ids[], position, metadata                          │   │
│  │ }                                                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              │ entity_ids (cross-reference)         │
│                              ▼                                      │
│  LEVEL 2: ENTITY INDEX (Qdrant: mirage_entities)                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Vector: entity_embedding (1024d)                             │   │
│  │ Payload: {                                                   │   │
│  │   entity_id, name, type, description, chunk_ids[],          │   │
│  │   document_ids[], confidence, attributes, aliases[]         │   │
│  │ }                                                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              │ relationship references              │
│                              ▼                                      │
│  LEVEL 3: RELATIONSHIP INDEX (Qdrant: mirage_relationships)        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Vector: relationship_embedding (1024d)                       │   │
│  │ Payload: {                                                   │   │
│  │   relationship_id, source_id, target_id, type, description, │   │
│  │   chunk_ids[], strength, confidence, attributes             │   │
│  │ }                                                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              │ All stored in graph                  │
│                              ▼                                      │
│  GRAPH LAYER (Neo4j)                                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ (:Entity)-[:RELATED_TO]->(:Entity)                          │   │
│  │ (:Entity)-[:COOCCURS_WITH]->(:Entity)                       │   │
│  │ (:Entity)-[:SIMILAR_TO]->(:Entity)                          │   │
│  │ (:Chunk)-[:MENTIONS]->(:Entity)                             │   │
│  │ (:Document)-[:CONTAINS]->(:Chunk)                           │   │
│  │ (:Community)-[:CONTAINS]->(:Entity)  [Phase 2]              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Index Manager Implementation

```python
# mirage/src/core/indexing/index_manager.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

@dataclass
class IndexConfig:
    """Configuration for an index"""
    collection_name: str
    vector_size: int
    distance_metric: str = "cosine"
    payload_schema: Dict = None

class IndexManager:
    """
    Manages triple-level indexing for MIRAGE v2.

    Provides unified interface for:
    - Chunk indexing (semantic search on text)
    - Entity indexing (search entities directly)
    - Relationship indexing (search relationships directly)
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.qdrant = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
        self.embedder = self._init_embedder(config.embedding)

        # Initialize collections
        self.chunk_index = ChunkIndex(self.qdrant, self.embedder, config)
        self.entity_index = EntityIndex(self.qdrant, self.embedder, config)
        self.relationship_index = RelationshipIndex(self.qdrant, self.embedder, config)

    async def index_document(self, document: Document) -> IndexingResult:
        """
        Full indexing pipeline for a document.

        1. Chunk the document
        2. Extract entities and relationships
        3. Generate embeddings for all components
        4. Store in triple-level index
        5. Update graph structure
        """
        # Step 1: Chunk
        chunker = self._get_chunker(self.config.chunking.strategy)
        chunks = chunker.chunk(document.text)

        # Step 2: Extract entities and relationships
        extractor = self._get_extractor(self.config.extraction.method)
        extraction_result = await extractor.extract(chunks)

        # Step 3: Generate embeddings
        chunk_embeddings = self.embedder.embed_chunks(chunks)
        entity_embeddings = self.embedder.embed_entities(extraction_result.entities)
        relationship_embeddings = self.embedder.embed_relationships(extraction_result.relationships)

        # Step 4: Index all levels
        chunk_ids = await self.chunk_index.index(chunks, chunk_embeddings, document.id)
        entity_ids = await self.entity_index.index(
            extraction_result.entities,
            entity_embeddings,
            chunk_ids  # Cross-reference
        )
        relationship_ids = await self.relationship_index.index(
            extraction_result.relationships,
            relationship_embeddings,
            entity_ids  # Cross-reference
        )

        # Step 5: Update graph
        await self._update_graph(
            document, chunks, chunk_ids,
            extraction_result.entities, entity_ids,
            extraction_result.relationships, relationship_ids
        )

        return IndexingResult(
            document_id=document.id,
            chunk_count=len(chunks),
            entity_count=len(extraction_result.entities),
            relationship_count=len(extraction_result.relationships)
        )
```

---

## Part 3: Seven Query Modes

### 3.1 Mode Definitions

| Mode | Description | When to Use | Components Used |
|------|-------------|-------------|-----------------|
| **naive** | Vector search on chunks only | Simple factual lookups | Chunk Index |
| **local** | Entity-focused with graph expansion | "Who is X?" / Specific entities | Entity Index + Graph |
| **global** | Relationship-focused across graph | "What are the themes?" / Abstract | Relationship Index + Graph |
| **hybrid** | Local + Global combined | General queries | Entity + Relationship Index |
| **mix** | All modes combined with fusion | Complex multi-faceted queries | All Indices |
| **semantic** | Deep semantic matching with context | Nuanced understanding needed | All Indices + Reranking |
| **bypass** | Direct LLM, no retrieval | Baseline comparison / Testing | None (LLM only) |

### 3.2 Query Router

```python
# mirage/src/core/retrieval/query_router.py

from enum import Enum
from typing import Optional
import re

class QueryMode(Enum):
    NAIVE = "naive"
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    MIX = "mix"
    SEMANTIC = "semantic"
    BYPASS = "bypass"

class QueryRouter:
    """
    Intelligent query routing to select optimal retrieval mode.

    Uses keyword patterns, query structure, and optionally LLM classification
    to route queries to the most appropriate retrieval mode.
    """

    # English patterns
    LOCAL_PATTERNS_EN = [
        r"\bwho\b", r"\bwhom\b", r"\bwhose\b",
        r"\bwhat is\b", r"\bwhat are\b",
        r"\bwhere\b", r"\bwhen\b",
        r"\bspecific\b", r"\bexactly\b",
        r"\bname of\b", r"\blist\b",
    ]

    GLOBAL_PATTERNS_EN = [
        r"\btheme\b", r"\bthemes\b",
        r"\boverview\b", r"\bsummary\b", r"\bsummarize\b",
        r"\bmain\b", r"\bgeneral\b", r"\boverall\b",
        r"\btrends?\b", r"\bpatterns?\b",
        r"\brelationship between\b", r"\bconnection\b",
        r"\bhow does .* relate\b", r"\bimpact\b",
    ]

    # Arabic patterns
    LOCAL_PATTERNS_AR = [
        r"\bمن\b", r"\bمن هو\b", r"\bمن هي\b",
        r"\bما هو\b", r"\bما هي\b",
        r"\bأين\b", r"\bمتى\b",
        r"\bاسم\b", r"\bقائمة\b",
    ]

    GLOBAL_PATTERNS_AR = [
        r"\bموضوع\b", r"\bمواضيع\b",
        r"\bملخص\b", r"\bلخص\b",
        r"\bعام\b", r"\bرئيسي\b",
        r"\bاتجاه\b", r"\bنمط\b",
        r"\bعلاقة\b", r"\bتأثير\b",
    ]

    def __init__(self, config: RetrievalConfig, use_llm_classification: bool = False):
        self.config = config
        self.use_llm = use_llm_classification
        self.llm_client = None  # Lazy initialization

    def route(self, query: str, explicit_mode: Optional[str] = None) -> QueryMode:
        """
        Route query to appropriate mode.

        Args:
            query: User query string
            explicit_mode: Override mode (if user specifies)

        Returns:
            QueryMode enum value
        """
        # Explicit override
        if explicit_mode:
            return QueryMode(explicit_mode)

        # Use LLM classification if enabled
        if self.use_llm:
            return self._llm_classify(query)

        # Rule-based classification
        return self._rule_based_classify(query)

    def _rule_based_classify(self, query: str) -> QueryMode:
        """Rule-based query classification"""
        query_lower = query.lower()

        # Score each mode
        local_score = sum(
            1 for pattern in self.LOCAL_PATTERNS_EN + self.LOCAL_PATTERNS_AR
            if re.search(pattern, query_lower)
        )

        global_score = sum(
            1 for pattern in self.GLOBAL_PATTERNS_EN + self.GLOBAL_PATTERNS_AR
            if re.search(pattern, query_lower)
        )

        # Decision logic
        if local_score > 0 and global_score == 0:
            return QueryMode.LOCAL
        elif global_score > 0 and local_score == 0:
            return QueryMode.GLOBAL
        elif local_score > 0 and global_score > 0:
            return QueryMode.MIX  # Complex query needing both
        elif self._is_complex_query(query):
            return QueryMode.SEMANTIC
        else:
            return QueryMode.HYBRID  # Default

    def _is_complex_query(self, query: str) -> bool:
        """Detect complex queries needing semantic mode"""
        # Long queries
        if len(query.split()) > 15:
            return True
        # Multiple question marks
        if query.count("?") > 1:
            return True
        # Contains comparative language
        if re.search(r"\b(compare|versus|vs|difference|similar)\b", query.lower()):
            return True
        return False

    def _llm_classify(self, query: str) -> QueryMode:
        """Use LLM to classify query type"""
        # Implementation for LLM-based classification
        pass
```

### 3.3 Multi-Mode Retriever

```python
# mirage/src/core/retrieval/multimode_retriever.py

from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod

@dataclass
class RetrievalResult:
    """Result from retrieval operation"""
    chunks: List[Dict]
    entities: List[Dict]
    relationships: List[Dict]
    mode: str
    scores: Dict[str, float]
    latency_ms: int
    metadata: Dict

class BaseRetriever(ABC):
    """Base class for all retrievers"""

    @abstractmethod
    async def retrieve(self, query: str, config: Dict) -> RetrievalResult:
        pass

class MultiModeRetriever:
    """
    Orchestrates retrieval across all modes.

    This is the main entry point for retrieval in MIRAGE v2.
    It routes queries to appropriate mode(s) and handles fusion.
    """

    def __init__(
        self,
        chunk_index: ChunkIndex,
        entity_index: EntityIndex,
        relationship_index: RelationshipIndex,
        graph_client: Neo4jClient,
        config: RetrievalConfig
    ):
        self.chunk_index = chunk_index
        self.entity_index = entity_index
        self.relationship_index = relationship_index
        self.graph = graph_client
        self.config = config

        # Initialize mode-specific retrievers
        self.retrievers = {
            QueryMode.NAIVE: NaiveRetriever(chunk_index, config),
            QueryMode.LOCAL: LocalRetriever(entity_index, graph_client, chunk_index, config),
            QueryMode.GLOBAL: GlobalRetriever(relationship_index, graph_client, chunk_index, config),
            QueryMode.HYBRID: HybridRetriever(entity_index, relationship_index, graph_client, chunk_index, config),
            QueryMode.MIX: MixRetriever(self, config),  # Uses all other retrievers
            QueryMode.SEMANTIC: SemanticRetriever(entity_index, relationship_index, graph_client, chunk_index, config),
            QueryMode.BYPASS: BypassRetriever(),
        }

        self.router = QueryRouter(config)
        self.fusion = ResultFusion(config)
        self.reranker = Reranker(config) if config.reranking.enabled else None

    async def retrieve(
        self,
        query: str,
        mode: Optional[str] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Main retrieval entry point.

        Args:
            query: User query
            mode: Explicit mode override (optional)
            **kwargs: Additional retrieval parameters

        Returns:
            RetrievalResult with chunks, entities, relationships
        """
        import time
        start_time = time.time()

        # Route query
        selected_mode = self.router.route(query, mode)

        # Get retriever
        retriever = self.retrievers[selected_mode]

        # Execute retrieval
        result = await retriever.retrieve(query, self.config.modes[selected_mode.value])

        # Rerank if enabled
        if self.reranker and selected_mode != QueryMode.BYPASS:
            result = await self.reranker.rerank(query, result)

        # Add metadata
        result.latency_ms = int((time.time() - start_time) * 1000)
        result.mode = selected_mode.value

        return result


class NaiveRetriever(BaseRetriever):
    """
    Naive mode: Vector search on chunks only.

    Equivalent to traditional RAG without graph.
    Useful as baseline and for simple queries.
    """

    def __init__(self, chunk_index: ChunkIndex, config: RetrievalConfig):
        self.chunk_index = chunk_index
        self.config = config

    async def retrieve(self, query: str, mode_config: Dict) -> RetrievalResult:
        # Direct vector search
        chunks = await self.chunk_index.search(
            query,
            limit=mode_config.get("top_k", 10),
            score_threshold=mode_config.get("score_threshold", 0.5)
        )

        return RetrievalResult(
            chunks=chunks,
            entities=[],
            relationships=[],
            mode="naive",
            scores={"vector": sum(c["score"] for c in chunks) / len(chunks) if chunks else 0},
            latency_ms=0,
            metadata={"search_type": "vector_only"}
        )


class LocalRetriever(BaseRetriever):
    """
    Local mode: Entity-focused retrieval with graph expansion.

    1. Search entity index for relevant entities
    2. Get chunks that mention these entities
    3. Expand via 1-hop graph traversal
    4. Return enriched context
    """

    def __init__(self, entity_index, graph_client, chunk_index, config):
        self.entity_index = entity_index
        self.graph = graph_client
        self.chunk_index = chunk_index
        self.config = config

    async def retrieve(self, query: str, mode_config: Dict) -> RetrievalResult:
        # Step 1: Search entities
        entities = await self.entity_index.search(
            query,
            limit=mode_config.get("entity_top_k", 5),
            score_threshold=mode_config.get("score_threshold", 0.4)
        )

        if not entities:
            # Fallback to naive if no entities found
            return await NaiveRetriever(self.chunk_index, self.config).retrieve(query, mode_config)

        # Step 2: Get chunks mentioning these entities
        entity_names = [e["name"] for e in entities]
        chunks = await self._get_entity_chunks(
            entity_names,
            limit_per_entity=mode_config.get("chunk_per_entity", 3)
        )

        # Step 3: Graph expansion
        expanded_entities = await self._expand_via_graph(
            entity_names,
            hops=mode_config.get("graph_hops", 1)
        )

        # Add expanded entity chunks
        if expanded_entities:
            expanded_chunks = await self._get_entity_chunks(
                [e["name"] for e in expanded_entities],
                limit_per_entity=2
            )
            # Apply decay to expanded chunks
            decay = self.config.graph.traversal.relationship_decay
            for chunk in expanded_chunks:
                chunk["score"] *= decay
            chunks.extend(expanded_chunks)

        # Deduplicate and sort
        chunks = self._deduplicate_chunks(chunks)

        return RetrievalResult(
            chunks=chunks,
            entities=entities + expanded_entities,
            relationships=[],
            mode="local",
            scores={
                "entity_relevance": sum(e["score"] for e in entities) / len(entities),
                "chunk_relevance": sum(c["score"] for c in chunks) / len(chunks) if chunks else 0
            },
            latency_ms=0,
            metadata={"expansion_hops": mode_config.get("graph_hops", 1)}
        )

    async def _get_entity_chunks(self, entity_names: List[str], limit_per_entity: int) -> List[Dict]:
        """Get chunks that mention specific entities via MENTIONS relationship"""
        query = """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE e.name IN $entity_names
        RETURN c.id as chunk_id, c.text as text, e.name as entity, 1.0 as score
        LIMIT $limit
        """
        # Execute query and return chunks
        pass

    async def _expand_via_graph(self, entity_names: List[str], hops: int) -> List[Dict]:
        """Expand to related entities via graph traversal"""
        query = f"""
        MATCH (e:Entity)-[r*1..{hops}]-(related:Entity)
        WHERE e.name IN $entity_names
        AND NOT related.name IN $entity_names
        RETURN DISTINCT related.name as name, related.type as type,
               related.description as description,
               min(length(r)) as distance
        ORDER BY distance
        LIMIT 20
        """
        # Execute and apply decay based on distance
        pass


class GlobalRetriever(BaseRetriever):
    """
    Global mode: Relationship-focused retrieval.

    Key innovation from LightRAG: Search relationships directly.

    1. Search relationship index for relevant relationships
    2. Extract entities from matched relationships
    3. Get chunks for relationship context
    4. Return broad thematic context
    """

    def __init__(self, relationship_index, graph_client, chunk_index, config):
        self.relationship_index = relationship_index
        self.graph = graph_client
        self.chunk_index = chunk_index
        self.config = config

    async def retrieve(self, query: str, mode_config: Dict) -> RetrievalResult:
        # Step 1: Search relationships directly
        relationships = await self.relationship_index.search(
            query,
            limit=mode_config.get("relationship_top_k", 10),
            score_threshold=mode_config.get("score_threshold", 0.3)
        )

        if not relationships:
            return await NaiveRetriever(self.chunk_index, self.config).retrieve(query, mode_config)

        # Step 2: Extract entities from relationships
        entities = set()
        for rel in relationships:
            entities.add(rel["source_id"])
            entities.add(rel["target_id"])

        entity_details = await self._get_entity_details(list(entities))

        # Step 3: Get chunks from relationship context
        relationship_chunk_ids = set()
        for rel in relationships:
            relationship_chunk_ids.update(rel.get("chunk_ids", []))

        chunks = await self._get_chunks_by_ids(list(relationship_chunk_ids))

        return RetrievalResult(
            chunks=chunks,
            entities=entity_details,
            relationships=relationships,
            mode="global",
            scores={
                "relationship_relevance": sum(r["score"] for r in relationships) / len(relationships),
                "coverage": len(entities) / 10  # Normalized by expected entity count
            },
            latency_ms=0,
            metadata={"relationships_found": len(relationships)}
        )


class HybridRetriever(BaseRetriever):
    """
    Hybrid mode: Combines Local and Global.

    Executes both strategies and fuses results.
    """

    def __init__(self, entity_index, relationship_index, graph_client, chunk_index, config):
        self.local = LocalRetriever(entity_index, graph_client, chunk_index, config)
        self.global_ = GlobalRetriever(relationship_index, graph_client, chunk_index, config)
        self.config = config
        self.fusion = ResultFusion(config)

    async def retrieve(self, query: str, mode_config: Dict) -> RetrievalResult:
        import asyncio

        # Execute local and global in parallel
        local_result, global_result = await asyncio.gather(
            self.local.retrieve(query, mode_config),
            self.global_.retrieve(query, mode_config)
        )

        # Fuse results
        fused = self.fusion.fuse(
            [local_result, global_result],
            weights=[mode_config.get("local_weight", 0.5), mode_config.get("global_weight", 0.5)],
            method=mode_config.get("fusion", "rrf")
        )

        fused.mode = "hybrid"
        return fused


class MixRetriever(BaseRetriever):
    """
    Mix mode: Combines ALL retrieval strategies.

    This is the most comprehensive mode:
    - Naive (vector on chunks)
    - Local (entity-focused)
    - Global (relationship-focused)

    Uses sophisticated fusion (RRF or learned weights).
    """

    def __init__(self, multi_retriever: MultiModeRetriever, config: RetrievalConfig):
        self.multi_retriever = multi_retriever
        self.config = config
        self.fusion = ResultFusion(config)

    async def retrieve(self, query: str, mode_config: Dict) -> RetrievalResult:
        import asyncio

        # Execute all three base modes in parallel
        naive_result, local_result, global_result = await asyncio.gather(
            self.multi_retriever.retrievers[QueryMode.NAIVE].retrieve(
                query, self.config.modes["naive"]
            ),
            self.multi_retriever.retrievers[QueryMode.LOCAL].retrieve(
                query, self.config.modes["local"]
            ),
            self.multi_retriever.retrievers[QueryMode.GLOBAL].retrieve(
                query, self.config.modes["global"]
            )
        )

        # Fuse with configured weights
        fused = self.fusion.fuse(
            [naive_result, local_result, global_result],
            weights=[
                mode_config.get("naive_weight", 0.2),
                mode_config.get("local_weight", 0.4),
                mode_config.get("global_weight", 0.4)
            ],
            method=mode_config.get("fusion", "rrf")
        )

        fused.mode = "mix"
        fused.metadata["component_scores"] = {
            "naive": naive_result.scores,
            "local": local_result.scores,
            "global": global_result.scores
        }

        return fused


class SemanticRetriever(BaseRetriever):
    """
    Semantic mode: Deep semantic understanding with context.

    Beyond LightRAG - adds:
    - Cross-encoder reranking
    - Semantic similarity between query and entity descriptions
    - Relationship chain reasoning
    """

    async def retrieve(self, query: str, mode_config: Dict) -> RetrievalResult:
        # Similar to hybrid but with:
        # 1. Lower thresholds for initial retrieval
        # 2. More aggressive graph expansion
        # 3. Mandatory reranking
        # 4. Entity description matching
        pass


class BypassRetriever(BaseRetriever):
    """
    Bypass mode: No retrieval, direct to LLM.

    Use cases:
    - Baseline comparison (LLM without RAG)
    - Testing LLM capabilities
    - Questions that don't need retrieval
    """

    async def retrieve(self, query: str, mode_config: Dict) -> RetrievalResult:
        return RetrievalResult(
            chunks=[],
            entities=[],
            relationships=[],
            mode="bypass",
            scores={},
            latency_ms=0,
            metadata={"reason": "Direct LLM query, no retrieval"}
        )
```

### 3.4 Result Fusion

```python
# mirage/src/core/retrieval/fusion.py

from typing import List, Dict
from dataclasses import dataclass

class ResultFusion:
    """
    Fuses results from multiple retrieval modes.

    Supports multiple fusion strategies:
    - RRF (Reciprocal Rank Fusion): Best for combining ranked lists
    - Weighted: Simple weighted combination
    - Max: Takes max score for each item
    - Learned: Uses trained model to weight results (future)
    """

    def __init__(self, config: RetrievalConfig):
        self.config = config

    def fuse(
        self,
        results: List[RetrievalResult],
        weights: List[float],
        method: str = "rrf"
    ) -> RetrievalResult:
        """
        Fuse multiple retrieval results.

        Args:
            results: List of RetrievalResult from different modes
            weights: Weight for each result (should sum to 1)
            method: Fusion method (rrf | weighted | max)

        Returns:
            Fused RetrievalResult
        """
        if method == "rrf":
            return self._rrf_fusion(results, weights)
        elif method == "weighted":
            return self._weighted_fusion(results, weights)
        elif method == "max":
            return self._max_fusion(results)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def _rrf_fusion(self, results: List[RetrievalResult], weights: List[float]) -> RetrievalResult:
        """
        Reciprocal Rank Fusion.

        RRF score = sum(weight_i / (k + rank_i)) for each result set
        where k = 60 (standard constant)
        """
        k = 60
        chunk_scores = {}
        entity_scores = {}
        relationship_scores = {}

        for result, weight in zip(results, weights):
            # Score chunks
            for rank, chunk in enumerate(result.chunks, 1):
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {"chunk": chunk, "score": 0}
                chunk_scores[chunk_id]["score"] += weight / (k + rank)

            # Score entities
            for rank, entity in enumerate(result.entities, 1):
                entity_id = entity.get("entity_id") or entity.get("name")
                if entity_id not in entity_scores:
                    entity_scores[entity_id] = {"entity": entity, "score": 0}
                entity_scores[entity_id]["score"] += weight / (k + rank)

            # Score relationships
            for rank, rel in enumerate(result.relationships, 1):
                rel_id = rel.get("relationship_id") or f"{rel['source_id']}_{rel['target_id']}"
                if rel_id not in relationship_scores:
                    relationship_scores[rel_id] = {"relationship": rel, "score": 0}
                relationship_scores[rel_id]["score"] += weight / (k + rank)

        # Sort by fused score
        fused_chunks = sorted(
            [{"score": v["score"], **v["chunk"]} for v in chunk_scores.values()],
            key=lambda x: x["score"],
            reverse=True
        )

        fused_entities = sorted(
            [{"score": v["score"], **v["entity"]} for v in entity_scores.values()],
            key=lambda x: x["score"],
            reverse=True
        )

        fused_relationships = sorted(
            [{"score": v["score"], **v["relationship"]} for v in relationship_scores.values()],
            key=lambda x: x["score"],
            reverse=True
        )

        return RetrievalResult(
            chunks=fused_chunks[:self.config.modes["mix"].get("max_chunks", 10)],
            entities=fused_entities[:20],
            relationships=fused_relationships[:15],
            mode="fused",
            scores={"rrf_fusion": True},
            latency_ms=0,
            metadata={"fusion_method": "rrf", "weights": weights}
        )
```

---

## Part 4: Evaluation Framework

### 4.1 Metrics Definition

```python
# mirage/src/evaluation/metrics/definitions.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class MetricType(Enum):
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    LATENCY = "latency"
    EFFICIENCY = "efficiency"

@dataclass
class MetricDefinition:
    name: str
    type: MetricType
    description: str
    higher_is_better: bool
    range: tuple  # (min, max)

# All metrics we track
METRICS = {
    # Retrieval Metrics
    "retrieval_precision": MetricDefinition(
        name="Retrieval Precision",
        type=MetricType.RETRIEVAL,
        description="Fraction of retrieved chunks that are relevant",
        higher_is_better=True,
        range=(0, 1)
    ),
    "retrieval_recall": MetricDefinition(
        name="Retrieval Recall",
        type=MetricType.RETRIEVAL,
        description="Fraction of relevant chunks that were retrieved",
        higher_is_better=True,
        range=(0, 1)
    ),
    "retrieval_f1": MetricDefinition(
        name="Retrieval F1",
        type=MetricType.RETRIEVAL,
        description="Harmonic mean of precision and recall",
        higher_is_better=True,
        range=(0, 1)
    ),
    "retrieval_mrr": MetricDefinition(
        name="Mean Reciprocal Rank",
        type=MetricType.RETRIEVAL,
        description="Average reciprocal rank of first relevant result",
        higher_is_better=True,
        range=(0, 1)
    ),
    "retrieval_ndcg": MetricDefinition(
        name="NDCG@k",
        type=MetricType.RETRIEVAL,
        description="Normalized Discounted Cumulative Gain",
        higher_is_better=True,
        range=(0, 1)
    ),
    "entity_coverage": MetricDefinition(
        name="Entity Coverage",
        type=MetricType.RETRIEVAL,
        description="Fraction of ground truth entities found",
        higher_is_better=True,
        range=(0, 1)
    ),
    "relationship_coverage": MetricDefinition(
        name="Relationship Coverage",
        type=MetricType.RETRIEVAL,
        description="Fraction of ground truth relationships found",
        higher_is_better=True,
        range=(0, 1)
    ),

    # Generation Metrics
    "answer_relevancy": MetricDefinition(
        name="Answer Relevancy",
        type=MetricType.GENERATION,
        description="How relevant the answer is to the question (LLM-judged)",
        higher_is_better=True,
        range=(0, 1)
    ),
    "answer_faithfulness": MetricDefinition(
        name="Answer Faithfulness",
        type=MetricType.GENERATION,
        description="Whether answer is grounded in retrieved context",
        higher_is_better=True,
        range=(0, 1)
    ),
    "answer_completeness": MetricDefinition(
        name="Answer Completeness",
        type=MetricType.GENERATION,
        description="Whether answer covers all aspects of the question",
        higher_is_better=True,
        range=(0, 1)
    ),
    "hallucination_rate": MetricDefinition(
        name="Hallucination Rate",
        type=MetricType.GENERATION,
        description="Fraction of claims not supported by context",
        higher_is_better=False,
        range=(0, 1)
    ),
    "citation_accuracy": MetricDefinition(
        name="Citation Accuracy",
        type=MetricType.GENERATION,
        description="Accuracy of source citations",
        higher_is_better=True,
        range=(0, 1)
    ),

    # Latency Metrics
    "latency_p50": MetricDefinition(
        name="Latency P50",
        type=MetricType.LATENCY,
        description="50th percentile response time (ms)",
        higher_is_better=False,
        range=(0, float('inf'))
    ),
    "latency_p95": MetricDefinition(
        name="Latency P95",
        type=MetricType.LATENCY,
        description="95th percentile response time (ms)",
        higher_is_better=False,
        range=(0, float('inf'))
    ),
    "latency_p99": MetricDefinition(
        name="Latency P99",
        type=MetricType.LATENCY,
        description="99th percentile response time (ms)",
        higher_is_better=False,
        range=(0, float('inf'))
    ),

    # Efficiency Metrics
    "token_usage_retrieval": MetricDefinition(
        name="Retrieval Token Usage",
        type=MetricType.EFFICIENCY,
        description="Tokens used in retrieval phase",
        higher_is_better=False,
        range=(0, float('inf'))
    ),
    "token_usage_generation": MetricDefinition(
        name="Generation Token Usage",
        type=MetricType.EFFICIENCY,
        description="Tokens used in generation phase",
        higher_is_better=False,
        range=(0, float('inf'))
    ),
    "cost_per_query": MetricDefinition(
        name="Cost per Query",
        type=MetricType.EFFICIENCY,
        description="Estimated cost in USD per query",
        higher_is_better=False,
        range=(0, float('inf'))
    ),
}
```

### 4.2 Experiment Runner

```python
# mirage/src/evaluation/experiment_runner.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json
import hashlib

@dataclass
class ExperimentResult:
    """Result of a single experiment run"""
    experiment_id: str
    config_hash: str
    timestamp: datetime
    metrics: Dict[str, float]
    per_query_results: List[Dict]
    config: Dict
    duration_seconds: float

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "per_query_results": self.per_query_results,
            "config": self.config,
            "duration_seconds": self.duration_seconds
        }

class ExperimentRunner:
    """
    Runs experiments with different configurations and collects metrics.

    Features:
    - Versioned configurations
    - Reproducible results (config hash)
    - Parallel evaluation
    - Automatic result storage
    - Comparison reports
    """

    def __init__(self, base_config_path: str):
        self.base_config = self._load_config(base_config_path)
        self.results_dir = Path("eval/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _compute_config_hash(self, config: Dict) -> str:
        """Compute deterministic hash of config for reproducibility"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    async def run_experiment(
        self,
        experiment_config: ExperimentConfig,
        dataset: EvaluationDataset,
        experiment_id: Optional[str] = None
    ) -> ExperimentResult:
        """
        Run a complete experiment.

        Args:
            experiment_config: Full experiment configuration
            dataset: Ground truth dataset for evaluation
            experiment_id: Optional ID (auto-generated if not provided)

        Returns:
            ExperimentResult with all metrics
        """
        import time
        start_time = time.time()

        # Generate experiment ID
        if not experiment_id:
            config_hash = self._compute_config_hash(experiment_config.to_dict())
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config_hash[:8]}"

        # Initialize system with this config
        system = RAGSystem(experiment_config)

        # Run evaluation on all queries
        per_query_results = []
        all_retrieval_metrics = []
        all_generation_metrics = []
        all_latencies = []

        for query_item in dataset.queries:
            # Run retrieval
            retrieval_result = await system.retrieve(
                query_item.query,
                mode=experiment_config.retrieval.default_mode
            )

            # Compute retrieval metrics
            retrieval_metrics = self._compute_retrieval_metrics(
                retrieval_result,
                query_item.ground_truth
            )

            # Run generation
            generation_result = await system.generate(
                query_item.query,
                retrieval_result
            )

            # Compute generation metrics
            generation_metrics = await self._compute_generation_metrics(
                generation_result,
                query_item.expected_answer,
                retrieval_result
            )

            # Store per-query result
            per_query_results.append({
                "query_id": query_item.id,
                "query": query_item.query,
                "retrieval_metrics": retrieval_metrics,
                "generation_metrics": generation_metrics,
                "latency_ms": retrieval_result.latency_ms + generation_result.latency_ms,
                "mode": retrieval_result.mode
            })

            all_retrieval_metrics.append(retrieval_metrics)
            all_generation_metrics.append(generation_metrics)
            all_latencies.append(retrieval_result.latency_ms + generation_result.latency_ms)

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(
            all_retrieval_metrics,
            all_generation_metrics,
            all_latencies
        )

        duration = time.time() - start_time

        result = ExperimentResult(
            experiment_id=experiment_id,
            config_hash=self._compute_config_hash(experiment_config.to_dict()),
            timestamp=datetime.now(),
            metrics=aggregated_metrics,
            per_query_results=per_query_results,
            config=experiment_config.to_dict(),
            duration_seconds=duration
        )

        # Save result
        self._save_result(result)

        return result

    def _compute_retrieval_metrics(
        self,
        result: RetrievalResult,
        ground_truth: GroundTruth
    ) -> Dict[str, float]:
        """Compute retrieval metrics against ground truth"""
        retrieved_chunk_ids = {c["chunk_id"] for c in result.chunks}
        relevant_chunk_ids = set(ground_truth.relevant_chunk_ids)

        # Precision, Recall, F1
        if retrieved_chunk_ids:
            precision = len(retrieved_chunk_ids & relevant_chunk_ids) / len(retrieved_chunk_ids)
        else:
            precision = 0

        if relevant_chunk_ids:
            recall = len(retrieved_chunk_ids & relevant_chunk_ids) / len(relevant_chunk_ids)
        else:
            recall = 1.0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # MRR
        mrr = 0
        for rank, chunk in enumerate(result.chunks, 1):
            if chunk["chunk_id"] in relevant_chunk_ids:
                mrr = 1 / rank
                break

        # Entity coverage
        retrieved_entities = {e["name"] for e in result.entities}
        relevant_entities = set(ground_truth.relevant_entities)
        entity_coverage = len(retrieved_entities & relevant_entities) / len(relevant_entities) if relevant_entities else 1.0

        # Relationship coverage
        retrieved_rels = {(r["source_id"], r["target_id"]) for r in result.relationships}
        relevant_rels = {(r["source"], r["target"]) for r in ground_truth.relevant_relationships}
        rel_coverage = len(retrieved_rels & relevant_rels) / len(relevant_rels) if relevant_rels else 1.0

        return {
            "retrieval_precision": precision,
            "retrieval_recall": recall,
            "retrieval_f1": f1,
            "retrieval_mrr": mrr,
            "entity_coverage": entity_coverage,
            "relationship_coverage": rel_coverage
        }

    async def _compute_generation_metrics(
        self,
        generation_result: GenerationResult,
        expected_answer: str,
        retrieval_result: RetrievalResult
    ) -> Dict[str, float]:
        """Compute generation metrics using LLM-as-judge"""
        # Use a judge LLM to evaluate
        judge = LLMJudge(model="gpt-4o")

        relevancy = await judge.evaluate_relevancy(
            generation_result.query,
            generation_result.answer
        )

        faithfulness = await judge.evaluate_faithfulness(
            generation_result.answer,
            retrieval_result.chunks
        )

        completeness = await judge.evaluate_completeness(
            generation_result.query,
            generation_result.answer,
            expected_answer
        )

        hallucination_rate = await judge.detect_hallucinations(
            generation_result.answer,
            retrieval_result.chunks
        )

        return {
            "answer_relevancy": relevancy,
            "answer_faithfulness": faithfulness,
            "answer_completeness": completeness,
            "hallucination_rate": hallucination_rate
        }

    def _aggregate_metrics(
        self,
        retrieval_metrics: List[Dict],
        generation_metrics: List[Dict],
        latencies: List[float]
    ) -> Dict[str, float]:
        """Aggregate per-query metrics into summary statistics"""
        import numpy as np

        aggregated = {}

        # Retrieval metrics - mean
        for key in retrieval_metrics[0].keys():
            values = [m[key] for m in retrieval_metrics]
            aggregated[key] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)

        # Generation metrics - mean
        for key in generation_metrics[0].keys():
            values = [m[key] for m in generation_metrics]
            aggregated[key] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)

        # Latency percentiles
        aggregated["latency_p50"] = np.percentile(latencies, 50)
        aggregated["latency_p95"] = np.percentile(latencies, 95)
        aggregated["latency_p99"] = np.percentile(latencies, 99)
        aggregated["latency_mean"] = np.mean(latencies)

        return aggregated

    def _save_result(self, result: ExperimentResult):
        """Save experiment result to disk"""
        result_path = self.results_dir / f"{result.experiment_id}.json"
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
```

### 4.3 Comparison Report Generator

```python
# mirage/src/evaluation/comparison_report.py

from typing import List, Dict
from dataclasses import dataclass
import pandas as pd

@dataclass
class ComparisonReport:
    """Comparison report between multiple experiments"""
    experiments: List[str]  # Experiment IDs
    metrics_comparison: pd.DataFrame
    best_config: Dict
    recommendations: List[str]

class ReportGenerator:
    """
    Generates comparison reports across experiments.

    Features:
    - Side-by-side metric comparison
    - Statistical significance testing
    - Best configuration identification
    - Visualization generation
    """

    def generate_comparison(
        self,
        experiment_ids: List[str],
        focus_metrics: List[str] = None
    ) -> ComparisonReport:
        """
        Generate comparison report for multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare
            focus_metrics: Optional list of metrics to focus on

        Returns:
            ComparisonReport with detailed comparison
        """
        # Load all experiment results
        results = [self._load_result(exp_id) for exp_id in experiment_ids]

        # Build comparison dataframe
        if focus_metrics is None:
            focus_metrics = list(METRICS.keys())

        comparison_data = []
        for result in results:
            row = {"experiment_id": result.experiment_id}

            # Add config summary
            row["llm_model"] = result.config["llm"]["model"]
            row["embedding_model"] = result.config["embedding"]["model"]
            row["chunking_strategy"] = result.config["chunking"]["strategy"]
            row["retrieval_mode"] = result.config["retrieval"]["default_mode"]

            # Add metrics
            for metric in focus_metrics:
                if metric in result.metrics:
                    row[metric] = result.metrics[metric]

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Identify best config
        best_config = self._identify_best_config(df, focus_metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(df, focus_metrics)

        return ComparisonReport(
            experiments=experiment_ids,
            metrics_comparison=df,
            best_config=best_config,
            recommendations=recommendations
        )

    def generate_ablation_study(
        self,
        baseline_id: str,
        variant_ids: List[str],
        varying_parameter: str
    ) -> Dict:
        """
        Generate ablation study report.

        Compares variants against baseline to understand
        impact of a specific parameter.
        """
        baseline = self._load_result(baseline_id)
        variants = [self._load_result(vid) for vid in variant_ids]

        ablation_results = {
            "baseline": baseline.experiment_id,
            "parameter": varying_parameter,
            "variants": []
        }

        for variant in variants:
            # Compute delta from baseline for each metric
            deltas = {}
            for metric in baseline.metrics:
                if metric in variant.metrics:
                    baseline_val = baseline.metrics[metric]
                    variant_val = variant.metrics[metric]

                    if baseline_val != 0:
                        delta_pct = ((variant_val - baseline_val) / baseline_val) * 100
                    else:
                        delta_pct = 0

                    deltas[metric] = {
                        "baseline": baseline_val,
                        "variant": variant_val,
                        "delta": variant_val - baseline_val,
                        "delta_pct": delta_pct
                    }

            # Get the varying parameter value
            param_value = self._extract_param_value(variant.config, varying_parameter)

            ablation_results["variants"].append({
                "experiment_id": variant.experiment_id,
                "parameter_value": param_value,
                "deltas": deltas
            })

        return ablation_results

    def _generate_recommendations(
        self,
        df: pd.DataFrame,
        metrics: List[str]
    ) -> List[str]:
        """Generate actionable recommendations from comparison"""
        recommendations = []

        # Find patterns
        for metric in metrics:
            if metric not in df.columns:
                continue

            metric_def = METRICS.get(metric)
            if not metric_def:
                continue

            # Best performing config for this metric
            if metric_def.higher_is_better:
                best_idx = df[metric].idxmax()
            else:
                best_idx = df[metric].idxmin()

            best_row = df.iloc[best_idx]

            recommendations.append(
                f"For {metric_def.name}: Best result with "
                f"LLM={best_row['llm_model']}, "
                f"Embedding={best_row['embedding_model']}, "
                f"Mode={best_row['retrieval_mode']} "
                f"(score: {best_row[metric]:.4f})"
            )

        return recommendations
```

---

## Part 5: Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goal:** Configuration system + Triple-level indexing

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Configuration management system | `config/`, `ExperimentConfig`, `ConfigLoader` |
| 3-4 | Entity vector index | `EntityIndex` class, Qdrant collection |
| 5-6 | Relationship vector index | `RelationshipIndex` class, Qdrant collection |
| 7-8 | Index manager integration | `IndexManager` orchestrating all three indices |
| 9-10 | Update ingestion pipeline | Modified document processor using new indices |

**Validation:**
- [ ] Can load versioned configs
- [ ] Can index entities with embeddings
- [ ] Can index relationships with embeddings
- [ ] Can search all three indices independently

### Phase 2: Multi-Mode Retrieval (Week 3-4)

**Goal:** All 7 query modes working

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Query router | `QueryRouter` with bilingual patterns |
| 3-4 | Naive + Local retrievers | `NaiveRetriever`, `LocalRetriever` |
| 5-6 | Global + Hybrid retrievers | `GlobalRetriever`, `HybridRetriever` |
| 7-8 | Mix + Semantic retrievers | `MixRetriever`, `SemanticRetriever` |
| 9-10 | Result fusion + Reranking | `ResultFusion`, `Reranker` |

**Validation:**
- [ ] Query correctly routed to appropriate mode
- [ ] Each mode returns valid results
- [ ] Mix mode combines all modes correctly
- [ ] Fusion produces ranked results

### Phase 3: Evaluation Framework (Week 5-6)

**Goal:** Full evaluation capability

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Metrics implementation | All retrieval + generation metrics |
| 3-4 | Ground truth dataset format | Dataset schema, loader |
| 5-6 | Experiment runner | `ExperimentRunner` with parallel eval |
| 7-8 | Comparison reports | `ReportGenerator`, visualizations |
| 9-10 | CLI tools | `mirage eval run`, `mirage eval compare` |

**Validation:**
- [ ] Can run experiment with any config
- [ ] Metrics computed correctly
- [ ] Results saved and loadable
- [ ] Comparison reports generated

### Phase 4: LLM/Embedding Abstraction (Week 7)

**Goal:** Pluggable models for evaluation

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | LLM abstraction layer | `BaseLLM`, `TGIClient`, `OpenAIClient` |
| 3-4 | Embedding abstraction | `BaseEmbedder`, `JinaEmbedder`, `BGEEmbedder` |
| 5 | Model registry | YAML configs for all models |

**Validation:**
- [ ] Can switch LLM with config change
- [ ] Can switch embedding model with config change
- [ ] All models produce compatible outputs

### Phase 5: Advanced Features (Week 8)

**Goal:** Features beyond LightRAG

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Token-based chunking | `TokenChunker` with tiktoken |
| 3-4 | Hybrid chunking | `HybridChunker` (token + semantic) |
| 5-6 | Gleaning mechanism | Retry logic for extraction |
| 7 | Cross-encoder reranking | Integration with sentence-transformers |

### Phase 6: Evaluation Runs (Week 9-10)

**Goal:** Comprehensive evaluation

| Experiment | Variables | Questions Answered |
|------------|-----------|-------------------|
| LLM Comparison | Allam-7B, Llama3-8B, Qwen2-7B | Which LLM extracts best? |
| Embedding Comparison | Jina-v3, Jina-v4, BGE-M3, E5-multilingual | Which embeddings work best? |
| Chunking Comparison | Token, Semantic, Hybrid | Which chunking strategy? |
| Mode Comparison | All 7 modes | Which mode for which query type? |
| Depth Comparison | 1-hop, 2-hop, 3-hop | Optimal traversal depth? |

---

## Part 6: Exceeding LightRAG

### Features We Add Beyond LightRAG

| Feature | LightRAG | MIRAGE v2 |
|---------|----------|-----------|
| **Multilingual** | English only | Arabic + English + extensible |
| **Query Modes** | 6 | 7 (add Semantic mode) |
| **Chunking** | Token only | Token + Semantic + Hybrid |
| **Evaluation** | None | Full framework |
| **Versioning** | None | Complete config versioning |
| **Arabic NLP** | None | CAMeL Tools + Arabic normalization |
| **Reranking** | None | Cross-encoder reranking |
| **Prompt Management** | Basic | Versioned prompt registry |
| **Experiment Tracking** | None | Full MLOps-style tracking |

### Performance Targets

| Metric | LightRAG (reported) | MIRAGE v2 Target |
|--------|---------------------|------------------|
| Retrieval Latency | ~80ms | <100ms |
| Answer Relevancy | ~0.75 | >0.80 |
| Entity Coverage | Unknown | >0.85 |
| Arabic Support | 0% | 100% |
| Token Efficiency | <100/query | <150/query |

---

## Part 7: Directory Structure (Final)

```
mirage/
├── config/
│   ├── experiments/
│   │   ├── baseline_lightrag.yaml
│   │   ├── llm_comparison.yaml
│   │   ├── embedding_comparison.yaml
│   │   └── chunking_ablation.yaml
│   ├── prompts/
│   │   ├── entity_extraction/
│   │   │   ├── v1_english.yaml
│   │   │   ├── v2_arabic.yaml
│   │   │   └── v3_bilingual.yaml
│   │   ├── relationship_extraction/
│   │   ├── generation/
│   │   └── query_analysis/
│   ├── models/
│   │   ├── llm_registry.yaml
│   │   └── embedding_registry.yaml
│   └── defaults.yaml
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── experiment_config.py
│   │   ├── config_loader.py
│   │   └── config_validator.py
│   ├── core/
│   │   ├── chunking/
│   │   │   ├── __init__.py
│   │   │   ├── base_chunker.py
│   │   │   ├── token_chunker.py
│   │   │   ├── semantic_chunker.py
│   │   │   └── hybrid_chunker.py
│   │   ├── extraction/
│   │   │   ├── __init__.py
│   │   │   ├── base_extractor.py
│   │   │   ├── llm_extractor.py
│   │   │   ├── spacy_extractor.py
│   │   │   ├── camel_extractor.py
│   │   │   └── gleaning.py
│   │   ├── indexing/
│   │   │   ├── __init__.py
│   │   │   ├── chunk_index.py
│   │   │   ├── entity_index.py
│   │   │   ├── relationship_index.py
│   │   │   └── index_manager.py
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── query_router.py
│   │   │   ├── base_retriever.py
│   │   │   ├── naive_retriever.py
│   │   │   ├── local_retriever.py
│   │   │   ├── global_retriever.py
│   │   │   ├── hybrid_retriever.py
│   │   │   ├── mix_retriever.py
│   │   │   ├── semantic_retriever.py
│   │   │   ├── bypass_retriever.py
│   │   │   ├── fusion.py
│   │   │   └── reranker.py
│   │   ├── generation/
│   │   │   ├── __init__.py
│   │   │   ├── context_builder.py
│   │   │   ├── prompt_manager.py
│   │   │   └── generator.py
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── base_llm.py
│   │   │   ├── tgi_client.py
│   │   │   ├── openai_client.py
│   │   │   ├── anthropic_client.py
│   │   │   └── ollama_client.py
│   │   ├── embeddings/
│   │   │   ├── __init__.py
│   │   │   ├── base_embedder.py
│   │   │   ├── jina_embedder.py
│   │   │   ├── bge_embedder.py
│   │   │   └── e5_embedder.py
│   │   └── graph/
│   │       ├── __init__.py
│   │       ├── neo4j_client.py
│   │       ├── traversal.py
│   │       └── community_detection.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics/
│   │   │   ├── retrieval_metrics.py
│   │   │   ├── generation_metrics.py
│   │   │   └── latency_metrics.py
│   │   ├── datasets/
│   │   │   ├── ground_truth_loader.py
│   │   │   └── synthetic_generator.py
│   │   ├── experiment_runner.py
│   │   ├── comparison_report.py
│   │   ├── llm_judge.py
│   │   └── visualization.py
│   └── api/
│       └── (existing API code)
├── eval/
│   ├── ground_truth/
│   │   └── v1_arabic_english.json
│   ├── results/
│   └── reports/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
└── scripts/
    ├── run_experiment.py
    ├── compare_experiments.py
    └── generate_report.py
```

---

## Summary

This plan creates a **research-grade RAG system** that:

1. **Exceeds LightRAG** with 7 modes, multilingual support, and evaluation framework
2. **Enables rigorous evaluation** with versioned configs and reproducible experiments
3. **Supports model comparison** across LLMs, embeddings, chunking strategies
4. **Maintains Arabic support** as a key differentiator
5. **Provides MLOps-grade tracking** for experiments and results

**Timeline:** 10 weeks for full implementation
**Risk:** Medium (architectural changes, but incremental)
**Impact:** High (research-publishable system)
