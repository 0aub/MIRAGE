# MIRAGE v2: Complete Implementation Plan

> **Version**: 2.0 | **Last Updated**: 2024-12-03 | **Status**: Planning

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Configuration System](#3-configuration-system)
4. [Arabic Processing Pipeline](#4-arabic-processing-pipeline)
5. [Chunking Strategies](#5-chunking-strategies)
6. [Dynamic Model Management](#6-dynamic-model-management)
7. [Triple-Level Indexing](#7-triple-level-indexing)
8. [Graph Features](#8-graph-features)
9. [Retrieval Engine](#9-retrieval-engine)
10. [Prompt System](#10-prompt-system)
11. [Evaluation Framework](#11-evaluation-framework)
12. [Implementation Phases](#12-implementation-phases)
13. [Directory Structure](#13-directory-structure)

---

## 1. Executive Summary

### 1.1 Vision
Build the **strongest Arabic-English hybrid RAG system** that exceeds LightRAG and GraphRAG capabilities with:
- Production-ready Arabic understanding (MSA + dialects)
- 7 retrieval modes with intelligent routing
- Full experiment framework for scientific evaluation
- Dynamic model loading from HuggingFace

### 1.2 Key Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| GPU | RTX 4090 24GB | Focus on SLMs (7B-13B models) |
| Model Cache | `/data/models_cache` | Persistent local storage |
| Inference Backend | TGI only | Best for production serving |
| Users | Single user | No caching/rate limiting needed |
| External APIs | Disabled | Local inference only |
| Arabic NLP | Configurable (CAMeL, Stanza, etc.) | Compare in evaluation |

### 1.3 What We Build Beyond LightRAG

| Feature | LightRAG | GraphRAG | MIRAGE v2 |
|---------|----------|----------|-----------|
| Arabic Support | None | None | **Full (MSA + dialects)** |
| Query Modes | 6 | 2 (local/global) | **7 modes** |
| Community Detection | None | Yes | **Yes + temporal** |
| Hierarchical Summary | None | Yes | **Yes** |
| Confidence Decay | None | None | **Yes** |
| Chunking Options | 1 (token) | 1 | **5 strategies** |
| Evaluation Framework | None | None | **Complete** |
| Model Hot-swap | None | None | **Yes (HuggingFace)** |
| Data Versioning | None | None | **Yes** |

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              MIRAGE v2 ARCHITECTURE                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         CONFIGURATION LAYER                            │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │ │
│  │  │  Experiment  │ │    Prompt    │ │    Model     │ │   Arabic     │  │ │
│  │  │    Config    │ │   Registry   │ │   Registry   │ │   Config     │  │ │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                       │
│                                      ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         MODEL MANAGEMENT                               │ │
│  │                                                                        │ │
│  │   HuggingFace Hub ──► Local Cache (/data/models_cache) ──► TGI/Local  │ │
│  │                                                                        │ │
│  │   Supports: LLMs (Allam, Llama3, Qwen) | Embeddings (Jina, BGE, E5)   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                       │
│                                      ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         INGESTION PIPELINE                             │ │
│  │                                                                        │ │
│  │  Document ──► Language Detection ──► Arabic Processing (configurable) │ │
│  │      │                                       │                         │ │
│  │      ▼                                       ▼                         │ │
│  │  Chunking (5 strategies) ──► Entity Extraction ──► Relationship Ext.  │ │
│  │      │                              │                    │             │ │
│  │      ▼                              ▼                    ▼             │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                    TRIPLE-LEVEL INDEX                           │  │ │
│  │  │  ┌─────────────┐  ┌─────────────────┐  ┌───────────────────┐   │  │ │
│  │  │  │   Chunks    │  │    Entities     │  │   Relationships   │   │  │ │
│  │  │  │  (Qdrant)   │  │    (Qdrant)     │  │     (Qdrant)      │   │  │ │
│  │  │  └─────────────┘  └─────────────────┘  └───────────────────┘   │  │ │
│  │  │         │                 │                     │              │  │ │
│  │  │         └─────────────────┼─────────────────────┘              │  │ │
│  │  │                           ▼                                    │  │ │
│  │  │              ┌───────────────────────┐                         │  │ │
│  │  │              │   Knowledge Graph     │                         │  │ │
│  │  │              │      (Neo4j)          │                         │  │ │
│  │  │              │  + Communities        │                         │  │ │
│  │  │              │  + Temporal Relations │                         │  │ │
│  │  │              │  + Hierarchical Sums  │                         │  │ │
│  │  │              └───────────────────────┘                         │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                       │
│                                      ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         RETRIEVAL ENGINE                               │ │
│  │                                                                        │ │
│  │  Query ──► Router ──► Mode Selection ──► Multi-Index Search ──► Fusion│ │
│  │                                                                        │ │
│  │  7 Modes: naive | local | global | hybrid | mix | semantic | bypass   │ │
│  │                                                                        │ │
│  │  Features: Community traversal, Temporal filtering, Confidence decay  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                       │
│                                      ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         GENERATION LAYER                               │ │
│  │                                                                        │ │
│  │  Context Assembly ──► Prompt (CoT + Few-shot) ──► LLM ──► Response    │ │
│  │                                                                        │ │
│  │  Output: Answer + Citations + Confidence + Debug Info                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                         EVALUATION FRAMEWORK (Separate)                      │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Experiment Runner ──► Metrics Collection ──► Analysis & Reports      │ │
│  │                                                                        │ │
│  │  Compares: LLMs | Embeddings | Chunking | Arabic NLP | Retrieval Modes│ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Configuration System

### 3.1 Configuration Hierarchy

```
config/
├── core/                      # Core system configuration (rarely changed)
│   ├── defaults.yaml          # Default values for all parameters
│   ├── models.yaml            # Model registry (HuggingFace IDs)
│   └── storage.yaml           # Database connections
│
├── components/                # Component-specific configs
│   ├── chunking/
│   │   ├── token.yaml
│   │   ├── semantic.yaml
│   │   ├── sentence.yaml
│   │   ├── recursive.yaml
│   │   └── late.yaml
│   ├── arabic/
│   │   ├── camel.yaml
│   │   ├── stanza.yaml
│   │   └── disabled.yaml
│   ├── retrieval/
│   │   └── modes.yaml
│   └── prompts/
│       ├── extraction/
│       ├── generation/
│       └── analysis/
│
└── experiments/               # Experiment configurations (evaluation)
    ├── baselines/
    │   ├── naive_rag.yaml
    │   ├── lightrag_style.yaml
    │   └── graphrag_style.yaml
    ├── ablations/
    │   ├── llm_comparison.yaml
    │   ├── embedding_comparison.yaml
    │   ├── chunking_comparison.yaml
    │   └── arabic_nlp_comparison.yaml
    └── production/
        └── best_config.yaml
```

### 3.2 Core Configuration Schema

```yaml
# config/core/defaults.yaml
version: "2.0.0"

# =============================================================================
# STORAGE CONFIGURATION
# =============================================================================
storage:
  models_cache: "/data/models_cache"
  qdrant:
    host: "qdrant"
    port: 6333
    collections:
      chunks: "mirage_chunks"
      entities: "mirage_entities"
      relationships: "mirage_relationships"
  neo4j:
    uri: "bolt://neo4j:7687"
    user: "neo4j"
    password: "${NEO4J_PASSWORD}"
  data_versioning:
    enabled: true
    track_documents: true
    track_config_changes: true

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
models:
  # LLM Configuration
  llm:
    # HuggingFace model ID - will be auto-downloaded
    model_id: "ALLaM-AI/ALLaM-7B-Instruct-preview"
    backend: "tgi"  # Only TGI supported
    quantization: null  # null | "4bit" | "8bit"
    max_tokens: 4096
    temperature: 0.1
    top_p: 0.95

  # Embedding Configuration
  embedding:
    model_id: "jinaai/jina-embeddings-v3"
    dimensions: 1024
    normalize: true
    batch_size: 32
    # Task-specific prefixes (Jina-style)
    tasks:
      passage: "retrieval.passage"
      query: "retrieval.query"
      entity: "classification"

# =============================================================================
# ARABIC PROCESSING CONFIGURATION
# =============================================================================
arabic:
  enabled: true
  processor: "camel"  # "camel" | "stanza" | "none"

  # CAMeL Tools Configuration
  camel:
    # Morphological analysis for better tokenization
    morphological_analysis: true
    # Diacritization for disambiguation
    diacritization: false  # Expensive, enable for high accuracy
    # Dialect identification
    dialect_identification: true
    supported_dialects:
      - "MSA"      # Modern Standard Arabic
      - "EGY"      # Egyptian
      - "GLF"      # Gulf
      - "LEV"      # Levantine
    # Named Entity Recognition
    ner:
      enabled: true
      model: "camel-ner"
    # Normalization
    normalization:
      alef: true           # أ إ آ → ا
      yaa: true            # ى → ي
      taa_marbuta: true    # ة → ه (optional)
      remove_diacritics: true
      remove_tatweel: true  # ـــ

  # Stanza Configuration (alternative)
  stanza:
    language: "ar"
    processors: "tokenize,mwt,pos,lemma,ner"

  # Entity Normalization (Arabic-specific)
  entity_normalization:
    remove_titles:
      - "الدكتور"
      - "الأستاذ"
      - "المهندس"
      - "الشيخ"
      - "السيد"
      - "الحاج"
      - "الأمير"
      - "الملك"
    organization_suffixes:
      - "المحدودة"
      - "للتجارة"
      - "القابضة"

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================
chunking:
  strategy: "semantic"  # See Section 5 for all strategies

  # Each strategy has its own parameters
  strategies:
    token:
      max_tokens: 600
      overlap_tokens: 50
      tokenizer: "cl100k_base"

    semantic:
      similarity_threshold: 0.7
      min_chunk_tokens: 100
      max_chunk_tokens: 800
      embedding_model: "${models.embedding.model_id}"

    sentence:
      sentences_per_chunk: 5
      overlap_sentences: 1
      respect_paragraphs: true

    recursive:
      chunk_size: 1000
      chunk_overlap: 200
      separators:
        - "\n\n"
        - "\n"
        - ". "
        - " "

    late:
      # Jina's late chunking approach
      max_document_tokens: 8192
      chunk_after_embedding: true
      boundary_tokens: 256

# =============================================================================
# EXTRACTION CONFIGURATION
# =============================================================================
extraction:
  # Entity extraction method
  entity_method: "llm"  # "llm" | "ner" | "hybrid"

  # LLM extraction settings
  llm:
    max_tokens_per_chunk: 600
    gleaning:
      enabled: true
      max_retries: 2
    entity_types:
      - Person
      - Organization
      - Location
      - Event
      - Technology
      - Policy
      - Product
      - Date
      - Money
    relationship_types:
      - leads
      - manages
      - founded
      - works_for
      - located_in
      - participated_in
      - developed
      - announced
      - invested_in
      - partnered_with

  # Quality filters
  filters:
    min_entity_length: 2
    max_entity_length: 100
    min_confidence: 0.5
    blacklisted_entities:
      - "something"
      - "thing"
      - "aspect"
      - "way"
      - "شيء"
      - "موضوع"
    blacklisted_relationships:
      - "related_to"
      - "associated_with"
      - "connected_to"

# =============================================================================
# GRAPH CONFIGURATION
# =============================================================================
graph:
  # Community detection (Leiden algorithm)
  community_detection:
    enabled: true
    algorithm: "leiden"
    resolution: 1.0
    min_community_size: 3

  # Hierarchical summaries
  hierarchical_summaries:
    enabled: true
    levels: 3  # Document → Community → Global
    summary_max_tokens: 500

  # Temporal relationships
  temporal:
    enabled: true
    track_creation_time: true
    track_mention_time: true
    enable_temporal_queries: true

  # Confidence decay
  confidence_decay:
    enabled: true
    decay_function: "exponential"  # "linear" | "exponential"
    half_life_days: 30  # Confidence halves every 30 days without reinforcement
    min_confidence: 0.1
    reinforcement_boost: 0.2  # Boost when relationship is seen again

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================
retrieval:
  default_mode: "mix"

  modes:
    naive:
      enabled: true
      description: "Vector search on chunks only (traditional RAG)"
      top_k: 10
      score_threshold: 0.5

    local:
      enabled: true
      description: "Entity-focused with graph expansion"
      entity_top_k: 5
      chunks_per_entity: 3
      graph_hops: 2
      use_communities: true

    global:
      enabled: true
      description: "Relationship-focused across communities"
      relationship_top_k: 10
      use_hierarchical_summaries: true
      community_top_k: 3

    hybrid:
      enabled: true
      description: "Local + Global combined"
      local_weight: 0.5
      global_weight: 0.5
      fusion_method: "rrf"

    mix:
      enabled: true
      description: "All modes combined (most comprehensive)"
      naive_weight: 0.2
      local_weight: 0.4
      global_weight: 0.4
      fusion_method: "rrf"

    semantic:
      enabled: true
      description: "Deep semantic matching with reranking"
      initial_top_k: 20
      rerank_top_k: 5
      use_cross_encoder: true

    bypass:
      enabled: true
      description: "No retrieval, direct LLM (for baseline)"

  # Temporal filtering
  temporal_filter:
    enabled: false  # Enable per-query
    default_window_days: null  # null = all time

  # Confidence filtering
  confidence_filter:
    min_entity_confidence: 0.3
    min_relationship_confidence: 0.3

# =============================================================================
# GENERATION CONFIGURATION
# =============================================================================
generation:
  prompt_version: "v1"

  chain_of_thought:
    enabled: true
    steps:
      - "understand_query"
      - "identify_relevant_context"
      - "synthesize_answer"
      - "cite_sources"

  few_shot:
    enabled: true
    num_examples: 2
    examples_source: "config/prompts/generation/examples.yaml"

  output:
    include_citations: true
    citation_format: "inline"  # [1], [2], etc.
    include_confidence: true
    include_debug_info: true
    language: "auto"  # Respond in query language

# =============================================================================
# DATA VERSIONING
# =============================================================================
versioning:
  enabled: true
  track:
    - documents
    - entities
    - relationships
    - communities
    - config_changes
  storage: "neo4j"  # Store version info in Neo4j
```

---

## 4. Arabic Processing Pipeline

### 4.1 Overview

Arabic processing is **configurable** to allow comparison between different approaches:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ARABIC PROCESSING PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Text                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────┐                                               │
│  │ Language        │──► Arabic detected? ──► No ──► Skip Arabic    │
│  │ Detection       │                                   processing  │
│  └─────────────────┘                                               │
│      │ Yes                                                          │
│      ▼                                                              │
│  ┌─────────────────┐                                               │
│  │ Dialect         │──► Identifies: MSA, Egyptian, Gulf, Levantine │
│  │ Identification  │    (Used for processor selection)             │
│  └─────────────────┘                                               │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              CONFIGURABLE PROCESSOR                          │   │
│  │                                                              │   │
│  │  Option A: CAMeL Tools (Recommended)                        │   │
│  │  ├── Morphological Analysis                                 │   │
│  │  ├── Tokenization (Arabic-aware)                           │   │
│  │  ├── Named Entity Recognition                              │   │
│  │  └── Normalization                                          │   │
│  │                                                              │   │
│  │  Option B: Stanza                                           │   │
│  │  ├── Tokenization                                           │   │
│  │  ├── POS Tagging                                            │   │
│  │  ├── Lemmatization                                          │   │
│  │  └── NER                                                     │   │
│  │                                                              │   │
│  │  Option C: None (LLM handles everything)                    │   │
│  │  └── Raw text to LLM                                         │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────┐                                               │
│  │ Arabic-specific │                                               │
│  │ Normalization   │                                               │
│  │ ├── Alef: أإآ→ا │                                               │
│  │ ├── Yaa: ى→ي   │                                               │
│  │ ├── Remove تشكيل│                                               │
│  │ └── Remove ـــ  │                                               │
│  └─────────────────┘                                               │
│      │                                                              │
│      ▼                                                              │
│  Output: Processed Text + Metadata                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Processor Implementations

```python
# src/core/arabic/base_processor.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum

class ArabicDialect(Enum):
    MSA = "msa"           # Modern Standard Arabic
    EGYPTIAN = "egy"      # Egyptian Arabic
    GULF = "glf"          # Gulf Arabic
    LEVANTINE = "lev"     # Levantine Arabic
    MAGHREBI = "mgr"      # Maghrebi Arabic
    UNKNOWN = "unknown"

@dataclass
class ArabicProcessingResult:
    """Result of Arabic text processing"""
    original_text: str
    normalized_text: str
    dialect: ArabicDialect
    tokens: List[str]
    entities: List[Dict]  # NER results if available
    morphological_analysis: Optional[Dict] = None
    metadata: Dict = None

class BaseArabicProcessor(ABC):
    """Base class for Arabic text processors"""

    @abstractmethod
    def process(self, text: str) -> ArabicProcessingResult:
        """Process Arabic text"""
        pass

    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from Arabic text"""
        pass

    def normalize(self, text: str) -> str:
        """
        Standard Arabic normalization.
        Applied regardless of processor choice.
        """
        import re

        # Alef normalization: أ إ آ ٱ → ا
        text = re.sub(r'[أإآٱ]', 'ا', text)

        # Yaa normalization: ى → ي
        text = re.sub(r'ى', 'ي', text)

        # Taa marbuta: Optionally ة → ه
        # text = re.sub(r'ة', 'ه', text)  # Usually keep as is

        # Remove diacritics (tashkeel)
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)

        # Remove tatweel (kashida)
        text = re.sub(r'\u0640', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def detect_dialect(self, text: str) -> ArabicDialect:
        """
        Detect Arabic dialect.
        Default implementation - can be overridden.
        """
        # Dialect markers (simplified)
        egyptian_markers = ['ده', 'دي', 'كده', 'ازيك', 'عايز']
        gulf_markers = ['شلونك', 'زين', 'وايد', 'شنو']
        levantine_markers = ['هيك', 'شو', 'كيفك', 'منيح']

        text_lower = text.lower()

        if any(marker in text_lower for marker in egyptian_markers):
            return ArabicDialect.EGYPTIAN
        elif any(marker in text_lower for marker in gulf_markers):
            return ArabicDialect.GULF
        elif any(marker in text_lower for marker in levantine_markers):
            return ArabicDialect.LEVANTINE
        else:
            return ArabicDialect.MSA  # Default to MSA


# src/core/arabic/camel_processor.py

class CamelArabicProcessor(BaseArabicProcessor):
    """
    Arabic processor using CAMeL Tools.

    CAMeL Tools provides:
    - State-of-the-art Arabic NLP
    - Morphological analysis
    - Dialect identification
    - Named Entity Recognition
    - Diacritization

    Best for: Production Arabic processing with high accuracy
    Tradeoff: Slower than simpler methods
    """

    def __init__(self, config: Dict):
        self.config = config
        self._init_camel()

    def _init_camel(self):
        """Lazy initialization of CAMeL components"""
        try:
            from camel_tools.tokenizers.word import simple_word_tokenize
            from camel_tools.utils.normalize import normalize_alef_maksura_ar
            from camel_tools.utils.normalize import normalize_alef_ar
            from camel_tools.utils.normalize import normalize_teh_marbuta_ar
            from camel_tools.utils.dediac import dediac_ar

            self.tokenize = simple_word_tokenize
            self.normalize_alef = normalize_alef_ar
            self.normalize_yaa = normalize_alef_maksura_ar
            self.dediac = dediac_ar

            # Optional: Morphological analyzer
            if self.config.get('morphological_analysis', False):
                from camel_tools.morphology.database import MorphologyDB
                from camel_tools.morphology.analyzer import Analyzer
                self.morph_db = MorphologyDB.builtin_db()
                self.analyzer = Analyzer(self.morph_db)

            # Optional: NER
            if self.config.get('ner', {}).get('enabled', False):
                from camel_tools.ner import NERecognizer
                self.ner = NERecognizer.pretrained()

            # Optional: Dialect identification
            if self.config.get('dialect_identification', False):
                from camel_tools.dialectid import DialectIdentifier
                self.dialect_id = DialectIdentifier.pretrained()

        except ImportError as e:
            raise ImportError(f"CAMeL Tools not installed: {e}")

    def process(self, text: str) -> ArabicProcessingResult:
        # Detect dialect
        dialect = self._detect_dialect_camel(text)

        # Normalize
        normalized = self.normalize(text)

        # Tokenize
        tokens = self.tokenize(normalized)

        # Extract entities if NER enabled
        entities = self.extract_entities(text) if hasattr(self, 'ner') else []

        # Morphological analysis if enabled
        morph = self._analyze_morphology(text) if hasattr(self, 'analyzer') else None

        return ArabicProcessingResult(
            original_text=text,
            normalized_text=normalized,
            dialect=dialect,
            tokens=tokens,
            entities=entities,
            morphological_analysis=morph
        )

    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities using CAMeL NER"""
        if not hasattr(self, 'ner'):
            return []

        # CAMeL NER expects tokenized input
        tokens = self.tokenize(text)
        labels = self.ner.predict(tokens)

        # Convert to entity list
        entities = []
        current_entity = None

        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token,
                    'type': label[2:],
                    'source': 'camel_ner'
                }
            elif label.startswith('I-') and current_entity:
                current_entity['text'] += ' ' + token
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities


# src/core/arabic/stanza_processor.py

class StanzaArabicProcessor(BaseArabicProcessor):
    """
    Arabic processor using Stanza.

    Stanza provides:
    - Neural pipeline for NLP
    - Tokenization, POS, Lemmatization
    - NER
    - Dependency parsing

    Best for: Balanced speed/accuracy
    Tradeoff: Less Arabic-specific than CAMeL
    """

    def __init__(self, config: Dict):
        self.config = config
        self._init_stanza()

    def _init_stanza(self):
        import stanza

        # Download Arabic model if needed
        stanza.download('ar', processors='tokenize,mwt,pos,lemma,ner')

        # Initialize pipeline
        self.nlp = stanza.Pipeline(
            'ar',
            processors=self.config.get('processors', 'tokenize,pos,lemma,ner')
        )

    def process(self, text: str) -> ArabicProcessingResult:
        # Normalize first
        normalized = self.normalize(text)

        # Process with Stanza
        doc = self.nlp(normalized)

        # Extract tokens
        tokens = [word.text for sent in doc.sentences for word in sent.words]

        # Extract entities
        entities = self.extract_entities(text)

        return ArabicProcessingResult(
            original_text=text,
            normalized_text=normalized,
            dialect=self.detect_dialect(text),
            tokens=tokens,
            entities=entities
        )

    def extract_entities(self, text: str) -> List[Dict]:
        doc = self.nlp(self.normalize(text))

        entities = []
        for sent in doc.sentences:
            for ent in sent.ents:
                entities.append({
                    'text': ent.text,
                    'type': ent.type,
                    'source': 'stanza_ner'
                })

        return entities


# src/core/arabic/disabled_processor.py

class DisabledArabicProcessor(BaseArabicProcessor):
    """
    Minimal processor - only basic normalization.

    Use when:
    - Testing LLM's native Arabic understanding
    - Comparing with/without Arabic NLP
    - Speed is critical

    Note: Still applies basic normalization for consistency
    """

    def process(self, text: str) -> ArabicProcessingResult:
        normalized = self.normalize(text)

        return ArabicProcessingResult(
            original_text=text,
            normalized_text=normalized,
            dialect=ArabicDialect.UNKNOWN,
            tokens=normalized.split(),  # Simple whitespace tokenization
            entities=[]  # No NER
        )

    def extract_entities(self, text: str) -> List[Dict]:
        return []  # LLM will handle entity extraction
```

### 4.3 Arabic Processing Factory

```python
# src/core/arabic/__init__.py

from typing import Dict
from .base_processor import BaseArabicProcessor
from .camel_processor import CamelArabicProcessor
from .stanza_processor import StanzaArabicProcessor
from .disabled_processor import DisabledArabicProcessor

def create_arabic_processor(config: Dict) -> BaseArabicProcessor:
    """
    Factory function to create Arabic processor based on config.

    Usage:
        processor = create_arabic_processor(config['arabic'])
        result = processor.process(text)
    """
    if not config.get('enabled', True):
        return DisabledArabicProcessor(config)

    processor_type = config.get('processor', 'camel')

    processors = {
        'camel': CamelArabicProcessor,
        'stanza': StanzaArabicProcessor,
        'none': DisabledArabicProcessor,
        'disabled': DisabledArabicProcessor,
    }

    if processor_type not in processors:
        raise ValueError(f"Unknown Arabic processor: {processor_type}")

    processor_config = config.get(processor_type, {})
    return processors[processor_type](processor_config)
```

---

## 5. Chunking Strategies

### 5.1 Strategy Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHUNKING STRATEGIES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STRATEGY 1: TOKEN-BASED                                                    │
│  ════════════════════════                                                   │
│  Description: Fixed token count with overlap                                │
│  How it works:                                                              │
│    - Count tokens using tiktoken (cl100k_base)                             │
│    - Split at token boundaries                                              │
│    - Maintain overlap for context continuity                                │
│  Best for: Consistent LLM context windows, predictable behavior            │
│  Drawback: May split mid-sentence or mid-thought                           │
│  Config: max_tokens=600, overlap_tokens=50                                 │
│                                                                             │
│  Example:                                                                   │
│    Text: "The quick brown fox jumps over the lazy dog. It was sunny."      │
│    Chunk 1: "The quick brown fox jumps over" (tokens 1-50)                 │
│    Chunk 2: "jumps over the lazy dog. It was sunny." (tokens 40-90)        │
│             ↑ overlap                                                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STRATEGY 2: SEMANTIC                                                       │
│  ═══════════════════                                                        │
│  Description: Group sentences by semantic similarity                        │
│  How it works:                                                              │
│    - Split into sentences                                                   │
│    - Embed each sentence                                                    │
│    - Group sentences with similarity > threshold                           │
│    - Each group becomes a chunk                                             │
│  Best for: Maintaining topical coherence within chunks                     │
│  Drawback: Variable chunk sizes, more compute (embeddings)                 │
│  Config: similarity_threshold=0.7, max_chunk_tokens=800                    │
│                                                                             │
│  Example:                                                                   │
│    Sentences: [S1: climate], [S2: climate], [S3: economy], [S4: economy]   │
│    Similarity: S1↔S2=0.9, S2↔S3=0.3, S3↔S4=0.85                           │
│    Result: Chunk1=[S1,S2], Chunk2=[S3,S4]                                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STRATEGY 3: SENTENCE-BASED                                                 │
│  ══════════════════════════                                                 │
│  Description: Fixed number of sentences per chunk                          │
│  How it works:                                                              │
│    - Split into sentences (respects Arabic sentence boundaries)            │
│    - Group N sentences together                                             │
│    - Optional: respect paragraph boundaries                                 │
│  Best for: Documents with clear sentence structure                         │
│  Drawback: Ignores semantic relationships                                  │
│  Config: sentences_per_chunk=5, overlap_sentences=1                        │
│                                                                             │
│  Example:                                                                   │
│    Sentences: [S1, S2, S3, S4, S5, S6, S7]                                 │
│    Chunk 1: [S1, S2, S3, S4, S5]                                           │
│    Chunk 2: [S5, S6, S7, ...]  ← S5 is overlap                             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STRATEGY 4: RECURSIVE                                                      │
│  ═════════════════════                                                      │
│  Description: Try multiple separators in order until chunk fits            │
│  How it works:                                                              │
│    - Start with largest separator (\n\n = paragraphs)                      │
│    - If chunk too big, try next separator (\n = lines)                     │
│    - Continue until chunk_size met (. = sentences, then space)             │
│  Best for: Documents with mixed structure (headers, paragraphs, lists)     │
│  Drawback: Less predictable chunk boundaries                               │
│  Config: chunk_size=1000, separators=["\n\n", "\n", ". ", " "]            │
│                                                                             │
│  Example:                                                                   │
│    Text: "# Header\n\nParagraph 1...\n\nParagraph 2..."                    │
│    First try: Split by \n\n (paragraphs)                                   │
│    If paragraph > chunk_size: split by \n (lines)                          │
│    If line > chunk_size: split by ". " (sentences)                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STRATEGY 5: LATE CHUNKING (Jina-style)                                    │
│  ═══════════════════════════════════════                                    │
│  Description: Embed full document first, then chunk                        │
│  How it works:                                                              │
│    - Pass entire document to embedding model (up to 8192 tokens)           │
│    - Model embeds with full context awareness                               │
│    - Then split into chunks, each inheriting contextual embedding          │
│  Best for: Long documents where context matters across sections            │
│  Drawback: Requires long-context embedding model (Jina v3+)                │
│  Config: max_document_tokens=8192, boundary_tokens=256                     │
│                                                                             │
│  Example:                                                                   │
│    Document: [Full 5000 token document]                                    │
│    Step 1: Embed entire document → contextual token embeddings             │
│    Step 2: Split at natural boundaries → chunks with context-aware embeds │
│    Result: Each chunk "knows" about the whole document                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Chunker Implementations

```python
# src/core/chunking/base_chunker.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Chunk:
    """A single chunk of text"""
    id: str
    text: str
    token_count: int
    char_count: int
    metadata: Dict
    embedding: Optional[List[float]] = None

    # Position info
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0

class BaseChunker(ABC):
    """
    Base class for all chunking strategies.

    All chunkers must implement:
    - chunk(): Split text into chunks
    - name: Strategy name for logging/config
    """

    name: str = "base"

    @abstractmethod
    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Full document text
            document_id: Document identifier for chunk IDs

        Returns:
            List of Chunk objects
        """
        pass

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))


# src/core/chunking/token_chunker.py

class TokenChunker(BaseChunker):
    """
    TOKEN-BASED CHUNKING

    Splits text into fixed-size token chunks with overlap.
    Most predictable strategy - every chunk has similar token count.

    Use when:
    - You need consistent context window usage
    - Document structure doesn't matter
    - Speed is important (no embeddings needed)

    Parameters:
    - max_tokens: Maximum tokens per chunk (default: 600)
    - overlap_tokens: Overlap between chunks (default: 50)
    - tokenizer: Tiktoken encoding (default: cl100k_base)
    """

    name = "token"

    def __init__(self, config: Dict):
        self.max_tokens = config.get('max_tokens', 600)
        self.overlap_tokens = config.get('overlap_tokens', 50)
        self.tokenizer_name = config.get('tokenizer', 'cl100k_base')

        import tiktoken
        self.encoder = tiktoken.get_encoding(self.tokenizer_name)

    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        tokens = self.encoder.encode(text)
        chunks = []

        i = 0
        chunk_index = 0

        while i < len(tokens):
            # Get chunk tokens
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunk_text = self.encoder.decode(chunk_tokens)

            # Calculate character positions (approximate)
            start_char = len(self.encoder.decode(tokens[:i]))
            end_char = start_char + len(chunk_text)

            chunks.append(Chunk(
                id=f"{document_id}_chunk_{chunk_index}",
                text=chunk_text,
                token_count=len(chunk_tokens),
                char_count=len(chunk_text),
                metadata={'strategy': self.name},
                start_char=start_char,
                end_char=end_char,
                chunk_index=chunk_index
            ))

            # Move forward with overlap
            i += self.max_tokens - self.overlap_tokens
            chunk_index += 1

        return chunks


# src/core/chunking/semantic_chunker.py

class SemanticChunker(BaseChunker):
    """
    SEMANTIC CHUNKING

    Groups sentences by semantic similarity.
    Produces topically coherent chunks.

    Use when:
    - Topical coherence is important
    - Documents have clear topic transitions
    - You can afford embedding computation

    Parameters:
    - similarity_threshold: Min cosine similarity to group (default: 0.7)
    - min_chunk_tokens: Minimum tokens per chunk (default: 100)
    - max_chunk_tokens: Maximum tokens per chunk (default: 800)
    - embedding_model: Model for sentence embeddings
    """

    name = "semantic"

    def __init__(self, config: Dict, embedder=None):
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.min_chunk_tokens = config.get('min_chunk_tokens', 100)
        self.max_chunk_tokens = config.get('max_chunk_tokens', 800)
        self.embedder = embedder

    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        # Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        # Embed all sentences
        embeddings = self.embedder.embed_batch([s['text'] for s in sentences])

        # Group by similarity
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        current_tokens = sentences[0]['tokens']

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            embedding = embeddings[i]

            # Compute similarity to chunk average
            similarity = self._cosine_similarity(current_embedding, embedding)

            # Check if should add to current chunk
            new_tokens = current_tokens + sentence['tokens']

            if (similarity >= self.similarity_threshold and
                new_tokens <= self.max_chunk_tokens):
                # Add to current chunk
                current_chunk.append(sentence)
                current_tokens = new_tokens
                # Update chunk embedding (running average)
                current_embedding = self._average_embeddings(
                    [embeddings[j] for j in range(i - len(current_chunk) + 1, i + 1)]
                )
            else:
                # Start new chunk (only if current meets minimum)
                if current_tokens >= self.min_chunk_tokens:
                    chunks.append(self._create_chunk(
                        current_chunk, document_id, len(chunks), current_embedding
                    ))
                    current_chunk = [sentence]
                    current_embedding = embedding
                    current_tokens = sentence['tokens']
                else:
                    # Force add to meet minimum
                    current_chunk.append(sentence)
                    current_tokens = new_tokens

        # Don't forget last chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, document_id, len(chunks), current_embedding
            ))

        return chunks

    def _split_sentences(self, text: str) -> List[Dict]:
        """Split text into sentences with token counts"""
        import re

        # Sentence boundaries (handles Arabic and English)
        pattern = r'[.!?؟।।\u0964\u0965]+[\s\n]+'

        sentences = []
        for sent in re.split(pattern, text):
            sent = sent.strip()
            if sent:
                sentences.append({
                    'text': sent,
                    'tokens': self.count_tokens(sent)
                })

        return sentences


# src/core/chunking/sentence_chunker.py

class SentenceChunker(BaseChunker):
    """
    SENTENCE-BASED CHUNKING

    Groups fixed number of sentences.
    Simple and respects natural boundaries.

    Use when:
    - Documents have clear sentence structure
    - You want predictable boundaries
    - Semantic similarity not needed

    Parameters:
    - sentences_per_chunk: Number of sentences (default: 5)
    - overlap_sentences: Overlap sentences (default: 1)
    - respect_paragraphs: Don't split paragraphs (default: True)
    """

    name = "sentence"

    def __init__(self, config: Dict):
        self.sentences_per_chunk = config.get('sentences_per_chunk', 5)
        self.overlap_sentences = config.get('overlap_sentences', 1)
        self.respect_paragraphs = config.get('respect_paragraphs', True)

    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        sentences = self._split_sentences(text)
        chunks = []

        i = 0
        chunk_index = 0

        while i < len(sentences):
            chunk_sentences = sentences[i:i + self.sentences_per_chunk]
            chunk_text = ' '.join(chunk_sentences)

            chunks.append(Chunk(
                id=f"{document_id}_chunk_{chunk_index}",
                text=chunk_text,
                token_count=self.count_tokens(chunk_text),
                char_count=len(chunk_text),
                metadata={'strategy': self.name, 'sentence_count': len(chunk_sentences)},
                chunk_index=chunk_index
            ))

            i += self.sentences_per_chunk - self.overlap_sentences
            chunk_index += 1

        return chunks


# src/core/chunking/recursive_chunker.py

class RecursiveChunker(BaseChunker):
    """
    RECURSIVE CHUNKING

    Tries multiple separators in order to fit chunk size.
    Adapts to document structure.

    Use when:
    - Documents have mixed structure
    - You want to respect natural boundaries
    - Flexibility is needed

    Parameters:
    - chunk_size: Target chunk size in characters (default: 1000)
    - chunk_overlap: Overlap in characters (default: 200)
    - separators: List of separators to try (default: ["\n\n", "\n", ". ", " "])
    """

    name = "recursive"

    def __init__(self, config: Dict):
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        self.separators = config.get('separators', ["\n\n", "\n", ". ", " "])

    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        chunks_text = self._recursive_split(text, self.separators)

        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            chunks.append(Chunk(
                id=f"{document_id}_chunk_{i}",
                text=chunk_text,
                token_count=self.count_tokens(chunk_text),
                char_count=len(chunk_text),
                metadata={'strategy': self.name},
                chunk_index=i
            ))

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators"""
        if not separators:
            # No more separators, force split by character
            return self._split_by_size(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        splits = text.split(separator)

        chunks = []
        current_chunk = ""

        for split in splits:
            if len(current_chunk) + len(split) <= self.chunk_size:
                current_chunk += (separator if current_chunk else "") + split
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # If split itself is too big, recurse with smaller separators
                if len(split) > self.chunk_size:
                    chunks.extend(self._recursive_split(split, remaining_separators))
                    current_chunk = ""
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


# src/core/chunking/late_chunker.py

class LateChunker(BaseChunker):
    """
    LATE CHUNKING (Jina-style)

    Embeds entire document first, then chunks.
    Each chunk inherits contextual understanding.

    Use when:
    - Long documents with interconnected content
    - Context matters across sections
    - You have long-context embedding model (Jina v3+)

    Parameters:
    - max_document_tokens: Max tokens for full doc (default: 8192)
    - boundary_tokens: Tokens per chunk boundary (default: 256)

    Requires: Long-context embedding model
    """

    name = "late"

    def __init__(self, config: Dict, embedder=None):
        self.max_document_tokens = config.get('max_document_tokens', 8192)
        self.boundary_tokens = config.get('boundary_tokens', 256)
        self.embedder = embedder

    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        # Truncate if too long
        tokens = self.count_tokens(text)
        if tokens > self.max_document_tokens:
            # Fall back to regular chunking for very long docs
            from .token_chunker import TokenChunker
            fallback = TokenChunker({'max_tokens': 600, 'overlap_tokens': 50})
            return fallback.chunk(text, document_id)

        # Embed entire document with late interaction
        # Jina models support this via task="retrieval.passage" with full doc
        doc_embedding = self.embedder.embed_late(text)

        # Find natural boundaries (paragraphs, sections)
        boundaries = self._find_boundaries(text)

        # Create chunks at boundaries, each with contextual embedding
        chunks = []
        for i, (start, end) in enumerate(boundaries):
            chunk_text = text[start:end]

            # Extract chunk embedding from doc embedding (positional)
            chunk_embedding = self._extract_chunk_embedding(doc_embedding, start, end, len(text))

            chunks.append(Chunk(
                id=f"{document_id}_chunk_{i}",
                text=chunk_text,
                token_count=self.count_tokens(chunk_text),
                char_count=len(chunk_text),
                metadata={'strategy': self.name, 'has_context': True},
                embedding=chunk_embedding,
                start_char=start,
                end_char=end,
                chunk_index=i
            ))

        return chunks
```

### 5.3 Chunker Factory

```python
# src/core/chunking/__init__.py

from typing import Dict
from .base_chunker import BaseChunker, Chunk
from .token_chunker import TokenChunker
from .semantic_chunker import SemanticChunker
from .sentence_chunker import SentenceChunker
from .recursive_chunker import RecursiveChunker
from .late_chunker import LateChunker

CHUNKERS = {
    'token': TokenChunker,
    'semantic': SemanticChunker,
    'sentence': SentenceChunker,
    'recursive': RecursiveChunker,
    'late': LateChunker,
}

def create_chunker(config: Dict, embedder=None) -> BaseChunker:
    """
    Factory function to create chunker based on config.

    Usage:
        chunker = create_chunker(config['chunking'], embedder)
        chunks = chunker.chunk(text, document_id)
    """
    strategy = config.get('strategy', 'semantic')

    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown chunking strategy: {strategy}. "
                        f"Available: {list(CHUNKERS.keys())}")

    strategy_config = config.get('strategies', {}).get(strategy, {})

    # Some chunkers need embedder
    if strategy in ['semantic', 'late']:
        return CHUNKERS[strategy](strategy_config, embedder)
    else:
        return CHUNKERS[strategy](strategy_config)
```

---

## 6. Dynamic Model Management

### 6.1 Overview

All models (LLM and Embedding) are loaded dynamically from HuggingFace with local caching.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL MANAGEMENT SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Configuration                                                              │
│  ─────────────                                                              │
│  models:                                                                    │
│    llm:                                                                     │
│      model_id: "ALLaM-AI/ALLaM-7B-Instruct-preview"  ◄── HuggingFace ID    │
│    embedding:                                                               │
│      model_id: "jinaai/jina-embeddings-v3"           ◄── HuggingFace ID    │
│                                                                             │
│                           │                                                 │
│                           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      MODEL REGISTRY                                  │   │
│  │                                                                      │   │
│  │  1. Check local cache: /data/models_cache/{model_id}                │   │
│  │  2. If not exists: Download from HuggingFace Hub                    │   │
│  │  3. Load model with appropriate backend                             │   │
│  │  4. Return ready-to-use model interface                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│           ┌───────────────┼───────────────┐                                │
│           ▼               ▼               ▼                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                        │
│  │    LLM      │  │  Embedding  │  │  Reranker   │                        │
│  │   (TGI)     │  │   (Local)   │  │   (Local)   │                        │
│  └─────────────┘  └─────────────┘  └─────────────┘                        │
│                                                                             │
│  Local Cache Structure:                                                    │
│  /data/models_cache/                                                       │
│  ├── llm/                                                                  │
│  │   ├── ALLaM-AI--ALLaM-7B-Instruct-preview/                             │
│  │   ├── meta-llama--Llama-3.1-8B-Instruct/                               │
│  │   └── Qwen--Qwen2.5-7B-Instruct/                                       │
│  ├── embedding/                                                            │
│  │   ├── jinaai--jina-embeddings-v3/                                      │
│  │   ├── BAAI--bge-m3/                                                    │
│  │   └── intfloat--multilingual-e5-large/                                 │
│  └── reranker/                                                             │
│      └── BAAI--bge-reranker-v2-m3/                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Model Registry

```python
# src/core/models/registry.py

import os
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import hashlib

class ModelType(Enum):
    LLM = "llm"
    EMBEDDING = "embedding"
    RERANKER = "reranker"

@dataclass
class ModelInfo:
    """Information about a registered model"""
    model_id: str              # HuggingFace ID: "org/model-name"
    model_type: ModelType
    local_path: Optional[Path] = None
    is_downloaded: bool = False
    config: Dict = None

class ModelRegistry:
    """
    Central registry for all models.

    Handles:
    - Model discovery from HuggingFace
    - Local caching
    - Version tracking
    - Lazy loading
    """

    def __init__(self, cache_dir: str = "/data/models_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._models: Dict[str, ModelInfo] = {}
        self._loaded_models: Dict[str, Any] = {}

    def _get_cache_path(self, model_id: str, model_type: ModelType) -> Path:
        """Get local cache path for a model"""
        # Convert org/model to org--model for filesystem
        safe_name = model_id.replace("/", "--")
        return self.cache_dir / model_type.value / safe_name

    def register(self, model_id: str, model_type: ModelType, config: Dict = None) -> ModelInfo:
        """
        Register a model for use.

        Args:
            model_id: HuggingFace model ID (e.g., "jinaai/jina-embeddings-v3")
            model_type: Type of model (LLM, EMBEDDING, RERANKER)
            config: Additional configuration

        Returns:
            ModelInfo with registration details
        """
        cache_path = self._get_cache_path(model_id, model_type)
        is_downloaded = cache_path.exists() and any(cache_path.iterdir())

        info = ModelInfo(
            model_id=model_id,
            model_type=model_type,
            local_path=cache_path,
            is_downloaded=is_downloaded,
            config=config or {}
        )

        self._models[model_id] = info
        return info

    def download(self, model_id: str) -> Path:
        """
        Download model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Path to downloaded model
        """
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not registered")

        info = self._models[model_id]

        if info.is_downloaded:
            print(f"Model {model_id} already cached at {info.local_path}")
            return info.local_path

        print(f"Downloading {model_id} from HuggingFace Hub...")

        from huggingface_hub import snapshot_download

        # Download to cache directory
        info.local_path.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=model_id,
            local_dir=info.local_path,
            local_dir_use_symlinks=False
        )

        info.is_downloaded = True
        print(f"Downloaded {model_id} to {info.local_path}")

        return info.local_path

    def get_model(self, model_id: str, force_download: bool = False) -> Any:
        """
        Get a loaded model, downloading if necessary.

        Args:
            model_id: HuggingFace model ID
            force_download: Re-download even if cached

        Returns:
            Loaded model object
        """
        if model_id in self._loaded_models and not force_download:
            return self._loaded_models[model_id]

        info = self._models.get(model_id)
        if not info:
            raise ValueError(f"Model {model_id} not registered")

        # Download if needed
        if not info.is_downloaded or force_download:
            self.download(model_id)

        # Load based on type
        if info.model_type == ModelType.EMBEDDING:
            model = self._load_embedding_model(info)
        elif info.model_type == ModelType.LLM:
            model = self._load_llm_model(info)
        elif info.model_type == ModelType.RERANKER:
            model = self._load_reranker_model(info)
        else:
            raise ValueError(f"Unknown model type: {info.model_type}")

        self._loaded_models[model_id] = model
        return model

    def _load_embedding_model(self, info: ModelInfo):
        """Load embedding model using sentence-transformers"""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            str(info.local_path),
            trust_remote_code=True,
            device="cuda"
        )

        return model

    def _load_reranker_model(self, info: ModelInfo):
        """Load reranker model"""
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            str(info.local_path),
            trust_remote_code=True,
            device="cuda"
        )

        return model

    def _load_llm_model(self, info: ModelInfo):
        """
        For LLMs, we don't load directly - TGI serves them.
        Return a client pointing to TGI.
        """
        from .llm_client import TGIClient

        return TGIClient(
            model_path=str(info.local_path),
            config=info.config
        )

    def list_models(self) -> Dict[str, ModelInfo]:
        """List all registered models"""
        return self._models.copy()

    def get_disk_usage(self) -> Dict[str, int]:
        """Get disk usage for each model"""
        usage = {}
        for model_id, info in self._models.items():
            if info.local_path and info.local_path.exists():
                size = sum(f.stat().st_size for f in info.local_path.rglob("*") if f.is_file())
                usage[model_id] = size
        return usage
```

### 6.3 Embedding Manager

```python
# src/core/models/embedding_manager.py

from typing import List, Dict, Optional, Union
import numpy as np
from .registry import ModelRegistry, ModelType

class EmbeddingManager:
    """
    Unified interface for embedding models.

    Features:
    - Dynamic model loading from HuggingFace
    - Task-specific embeddings (query vs passage)
    - Batch processing
    - Caching
    """

    def __init__(self, config: Dict, registry: ModelRegistry):
        self.config = config
        self.registry = registry

        # Register the embedding model
        self.model_id = config.get('model_id', 'jinaai/jina-embeddings-v3')
        self.registry.register(self.model_id, ModelType.EMBEDDING, config)

        self.dimensions = config.get('dimensions', 1024)
        self.normalize = config.get('normalize', True)
        self.batch_size = config.get('batch_size', 32)

        self._model = None

    @property
    def model(self):
        """Lazy load model"""
        if self._model is None:
            self._model = self.registry.get_model(self.model_id)
        return self._model

    def embed(self, texts: Union[str, List[str]], task: str = "passage") -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts
            task: Task type for task-specific embeddings
                  - "passage": For documents/chunks
                  - "query": For user queries
                  - "entity": For entity descriptions

        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Get task prefix from config
        task_prefix = self.config.get('tasks', {}).get(task, '')

        # Add task prefix if model supports it (Jina-style)
        if task_prefix and 'jina' in self.model_id.lower():
            texts = [f"{task_prefix}: {t}" for t in texts]

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=self.normalize,
                show_progress_bar=False
            )
            all_embeddings.append(embeddings)

        result = np.vstack(all_embeddings)

        # Ensure correct dimensions
        if result.shape[1] != self.dimensions:
            # Some models support dimension reduction
            result = result[:, :self.dimensions]

        return result

    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """Embed chunks with passage task"""
        texts = [c.get('text', c) if isinstance(c, dict) else c for c in chunks]
        return self.embed(texts, task="passage")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed query with query task"""
        return self.embed(query, task="query")[0]

    def embed_entities(self, entities: List[Dict]) -> np.ndarray:
        """Embed entities with entity task"""
        texts = [
            f"{e['name']} ({e.get('type', 'Entity')}): {e.get('description', '')}"
            for e in entities
        ]
        return self.embed(texts, task="entity")

    def embed_relationships(self, relationships: List[Dict]) -> np.ndarray:
        """Embed relationships"""
        texts = [
            f"{r['source']} {r['type']} {r['target']}: {r.get('description', '')}"
            for r in relationships
        ]
        return self.embed(texts, task="passage")

    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between embeddings"""
        # Normalize if not already
        if not self.normalize:
            embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

        return np.dot(embeddings1, embeddings2.T)
```

### 6.4 LLM Manager

```python
# src/core/models/llm_manager.py

from typing import Dict, List, Optional, AsyncIterator
import httpx
from .registry import ModelRegistry, ModelType

class LLMManager:
    """
    Unified interface for LLM inference via TGI.

    Features:
    - Dynamic model configuration
    - Streaming support
    - Token counting
    - Error handling
    """

    def __init__(self, config: Dict, registry: ModelRegistry):
        self.config = config
        self.registry = registry

        self.model_id = config.get('model_id', 'ALLaM-AI/ALLaM-7B-Instruct-preview')
        self.registry.register(self.model_id, ModelType.LLM, config)

        self.tgi_endpoint = config.get('tgi_endpoint', 'http://tgi:80')
        self.max_tokens = config.get('max_tokens', 4096)
        self.temperature = config.get('temperature', 0.1)
        self.top_p = config.get('top_p', 0.95)

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            stream: Whether to stream response

        Returns:
            Generated text
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "top_p": self.top_p,
                "do_sample": (temperature or self.temperature) > 0,
                "return_full_text": False
            }
        }

        async with httpx.AsyncClient(timeout=120) as client:
            if stream:
                return self._stream_generate(client, payload)
            else:
                response = await client.post(
                    f"{self.tgi_endpoint}/generate",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                return result.get("generated_text", "")

    async def _stream_generate(
        self,
        client: httpx.AsyncClient,
        payload: Dict
    ) -> AsyncIterator[str]:
        """Stream generation token by token"""
        async with client.stream(
            "POST",
            f"{self.tgi_endpoint}/generate_stream",
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = json.loads(line[5:])
                    if token := data.get("token", {}).get("text"):
                        yield token

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Chat-style generation with message history.

        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            **kwargs: Additional generation parameters
        """
        # Format messages into prompt
        prompt = self._format_chat_messages(messages)
        return await self.generate(prompt, **kwargs)

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for the model's expected format"""
        # Different models have different chat templates
        # This is a generic format - override for specific models
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n"

        formatted += "<|assistant|>\n"
        return formatted

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        import tiktoken
        try:
            encoder = tiktoken.encoding_for_model("gpt-4")
        except:
            encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
```

---

## 7. Triple-Level Indexing

### 7.1 Index Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRIPLE-LEVEL INDEX                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LEVEL 1: CHUNK INDEX                                                       │
│  ═════════════════════                                                      │
│  Collection: mirage_chunks                                                  │
│  Purpose: Semantic search on document content                               │
│                                                                             │
│  Schema:                                                                    │
│  {                                                                          │
│    "id": "doc123_chunk_0",        // Unique chunk ID                       │
│    "vector": [0.1, 0.2, ...],     // 1024d embedding                       │
│    "payload": {                                                            │
│      "document_id": "doc123",                                              │
│      "text": "Chunk content...",                                           │
│      "token_count": 450,                                                   │
│      "chunk_index": 0,                                                     │
│      "entity_ids": ["ent_1", "ent_2"],  // Cross-reference to entities    │
│      "metadata": {...}                                                     │
│    }                                                                        │
│  }                                                                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LEVEL 2: ENTITY INDEX                                                      │
│  ══════════════════════                                                     │
│  Collection: mirage_entities                                                │
│  Purpose: Direct entity search, entity-based retrieval                      │
│                                                                             │
│  Schema:                                                                    │
│  {                                                                          │
│    "id": "ent_ahmed_hassan",      // Normalized entity ID                  │
│    "vector": [0.3, 0.4, ...],     // Entity embedding                      │
│    "payload": {                                                            │
│      "name": "Ahmed Hassan",                                               │
│      "normalized_name": "ahmed_hassan",                                    │
│      "type": "Person",                                                     │
│      "description": "CEO of...",                                           │
│      "chunk_ids": ["doc123_chunk_0", "doc456_chunk_2"],                   │
│      "document_ids": ["doc123", "doc456"],                                │
│      "confidence": 0.95,                                                   │
│      "community_id": "comm_1",      // Community membership               │
│      "first_seen": "2024-01-15",    // Temporal tracking                  │
│      "last_seen": "2024-12-03",                                           │
│      "mention_count": 15                                                  │
│    }                                                                        │
│  }                                                                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LEVEL 3: RELATIONSHIP INDEX                                                │
│  ═══════════════════════════                                                │
│  Collection: mirage_relationships                                           │
│  Purpose: Direct relationship search, global queries                        │
│                                                                             │
│  Schema:                                                                    │
│  {                                                                          │
│    "id": "rel_ahmed_leads_company",  // Relationship ID                    │
│    "vector": [0.5, 0.6, ...],        // Relationship embedding             │
│    "payload": {                                                            │
│      "source_id": "ent_ahmed_hassan",                                      │
│      "target_id": "ent_tech_company",                                      │
│      "type": "leads",                                                      │
│      "description": "Ahmed Hassan leads Tech Company since 2020",         │
│      "chunk_ids": ["doc123_chunk_0"],                                     │
│      "confidence": 0.9,                                                   │
│      "strength": "high",              // low, medium, high                │
│      "first_seen": "2024-01-15",                                          │
│      "last_seen": "2024-12-03",                                           │
│      "mention_count": 8,                                                  │
│      "decayed_confidence": 0.85       // After confidence decay           │
│    }                                                                        │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Index Manager

```python
# src/core/indexing/index_manager.py

from typing import List, Dict, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

@dataclass
class IndexingResult:
    """Result of indexing operation"""
    document_id: str
    chunks_indexed: int
    entities_indexed: int
    relationships_indexed: int
    communities_updated: int
    duration_ms: int

class IndexManager:
    """
    Manages triple-level indexing for MIRAGE v2.

    Coordinates:
    - Chunk index (semantic search)
    - Entity index (entity search)
    - Relationship index (relationship search)
    - Graph storage (Neo4j)
    """

    def __init__(self, config: Dict, embedder, neo4j_client):
        self.config = config
        self.embedder = embedder
        self.neo4j = neo4j_client

        # Initialize Qdrant
        self.qdrant = QdrantClient(
            host=config['storage']['qdrant']['host'],
            port=config['storage']['qdrant']['port']
        )

        # Collection names
        self.chunk_collection = config['storage']['qdrant']['collections']['chunks']
        self.entity_collection = config['storage']['qdrant']['collections']['entities']
        self.relationship_collection = config['storage']['qdrant']['collections']['relationships']

        # Ensure collections exist
        self._ensure_collections()

    def _ensure_collections(self):
        """Create collections if they don't exist"""
        vector_size = self.config['models']['embedding']['dimensions']

        collections = [
            self.chunk_collection,
            self.entity_collection,
            self.relationship_collection
        ]

        existing = {c.name for c in self.qdrant.get_collections().collections}

        for collection in collections:
            if collection not in existing:
                self.qdrant.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )

    async def index_document(
        self,
        document_id: str,
        chunks: List[Dict],
        entities: List[Dict],
        relationships: List[Dict]
    ) -> IndexingResult:
        """
        Index a document with all its components.

        Args:
            document_id: Document identifier
            chunks: List of chunks with text
            entities: Extracted entities
            relationships: Extracted relationships

        Returns:
            IndexingResult with counts
        """
        import time
        start = time.time()

        # 1. Generate embeddings
        chunk_embeddings = self.embedder.embed_chunks(chunks)
        entity_embeddings = self.embedder.embed_entities(entities)
        relationship_embeddings = self.embedder.embed_relationships(relationships)

        # 2. Index chunks
        chunk_points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_id = f"{document_id}_chunk_{i}"

            # Find entities mentioned in this chunk
            entity_ids = self._find_entities_in_chunk(chunk['text'], entities)

            chunk_points.append(PointStruct(
                id=self._hash_id(chunk_id),
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "text": chunk['text'],
                    "token_count": chunk.get('token_count', 0),
                    "chunk_index": i,
                    "entity_ids": entity_ids,
                    "metadata": chunk.get('metadata', {})
                }
            ))

        self.qdrant.upsert(
            collection_name=self.chunk_collection,
            points=chunk_points
        )

        # 3. Index entities
        entity_points = []
        for entity, embedding in zip(entities, entity_embeddings):
            entity_id = self._normalize_entity_id(entity['name'])

            # Find chunks mentioning this entity
            chunk_ids = [
                f"{document_id}_chunk_{i}"
                for i, chunk in enumerate(chunks)
                if entity['name'].lower() in chunk['text'].lower()
            ]

            entity_points.append(PointStruct(
                id=self._hash_id(entity_id),
                vector=embedding.tolist(),
                payload={
                    "entity_id": entity_id,
                    "name": entity['name'],
                    "type": entity.get('type', 'Entity'),
                    "description": entity.get('description', ''),
                    "chunk_ids": chunk_ids,
                    "document_ids": [document_id],
                    "confidence": entity.get('confidence', 1.0),
                    "first_seen": self._now(),
                    "last_seen": self._now(),
                    "mention_count": len(chunk_ids)
                }
            ))

        self.qdrant.upsert(
            collection_name=self.entity_collection,
            points=entity_points
        )

        # 4. Index relationships
        rel_points = []
        for rel, embedding in zip(relationships, relationship_embeddings):
            rel_id = f"rel_{self._normalize_entity_id(rel['source'])}_{rel['type']}_{self._normalize_entity_id(rel['target'])}"

            # Find chunks mentioning both entities
            chunk_ids = [
                f"{document_id}_chunk_{i}"
                for i, chunk in enumerate(chunks)
                if rel['source'].lower() in chunk['text'].lower()
                and rel['target'].lower() in chunk['text'].lower()
            ]

            rel_points.append(PointStruct(
                id=self._hash_id(rel_id),
                vector=embedding.tolist(),
                payload={
                    "relationship_id": rel_id,
                    "source_id": self._normalize_entity_id(rel['source']),
                    "target_id": self._normalize_entity_id(rel['target']),
                    "type": rel['type'],
                    "description": rel.get('description', ''),
                    "chunk_ids": chunk_ids,
                    "confidence": rel.get('confidence', 1.0),
                    "strength": rel.get('strength', 'medium'),
                    "first_seen": self._now(),
                    "last_seen": self._now(),
                    "mention_count": len(chunk_ids)
                }
            ))

        self.qdrant.upsert(
            collection_name=self.relationship_collection,
            points=rel_points
        )

        # 5. Update graph (Neo4j)
        await self._update_graph(document_id, chunks, entities, relationships)

        # 6. Update communities
        communities_updated = 0
        if self.config['graph']['community_detection']['enabled']:
            communities_updated = await self._update_communities()

        duration = int((time.time() - start) * 1000)

        return IndexingResult(
            document_id=document_id,
            chunks_indexed=len(chunks),
            entities_indexed=len(entities),
            relationships_indexed=len(relationships),
            communities_updated=communities_updated,
            duration_ms=duration
        )
```

---

## 8. Graph Features

### 8.1 Community Detection

```python
# src/core/graph/community_detection.py

from typing import List, Dict, Optional
import networkx as nx
from dataclasses import dataclass

@dataclass
class Community:
    """A community of related entities"""
    id: str
    entities: List[str]
    relationships: List[Dict]
    summary: Optional[str] = None
    level: int = 0  # Hierarchy level
    parent_id: Optional[str] = None

class CommunityDetector:
    """
    Detects communities in the entity graph using Leiden algorithm.

    Communities help answer global queries by grouping related entities.
    """

    def __init__(self, config: Dict, neo4j_client, llm_manager):
        self.config = config
        self.neo4j = neo4j_client
        self.llm = llm_manager

        self.resolution = config.get('resolution', 1.0)
        self.min_community_size = config.get('min_community_size', 3)

    async def detect_communities(self) -> List[Community]:
        """
        Detect communities in the entity graph.

        Returns:
            List of Community objects
        """
        # 1. Build NetworkX graph from Neo4j
        G = await self._build_networkx_graph()

        if G.number_of_nodes() < self.min_community_size:
            return []

        # 2. Run Leiden algorithm
        try:
            import leidenalg
            import igraph as ig

            # Convert to igraph
            ig_graph = ig.Graph.from_networkx(G)

            # Run Leiden
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                resolution_parameter=self.resolution
            )

            # Convert to communities
            communities = []
            for i, members in enumerate(partition):
                if len(members) >= self.min_community_size:
                    entity_names = [ig_graph.vs[m]['name'] for m in members]

                    # Get relationships within community
                    relationships = await self._get_community_relationships(entity_names)

                    communities.append(Community(
                        id=f"comm_{i}",
                        entities=entity_names,
                        relationships=relationships,
                        level=0
                    ))

        except ImportError:
            # Fallback to Louvain if Leiden not available
            import community as community_louvain

            partition = community_louvain.best_partition(G, resolution=self.resolution)

            # Group by community
            comm_members = {}
            for node, comm_id in partition.items():
                if comm_id not in comm_members:
                    comm_members[comm_id] = []
                comm_members[comm_id].append(node)

            communities = []
            for comm_id, members in comm_members.items():
                if len(members) >= self.min_community_size:
                    relationships = await self._get_community_relationships(members)

                    communities.append(Community(
                        id=f"comm_{comm_id}",
                        entities=members,
                        relationships=relationships,
                        level=0
                    ))

        # 3. Generate summaries if enabled
        if self.config.get('hierarchical_summaries', {}).get('enabled', False):
            for community in communities:
                community.summary = await self._generate_community_summary(community)

        # 4. Store communities in Neo4j
        await self._store_communities(communities)

        return communities

    async def _generate_community_summary(self, community: Community) -> str:
        """Generate a summary of the community using LLM"""
        # Build context from entities and relationships
        context = f"Entities: {', '.join(community.entities[:20])}\n\n"
        context += "Relationships:\n"
        for rel in community.relationships[:15]:
            context += f"- {rel['source']} {rel['type']} {rel['target']}\n"

        prompt = f"""Summarize the following group of related entities and their relationships in 2-3 sentences.
Focus on the main theme or topic that connects them.

{context}

Summary:"""

        summary = await self.llm.generate(prompt, max_tokens=200)
        return summary.strip()

    async def build_hierarchy(self, communities: List[Community]) -> List[Community]:
        """
        Build hierarchical community structure.

        Level 0: Base communities (from Leiden)
        Level 1: Groups of related communities
        Level 2: Global summary
        """
        max_levels = self.config.get('hierarchical_summaries', {}).get('levels', 3)

        all_communities = communities.copy()
        current_level_communities = communities

        for level in range(1, max_levels):
            if len(current_level_communities) < 2:
                break

            # Merge small communities into larger ones
            higher_communities = await self._merge_communities(
                current_level_communities, level
            )

            all_communities.extend(higher_communities)
            current_level_communities = higher_communities

        return all_communities
```

### 8.2 Temporal Relationships

```python
# src/core/graph/temporal.py

from datetime import datetime, timedelta
from typing import List, Dict, Optional

class TemporalManager:
    """
    Manages temporal aspects of the knowledge graph.

    Features:
    - Track when entities/relationships first appeared
    - Track when they were last mentioned
    - Enable temporal queries ("What happened last month?")
    - Apply confidence decay over time
    """

    def __init__(self, config: Dict, neo4j_client):
        self.config = config
        self.neo4j = neo4j_client

        self.decay_enabled = config.get('confidence_decay', {}).get('enabled', False)
        self.decay_function = config.get('confidence_decay', {}).get('decay_function', 'exponential')
        self.half_life_days = config.get('confidence_decay', {}).get('half_life_days', 30)
        self.min_confidence = config.get('confidence_decay', {}).get('min_confidence', 0.1)
        self.reinforcement_boost = config.get('confidence_decay', {}).get('reinforcement_boost', 0.2)

    async def update_temporal_info(
        self,
        entity_id: str = None,
        relationship_id: str = None,
        timestamp: datetime = None
    ):
        """
        Update temporal information when entity/relationship is mentioned.

        Args:
            entity_id: Entity to update (optional)
            relationship_id: Relationship to update (optional)
            timestamp: When it was mentioned (default: now)
        """
        timestamp = timestamp or datetime.now()

        if entity_id:
            await self._update_entity_temporal(entity_id, timestamp)

        if relationship_id:
            await self._update_relationship_temporal(relationship_id, timestamp)

    async def _update_entity_temporal(self, entity_id: str, timestamp: datetime):
        """Update entity temporal info"""
        query = """
        MATCH (e:Entity {id: $entity_id})
        SET e.last_seen = $timestamp,
            e.mention_count = COALESCE(e.mention_count, 0) + 1
        WITH e
        // Set first_seen if not set
        WHERE e.first_seen IS NULL
        SET e.first_seen = $timestamp
        """
        await self.neo4j.run_query(query, {
            'entity_id': entity_id,
            'timestamp': timestamp.isoformat()
        })

    def calculate_decayed_confidence(
        self,
        original_confidence: float,
        last_seen: datetime,
        current_time: datetime = None
    ) -> float:
        """
        Calculate confidence after decay.

        Exponential decay: C(t) = C_0 * 0.5^(t/half_life)
        Linear decay: C(t) = C_0 * (1 - t/max_age)
        """
        if not self.decay_enabled:
            return original_confidence

        current_time = current_time or datetime.now()
        days_since_seen = (current_time - last_seen).days

        if self.decay_function == 'exponential':
            decay_factor = 0.5 ** (days_since_seen / self.half_life_days)
        else:  # linear
            max_age = self.half_life_days * 3
            decay_factor = max(0, 1 - days_since_seen / max_age)

        decayed = original_confidence * decay_factor
        return max(decayed, self.min_confidence)

    def apply_reinforcement(self, current_confidence: float) -> float:
        """
        Boost confidence when entity/relationship is seen again.

        Prevents valuable information from decaying too much.
        """
        boosted = current_confidence + self.reinforcement_boost
        return min(boosted, 1.0)  # Cap at 1.0

    async def filter_by_time_range(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        entity_type: str = None
    ) -> List[Dict]:
        """
        Get entities/relationships within a time range.

        Useful for queries like "What happened last month?"
        """
        query = """
        MATCH (e:Entity)
        WHERE ($start_date IS NULL OR e.first_seen >= $start_date)
          AND ($end_date IS NULL OR e.first_seen <= $end_date)
          AND ($entity_type IS NULL OR e.type = $entity_type)
        RETURN e
        ORDER BY e.first_seen DESC
        """
        return await self.neo4j.run_query(query, {
            'start_date': start_date.isoformat() if start_date else None,
            'end_date': end_date.isoformat() if end_date else None,
            'entity_type': entity_type
        })

    async def apply_decay_to_all(self):
        """
        Apply confidence decay to all entities and relationships.

        Should be run periodically (e.g., daily).
        """
        current_time = datetime.now()

        # Update entities
        entity_query = """
        MATCH (e:Entity)
        WHERE e.last_seen IS NOT NULL
        SET e.decayed_confidence = $min_confidence +
            (e.confidence - $min_confidence) *
            (0.5 ^ (duration.inDays(datetime(e.last_seen), datetime()).days / $half_life))
        RETURN count(e) as updated
        """

        # Update relationships
        rel_query = """
        MATCH ()-[r]->()
        WHERE r.last_seen IS NOT NULL
        SET r.decayed_confidence = $min_confidence +
            (r.confidence - $min_confidence) *
            (0.5 ^ (duration.inDays(datetime(r.last_seen), datetime()).days / $half_life))
        RETURN count(r) as updated
        """

        await self.neo4j.run_query(entity_query, {
            'min_confidence': self.min_confidence,
            'half_life': self.half_life_days
        })

        await self.neo4j.run_query(rel_query, {
            'min_confidence': self.min_confidence,
            'half_life': self.half_life_days
        })
```

---

## 9. Retrieval Engine

### 9.1 Strongest Retrieval Design

After analyzing LightRAG, GraphRAG, and current MIRAGE, here's the **strongest retrieval design**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MIRAGE v2 RETRIEVAL ENGINE                                │
│                    "STRONGEST HYBRID RETRIEVAL"                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: User Query                                                          │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    QUERY ANALYSIS                                    │   │
│  │                                                                      │   │
│  │  1. Language Detection (Arabic/English/Mixed)                       │   │
│  │  2. Query Type Classification (factual/analytical/temporal/global)  │   │
│  │  3. Keyword Extraction (bilingual)                                  │   │
│  │  4. Entity Mention Detection                                        │   │
│  │  5. Temporal Expression Parsing ("last month", "في 2023")           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MODE SELECTION                                    │   │
│  │                                                                      │   │
│  │  Router selects mode based on query analysis:                       │   │
│  │                                                                      │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                   │   │
│  │  │  NAIVE  │ │  LOCAL  │ │ GLOBAL  │ │ HYBRID  │                   │   │
│  │  │ Vector  │ │ Entity  │ │ Relat.  │ │  L+G    │                   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘                   │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                               │   │
│  │  │   MIX   │ │SEMANTIC │ │ BYPASS  │                               │   │
│  │  │  All    │ │ Deep    │ │ No RAG  │                               │   │
│  │  └─────────┘ └─────────┘ └─────────┘                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PARALLEL RETRIEVAL                                │   │
│  │                                                                      │   │
│  │  ┌───────────────────┐  ┌───────────────────┐  ┌─────────────────┐ │   │
│  │  │   CHUNK SEARCH    │  │   ENTITY SEARCH   │  │   REL SEARCH    │ │   │
│  │  │                   │  │                   │  │                 │ │   │
│  │  │ Vector similarity │  │ Vector similarity │  │ Vector similar. │ │   │
│  │  │ on chunk index    │  │ on entity index   │  │ on rel index    │ │   │
│  │  │                   │  │                   │  │                 │ │   │
│  │  │ Returns: chunks   │  │ Returns: entities │  │ Returns: rels   │ │   │
│  │  └─────────┬─────────┘  └─────────┬─────────┘  └────────┬────────┘ │   │
│  │            │                      │                      │          │   │
│  │            ▼                      ▼                      ▼          │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │                    GRAPH EXPANSION                           │   │   │
│  │  │                                                              │   │   │
│  │  │  From entities found:                                        │   │   │
│  │  │  1. Traverse to related entities (1-2 hops)                 │   │   │
│  │  │  2. Include community members                                │   │   │
│  │  │  3. Follow high-confidence relationships                    │   │   │
│  │  │  4. Apply temporal filtering if query has time aspect       │   │   │
│  │  │  5. Apply confidence decay                                   │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │            │                                                        │   │
│  │            ▼                                                        │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │                    COMMUNITY CONTEXT                         │   │   │
│  │  │                                                              │   │   │
│  │  │  If global query or entities span multiple communities:     │   │   │
│  │  │  1. Identify relevant communities                           │   │   │
│  │  │  2. Include community summaries                              │   │   │
│  │  │  3. Add hierarchical context if needed                      │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    FUSION & RANKING                                  │   │
│  │                                                                      │   │
│  │  1. Reciprocal Rank Fusion (RRF) across all sources                │   │
│  │  2. Apply mode-specific weights                                     │   │
│  │  3. Cross-encoder reranking (if enabled)                           │   │
│  │  4. Deduplicate overlapping chunks                                  │   │
│  │  5. Enforce token budget for context                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│      │                                                                      │
│      ▼                                                                      │
│  OUTPUT: Ranked chunks + entities + relationships + community context      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Mode Implementations

```python
# src/core/retrieval/mix_retriever.py

class MixRetriever(BaseRetriever):
    """
    MIX MODE: The most comprehensive retrieval.

    Combines all retrieval strategies:
    1. Naive (vector on chunks) - catches direct semantic matches
    2. Local (entity-focused) - finds specific entity information
    3. Global (relationship-focused) - captures thematic connections

    This is the STRONGEST mode for complex queries.
    """

    name = "mix"

    def __init__(
        self,
        chunk_index,
        entity_index,
        relationship_index,
        graph_client,
        community_detector,
        temporal_manager,
        config
    ):
        self.chunk_index = chunk_index
        self.entity_index = entity_index
        self.relationship_index = relationship_index
        self.graph = graph_client
        self.communities = community_detector
        self.temporal = temporal_manager
        self.config = config

        # Mode-specific weights
        self.naive_weight = config.get('naive_weight', 0.2)
        self.local_weight = config.get('local_weight', 0.4)
        self.global_weight = config.get('global_weight', 0.4)

    async def retrieve(self, query: str, query_analysis: Dict) -> RetrievalResult:
        """
        Execute mix retrieval.

        Args:
            query: User query
            query_analysis: Pre-analyzed query info (language, type, entities, etc.)

        Returns:
            RetrievalResult with all components
        """
        import asyncio

        # 1. Parallel search across all indices
        naive_task = self._naive_search(query)
        local_task = self._local_search(query, query_analysis)
        global_task = self._global_search(query, query_analysis)

        naive_result, local_result, global_result = await asyncio.gather(
            naive_task, local_task, global_task
        )

        # 2. Temporal filtering if query has time aspect
        if query_analysis.get('has_temporal'):
            time_range = query_analysis.get('time_range')
            naive_result = self._apply_temporal_filter(naive_result, time_range)
            local_result = self._apply_temporal_filter(local_result, time_range)
            global_result = self._apply_temporal_filter(global_result, time_range)

        # 3. Community context for global understanding
        community_context = []
        if query_analysis.get('query_type') in ['global', 'analytical']:
            community_context = await self._get_community_context(
                local_result.entities + global_result.entities
            )

        # 4. RRF Fusion
        fused = self._rrf_fusion([
            (naive_result, self.naive_weight),
            (local_result, self.local_weight),
            (global_result, self.global_weight)
        ])

        # 5. Add community summaries to context
        fused.community_summaries = community_context

        # 6. Apply confidence decay
        fused = self._apply_confidence_decay(fused)

        return fused

    async def _naive_search(self, query: str) -> RetrievalResult:
        """Direct vector search on chunks"""
        chunks = await self.chunk_index.search(
            query,
            limit=self.config.get('naive', {}).get('top_k', 10)
        )

        return RetrievalResult(
            chunks=chunks,
            entities=[],
            relationships=[],
            mode="naive"
        )

    async def _local_search(self, query: str, analysis: Dict) -> RetrievalResult:
        """Entity-focused search with graph expansion"""
        # Search entity index
        entities = await self.entity_index.search(
            query,
            limit=self.config.get('local', {}).get('entity_top_k', 5)
        )

        # Also search for mentioned entities
        if analysis.get('mentioned_entities'):
            for entity_name in analysis['mentioned_entities']:
                exact_match = await self.entity_index.get_by_name(entity_name)
                if exact_match and exact_match not in entities:
                    entities.append(exact_match)

        # Graph expansion
        expanded_entities = await self._expand_via_graph(
            entities,
            hops=self.config.get('local', {}).get('graph_hops', 2)
        )

        # Get chunks for all entities
        all_entities = entities + expanded_entities
        chunks = await self._get_chunks_for_entities(all_entities)

        # Get relationships between found entities
        relationships = await self._get_relationships_between(all_entities)

        return RetrievalResult(
            chunks=chunks,
            entities=all_entities,
            relationships=relationships,
            mode="local"
        )

    async def _global_search(self, query: str, analysis: Dict) -> RetrievalResult:
        """Relationship-focused search"""
        # Search relationship index
        relationships = await self.relationship_index.search(
            query,
            limit=self.config.get('global', {}).get('relationship_top_k', 10)
        )

        # Extract entities from relationships
        entity_ids = set()
        for rel in relationships:
            entity_ids.add(rel['source_id'])
            entity_ids.add(rel['target_id'])

        entities = await self._get_entities_by_ids(list(entity_ids))

        # Get community context
        if self.config.get('global', {}).get('use_hierarchical_summaries', True):
            communities = await self._get_communities_for_entities(entities)
            community_chunks = [
                {"text": c.summary, "type": "community_summary", "community_id": c.id}
                for c in communities if c.summary
            ]
        else:
            community_chunks = []

        # Get chunks for relationships
        chunks = await self._get_chunks_for_relationships(relationships)
        chunks.extend(community_chunks)

        return RetrievalResult(
            chunks=chunks,
            entities=entities,
            relationships=relationships,
            mode="global"
        )

    async def _expand_via_graph(
        self,
        entities: List[Dict],
        hops: int = 2
    ) -> List[Dict]:
        """Expand to related entities via graph traversal"""
        entity_names = [e['name'] for e in entities]

        query = f"""
        MATCH (e:Entity)-[r*1..{hops}]-(related:Entity)
        WHERE e.name IN $entity_names
          AND NOT related.name IN $entity_names
          AND (related.decayed_confidence IS NULL OR related.decayed_confidence >= $min_confidence)
        WITH related, min(length(r)) as distance
        RETURN DISTINCT related.name as name,
               related.type as type,
               related.description as description,
               related.confidence as confidence,
               related.decayed_confidence as decayed_confidence,
               related.community_id as community_id,
               distance
        ORDER BY distance, related.decayed_confidence DESC
        LIMIT 30
        """

        results = await self.graph.run_query(query, {
            'entity_names': entity_names,
            'min_confidence': self.config.get('confidence_filter', {}).get('min_entity_confidence', 0.3)
        })

        # Apply distance-based scoring
        decay = self.config.get('graph', {}).get('traversal', {}).get('relationship_decay', 0.7)
        expanded = []
        for r in results:
            r['expansion_score'] = decay ** r['distance']
            expanded.append(r)

        return expanded

    async def _get_community_context(self, entities: List[Dict]) -> List[str]:
        """Get community summaries for relevant entities"""
        community_ids = set()
        for entity in entities:
            if entity.get('community_id'):
                community_ids.add(entity['community_id'])

        if not community_ids:
            return []

        query = """
        MATCH (c:Community)
        WHERE c.id IN $community_ids
        RETURN c.id as id, c.summary as summary, c.level as level
        ORDER BY c.level DESC
        """

        results = await self.graph.run_query(query, {
            'community_ids': list(community_ids)
        })

        return [r['summary'] for r in results if r.get('summary')]

    def _rrf_fusion(
        self,
        results_with_weights: List[tuple]
    ) -> RetrievalResult:
        """
        Reciprocal Rank Fusion across multiple result sets.

        RRF score = sum(weight_i / (k + rank_i))
        k = 60 (standard constant)
        """
        k = 60
        chunk_scores = {}
        entity_scores = {}
        relationship_scores = {}

        for result, weight in results_with_weights:
            # Score chunks
            for rank, chunk in enumerate(result.chunks, 1):
                chunk_id = chunk.get('chunk_id', chunk.get('id', str(rank)))
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {'chunk': chunk, 'score': 0}
                chunk_scores[chunk_id]['score'] += weight / (k + rank)

            # Score entities
            for rank, entity in enumerate(result.entities, 1):
                entity_id = entity.get('entity_id', entity.get('name'))
                if entity_id not in entity_scores:
                    entity_scores[entity_id] = {'entity': entity, 'score': 0}
                entity_scores[entity_id]['score'] += weight / (k + rank)

            # Score relationships
            for rank, rel in enumerate(result.relationships, 1):
                rel_id = rel.get('relationship_id', f"{rel.get('source_id')}_{rel.get('target_id')}")
                if rel_id not in relationship_scores:
                    relationship_scores[rel_id] = {'relationship': rel, 'score': 0}
                relationship_scores[rel_id]['score'] += weight / (k + rank)

        # Sort by score
        sorted_chunks = sorted(
            [{'score': v['score'], **v['chunk']} for v in chunk_scores.values()],
            key=lambda x: x['score'],
            reverse=True
        )

        sorted_entities = sorted(
            [{'score': v['score'], **v['entity']} for v in entity_scores.values()],
            key=lambda x: x['score'],
            reverse=True
        )

        sorted_relationships = sorted(
            [{'score': v['score'], **v['relationship']} for v in relationship_scores.values()],
            key=lambda x: x['score'],
            reverse=True
        )

        return RetrievalResult(
            chunks=sorted_chunks[:20],
            entities=sorted_entities[:15],
            relationships=sorted_relationships[:10],
            mode="mix"
        )
```

---

## 10. Prompt System

### 10.1 Prompt Registry Structure

```yaml
# config/prompts/extraction/entity_v1.yaml
name: "entity_extraction_v1"
version: "1.0.0"
language: "bilingual"  # ar, en, bilingual
description: "Standard entity extraction prompt with few-shot examples"

system_prompt: |
  You are an expert at extracting entities and relationships from text.
  Extract ALL entities and their relationships from the given text.

  Entity Types: Person, Organization, Location, Event, Technology, Policy, Product, Date
  Relationship Types: leads, manages, founded, works_for, located_in, participated_in, developed, announced

few_shot_examples:
  - input: |
      Ahmed Hassan, CEO of Tech Company, announced the new AI product at the Dubai Conference.
    output: |
      {
        "entities": [
          {"name": "Ahmed Hassan", "type": "Person", "description": "CEO of Tech Company"},
          {"name": "Tech Company", "type": "Organization"},
          {"name": "Dubai Conference", "type": "Event"},
          {"name": "AI product", "type": "Product"}
        ],
        "relationships": [
          {"source": "Ahmed Hassan", "type": "leads", "target": "Tech Company"},
          {"source": "Ahmed Hassan", "type": "announced", "target": "AI product"},
          {"source": "AI product", "type": "announced_at", "target": "Dubai Conference"}
        ]
      }

  - input: |
      أعلن الدكتور محمد العلي، رئيس شركة الابتكار، عن إطلاق منتج جديد في مؤتمر الرياض للتقنية.
    output: |
      {
        "entities": [
          {"name": "محمد العلي", "type": "Person", "description": "رئيس شركة الابتكار"},
          {"name": "شركة الابتكار", "type": "Organization"},
          {"name": "مؤتمر الرياض للتقنية", "type": "Event"},
          {"name": "منتج جديد", "type": "Product"}
        ],
        "relationships": [
          {"source": "محمد العلي", "type": "leads", "target": "شركة الابتكار"},
          {"source": "محمد العلي", "type": "announced", "target": "منتج جديد"},
          {"source": "منتج جديد", "type": "announced_at", "target": "مؤتمر الرياض للتقنية"}
        ]
      }

chain_of_thought:
  enabled: true
  steps:
    - "First, identify all named entities in the text"
    - "Classify each entity by type"
    - "Find relationships between entities"
    - "Verify each relationship has supporting evidence"
```

### 10.2 Generation Prompt with CoT

```yaml
# config/prompts/generation/answer_v1.yaml
name: "answer_generation_v1"
version: "1.0.0"
language: "bilingual"

system_prompt: |
  You are a helpful assistant that answers questions based on provided context.
  Always cite your sources using [1], [2], etc.
  If the context doesn't contain enough information, say so clearly.
  Respond in the same language as the question.

chain_of_thought:
  enabled: true
  template: |
    Let me analyze this step by step:

    1. UNDERSTAND THE QUERY:
    {query_analysis}

    2. RELEVANT CONTEXT:
    {context_summary}

    3. KEY ENTITIES INVOLVED:
    {entity_list}

    4. RELATIONSHIPS:
    {relationship_summary}

    5. SYNTHESIZED ANSWER:
    {answer}

    CITATIONS:
    {citations}

few_shot_examples:
  - query: "Who leads Tech Company?"
    context: |
      [1] Ahmed Hassan is the CEO of Tech Company since 2020.
      [2] Tech Company was founded in Dubai.
    answer: |
      Ahmed Hassan leads Tech Company as CEO since 2020 [1].

  - query: "من يقود شركة التقنية؟"
    context: |
      [1] أحمد حسن هو الرئيس التنفيذي لشركة التقنية منذ 2020.
      [2] تأسست شركة التقنية في دبي.
    answer: |
      يقود أحمد حسن شركة التقنية كرئيس تنفيذي منذ عام 2020 [1].
```

### 10.3 Prompt Manager

```python
# src/core/generation/prompt_manager.py

from pathlib import Path
from typing import Dict, Optional
import yaml

class PromptManager:
    """
    Manages versioned prompts for all system components.

    Features:
    - Load prompts from YAML files
    - Version tracking
    - Few-shot example injection
    - Chain-of-thought formatting
    - Bilingual support
    """

    def __init__(self, prompts_dir: str = "config/prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, Dict] = {}

    def load_prompt(self, category: str, name: str) -> Dict:
        """
        Load a prompt by category and name.

        Args:
            category: Prompt category (extraction, generation, analysis)
            name: Prompt name (e.g., entity_v1, answer_v1)

        Returns:
            Prompt configuration dict
        """
        cache_key = f"{category}/{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt_path = self.prompts_dir / category / f"{name}.yaml"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")

        with open(prompt_path) as f:
            prompt = yaml.safe_load(f)

        self._cache[cache_key] = prompt
        return prompt

    def format_extraction_prompt(
        self,
        text: str,
        prompt_name: str = "entity_v1",
        include_examples: bool = True,
        use_cot: bool = True
    ) -> str:
        """
        Format extraction prompt with text.

        Args:
            text: Text to extract from
            prompt_name: Which prompt version to use
            include_examples: Include few-shot examples
            use_cot: Use chain-of-thought

        Returns:
            Formatted prompt string
        """
        prompt = self.load_prompt("extraction", prompt_name)

        parts = [prompt['system_prompt']]

        if include_examples and prompt.get('few_shot_examples'):
            parts.append("\n## Examples:\n")
            for i, example in enumerate(prompt['few_shot_examples'], 1):
                parts.append(f"### Example {i}:")
                parts.append(f"Input:\n{example['input']}")
                parts.append(f"Output:\n{example['output']}\n")

        if use_cot and prompt.get('chain_of_thought', {}).get('enabled'):
            parts.append("\n## Instructions:")
            for step in prompt['chain_of_thought']['steps']:
                parts.append(f"- {step}")

        parts.append(f"\n## Text to analyze:\n{text}")
        parts.append("\n## Your extraction (JSON format):")

        return "\n".join(parts)

    def format_generation_prompt(
        self,
        query: str,
        context: str,
        entities: list,
        relationships: list,
        prompt_name: str = "answer_v1",
        use_cot: bool = True
    ) -> str:
        """
        Format generation prompt with context.

        Args:
            query: User query
            context: Retrieved context (chunks)
            entities: Found entities
            relationships: Found relationships
            prompt_name: Which prompt version
            use_cot: Use chain-of-thought

        Returns:
            Formatted prompt string
        """
        prompt = self.load_prompt("generation", prompt_name)

        parts = [prompt['system_prompt']]

        if use_cot and prompt.get('chain_of_thought', {}).get('enabled'):
            # Format CoT template
            cot = prompt['chain_of_thought']['template']
            cot = cot.replace("{query_analysis}", f"Query: {query}")
            cot = cot.replace("{context_summary}", "See context below")
            cot = cot.replace("{entity_list}", ", ".join(e.get('name', str(e)) for e in entities[:10]))
            cot = cot.replace("{relationship_summary}",
                            "\n".join(f"- {r.get('source')} {r.get('type')} {r.get('target')}"
                                     for r in relationships[:5]))
            parts.append(cot)

        parts.append(f"\n## Context:\n{context}")
        parts.append(f"\n## Question:\n{query}")
        parts.append("\n## Answer:")

        return "\n".join(parts)
```

---

## 11. Evaluation Framework

### 11.1 Separation of Core and Evaluation

```
mirage/
├── src/                    # CORE SYSTEM (production code)
│   └── ...
│
└── evaluation/             # EVALUATION FRAMEWORK (separate)
    ├── config/
    │   └── experiments/    # Experiment configs
    ├── datasets/           # Test datasets
    ├── metrics/            # Metric implementations
    ├── runners/            # Experiment runners
    ├── reports/            # Generated reports
    └── analysis/           # Analysis tools
```

### 11.2 Experiment Configuration

```yaml
# evaluation/config/experiments/llm_comparison.yaml
experiment:
  id: "llm_comparison_001"
  name: "LLM Comparison: Allam vs Llama3 vs Qwen"
  description: "Compare extraction and generation quality across LLMs"
  created_at: "2024-12-03"

# What we're varying
variables:
  llm:
    - model_id: "ALLaM-AI/ALLaM-7B-Instruct-preview"
      name: "Allam-7B"
    - model_id: "meta-llama/Llama-3.1-8B-Instruct"
      name: "Llama3-8B"
    - model_id: "Qwen/Qwen2.5-7B-Instruct"
      name: "Qwen2.5-7B"

# What we're keeping constant
constants:
  embedding:
    model_id: "jinaai/jina-embeddings-v3"
  chunking:
    strategy: "semantic"
    params:
      similarity_threshold: 0.7
      max_chunk_tokens: 600
  retrieval:
    mode: "mix"
  arabic:
    processor: "camel"

# Test dataset
dataset:
  path: "evaluation/datasets/test_queries_v1.json"
  num_samples: 50

# Metrics to collect
metrics:
  - retrieval_precision
  - retrieval_recall
  - retrieval_f1
  - answer_relevancy
  - answer_faithfulness
  - entity_extraction_f1
  - relationship_extraction_f1
  - latency_p50
  - latency_p95

# Output
output:
  save_results: true
  generate_report: true
  report_format: "markdown"
```

### 11.3 Test Dataset Format

```json
// evaluation/datasets/test_queries_v1.json
{
  "version": "1.0",
  "created_at": "2024-12-03",
  "description": "Test queries for MIRAGE evaluation",
  "queries": [
    {
      "id": "q001",
      "query": "Who is Ahmed Hassan?",
      "query_ar": "من هو أحمد حسن؟",
      "language": "en",
      "complexity": "simple",  // simple, medium, complex
      "query_type": "factual",  // factual, analytical, temporal, global
      "ground_truth": {
        "relevant_chunks": ["doc1_chunk_0", "doc1_chunk_2"],
        "relevant_entities": ["Ahmed Hassan", "Tech Company"],
        "relevant_relationships": [
          {"source": "Ahmed Hassan", "type": "leads", "target": "Tech Company"}
        ],
        "expected_answer_keywords": ["CEO", "Tech Company", "2020"],
        "expected_answer": "Ahmed Hassan is the CEO of Tech Company since 2020."
      }
    },
    {
      "id": "q002",
      "query": "ما هي المواضيع الرئيسية في الوثائق؟",
      "language": "ar",
      "complexity": "complex",
      "query_type": "global",
      "ground_truth": {
        "relevant_chunks": ["doc1_chunk_0", "doc2_chunk_1", "doc3_chunk_0"],
        "relevant_entities": ["Tech Company", "Innovation", "AI"],
        "expected_answer_keywords": ["technology", "innovation", "AI", "تقنية", "ابتكار"]
      }
    }
  ]
}
```

### 11.4 Metrics Implementation

```python
# evaluation/metrics/retrieval_metrics.py

from typing import List, Dict, Set
from dataclasses import dataclass

@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics"""
    precision: float
    recall: float
    f1: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    entity_coverage: float
    relationship_coverage: float

def calculate_retrieval_metrics(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str],
    retrieved_entities: List[str],
    ground_truth_entities: List[str],
    retrieved_relationships: List[tuple],
    ground_truth_relationships: List[tuple]
) -> RetrievalMetrics:
    """
    Calculate retrieval metrics against ground truth.

    Args:
        retrieved_chunks: List of retrieved chunk IDs
        ground_truth_chunks: List of relevant chunk IDs
        retrieved_entities: List of retrieved entity names
        ground_truth_entities: List of relevant entity names
        retrieved_relationships: List of (source, type, target) tuples
        ground_truth_relationships: List of relevant relationships

    Returns:
        RetrievalMetrics with all scores
    """
    # Chunk metrics
    retrieved_set = set(retrieved_chunks)
    relevant_set = set(ground_truth_chunks)

    if retrieved_set:
        precision = len(retrieved_set & relevant_set) / len(retrieved_set)
    else:
        precision = 0.0

    if relevant_set:
        recall = len(retrieved_set & relevant_set) / len(relevant_set)
    else:
        recall = 1.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # MRR
    mrr = 0.0
    for rank, chunk_id in enumerate(retrieved_chunks, 1):
        if chunk_id in relevant_set:
            mrr = 1.0 / rank
            break

    # NDCG (simplified)
    import numpy as np
    relevance = [1 if c in relevant_set else 0 for c in retrieved_chunks]
    if sum(relevance) > 0:
        dcg = sum(r / np.log2(i + 2) for i, r in enumerate(relevance))
        ideal = sorted(relevance, reverse=True)
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
        ndcg = dcg / idcg if idcg > 0 else 0
    else:
        ndcg = 0.0

    # Entity coverage
    retrieved_ent_set = set(e.lower() for e in retrieved_entities)
    relevant_ent_set = set(e.lower() for e in ground_truth_entities)
    if relevant_ent_set:
        entity_coverage = len(retrieved_ent_set & relevant_ent_set) / len(relevant_ent_set)
    else:
        entity_coverage = 1.0

    # Relationship coverage
    retrieved_rel_set = set(retrieved_relationships)
    relevant_rel_set = set(ground_truth_relationships)
    if relevant_rel_set:
        relationship_coverage = len(retrieved_rel_set & relevant_rel_set) / len(relevant_rel_set)
    else:
        relationship_coverage = 1.0

    return RetrievalMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        mrr=mrr,
        ndcg=ndcg,
        entity_coverage=entity_coverage,
        relationship_coverage=relationship_coverage
    )
```

### 11.5 Experiment Runner

```python
# evaluation/runners/experiment_runner.py

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, asdict

@dataclass
class ExperimentResult:
    experiment_id: str
    variant_name: str
    config: Dict
    metrics: Dict
    per_query_results: List[Dict]
    duration_seconds: float
    timestamp: str

class ExperimentRunner:
    """
    Runs experiments comparing different configurations.

    Features:
    - Run single experiment
    - Run ablation studies (vary one parameter)
    - Run full comparison (vary multiple parameters)
    - Save results for analysis
    """

    def __init__(self, base_system, output_dir: str = "evaluation/results"):
        self.base_system = base_system
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_experiment(
        self,
        experiment_config: Dict,
        dataset: List[Dict]
    ) -> List[ExperimentResult]:
        """
        Run experiment with all variants.

        Args:
            experiment_config: Experiment configuration
            dataset: Test dataset

        Returns:
            List of results for each variant
        """
        results = []
        variables = experiment_config.get('variables', {})
        constants = experiment_config.get('constants', {})

        # Get all variants (e.g., different LLMs)
        for var_name, var_values in variables.items():
            for variant in var_values:
                # Build config for this variant
                config = {**constants, var_name: variant}

                print(f"Running variant: {variant.get('name', variant)}")

                # Run this variant
                result = await self._run_variant(
                    experiment_id=experiment_config['experiment']['id'],
                    variant_name=variant.get('name', str(variant)),
                    config=config,
                    dataset=dataset,
                    metrics_to_collect=experiment_config.get('metrics', [])
                )

                results.append(result)

                # Save intermediate result
                self._save_result(result)

        return results

    async def _run_variant(
        self,
        experiment_id: str,
        variant_name: str,
        config: Dict,
        dataset: List[Dict],
        metrics_to_collect: List[str]
    ) -> ExperimentResult:
        """Run a single variant and collect metrics"""
        import time
        start_time = time.time()

        # Reconfigure system with this variant's config
        system = self._configure_system(config)

        per_query_results = []
        all_metrics = {m: [] for m in metrics_to_collect}

        for query_item in dataset:
            # Run retrieval
            retrieval_result = await system.retrieve(
                query_item['query'],
                mode=config.get('retrieval', {}).get('mode', 'mix')
            )

            # Run generation
            generation_result = await system.generate(
                query_item['query'],
                retrieval_result
            )

            # Calculate metrics
            query_metrics = self._calculate_metrics(
                query_item,
                retrieval_result,
                generation_result,
                metrics_to_collect
            )

            per_query_results.append({
                'query_id': query_item['id'],
                'query': query_item['query'],
                'metrics': query_metrics,
                'retrieval': {
                    'chunks': len(retrieval_result.chunks),
                    'entities': len(retrieval_result.entities),
                    'relationships': len(retrieval_result.relationships)
                },
                'generation': {
                    'answer_length': len(generation_result.answer),
                    'citations': len(generation_result.citations)
                }
            })

            # Aggregate metrics
            for metric, value in query_metrics.items():
                if metric in all_metrics:
                    all_metrics[metric].append(value)

        # Compute aggregate metrics
        import numpy as np
        aggregated = {}
        for metric, values in all_metrics.items():
            if values:
                aggregated[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

        duration = time.time() - start_time

        return ExperimentResult(
            experiment_id=experiment_id,
            variant_name=variant_name,
            config=config,
            metrics=aggregated,
            per_query_results=per_query_results,
            duration_seconds=duration,
            timestamp=datetime.now().isoformat()
        )

    def _save_result(self, result: ExperimentResult):
        """Save result to disk"""
        filename = f"{result.experiment_id}_{result.variant_name}_{result.timestamp[:10]}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

    def generate_comparison_report(
        self,
        results: List[ExperimentResult],
        output_path: str = None
    ) -> str:
        """Generate markdown comparison report"""
        report = []
        report.append(f"# Experiment Comparison Report")
        report.append(f"\nGenerated: {datetime.now().isoformat()}")
        report.append(f"\nExperiment: {results[0].experiment_id}")

        # Summary table
        report.append("\n## Summary\n")
        report.append("| Variant | " + " | ".join(results[0].metrics.keys()) + " |")
        report.append("|" + "---|" * (len(results[0].metrics) + 1))

        for result in results:
            row = f"| {result.variant_name} |"
            for metric, values in result.metrics.items():
                row += f" {values['mean']:.4f} |"
            report.append(row)

        # Detailed analysis
        report.append("\n## Detailed Analysis\n")
        for result in results:
            report.append(f"### {result.variant_name}\n")
            report.append(f"- Duration: {result.duration_seconds:.2f}s")
            report.append(f"- Queries: {len(result.per_query_results)}")
            report.append("\nMetrics:")
            for metric, values in result.metrics.items():
                report.append(f"- {metric}: {values['mean']:.4f} (±{values['std']:.4f})")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)

        return report_text
```

---

## 12. Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Goal**: Configuration system + Model management

| Day | Task | Deliverables |
|-----|------|--------------|
| 1 | Create config directory structure | `config/core/`, `config/components/` |
| 2 | Implement ConfigLoader | `src/config/config_loader.py` |
| 3 | Implement ModelRegistry | `src/core/models/registry.py` |
| 4 | Implement EmbeddingManager | `src/core/models/embedding_manager.py` |
| 5 | Implement LLMManager | `src/core/models/llm_manager.py` |
| 6 | Test model download/loading | Integration tests |
| 7 | Create default configs | `defaults.yaml`, `models.yaml` |

**Validation**:
- [ ] Can load configs from YAML
- [ ] Can download models from HuggingFace
- [ ] Models cached in `/data/models_cache`
- [ ] Can switch models via config

### Phase 2: Arabic Processing (Week 3)
**Goal**: Configurable Arabic pipeline

| Day | Task | Deliverables |
|-----|------|--------------|
| 1-2 | Implement CAMeL processor | `src/core/arabic/camel_processor.py` |
| 3 | Implement Stanza processor | `src/core/arabic/stanza_processor.py` |
| 4 | Implement disabled processor | `src/core/arabic/disabled_processor.py` |
| 5 | Create processor factory | `src/core/arabic/__init__.py` |
| 6-7 | Test Arabic processing | Unit tests |

**Validation**:
- [ ] Can switch Arabic processor via config
- [ ] CAMeL NER working
- [ ] Normalization working
- [ ] Dialect detection working

### Phase 3: Chunking Strategies (Week 4)
**Goal**: 5 chunking strategies

| Day | Task | Deliverables |
|-----|------|--------------|
| 1 | Implement TokenChunker | `src/core/chunking/token_chunker.py` |
| 2 | Implement SemanticChunker | `src/core/chunking/semantic_chunker.py` |
| 3 | Implement SentenceChunker | `src/core/chunking/sentence_chunker.py` |
| 4 | Implement RecursiveChunker | `src/core/chunking/recursive_chunker.py` |
| 5 | Implement LateChunker | `src/core/chunking/late_chunker.py` |
| 6-7 | Test all chunkers | Unit tests, comparison |

**Validation**:
- [ ] Can switch chunker via config
- [ ] All 5 strategies produce valid chunks
- [ ] Token counts accurate

### Phase 4: Triple-Level Indexing (Week 5-6)
**Goal**: Entity and relationship indices

| Day | Task | Deliverables |
|-----|------|--------------|
| 1-2 | Create Qdrant collections | Entity + Relationship collections |
| 3-4 | Implement IndexManager | `src/core/indexing/index_manager.py` |
| 5-6 | Implement entity indexing | Entity embeddings + storage |
| 7-8 | Implement relationship indexing | Relationship embeddings + storage |
| 9-10 | Update ingestion pipeline | Integration |

**Validation**:
- [ ] 3 Qdrant collections working
- [ ] Entity search working
- [ ] Relationship search working
- [ ] Cross-references correct

### Phase 5: Graph Features (Week 7)
**Goal**: Community detection + temporal

| Day | Task | Deliverables |
|-----|------|--------------|
| 1-2 | Implement CommunityDetector | `src/core/graph/community_detection.py` |
| 3 | Implement hierarchy builder | Hierarchical summaries |
| 4-5 | Implement TemporalManager | `src/core/graph/temporal.py` |
| 6-7 | Test graph features | Integration tests |

**Validation**:
- [ ] Communities detected
- [ ] Summaries generated
- [ ] Temporal filtering works
- [ ] Confidence decay works

### Phase 6: Retrieval Engine (Week 8-9)
**Goal**: 7 retrieval modes

| Day | Task | Deliverables |
|-----|------|--------------|
| 1 | Implement QueryRouter | `src/core/retrieval/query_router.py` |
| 2 | Implement NaiveRetriever | Basic vector search |
| 3 | Implement LocalRetriever | Entity-focused |
| 4 | Implement GlobalRetriever | Relationship-focused |
| 5 | Implement HybridRetriever | Local + Global |
| 6-7 | Implement MixRetriever | All combined |
| 8 | Implement SemanticRetriever | Deep matching |
| 9 | Implement BypassRetriever | No retrieval |
| 10 | Implement RRF fusion | Result fusion |

**Validation**:
- [ ] All 7 modes working
- [ ] Query routing correct
- [ ] RRF fusion produces ranked results

### Phase 7: Prompt System (Week 10)
**Goal**: Versioned prompts with CoT

| Day | Task | Deliverables |
|-----|------|--------------|
| 1-2 | Create prompt templates | YAML prompts |
| 3-4 | Implement PromptManager | `src/core/generation/prompt_manager.py` |
| 5-7 | Create few-shot examples | Bilingual examples |

**Validation**:
- [ ] Can load prompts from YAML
- [ ] CoT formatting works
- [ ] Few-shot injection works

### Phase 8: Evaluation Framework (Week 11-12)
**Goal**: Experiment runner + metrics

| Day | Task | Deliverables |
|-----|------|--------------|
| 1-2 | Create evaluation structure | `evaluation/` directory |
| 3-4 | Implement metrics | Retrieval + generation metrics |
| 5-6 | Implement ExperimentRunner | `evaluation/runners/` |
| 7-8 | Create test dataset | `evaluation/datasets/` |
| 9-10 | Run baseline experiments | Initial results |

**Validation**:
- [ ] Can run experiments
- [ ] Metrics calculated correctly
- [ ] Reports generated

---

## 13. Directory Structure (Final)

```
mirage/
├── config/
│   ├── core/
│   │   ├── defaults.yaml           # Default configuration
│   │   ├── models.yaml             # Model registry
│   │   └── storage.yaml            # Database configs
│   ├── components/
│   │   ├── arabic/
│   │   │   ├── camel.yaml
│   │   │   ├── stanza.yaml
│   │   │   └── disabled.yaml
│   │   ├── chunking/
│   │   │   ├── token.yaml
│   │   │   ├── semantic.yaml
│   │   │   ├── sentence.yaml
│   │   │   ├── recursive.yaml
│   │   │   └── late.yaml
│   │   ├── retrieval/
│   │   │   └── modes.yaml
│   │   └── prompts/
│   │       ├── extraction/
│   │       │   ├── entity_v1.yaml
│   │       │   └── entity_v2_cot.yaml
│   │       └── generation/
│   │           ├── answer_v1.yaml
│   │           └── answer_v2_cot.yaml
│   └── experiments/                 # For evaluation (separate)
│       ├── baselines/
│       └── ablations/
│
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config_loader.py
│   │   └── config_validator.py
│   ├── core/
│   │   ├── arabic/
│   │   │   ├── __init__.py
│   │   │   ├── base_processor.py
│   │   │   ├── camel_processor.py
│   │   │   ├── stanza_processor.py
│   │   │   └── disabled_processor.py
│   │   ├── chunking/
│   │   │   ├── __init__.py
│   │   │   ├── base_chunker.py
│   │   │   ├── token_chunker.py
│   │   │   ├── semantic_chunker.py
│   │   │   ├── sentence_chunker.py
│   │   │   ├── recursive_chunker.py
│   │   │   └── late_chunker.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── registry.py
│   │   │   ├── embedding_manager.py
│   │   │   └── llm_manager.py
│   │   ├── extraction/
│   │   │   ├── __init__.py
│   │   │   ├── entity_extractor.py
│   │   │   ├── relationship_extractor.py
│   │   │   └── gleaning.py
│   │   ├── indexing/
│   │   │   ├── __init__.py
│   │   │   ├── chunk_index.py
│   │   │   ├── entity_index.py
│   │   │   ├── relationship_index.py
│   │   │   └── index_manager.py
│   │   ├── graph/
│   │   │   ├── __init__.py
│   │   │   ├── neo4j_client.py
│   │   │   ├── community_detection.py
│   │   │   ├── temporal.py
│   │   │   └── traversal.py
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
│   │   └── generation/
│   │       ├── __init__.py
│   │       ├── prompt_manager.py
│   │       ├── context_builder.py
│   │       └── generator.py
│   └── api/
│       └── (existing API code)
│
├── evaluation/                      # SEPARATE from core
│   ├── config/
│   │   └── experiments/
│   │       ├── llm_comparison.yaml
│   │       ├── embedding_comparison.yaml
│   │       ├── chunking_comparison.yaml
│   │       └── arabic_nlp_comparison.yaml
│   ├── datasets/
│   │   ├── test_queries_v1.json
│   │   └── ground_truth/
│   ├── metrics/
│   │   ├── retrieval_metrics.py
│   │   ├── generation_metrics.py
│   │   └── extraction_metrics.py
│   ├── runners/
│   │   └── experiment_runner.py
│   ├── analysis/
│   │   └── report_generator.py
│   └── results/
│       └── (generated results)
│
├── data/
│   └── models_cache/               # Downloaded models
│       ├── llm/
│       ├── embedding/
│       └── reranker/
│
└── tests/
    ├── unit/
    ├── integration/
    └── evaluation/
```

---

## Summary

This comprehensive plan creates **MIRAGE v2**, a research-grade RAG system that exceeds LightRAG with:

1. **7 Retrieval Modes** - More than LightRAG's 6
2. **Full Arabic Support** - Production-ready with CAMeL/Stanza
3. **5 Chunking Strategies** - Configurable and comparable
4. **Dynamic Model Management** - HuggingFace integration
5. **Community Detection** - With hierarchical summaries
6. **Temporal Relationships** - Time-aware queries
7. **Confidence Decay** - Knowledge freshness
8. **Complete Evaluation Framework** - Separate from core
9. **Versioned Everything** - Configs, prompts, models

**Timeline**: 12 weeks
**Estimated Lines of Code**: ~15,000
**Risk Level**: Medium (modular approach reduces risk)
