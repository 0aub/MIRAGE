# MIRAGE Code Reference Guide

## SEARCH ENGINES

### 1. Local Search (Entity-focused neighborhood search)
- **Main**: `mirage/src/core/graph_builder/local_search.py`
- **API**: `mirage/src/api/graphrag_service.py:200`
- **Engine**: `mirage/src/core/graph_builder/hybrid_search.py:263`

### 2. Global Search (Community summaries search)
- **Main**: `mirage/src/core/retrieval/global_search.py` - Line 133 `search()`, Line 735 `search_auto_level()`
- **API**: `mirage/src/api/graphrag_service.py:162`
- **Engine**: `mirage/src/core/graph_builder/hybrid_search.py:241`

### 3. Vector Search (Qdrant embeddings)
- **Main**: `mirage/src/core/vector_store/qdrant_client.py`
- **Hybrid Retriever**: `mirage/src/core/retrieval/hybrid_retriever.py:113` - `_vector_search()`
- **Retrieval Engine**: `mirage/src/core/retrieval/retrieval_engine.py`

### 4. Hybrid Search (Combines all modes)
- **Main**: `mirage/src/core/graph_builder/hybrid_search.py:286` - `_hybrid_search()`
- **Enhanced Neo4j**: `mirage/src/core/graph_builder/enhanced_neo4j_client.py:406`

---

## ENTITY EXTRACTION PROMPTS

### File: `mirage/src/core/graph_builder/llm_entity_extractor.py`

**Arabic Prompt** (Line 304-337):
```
أنت مستخرج معرفة متخصص لبناء رسم بياني معرفي
```

**English Prompt** (Line 344-380):
```
You are a knowledge extractor for building a Knowledge Graph
```

**Entity Validation Prompt** (Line 879-903):
```
أنت مدقق كيانات / You are an entity validator
```

---

## ARABIC PROCESSING

### Arabic Processors (choose one):
- `mirage/src/core/arabic/camel_processor.py` - CAMeL Tools (morphology, NER)
- `mirage/src/core/arabic/stanza_processor.py` - Stanza NLP
- `mirage/src/core/arabic/base_processor.py` - Base interface
- `mirage/src/core/arabic/factory.py` - Factory pattern

### Entity Normalizer (Arabic name normalization):
- `mirage/src/core/graph_builder/entity_normalizer.py`

### Content Cleaner (Arabic text cleaning):
- `mirage/src/core/document_processor/content_cleaner.py`

### Punctuation Restorer:
- `mirage/src/core/document_processor/punctuation_restorer.py`

---

## PROMPTS CONFIG

- `mirage/src/config/prompts.yaml` - All system prompts
- `mirage/src/config/prompt_loader.py` - Prompt loading
- `mirage/src/core/generation/prompt_manager.py` - Prompt management

---

## KEY DIRECTORIES

```
mirage/src/
├── api/                    # FastAPI endpoints
│   ├── graphrag_service.py # Search APIs (local, global, hybrid)
│   ├── chat_service.py     # Chat API
│   ├── file_service.py     # File upload/processing
│   └── url_service.py      # URL ingestion
├── core/
│   ├── arabic/             # Arabic NLP processing
│   ├── graph_builder/      # Entity extraction, Neo4j, search engines
│   ├── retrieval/          # RAG retrieval engines
│   ├── vector_store/       # Qdrant vector storage
│   ├── embeddings/         # Jina embeddings
│   ├── document_processor/ # Chunking, cleaning
│   └── generation/         # Response generation
└── config/
    ├── prompts.yaml        # System prompts
    └── settings.py         # Configuration
```
