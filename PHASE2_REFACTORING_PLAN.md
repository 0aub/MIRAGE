# MIRAGE Phase 2 Refactoring Plan

**Created:** 2024-12-25
**Status:** PLANNING

---

## EXECUTIVE SUMMARY

Phase 1 completed successfully (24,803 lines deleted, ~70 files removed).
Phase 2 focuses on code quality improvements:
- Splitting 7 "monster" files (>1000 lines each)
- Removing or consolidating low-usage modules
- Improving code organization

---

## PRIORITY 1: SPLIT MONSTER FILES

### 1.1 neo4j_client.py (1480 lines)

**Current:** Single file handling graph operations, queries, and data persistence

**Proposed Split:**
```
mirage/src/core/graph_builder/
├── neo4j/
│   ├── __init__.py          # Re-exports for backwards compatibility
│   ├── client.py             # Core Neo4j connection & session management (~300 lines)
│   ├── entity_store.py       # Entity CRUD operations (~400 lines)
│   ├── relationship_store.py # Relationship CRUD operations (~300 lines)
│   ├── community_store.py    # Community data operations (~200 lines)
│   └── queries.py            # Cypher query templates (~280 lines)
```

**Risk:** HIGH - Central to entire system
**Approach:** Extract incrementally, maintain backward-compatible imports

---

### 1.2 retrieval_engine.py (1393 lines)

**Current:** God class with vector search, graph search, reranking, all retrieval modes

**Proposed Split:**
```
mirage/src/core/retrieval/
├── engine/
│   ├── __init__.py           # Main RetrievalEngine facade (~200 lines)
│   ├── base.py               # Base classes and interfaces (~150 lines)
│   ├── vector_retriever.py   # Vector/Qdrant retrieval (~300 lines)
│   ├── graph_retriever.py    # Graph-based retrieval (~300 lines)
│   ├── reranker.py           # Cross-encoder reranking (~200 lines)
│   └── fusion.py             # Result fusion/deduplication (~243 lines)
```

**Risk:** HIGH - Used by chat_service and benchmark_service
**Approach:** Create facade pattern, delegate to specialized modules

---

### 1.3 chat_service.py (1279 lines)

**Current:** API endpoints mixed with business logic

**Proposed Split:**
```
mirage/src/api/chat/
├── __init__.py               # Router exports
├── routes.py                 # FastAPI route definitions (~300 lines)
├── handlers.py               # Request handlers (~400 lines)
├── models.py                 # Pydantic request/response models (~200 lines)
└── logic.py                  # Business logic (retrieval, generation) (~379 lines)
```

**Risk:** MEDIUM - API layer, affects frontend
**Approach:** Keep route signatures identical, extract logic

---

### 1.4 benchmark_service.py (1305 lines)

**Current:** Evaluation logic mixed with API endpoints

**Proposed Split:**
```
mirage/src/api/benchmark/
├── __init__.py
├── routes.py                 # API endpoints (~250 lines)
├── models.py                 # Request/response models (~150 lines)
├── runners.py                # Benchmark execution logic (~400 lines)
└── metrics.py                # Metric calculation (~505 lines)
```

**Risk:** LOW - Separate feature, fewer dependencies
**Approach:** Good candidate for first refactoring

---

### 1.5 llm_entity_extractor.py (1145 lines)

**Current:** Multiple extraction strategies in one file

**Proposed Split:**
```
mirage/src/core/graph_builder/extraction/
├── __init__.py
├── base.py                   # Base extractor interface (~100 lines)
├── llm_extractor.py          # LLM-based extraction (~400 lines)
├── prompts.py                # Prompt templates (~200 lines)
├── parser.py                 # JSON/response parsing (~200 lines)
└── validator.py              # Entity validation (~245 lines)
```

**Risk:** MEDIUM - Core extraction pipeline
**Approach:** Prompts are hot-swappable, extract first

---

### 1.6 community_detector.py (1040 lines)

**Current:** Community detection algorithms mixed with Neo4j operations

**Proposed Split:**
```
mirage/src/core/graph_builder/community/
├── __init__.py
├── detector.py               # Main detection logic (~300 lines)
├── algorithms.py             # Louvain, label propagation (~300 lines)
├── summarizer.py             # Community summarization (~240 lines)
└── neo4j_integration.py      # Neo4j storage (~200 lines)
```

**Risk:** MEDIUM - Used by global search
**Approach:** Algorithms are pure functions, easy to extract

---

### 1.7 url_service.py (1019 lines)

**Current:** URL processing mixed with web scraping logic

**Proposed Split:**
```
mirage/src/api/url/
├── __init__.py
├── routes.py                 # API endpoints (~200 lines)
├── models.py                 # Request/response models (~100 lines)
├── scraper.py                # Web scraping logic (~300 lines)
├── youtube_handler.py        # YouTube processing (~219 lines)
└── processor.py              # Content processing (~200 lines)
```

**Risk:** LOW - Self-contained feature
**Approach:** Good candidate for early refactoring

---

## PRIORITY 2: MODULE CONSOLIDATION

### 2.1 Retrieval Modules Status

| Module | Lines | Usage | Action |
|--------|-------|-------|--------|
| retrieval_engine.py | 1393 | HIGH (main engine) | SPLIT |
| v5_engine.py | 506 | MEDIUM (experimental) | KEEP |
| dual_level_retrieval.py | ~400 | LOW (via v5_engine) | KEEP |
| hippocampal_retrieval.py | ~450 | LOW (via v5_engine) | KEEP |
| drift_search.py | ~500 | LOW (experimental) | KEEP |
| hyde.py | ~200 | LOW (optional enhancer) | KEEP |
| observability.py | ~300 | MEDIUM (tracing) | KEEP |

**Decision:** All retrieval modules are actively used through v5_engine.
No modules to remove - they form a coherent experimental retrieval system.

---

### 2.2 Graph Builder Modules Status

| Module | Lines | Usage | Action |
|--------|-------|-------|--------|
| entity_disambiguator.py | 587 | HIGH (retrieval_engine) | KEEP |
| relationship_normalizer.py | 519 | HIGH (neo4j_client) | KEEP |
| relationship_enricher.py | 569 | MEDIUM (internal) | KEEP |
| coreference_resolver.py | 519 | LOW (exported only) | EVALUATE |
| incremental_updater.py | 765 | LOW (exported only) | EVALUATE |
| community_visualizer.py | ~200 | LOW (debugging tool) | EVALUATE |

**Decision:** Core modules (disambiguator, normalizer, enricher) are essential.
Three modules need usage verification:
- `coreference_resolver` - May be dead code
- `incremental_updater` - May be dead code
- `community_visualizer` - Debugging only, keep but move to tools/

---

## PRIORITY 3: RECOMMENDED REFACTORING ORDER

### Phase 2A: Low-Risk Quick Wins
1. **benchmark_service.py** - Isolated, few dependencies
2. **url_service.py** - Self-contained feature
3. Extract prompts from llm_entity_extractor.py

### Phase 2B: Medium-Risk Core Improvements
4. **chat_service.py** - Keep API stable, extract logic
5. **community_detector.py** - Pure algorithms extractable
6. **llm_entity_extractor.py** - Complete split

### Phase 2C: High-Risk Critical Refactoring
7. **retrieval_engine.py** - Core system, needs careful facade
8. **neo4j_client.py** - Most critical, refactor last

---

## PRIORITY 4: DEAD CODE VERIFICATION

### Modules to Trace
```bash
# Check if coreference_resolver is ever called
grep -r "coreference_resolver\|CoreferenceResolver" mirage/ --include="*.py" | grep -v "__init__.py" | grep -v "coreference_resolver.py"

# Check if incremental_updater is ever called
grep -r "incremental_updater\|IncrementalGraphUpdater" mirage/ --include="*.py" | grep -v "__init__.py" | grep -v "incremental_updater.py"
```

### CONFIRMED DEAD CODE (No Usage Found):
- `mirage/src/core/graph_builder/coreference_resolver.py` (519 lines) - DELETE
- `mirage/src/core/graph_builder/incremental_updater.py` (765 lines) - DELETE
- `mirage/src/core/graph_builder/community_visualizer.py` (~200 lines) - DELETE or move to tools/

**Total Dead Code:** ~1,484 lines

---

## SUCCESS CRITERIA

1. No file over 500 lines (except data/config files)
2. Clear separation of concerns (API vs Logic vs Data)
3. All imports work with backwards compatibility
4. All tests pass
5. No dead code remaining

---

## NOTES

- Always maintain backward-compatible imports via `__init__.py`
- Use facade pattern for complex modules
- Write tests before refactoring high-risk modules
- One PR per module split to keep changes reviewable
