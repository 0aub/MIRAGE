# GraphRAG Enhancement Evaluation Report

**Date:** 2025-12-25
**Version:** MIRAGE V4+ GraphRAG

## Summary of Enhancements

This report documents the GraphRAG enhancements made to MIRAGE based on Microsoft's GraphRAG specification.

### 1. Local Search with Entity Embeddings

**File:** `mirage/src/core/retrieval/engine/local_mode.py`

**Enhancement:**
- Added semantic entity search using query embeddings
- Entities are matched by embedding similarity (not just name matching)
- Entity descriptions are included in retrieval metadata
- Results prioritized: semantic matches > disambiguated > chunk-linked

**New Methods Added:**
- `search_entities_by_embedding()` in `search_ops.py` - Cosine similarity search
- `get_entity_with_context()` - Full entity context (relationships, communities)

**Metrics to Validate:**
| Metric | Description | Target |
|--------|-------------|--------|
| `semantic_entities_found` | Count of entities found via embedding search | > 0 for entity queries |
| `entity_descriptions` | Entity descriptions in metadata | Present for semantic matches |
| `match_type: semantic` | Chunk metadata shows semantic match | Present in results |

### 2. Global Search with JSON Mode

**File:** `mirage/src/core/retrieval/global_search.py`

**Enhancement:**
- MAP phase uses JSON output format per GraphRAG spec
- Structured scoring (0-100) with explanation
- Falls back to text parsing if JSON fails

**JSON Output Format:**
```json
{
  "score": 75,
  "answer": "Partial answer from this community...",
  "points": ["Key point 1", "Key point 2"],
  "explanation": "Reasoning for the score"
}
```

**Metrics to Validate:**
| Metric | Description | Target |
|--------|-------------|--------|
| `communities_searched` | Communities processed in map phase | > 0 |
| `partial_answers_count` | Relevant partial answers | > 0 for thematic queries |
| `global_answer` | Synthesized answer | Present and coherent |

### 3. DRIFT Search (Prior Session)

**File:** `mirage/src/core/retrieval/drift_search.py`

**Enhancement:**
- Implemented 3-phase algorithm: PRIMER → FOLLOW_UP → OUTPUT
- LLM generates follow-up questions for iterative refinement
- Comprehensive answer synthesis from all iterations

**Metrics to Validate:**
| Metric | Description | Target |
|--------|-------------|--------|
| `iterations` | Number of DRIFT iterations | 1-3 |
| `follow_up_questions` | Generated follow-ups | > 0 for complex queries |
| `phase` | Final phase reached | OUTPUT |

### 4. Vector RAG (Renamed from Naive)

**File:** `mirage/src/core/retrieval/engine/vector_mode.py`

**Change:**
- Renamed `NAIVE` to `VECTOR` throughout codebase
- Added backward compatibility alias
- Updated all references in ~20 files

---

## Evaluation Test Cases

### Local Search Tests
```
LOCAL_001: "من هي الجهة المسؤولة عن تنظيم البيانات في المملكة؟"
- Expected: Entities like NDMO, SDAIA found via embedding search
- Success: semantic_entities > 0

LOCAL_002: "ما هي سدايا؟"
- Expected: SDAIA entity matched semantically
- Success: Entity description in metadata
```

### Global Search Tests
```
GLOBAL_001: "ما هي المبادئ الرئيسية لحماية الخصوصية في السياسات السعودية؟"
- Expected: Map-reduce over community summaries
- Success: global_answer present, communities_searched > 0

GLOBAL_002: "Summarize the key themes across all data governance policies."
- Expected: Thematic synthesis from multiple communities
- Success: themes array populated
```

### DRIFT Search Tests
```
DRIFT_001: "كيف تؤثر تصنيفات البيانات على شروط مشاركتها وما هي الجهات المسؤولة؟"
- Expected: Multi-hop reasoning with follow-up questions
- Success: iterations > 1, comprehensive answer
```

---

## Running the Evaluation

### Inside Docker Container

```bash
# Navigate to MIRAGE directory
cd /app

# Run GraphRAG evaluation
python evaluate_graphrag.py --api http://localhost:8000

# Full evaluation including DRIFT
python evaluate_graphrag.py --api http://localhost:8000 --full

# With custom output
python evaluate_graphrag.py --api http://localhost:8000 --output /app/results/graphrag_eval.json
```

### Expected Output

```
================================================================================
MIRAGE GraphRAG EVALUATION
================================================================================
API: http://localhost:8000
Test Cases: 9
Modes: vector, local, global, hybrid
================================================================================

[LOCAL_001] من هي الجهة المسؤولة عن تنظيم البيانات في المملكة؟...
  Expected mode: local | Difficulty: L1
  VECTOR   | Score: 0.65 | Latency: 1200ms | Entities: 2 (semantic: 0)
  LOCAL    | Score: 0.82 | Latency: 1500ms | Entities: 5 (semantic: 3)  <- Better with semantic
  GLOBAL   | Score: 0.70 | Latency: 2000ms | Entities: 1 (semantic: 0)
  HYBRID   | Score: 0.78 | Latency: 1800ms | Entities: 4 (semantic: 2)

================================================================================
SUMMARY
================================================================================
Mode       Avg Score    Avg Latency  Entities     Semantic     Success
----------------------------------------------------------------------
VECTOR     0.650        1200ms       2.0          0.0          9/9
LOCAL      0.820        1500ms       5.0          3.0          9/9    <- Best for entity queries
GLOBAL     0.750        2000ms       1.0          0.0          9/9    <- Best for thematic
HYBRID     0.780        1800ms       4.0          2.0          9/9

================================================================================
GRAPHRAG ENHANCEMENT METRICS
================================================================================
Local Search Semantic Entity Ratio: 60.00%    <- NEW: Semantic matching working
Global Search Avg Score: 0.750                <- JSON mode parsing working
```

---

## Key Improvements Expected

### Before GraphRAG Enhancements
- Local search used name matching only
- Global search used text format (less reliable parsing)
- No entity description embeddings
- No semantic entity matching

### After GraphRAG Enhancements
| Feature | Before | After |
|---------|--------|-------|
| Entity Matching | Name only | Embedding similarity |
| Entity Context | None | Description + relationships |
| Map Phase Output | Text | JSON structured |
| Score Parsing | Regex | JSON native |
| Semantic Entities | 0% | 50-80% expected |

---

## Files Changed

### Core Retrieval
- `local_mode.py` - Semantic entity search integration
- `global_search.py` - JSON mode map phase
- `drift_search.py` - 3-phase algorithm (prior session)
- `vector_mode.py` - Renamed from naive_mode.py

### Graph Builder
- `search_ops.py` - New methods for embedding search

### Configuration
- `constants.py` - MODE_WEIGHT_VECTOR (renamed)
- `__init__.py` - Updated exports

---

## Validation Checklist

- [ ] All Python files pass syntax check
- [ ] API responds to all retrieval modes
- [ ] Local search returns `semantic_entities_found > 0`
- [ ] Global search returns `communities_searched > 0`
- [ ] Entity descriptions appear in metadata
- [ ] JSON parsing works in map phase
- [ ] Backward compatibility: `NAIVE` alias works
