# MIRAGE GraphRAG Implementation Analysis

## Executive Summary

After studying Microsoft's official GraphRAG documentation and comparing with MIRAGE's current implementation, I identified several logic issues and missing features that need to be addressed.

---

## Microsoft GraphRAG Official Specification

### 1. Global Search (Map-Reduce)
**Purpose**: Answer broad, thematic questions across the entire knowledge base.

**Algorithm**:
1. **MAP Phase**:
   - Divide community reports into chunks
   - Each chunk generates intermediate response with key points
   - Each point has `description` and `score` (importance rating)
   - Uses JSON mode for structured output

2. **FILTER Phase**:
   - Filter points with `score = 0` (irrelevant)
   - Sort by descending importance scores

3. **REDUCE Phase**:
   - Aggregate most important points from all map responses
   - Synthesize into coherent final answer
   - Respect token limits

### 2. Local Search
**Purpose**: Answer specific, entity-focused questions.

**Algorithm**:
1. Extract entities semantically related to query (using entity description embeddings)
2. For each entity, retrieve:
   - Connected entities (via relationships)
   - Relationships (with descriptions)
   - Entity covariates (additional attributes)
   - Community reports (the entity belongs to)
   - Text units (source chunks mentioning the entity)
3. Rank and filter all candidate sources
4. Build context within defined token window
5. Generate response

### 3. DRIFT Search (Dynamic Reasoning and Inference with Flexible Traversal)
**Purpose**: Combine benefits of global and local search dynamically.

**Algorithm**:
1. **Primer Phase**:
   - Compare query against top K semantically relevant community reports
   - Generate initial broad answer AND follow-up questions

2. **Follow-Up Phase** (Iterative):
   - Use local search to refine each follow-up question
   - Produce intermediate answers
   - Generate MORE follow-up questions with increasing specificity
   - Confidence-weighted expansion decides when to continue

3. **Output Phase**:
   - Organize results hierarchically
   - Rank by relevance
   - Merge into final comprehensive answer

**Key Innovation**: DRIFT generates follow-up questions dynamically to explore the knowledge graph more thoroughly.

---

## Current MIRAGE Implementation Issues

### Issue 1: DRIFT Search is Incorrectly Implemented

**Current Implementation** (`drift_search.py`):
```
1. Global search → get community summaries
2. Extract entities from summaries
3. If confidence low → do local search
4. Optional claim search
5. Merge and generate answer
```

**Problems**:
- Missing **Primer Phase** with follow-up question generation
- No iterative refinement loop
- No confidence-weighted expansion
- Simply sequential (global then local), not truly "drifting"
- Missing the key innovation: dynamic follow-up question generation

**Correct Implementation Should**:
- Generate follow-up questions after primer phase
- Iterate with confidence checking
- Use local search to answer each follow-up question
- Aggregate answers hierarchically

### Issue 2: Local Search Missing Key Components

**Current Implementation** (`local_mode.py`):
- Gets entities from chunks ✓
- Retrieves entity-related chunks ✓
- Uses entity disambiguation ✓

**Missing**:
- Entity description embeddings for semantic entity matching
- Relationship descriptions in context
- Entity covariates (attributes)
- Community reports integration (entities should retrieve their community context)
- Proper context window budget management

### Issue 3: Global Search Minor Issues

**Current Implementation** (`global_search.py`):
- Map-reduce pattern ✓
- Parallel execution ✓
- Relevance filtering ✓

**Issues**:
- Not using JSON mode for structured map responses
- Missing explicit point scoring with integer scores
- Should filter `score = 0` points explicitly
- Reduce phase should work with "points" not full "answers"

---

## Target Features to Implement

### Priority 1: Fix DRIFT Search
1. Add Primer Phase with follow-up question generation
2. Implement iterative Follow-Up Phase with confidence tracking
3. Add confidence-weighted expansion logic
4. Generate dynamic follow-up questions using LLM

### Priority 2: Enhance Local Search
1. Add entity description embedding search
2. Include relationship descriptions in context
3. Add community report context for relevant entities
4. Implement proper context budget management

### Priority 3: Improve Global Search
1. Add JSON mode for map phase
2. Implement integer scoring for points
3. Add explicit score=0 filtering
4. Refactor reduce to work with points list

---

## Other Changes Needed

### Rename "Naive" to "Vector"
- `RetrievalMode.NAIVE` → `RetrievalMode.VECTOR`
- All references in code, comments, config
- More accurate terminology (it's vector similarity search)

### Remove REFRAG
REFRAG (Retrieval-Augmented Generation with Frozen Fragment Embeddings) is:
- Not part of Microsoft GraphRAG
- A separate Meta research project
- Not useful for this implementation
- Adding complexity without benefit

Files to remove:
- `mirage/src/core/refrag/` (entire directory)
- `mirage/src/api/refrag_service.py`
- All REFRAG references in other files

---

## Implementation Roadmap

1. **Phase 1**: Remove REFRAG, rename Naive→Vector
2. **Phase 2**: Fix DRIFT Search with proper algorithm
3. **Phase 3**: Enhance Local Search with missing components
4. **Phase 4**: Improve Global Search with structured output
