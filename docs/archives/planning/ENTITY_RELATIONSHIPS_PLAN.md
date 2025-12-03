# Entity Relationships Implementation Plan

## Goal
Implement 4 types of relationships for comprehensive hybrid vector-graph retrieval:
1. **MENTIONS** (Chunk → Entity) - ✅ Already implemented
2. **RELATED_TO** (Entity → Entity, LLM-extracted) - 🔨 Needs proper storage
3. **COOCCURS_WITH** (Entity → Entity, co-occurrence) - ⚠️ To implement
4. **SIMILAR_TO** (Entity → Entity, semantic similarity) - ⚠️ To implement

---

## Architecture

### Graph Schema
```cypher
// Nodes
(:Chunk {id, text, document_id, embedding})
(:Entity {id, name, type, description, document_id, embedding})

// Relationships
(:Chunk)-[:MENTIONS {chunk_id}]->(:Entity)
(:Entity)-[:RELATED_TO {type, confidence, source: "llm"}]->(:Entity)
(:Entity)-[:COOCCURS_WITH {frequency, chunks: [chunk_ids], source: "cooccurrence"}]->(:Entity)
(:Entity)-[:SIMILAR_TO {score, source: "semantic"}]->(:Entity)
```

### Relationship Types

#### 1. MENTIONS (Chunk → Entity) ✅
**Status:** Already implemented
**Purpose:** Links chunks to entities they mention
**Properties:**
- `chunk_id`: Which chunk mentions this entity

**Example:**
```cypher
(chunk_0)-[:MENTIONS]->(Claude AI)
(chunk_0)-[:MENTIONS]->(Anthropic)
```

#### 2. RELATED_TO (Entity → Entity, LLM) 🔨
**Status:** LLM extracts but not stored properly
**Purpose:** Explicit semantic relationships extracted by LLM
**Properties:**
- `type`: Relationship type (e.g., "WORKS_FOR", "LOCATED_IN", "PART_OF")
- `confidence`: LLM confidence score (0-1)
- `source`: "llm"

**Example:**
```cypher
(Claude AI)-[:RELATED_TO {type: "DEVELOPED_BY", confidence: 0.95, source: "llm"}]->(Anthropic)
(Artifacts)-[:RELATED_TO {type: "FEATURE_OF", confidence: 0.9, source: "llm"}]->(Claude AI)
```

#### 3. COOCCURS_WITH (Entity → Entity, Co-occurrence) ⚠️
**Status:** To implement
**Purpose:** Statistical co-occurrence in the same chunks
**Properties:**
- `frequency`: Number of chunks where both entities appear
- `chunks`: List of chunk IDs where they cooccur
- `source`: "cooccurrence"

**Example:**
```cypher
(Claude AI)-[:COOCCURS_WITH {frequency: 15, chunks: ["chunk_0", "chunk_1", ...], source: "cooccurrence"}]->(Anthropic)
```

**When to create:**
- Both entities mentioned in the same chunk
- Frequency ≥ 2 (appear together in at least 2 chunks)

#### 4. SIMILAR_TO (Entity → Entity, Semantic) ⚠️
**Status:** To implement
**Purpose:** Semantic similarity based on entity embeddings
**Properties:**
- `score`: Cosine similarity (0-1)
- `source`: "semantic"

**Example:**
```cypher
(Claude AI)-[:SIMILAR_TO {score: 0.87, source: "semantic"}]->(GPT-4)
(Azure)-[:SIMILAR_TO {score: 0.82, source: "semantic"}]->(AWS)
```

**When to create:**
- Similarity score ≥ 0.75 (configurable threshold)
- Limit to top 5 most similar entities per entity

---

## Implementation Phases

### Phase 1: Entity Embeddings Storage ⚠️
**Goal:** Store embeddings for entities to enable semantic similarity

**Tasks:**
1. Add `embedding` property to Entity nodes
2. Generate embeddings for entity names (using Jina)
3. Store embeddings in Neo4j during entity creation

**Files to modify:**
- `mirage/src/core/graph_builder/neo4j_client.py` - Add embedding parameter
- `mirage/src/core/graph_builder/entity_extractor.py` - Generate entity embeddings

**Code changes:**
```python
# entity_extractor.py
def extract_with_chunk_references(self, chunks, ...):
    entities_with_chunks = []
    for entity_name, entity_info in entity_chunk_map.items():
        # Generate entity embedding
        entity_embedding = self.embedder.embed([entity_name], task="retrieval.query")[0]

        entities_with_chunks.append({
            "name": entity_name,
            "type": entity_info["type"],
            "chunks": entity_info["chunks"],
            "confidence": entity_info["confidence"],
            "embedding": entity_embedding  # NEW
        })
```

---

### Phase 2: LLM Relationship Storage 🔨
**Goal:** Store LLM-extracted relationships as RELATED_TO edges

**Tasks:**
1. Extract relationship type from LLM response
2. Create RELATED_TO relationships with metadata
3. Store confidence scores

**Files to modify:**
- `mirage/src/core/graph_builder/neo4j_client.py` - Add LLM relationship storage

**Current issue:**
- LLM extracts relationships: `[{"source": "Claude AI", "target": "Anthropic", "type": "DEVELOPED_BY"}]`
- But they're stored as generic relationships without type/confidence

**Fix:**
```python
# neo4j_client.py - store_chunks_with_entities()
for rel in relationships:
    query = """
    MATCH (e1:Entity {name: $source, document_id: $doc_id})
    MATCH (e2:Entity {name: $target, document_id: $doc_id})
    MERGE (e1)-[r:RELATED_TO]->(e2)
    SET r.type = $rel_type,
        r.confidence = $confidence,
        r.source = 'llm'
    """
```

---

### Phase 3: Co-occurrence Relationships ⚠️
**Goal:** Create COOCCURS_WITH edges for entities in the same chunks

**Tasks:**
1. Track entity co-occurrences during chunk processing
2. Calculate co-occurrence frequency
3. Create relationships with chunk references

**Algorithm:**
```python
def create_cooccurrence_relationships(chunks, entities_with_chunks):
    # Map: (entity1, entity2) -> [chunk_ids]
    cooccurrences = defaultdict(list)

    for chunk in chunks:
        chunk_id = chunk["id"]
        # Get entities mentioned in this chunk
        chunk_entities = [e for e in entities_with_chunks if chunk_id in e["chunks"]]

        # Create pairs
        for i, e1 in enumerate(chunk_entities):
            for e2 in chunk_entities[i+1:]:
                pair = tuple(sorted([e1["name"], e2["name"]]))
                cooccurrences[pair].append(chunk_id)

    # Create relationships for pairs with frequency ≥ 2
    relationships = []
    for (e1_name, e2_name), chunk_ids in cooccurrences.items():
        if len(chunk_ids) >= 2:
            relationships.append({
                "source": e1_name,
                "target": e2_name,
                "type": "COOCCURS_WITH",
                "frequency": len(chunk_ids),
                "chunks": chunk_ids,
                "source_type": "cooccurrence"
            })

    return relationships
```

**Files to create/modify:**
- `mirage/src/core/graph_builder/cooccurrence_extractor.py` - NEW
- `mirage/src/api/url_service.py` - Call cooccurrence extraction

---

### Phase 4: Semantic Similarity Relationships ⚠️
**Goal:** Create SIMILAR_TO edges based on entity embedding similarity

**Tasks:**
1. Compute pairwise cosine similarity between entity embeddings
2. Create relationships for top-k similar entities
3. Store similarity scores

**Algorithm:**
```python
def create_semantic_similarity_relationships(entities_with_embeddings, threshold=0.75, top_k=5):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Prepare entity embeddings matrix
    entity_names = [e["name"] for e in entities_with_embeddings]
    embeddings = np.array([e["embedding"] for e in entities_with_embeddings])

    # Compute pairwise similarity
    similarity_matrix = cosine_similarity(embeddings)

    relationships = []
    for i, entity_i in enumerate(entity_names):
        # Get top-k similar entities (excluding self)
        similarities = similarity_matrix[i]
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]

        for j in similar_indices:
            score = similarities[j]
            if score >= threshold:
                relationships.append({
                    "source": entity_i,
                    "target": entity_names[j],
                    "type": "SIMILAR_TO",
                    "score": float(score),
                    "source_type": "semantic"
                })

    return relationships
```

**Files to create/modify:**
- `mirage/src/core/graph_builder/semantic_similarity.py` - NEW
- `mirage/src/api/url_service.py` - Call semantic similarity

**Dependencies:**
- `scikit-learn` (already in requirements.txt)

---

### Phase 5: Hybrid Retrieval Enhancement 🔄
**Goal:** Update graph expansion to use all relationship types

**Current Cypher (1-hop only uses MENTIONS):**
```cypher
MATCH (anchor:Chunk)-[:MENTIONS]->(entity:Entity)
MATCH (same_entity_chunk:Chunk)-[:MENTIONS]->(entity)
```

**Enhanced Cypher (2-hop with all relationship types):**
```cypher
// 2-HOP: Use ALL relationship types
MATCH (entity1)-[rel:RELATED_TO|COOCCURS_WITH|SIMILAR_TO]-(entity2)
MATCH (related_chunk:Chunk)-[:MENTIONS]->(entity2)
WHERE rel.source IN ['llm', 'cooccurrence', 'semantic']
  AND (rel.confidence >= 0.7 OR rel.frequency >= 2 OR rel.score >= 0.75)
```

**Weighted scoring:**
```python
# Different weights for different relationship types
relationship_weights = {
    "llm": 1.0,           # Highest confidence (explicit semantic)
    "cooccurrence": 0.8,  # Strong signal (statistical)
    "semantic": 0.6       # Good signal (embedding similarity)
}

graph_score = base_score * relationship_weights[rel.source]
```

**Files to modify:**
- `mirage/src/core/retrieval/hybrid_retriever.py` - Update 2-hop Cypher query

---

## Implementation Order

### Step 1: Entity Embeddings (30 min)
- [x] Add embedding generation in entity_extractor.py
- [x] Store embeddings in neo4j_client.py

### Step 2: LLM Relationship Storage (20 min)
- [x] Fix relationship storage with type/confidence
- [x] Update Cypher queries in neo4j_client.py

### Step 3: Co-occurrence Extractor (45 min)
- [x] Create cooccurrence_extractor.py
- [x] Integrate in url_service.py
- [x] Test with existing documents

### Step 4: Semantic Similarity (45 min)
- [x] Create semantic_similarity.py
- [x] Integrate in url_service.py
- [x] Test with existing documents

### Step 5: Enhanced Retrieval (30 min)
- [x] Update hybrid_retriever.py Cypher
- [x] Add relationship type weighting
- [x] Test retrieval with all relationship types

### Step 6: Testing & Validation (30 min)
- [x] Reprocess YouTube video
- [x] Verify all 4 relationship types created
- [x] Test 2-hop retrieval
- [x] Compare results with/without relationships

**Total estimated time:** ~3 hours

---

## Expected Results

### Before (Current State)
```
Query: "What is Claude AI?"
- 5 anchor chunks (vector search)
- 3 chunks from 1-hop (same entities)
- 0 chunks from 2-hop (no entity relationships)
Total: 8 chunks
```

### After (With All Relationships)
```
Query: "What is Claude AI?"
- 5 anchor chunks (vector search)
- 8 chunks from 1-hop (same entities - more due to better entity extraction)
- 12 chunks from 2-hop via:
  - 5 via LLM relationships (e.g., Anthropic → Safety)
  - 4 via co-occurrence (e.g., Claude ↔ GPT-4)
  - 3 via semantic similarity (e.g., AI ↔ Machine Learning)
Total: 25 chunks (3x improvement!)
```

---

## Configuration

### Thresholds (configurable)
```python
RELATIONSHIP_THRESHOLDS = {
    "llm_confidence": 0.7,        # Minimum LLM confidence
    "cooccurrence_frequency": 2,   # Minimum co-occurrences
    "semantic_similarity": 0.75,   # Minimum cosine similarity
    "top_k_similar": 5             # Max similar entities per entity
}
```

### Relationship Weights (for scoring)
```python
RELATIONSHIP_WEIGHTS = {
    "llm": 1.0,           # Explicit semantic relationships
    "cooccurrence": 0.8,  # Statistical relationships
    "semantic": 0.6       # Embedding-based relationships
}
```

---

## Next Steps

1. ✅ Create this plan document
2. 🔨 Implement entity embeddings
3. 🔨 Fix LLM relationship storage
4. ⚠️ Create co-occurrence extractor
5. ⚠️ Create semantic similarity calculator
6. ⚠️ Update hybrid retrieval
7. ⚠️ Test with YouTube video
8. ⚠️ Measure performance improvement
