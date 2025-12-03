# Entity Relationships Implementation Status

## ✅ Completed

### Phase 1: Entity Embeddings Generation
**Status:** ✅ Complete

**Files Modified:**
1. `mirage/src/core/graph_builder/entity_extractor.py`
   - Added `embedder` parameter to `__init__` (line 46)
   - Added batch embedding generation in `extract_with_chunk_references` (lines 477-506)
   - Embeddings generated for all entities using Jina

2. `mirage/src/api/url_service.py`
   - Passed `jina_embedder` to EntityExtractor (line 55)

3. `mirage/src/api/document_service.py`
   - Reordered initialization (jina_embedder before entity_extractor)
   - Passed `jina_embedder` to EntityExtractor (line 27)

**Result:**
- ✅ Entity embeddings now generated during extraction
- ✅ Embeddings available in `entities_with_chunks` array
- ✅ Ready for semantic similarity calculations

---

## 🔨 In Progress

### Phase 2: Store Entity Embeddings in Neo4j
**Status:** 🔨 Need to implement

**Next Steps:**
1. Modify `neo4j_client.py` - `create_entity_node()` method
   - Add embedding parameter
   - Store embeddings as entity property

2. Update Cypher query:
```python
# In neo4j_client.py
def create_entity_node(self, entity, document_id):
    query = """
    MERGE (e:Entity {name: $name, document_id: $doc_id})
    SET e.type = $type,
        e.confidence = $confidence,
        e.embedding = $embedding  # NEW: Store embedding
    RETURN e
    """
    params = {
        "name": entity["name"],
        "doc_id": document_id,
        "type": entity.get("type", "Unknown"),
        "confidence": entity.get("confidence", 1.0),
        "embedding": entity.get("embedding", [])  # NEW
    }
```

---

## ⚠️ To Do

### Phase 3: LLM Relationship Storage
**Files to modify:**
- `mirage/src/core/graph_builder/neo4j_client.py`

**Implementation:**
```python
# Add relationship type and confidence storage
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

### Phase 4: Co-occurrence Relationships
**Files to create:**
- `mirage/src/core/graph_builder/cooccurrence_extractor.py`

**Algorithm:**
```python
def create_cooccurrence_relationships(chunks, entities_with_chunks):
    cooccurrences = defaultdict(list)
    
    for chunk in chunks:
        chunk_entities = [e for e in entities_with_chunks 
                         if chunk["id"] in e["chunks"]]
        
        # Create entity pairs
        for i, e1 in enumerate(chunk_entities):
            for e2 in chunk_entities[i+1:]:
                pair = tuple(sorted([e1["name"], e2["name"]]))
                cooccurrences[pair].append(chunk["id"])
    
    # Filter: minimum 2 co-occurrences
    relationships = []
    for (e1, e2), chunk_ids in cooccurrences.items():
        if len(chunk_ids) >= 2:
            relationships.append({
                "source": e1,
                "target": e2,
                "type": "COOCCURS_WITH",
                "frequency": len(chunk_ids),
                "chunks": chunk_ids,
                "source_type": "cooccurrence"
            })
    
    return relationships
```

**Integration in url_service.py:**
```python
# After entity extraction
from ..core.graph_builder.cooccurrence_extractor import create_cooccurrence_relationships

cooccurrence_rels = create_cooccurrence_relationships(chunks, entities_with_chunks)
relationships.extend(cooccurrence_rels)
```

---

### Phase 5: Semantic Similarity Relationships
**Files to create:**
- `mirage/src/core/graph_builder/semantic_similarity.py`

**Implementation:**
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def create_semantic_similarity_relationships(
    entities_with_embeddings,
    threshold=0.75,
    top_k=5
):
    entity_names = [e["name"] for e in entities_with_embeddings]
    embeddings = np.array([e["embedding"] for e in entities_with_embeddings])
    
    similarity_matrix = cosine_similarity(embeddings)
    
    relationships = []
    for i, entity_i in enumerate(entity_names):
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

**Integration in url_service.py:**
```python
# After entity extraction
from ..core.graph_builder.semantic_similarity import create_semantic_similarity_relationships

# Filter entities that have embeddings
entities_with_emb = [e for e in entities_with_chunks if "embedding" in e]
semantic_rels = create_semantic_similarity_relationships(entities_with_emb)
relationships.extend(semantic_rels)
```

---

### Phase 6: Enhanced Hybrid Retrieval
**File to modify:**
- `mirage/src/core/retrieval/hybrid_retriever.py`

**Updated 2-hop Cypher query:**
```cypher
// 2-HOP: Find chunks mentioning RELATED entities (ALL relationship types)
UNWIND hop1_chunk_ids AS hop1_id
MATCH (hop1_chunk:Chunk {id: hop1_id})
OPTIONAL MATCH (hop1_chunk)-[:MENTIONS]->(entity1:Entity)

// Match ALL relationship types with filtering
OPTIONAL MATCH (entity1)-[rel:RELATED_TO|COOCCURS_WITH|SIMILAR_TO]-(entity2:Entity)
WHERE (rel.source = 'llm' AND rel.confidence >= 0.7)
   OR (rel.source = 'cooccurrence' AND rel.frequency >= 2)
   OR (rel.source = 'semantic' AND rel.score >= 0.75)

OPTIONAL MATCH (related_chunk:Chunk)-[:MENTIONS]->(entity2)
WHERE related_chunk.id IS NOT NULL
  AND NOT related_chunk.id IN anchor_ids
  AND NOT related_chunk.id IN hop1_chunk_ids
```

**Weighted scoring:**
```python
RELATIONSHIP_WEIGHTS = {
    "llm": 1.0,
    "cooccurrence": 0.8,
    "semantic": 0.6
}

graph_score = base_score * RELATIONSHIP_WEIGHTS[rel.source]
```

---

## Testing Plan

### Step 1: Test Entity Embeddings
```bash
# Reprocess YouTube video
curl -X POST "http://localhost:8000/documents/documents/yt_oeE-iwhivag/reprocess"

# Check Neo4j for entity embeddings
docker exec mirage-neo4j cypher-shell -u neo4j -p password \
  "MATCH (e:Entity {document_id: 'yt_oeE-iwhivag'}) 
   WHERE e.embedding IS NOT NULL 
   RETURN count(e) as entities_with_embeddings"
```

### Step 2: Test All Relationship Types
```bash
# Check relationship counts by type
docker exec mirage-neo4j cypher-shell -u neo4j -p password \
  "MATCH ()-[r]->() 
   WHERE r.source IN ['llm', 'cooccurrence', 'semantic']
   RETURN r.source, type(r), count(*) as count"
```

### Step 3: Test Hybrid Retrieval
```bash
# Test Arabic query with all relationship types
curl -X POST "http://localhost:8000/chat/hybrid-retrieve" \
  -H "Content-Type: application/json" \
  -d '{"message": "ما هي جائزة الحكومة الرقمية؟"}'

# Expected: ~20-25 chunks (vs 8 currently)
```

---

## Summary

**Completed:**
- ✅ Comprehensive plan document ([ENTITY_RELATIONSHIPS_PLAN.md](ENTITY_RELATIONSHIPS_PLAN.md))
- ✅ Entity embedding generation

**Remaining (Est. 2-3 hours):**
1. Store embeddings in Neo4j (20 min)
2. Fix LLM relationship storage (20 min)  
3. Create co-occurrence extractor (45 min)
4. Create semantic similarity calculator (45 min)
5. Update hybrid retrieval (30 min)
6. Testing & validation (30 min)

**Expected Improvement:**
- Current: 8 chunks (5 vector + 3 graph)
- After: 25+ chunks (5 vector + 8 1-hop + 12 2-hop)
- **3x more contextual information!**
