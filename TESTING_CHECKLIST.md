# GraphRAG Phase 1-3 Testing Checklist

**Purpose**: Step-by-step testing guide for Phases 1-3
**Estimated Time**: 30-60 minutes
**Prerequisites**: Docker, Docker Compose installed

---

## Pre-Test Setup

### 1. Start Docker Services

```bash
cd /home/aub/boo/MIRAGE

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

**Expected Output**:
```
NAME              STATUS    PORTS
mirage-neo4j      Up        7474, 7687
mirage-qdrant     Up        6333
mirage-redis      Up        6379
mirage-tgi        Up        8765
mirage-api        Up        8000
mirage-ui         Up        3000
```

### 2. Wait for TGI to Load Model (~2-3 minutes)

```bash
# Watch TGI logs
docker-compose logs -f tgi

# Wait for this message:
# "Connected" or "Model loaded successfully"
```

**Troubleshooting**:
- If TGI crashes: Check GPU availability with `nvidia-smi`
- If model download fails: Check internet connection
- If out of memory: Reduce MAX_BATCH_PREFILL_TOKENS in docker-compose.yml

### 3. Verify Neo4j Has Data

```bash
# Open Neo4j Browser
# Visit: http://localhost:7474
# Login: neo4j / password

# Run this Cypher query:
MATCH (e:Entity)
RETURN COUNT(e) as entity_count
```

**Required**: At least 20 entities (100+ recommended)

**If No Data**:
- Upload documents via UI: http://localhost:3000
- Or run ingestion script (if available)

---

## Phase 1 Testing (Manual)

Phase 1 components are tested through integration with existing extraction.

### Test Entity Normalization

```bash
# In Neo4j Browser, check for duplicates
MATCH (e:Entity)
WHERE e.name CONTAINS 'Dr.' OR e.name CONTAINS 'الدكتور'
RETURN e.name
LIMIT 10
```

**Expected**: Should see normalized names (titles removed)

### Test Graph Traversal

```bash
# In Python shell
python3 << 'PYEOF'
import sys
sys.path.insert(0, 'mirage/src')

from core.graph_builder import Neo4jClient, GraphTraversal

client = Neo4jClient(uri='bolt://localhost:7687', user='neo4j', password='password')
traversal = GraphTraversal(client)

# Get first entity
result = client.execute_query("MATCH (e:Entity) RETURN e.name as name LIMIT 1")
if result:
    entity_name = result[0]['name']
    print(f"Testing traversal from: {entity_name}")

    # Traverse 2 hops
    trav_result = traversal.traverse_from_entities([entity_name], max_hops=2)
    print(f"Found {trav_result.total_nodes} nodes, {trav_result.total_edges} edges")

    if trav_result.total_nodes > 1:
        print("✓ Graph traversal working!")
    else:
        print("⚠ No connections found - graph may be sparse")
PYEOF
```

**Expected**: Should find connected entities

---

## Phase 2 Testing (Automated)

### Run Community Detection Test

```bash
cd /home/aub/boo/MIRAGE

# Run test script
python3 mirage/test_community_detection.py
```

### Expected Output

```
================================================================================
PHASE 2 TEST: Community Detection
================================================================================

Connecting to Neo4j...
✓ Found 150 entities in graph
✓ Found 320 relationships

Initializing community detector...
✓ Summarizer initialized

Running community detection...
  Parameters:
    - Resolution: 1.0
    - Hierarchy levels: 3
    - Min community size: 3

✓ Community detection complete!
  - Total communities: 42
  - Hierarchy levels: 3
  - Modularity: 0.456

✓ Communities stored successfully!
  - Community nodes created: 42
  - BELONGS_TO relationships: 450
  - PARENT_OF relationships: 15

================================================================================
VISUALIZATION: Community Structure
================================================================================

Level 0 (finest): 25 communities
Level 1: 12 communities
Level 2: 5 communities

================================================================================
VALIDATION
================================================================================

✓ All communities stored correctly
✓ Hierarchy structure created
✓ 145/150 entities assigned to communities

================================================================================
TEST SUMMARY
================================================================================

✓ ALL TESTS PASSED

Phase 2 (Community Detection) is working correctly!
```

### Validation Checklist

- [ ] Test script runs without errors
- [ ] Modularity > 0.3 (good community structure)
- [ ] 90%+ entities assigned to communities
- [ ] Hierarchy has 3 levels
- [ ] Community counts make sense (10-50 communities typical)

### Verify in Neo4j Browser

```cypher
// Check community nodes
MATCH (c:Community)
RETURN c.id, c.level, c.size
ORDER BY c.level, c.id
LIMIT 20

// Check hierarchy
MATCH (parent:Community)-[:PARENT_OF]->(child:Community)
RETURN parent.id, child.id
LIMIT 10

// Check entity assignments
MATCH (e:Entity)-[:BELONGS_TO]->(c:Community {level: 0})
RETURN e.name, c.id
LIMIT 10
```

**Expected**: Should see Community nodes, relationships, and entity assignments

---

## Phase 3 Testing (Automated)

### Run Community Summarization Test

```bash
cd /home/aub/boo/MIRAGE

# Run test script
python3 mirage/test_community_summarization.py
```

### Expected Output

```
================================================================================
PHASE 3 TEST: Community Summarization
================================================================================

Connecting to Neo4j...
✓ Found 42 communities

Community Structure:
  Level 0: 25 communities (avg size: 6.0)
  Level 1: 12 communities (avg size: 12.5)
  Level 2: 5 communities (avg size: 30.0)

Initializing community summarizer...
✓ Summarizer initialized

================================================================================
TEST: Single Community Summary
================================================================================

Testing with community: L0_C1
  Entities: 8

Generating summary...
✓ Summary generated successfully!

Summary:
────────────────────────────────────────────────────────────────────────────────
تتمحور هذه المجموعة حول التكنولوجيا والذكاء الاصطناعي في شركة IBM. تشمل
الكيانات الأساسية Ahmed Hassan الذي يعمل في IBM ويطور حلول الذكاء الاصطناعي.
العلاقات المهمة تظهر التعاون بين IBM وجامعة الملك عبد الله في مجال البحث
والتطوير.
────────────────────────────────────────────────────────────────────────────────

Metadata:
  Tokens: ~125
  Themes: التكنولوجيا, الذكاء الاصطناعي
  Key entities: Ahmed Hassan, IBM, AI, KAUST, Riyadh

================================================================================
TEST: Batch Summary Generation
================================================================================

Generating summaries (max 5 communities per level)...

Level 0: Processing 5 communities...
  1/5: L0_C1 ✓
  2/5: L0_C2 ✓
  3/5: L0_C3 ✓
  4/5: L0_C4 ✓
  5/5: L0_C5 ✓

Level 1: Processing 5 communities...
  1/5: L1_C1 ✓
  2/5: L1_C2 ✓
  3/5: L1_C3 ✓
  4/5: L1_C4 ✓
  5/5: L1_C5 ✓

✓ Generated 10 summaries

================================================================================
TEST: Summary Storage
================================================================================

Storing summaries in Neo4j...
✓ Storage complete!
  Stored: 10
  Failed: 0

================================================================================
VALIDATION
================================================================================

✓ All summaries stored successfully
✓ Token limits respected (max: 456 tokens)
✓ Summaries have sufficient content (avg: 342 chars)

================================================================================
TEST SUMMARY
================================================================================

✓ ALL TESTS PASSED

Phase 3 (Community Summarization) is working correctly!

Next steps:
  1. Phase 4: Implement global search over summaries
  2. Test map-reduce query answering
```

### Validation Checklist

- [ ] Test script runs without errors
- [ ] Allam generates coherent summaries
- [ ] Summaries are in Arabic (or English for English entities)
- [ ] Token count < 600 per summary
- [ ] All summaries stored in Neo4j
- [ ] Themes extracted from summaries

### Verify in Neo4j Browser

```cypher
// Check summaries
MATCH (c:Community)
WHERE c.summary IS NOT NULL
RETURN c.id, c.level, c.summary, c.themes
ORDER BY c.level
LIMIT 5

// Check summary statistics
MATCH (c:Community)
WHERE c.summary IS NOT NULL
RETURN
  c.level as level,
  COUNT(c) as summary_count,
  AVG(c.summary_token_count) as avg_tokens
ORDER BY level
```

**Expected**: Should see summaries stored with themes and token counts

---

## Troubleshooting

### Issue: "No entities in graph"

**Solution**:
1. Upload documents via UI: http://localhost:3000
2. Wait for processing to complete
3. Check Neo4j Browser for entities

### Issue: "Cannot connect to Neo4j"

**Solution**:
```bash
# Check if Neo4j is running
docker-compose ps neo4j

# If not running, start it
docker-compose up -d neo4j

# Check logs for errors
docker-compose logs neo4j
```

### Issue: "Cannot connect to TGI"

**Solution**:
```bash
# Check if TGI is running
docker-compose ps tgi

# Check logs
docker-compose logs tgi

# Look for "Connected" or error messages

# If crashed, check GPU
nvidia-smi

# If no GPU, you'll need to use API provider (OpenAI/Anthropic)
```

### Issue: "Summaries are nonsense"

**Possible Causes**:
1. Allam not loaded correctly
2. Prompt too long (> 2K tokens)
3. Temperature too high

**Solution**:
```bash
# Test Allam directly
curl -X POST http://localhost:8765/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "مرحبا، كيف حالك؟", "parameters": {"max_new_tokens": 50}}'

# Should get coherent Arabic response
```

### Issue: "Modularity too low (< 0.3)"

**Possible Causes**:
1. Graph has few connections (sparse)
2. No clear community structure
3. Too few entities

**Solution**:
- Add more documents to create denser graph
- Lower min_community_size parameter
- Acceptable if graph is naturally sparse

### Issue: Test script crashes with ImportError

**Solution**:
```bash
# Install dependencies
cd /home/aub/boo/MIRAGE/mirage
pip install -r requirements.txt

# Or use Docker environment
docker-compose exec mirage python test_community_detection.py
```

---

## Performance Benchmarks

### Expected Timing

| Operation                    | Entities | Time       |
|------------------------------|----------|------------|
| Community Detection          | 100      | < 5s       |
| Community Detection          | 1,000    | < 30s      |
| Community Detection          | 10,000   | < 2min     |
| Summary Generation (1 comm)  | -        | 2-5s       |
| Summary Generation (50 comm) | -        | 2-5min     |

### Resource Usage

| Service  | CPU   | Memory | GPU VRAM |
|----------|-------|--------|----------|
| Neo4j    | 10%   | 512MB  | -        |
| TGI      | 5%    | 4GB    | 14GB     |
| MIRAGE   | 5%    | 256MB  | -        |

---

## Success Criteria

### Phase 2 ✅

- [ ] Community detection runs without errors
- [ ] Modularity > 0.3
- [ ] Communities created in Neo4j
- [ ] Hierarchy has 3+ levels
- [ ] 90%+ entities assigned

### Phase 3 ✅

- [ ] Summaries generated successfully
- [ ] Summaries are coherent and relevant
- [ ] Token count < 600 per summary
- [ ] Themes extracted
- [ ] All summaries stored in Neo4j

### Overall ✅

- [ ] All test scripts pass
- [ ] No Python exceptions
- [ ] Neo4j data looks correct
- [ ] Allam responses are good quality
- [ ] Performance is acceptable

---

## Next Steps After Successful Testing

1. **Push to GitHub** (if not already done)
   ```bash
   git push origin main
   ```

2. **Proceed to Phase 4** (Global Search)
   - Map-reduce over community summaries
   - Theme-based queries
   - High-level question answering

3. **Document Findings**
   - Note any issues encountered
   - Record performance metrics
   - Capture example outputs

---

**Testing Duration**: 30-60 minutes
**Recommended**: Test Phase 2 first, then Phase 3
**Support**: Review PHASE_1_2_3_EVALUATION.md for detailed analysis

---
