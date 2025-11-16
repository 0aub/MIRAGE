# GraphRAG Analysis: Current State vs Best Practices

**Date:** January 2025
**Project:** MIRAGE - Multilingual Information Retrieval with Accelerated Graph Embeddings

## Executive Summary

After deep research into GraphRAG strategies, recent papers (including Microsoft's seminal work from April 2024 and January 2025 updates), and production implementations, this document analyzes MIRAGE's current architecture against GraphRAG best practices and identifies critical gaps.

**Key Finding:** MIRAGE currently implements a **basic hybrid approach** (vector + graph storage) but **lacks the core GraphRAG innovations** that enable global reasoning and complex query answering.

---

## Research Summary

### Core GraphRAG Papers & Sources

1. **Microsoft GraphRAG** (arXiv:2404.16130) - "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
2. **GraphRAG Survey** (arXiv:2501.00309) - Comprehensive 2025 survey
3. **GraphRAG Security** (arXiv:2501.14050) - Vulnerability analysis (Jan 2025)
4. **HybridRAG** (arXiv:2408.04948) - Integrating KG and Vector RAG
5. **Graph RAG Survey** (arXiv:2408.08921) - Comprehensive overview
6. Multiple production implementation guides from Neo4j, Databricks, AWS, etc.

### Key Insights from Research

#### What GraphRAG Actually Is

GraphRAG is **NOT** just using a graph database alongside vectors. The critical innovations are:

1. **Community Detection** - Using Leiden algorithm to detect hierarchical communities in the knowledge graph
2. **Community Summaries** - Pre-generating summaries of each community at multiple hierarchical levels
3. **Global Search** - Map-reduce over community summaries for holistic questions
4. **Local Search** - Entity-centric retrieval with graph traversal
5. **Hybrid Retrieval** - Combining vector similarity with graph relationships

#### Performance Metrics

- GraphRAG achieves **80% accuracy** vs traditional RAG's **50.83%**
- **70-80% win rate** on comprehensiveness and diversity
- HybridRAG (vector + graph) achieves **0.96 answer relevancy** vs GraphRAG alone (0.89) or VectorRAG alone (0.91)
- Global search answers questions that traditional RAG **cannot answer at all** (e.g., "What are the main themes?")

---

## Current MIRAGE Architecture

### What We Have ✅

1. **Vector Database (Qdrant)**
   - Semantic embeddings using Jina embeddings
   - Chunk-level vector storage
   - Basic similarity search

2. **Graph Database (Neo4j)**
   - Entity nodes with properties
   - Relationship edges
   - Basic graph storage

3. **Entity Extraction**
   - LLM-based entity extraction (`llm_entity_extractor.py`)
   - Relationship extraction
   - Content rewriting

4. **Content-Type Specific Chunking**
   - ChunkerFactory with semantic chunking
   - Different strategies for PDF, YouTube, URLs, text
   - Configurable chunk sizes

5. **Chat Interface**
   - Basic query interface
   - LLM integration (OpenAI, Anthropic, TGI)

### What We're Missing ❌

#### 1. **NO Community Detection**
- **Gap:** We extract entities and relationships but **never group them into communities**
- **Impact:** Cannot answer global questions about themes, patterns, or holistic insights
- **What's needed:** Implement Leiden algorithm for hierarchical community detection

#### 2. **NO Community Summaries**
- **Gap:** No pre-generated summaries of entity groups
- **Impact:** Every query requires real-time processing; no global context
- **What's needed:** Generate hierarchical summaries during indexing phase

#### 3. **NO Global Search**
- **Gap:** No map-reduce mechanism over community summaries
- **Impact:** Cannot answer "What are the main themes?" or similar global questions
- **What's needed:** Implement map-reduce query strategy with community summaries

#### 4. **NO Local Search with Graph Traversal**
- **Gap:** Chat service doesn't leverage graph relationships for context expansion
- **Impact:** Missing rich relational context around entities
- **What's needed:** Entity-centric retrieval with neighbor expansion

#### 5. **NO Hybrid Retrieval Strategy**
- **Gap:** Vector and graph queries are **separate**, not integrated
- **Impact:** Missing the 0.96 answer relevancy of true hybrid approach
- **What's needed:** Unified retrieval combining both approaches

#### 6. **NO Query Classification**
- **Gap:** All queries treated the same way
- **Impact:** Using wrong retrieval strategy for different question types
- **What's needed:** Route to global vs local vs hybrid search based on query

---

## Detailed Gap Analysis

### 1. Indexing Pipeline

| Component | Current State | GraphRAG Best Practice | Gap |
|-----------|--------------|----------------------|-----|
| Text Chunking | ✅ Semantic chunking | ✅ Semantic chunking | Minor - could optimize chunk size (50-100 tokens recommended) |
| Entity Extraction | ✅ LLM-based | ✅ LLM-based (or NLP) | Good - consider schema-guided extraction |
| Relationship Extraction | ✅ LLM-based | ✅ LLM-based | Good - consider co-occurrence relationships |
| Entity Normalization | ❌ Missing | ✅ Required | **Critical** - "Officer Johnson" vs "Inspector Johnson" |
| Graph Construction | ✅ Neo4j storage | ✅ Entity knowledge graph | Good |
| Community Detection | ❌ **Missing** | ✅ **Leiden algorithm** | **Critical** |
| Community Summaries | ❌ **Missing** | ✅ **Hierarchical summaries** | **Critical** |
| Vector Indexing | ✅ Qdrant | ✅ Vector DB | Good |

### 2. Query Pipeline

| Component | Current State | GraphRAG Best Practice | Gap |
|-----------|--------------|----------------------|-----|
| Query Classification | ❌ Missing | ✅ Global vs Local | **Critical** |
| Vector Search | ✅ Basic similarity | ✅ Semantic search | Good |
| Graph Traversal | ⚠️ Partial | ✅ Multi-hop reasoning | Needs expansion |
| Hybrid Retrieval | ❌ Missing | ✅ Combined retrieval | **Critical** |
| Global Search | ❌ **Missing** | ✅ **Map-reduce** | **Critical** |
| Local Search | ⚠️ Basic | ✅ Entity + neighbors | Needs improvement |
| Context Ranking | ❌ Missing | ✅ Score-based sorting | Important |
| Response Generation | ✅ LLM-based | ✅ LLM-based | Good |

### 3. Architecture Patterns

| Pattern | Current State | GraphRAG Best Practice | Gap |
|---------|--------------|----------------------|-----|
| Storage Strategy | Hybrid (Vector + Graph) | ✅ Hybrid | Good |
| Retrieval Strategy | Separate queries | Integrated hybrid | **Critical** |
| Community Structure | None | Hierarchical Leiden | **Critical** |
| Summary Generation | On-demand only | Pre-generated + on-demand | **Critical** |
| Query Routing | Single path | Multi-path (global/local/hybrid) | **Critical** |
| Cost Optimization | Basic caching | REFRAG + community reuse | Good foundation |

---

## Technical Deep Dive: What's Missing

### 1. Community Detection Implementation

**What it is:**
- Apply Leiden algorithm to entity knowledge graph
- Detect hierarchical communities (levels 0, 1, 2, ...)
- Each community = group of densely connected entities

**Why it matters:**
- Enables understanding of themes and topics
- Provides structure for summarization
- Essential for global search

**Implementation needed:**
```python
# Pseudo-code
def detect_communities(neo4j_client):
    # Project graph for community detection
    query = """
    CALL gds.graph.project(
        'entity-graph',
        'Entity',
        'RELATED_TO',
        {relationshipProperties: 'weight'}
    )
    """

    # Run Leiden algorithm
    leiden_query = """
    CALL gds.leiden.write('entity-graph', {
        writeProperty: 'community',
        includeIntermediateCommunities: true,
        relationshipWeightProperty: 'weight',
        resolution: 1.0
    })
    """

    # Extract hierarchical communities
    # Level 0 = finest granularity
    # Level N = highest level themes
```

**Key parameters:**
- `resolution`: Controls granularity (default 1.0)
- `includeIntermediateCommunities`: Enables hierarchical structure
- Typical output: 3-5 levels of hierarchy

### 2. Community Summary Generation

**What it is:**
- For each community at each level, generate LLM summary
- Summaries describe the community's entities and relationships
- Higher-level summaries recursively incorporate lower-level summaries

**Why it matters:**
- Enables answering global questions without reading all documents
- Provides pre-computed context for query time
- Core innovation of GraphRAG

**Implementation needed:**
```python
def generate_community_summaries(neo4j_client, llm_client):
    # For each hierarchical level (bottom-up)
    for level in range(max_level, -1, -1):
        communities = get_communities_at_level(level)

        for community_id in communities:
            # Get entities in community
            entities = get_community_entities(community_id, level)

            # Get relationships
            relationships = get_community_relationships(community_id, level)

            # Get child summaries (if not bottom level)
            child_summaries = []
            if level < max_level:
                child_summaries = get_child_summaries(community_id, level + 1)

            # Generate summary using LLM
            prompt = f"""
            Summarize this community of related entities and relationships.

            Entities: {entities}
            Relationships: {relationships}
            Sub-community summaries: {child_summaries}

            Provide a comprehensive summary describing:
            1. Main theme/topic of this community
            2. Key entities and their roles
            3. Important relationships and patterns
            4. Overall significance
            """

            summary = llm_client.generate(prompt)

            # Store summary
            store_community_summary(community_id, level, summary)
```

### 3. Global Search with Map-Reduce

**What it is:**
- Query answering using ALL community summaries
- Map phase: Each summary generates partial answer
- Reduce phase: Combine partial answers into final answer

**Why it matters:**
- Only way to answer "What are the main themes in the dataset?"
- Scales to millions of tokens
- 70-80% better than traditional RAG on global questions

**Implementation needed:**
```python
def global_search(query: str, community_level: int = 0):
    # 1. Get all community summaries at specified level
    summaries = get_all_community_summaries(level=community_level)

    # 2. Shuffle and split into chunks (token limit)
    chunks = shuffle_and_chunk(summaries, max_tokens=8000)

    # 3. MAP PHASE: Generate partial answers in parallel
    partial_answers = []
    for chunk in chunks:
        prompt = f"""
        Query: {query}

        Community summaries: {chunk}

        Generate a partial answer based on these summaries.
        Score: Rate helpfulness 0-100.
        """
        answer = llm_client.generate(prompt)
        partial_answers.append(answer)

    # 4. REDUCE PHASE: Sort by score and combine
    sorted_answers = sorted(partial_answers, key=lambda x: x.score, reverse=True)
    top_answers = sorted_answers[:10]  # Token limit

    final_prompt = f"""
    Query: {query}

    Partial answers from different communities: {top_answers}

    Synthesize these into a comprehensive final answer.
    """

    final_answer = llm_client.generate(final_prompt)
    return final_answer
```

### 4. Local Search with Graph Traversal

**What it is:**
- Start with entities matching query
- Expand to neighbors (1-hop, 2-hop)
- Retrieve associated text chunks
- Combine with community context

**Why it matters:**
- Provides rich relational context
- Enables multi-hop reasoning
- Better than pure vector search for entity-specific questions

**Implementation needed:**
```python
def local_search(query: str, max_hops: int = 2):
    # 1. Extract entities from query
    query_entities = extract_entities_from_query(query)

    # 2. Find matching entities in graph
    matched_entities = find_entities_in_graph(query_entities)

    # 3. Expand to neighbors (graph traversal)
    context_entities = []
    for entity in matched_entities:
        neighbors = get_neighbors(entity, max_hops=max_hops)
        context_entities.extend(neighbors)

    # 4. Get community context
    communities = get_entity_communities(matched_entities)
    community_summaries = [get_community_summary(c) for c in communities]

    # 5. Get associated text chunks (vector similarity)
    chunks = []
    for entity in context_entities:
        entity_chunks = get_entity_chunks(entity)
        chunks.extend(entity_chunks)

    # 6. Rank and filter by relevance
    ranked_chunks = rank_by_relevance(query, chunks)

    # 7. Combine context
    context = {
        'entities': context_entities,
        'relationships': get_relationships(context_entities),
        'chunks': ranked_chunks[:20],
        'community_summaries': community_summaries
    }

    # 8. Generate answer
    return generate_answer(query, context)
```

### 5. Hybrid Retrieval Strategy

**What it is:**
- Combine vector similarity AND graph relationships
- Query both databases simultaneously
- Merge and rank results

**Why it matters:**
- Achieves 0.96 answer relevancy (best of all approaches)
- Leverages strengths of both methods
- Industry best practice

**Implementation needed:**
```python
def hybrid_search(query: str):
    # 1. Vector search for similar chunks
    vector_results = qdrant_client.search(
        query_embedding=embed(query),
        limit=50
    )

    # 2. Graph search for related entities
    entities = extract_entities(query)
    graph_results = neo4j_client.find_related(
        entities=entities,
        max_hops=2
    )

    # 3. Get chunks associated with graph entities
    graph_chunks = []
    for entity in graph_results:
        chunks = get_entity_chunks(entity)
        graph_chunks.extend(chunks)

    # 4. Merge results with scoring
    all_results = []

    # Vector results: high semantic similarity
    for result in vector_results:
        all_results.append({
            'chunk': result.chunk,
            'vector_score': result.score,
            'graph_score': 0.0,
            'source': 'vector'
        })

    # Graph results: high relational relevance
    for chunk in graph_chunks:
        # Check if already in vector results
        existing = find_in_results(all_results, chunk)
        if existing:
            existing['graph_score'] = calculate_graph_relevance(chunk)
            existing['source'] = 'hybrid'
        else:
            all_results.append({
                'chunk': chunk,
                'vector_score': 0.0,
                'graph_score': calculate_graph_relevance(chunk),
                'source': 'graph'
            })

    # 5. Hybrid ranking
    for result in all_results:
        result['hybrid_score'] = (
            0.6 * result['vector_score'] +
            0.4 * result['graph_score']
        )

    # 6. Sort by hybrid score
    ranked_results = sorted(all_results, key=lambda x: x['hybrid_score'], reverse=True)

    return ranked_results[:20]
```

### 6. Query Classification and Routing

**What it is:**
- Analyze query to determine type
- Route to appropriate search strategy

**Query types:**
- **Global**: "What are the main themes?" → Use global search
- **Local**: "Tell me about entity X" → Use local search
- **Hybrid**: "How does X relate to Y?" → Use hybrid search
- **Basic**: Simple factual → Use vector search

**Implementation needed:**
```python
def classify_and_route(query: str):
    # Use LLM or heuristics to classify
    classification_prompt = f"""
    Classify this query:
    "{query}"

    Types:
    - GLOBAL: Asks about overall themes, patterns, summaries
    - LOCAL: Asks about specific entities or relationships
    - HYBRID: Asks about connections or comparisons
    - BASIC: Simple factual question

    Return: GLOBAL|LOCAL|HYBRID|BASIC
    """

    query_type = llm_client.generate(classification_prompt).strip()

    if query_type == "GLOBAL":
        return global_search(query)
    elif query_type == "LOCAL":
        return local_search(query)
    elif query_type == "HYBRID":
        return hybrid_search(query)
    else:
        return basic_vector_search(query)
```

---

## Current vs Ideal Architecture

### Current Architecture (Simplified)

```
Query
  ↓
Chat Service
  ↓
[Vector Search OR Graph Query]  ← Separate, not integrated
  ↓
Context
  ↓
LLM
  ↓
Answer
```

### Ideal GraphRAG Architecture

```
Query
  ↓
Query Classifier
  ├─→ Global Search
  │    ├─→ Get Community Summaries
  │    ├─→ Map-Reduce
  │    └─→ LLM → Answer
  │
  ├─→ Local Search
  │    ├─→ Entity Matching
  │    ├─→ Graph Traversal (neighbors)
  │    ├─→ Community Context
  │    ├─→ Associated Chunks
  │    └─→ LLM → Answer
  │
  └─→ Hybrid Search
       ├─→ Vector Search (semantic)
       ├─→ Graph Search (relational)
       ├─→ Merge & Rank
       └─→ LLM → Answer

[Indexing Pipeline]
Document
  ↓
Chunking (semantic, 50-100 tokens)
  ↓
Entity Extraction (LLM + normalization)
  ↓
Graph Construction (Neo4j)
  ↓
Community Detection (Leiden)
  ↓
Community Summaries (hierarchical)
  ↓
Vector Embedding (Qdrant)
```

---

## Answering Your Questions

### Q1: "I read that GraphRAG should return nodes, edges, AND context. Does that align with our approach?"

**Answer:** Partially, but we're missing critical pieces.

**What GraphRAG returns:**
- **Nodes & Edges**: Yes, we have this ✅
- **Context**: Yes, we have chunks ✅
- **Community Summaries**: No, we don't have this ❌
- **Hierarchical Context**: No, we don't have this ❌
- **Relational Context**: Partial, not integrated ⚠️

**The key difference:** GraphRAG doesn't just return nodes, edges, and chunks. It returns:
1. Relevant community summaries (global context)
2. Entity-centric subgraphs (local context)
3. Associated text chunks (detailed context)
4. All integrated and ranked together

We have the pieces but not the integration.

### Q2: "Is having both vector store and graph DB the same as GraphRAG?"

**Answer:** No. That's a necessary but not sufficient condition.

**What we have:**
- ✅ Vector database (Qdrant) for semantic search
- ✅ Graph database (Neo4j) for relationships
- ❌ Community detection and summaries
- ❌ Map-reduce global search
- ❌ Integrated hybrid retrieval
- ❌ Query routing

**Analogy:** Having both databases is like having a car engine and tires, but GraphRAG is the complete car with transmission, steering, and control systems that make it actually drive.

**The critical innovations are:**
1. **Community detection** - organizing the graph into meaningful groups
2. **Community summaries** - pre-computed global context
3. **Global search** - map-reduce over summaries
4. **Hybrid retrieval** - integrated vector + graph

Without these, we have a "hybrid storage system" but not "GraphRAG."

### Q3: "How can we make it more intelligent and connect the dots and reason?"

**Answer:** Implement the missing GraphRAG components. Specific recommendations:

**For Better Reasoning:**

1. **Multi-hop Graph Traversal** (Local Search)
   - Currently: We query entities independently
   - Needed: Follow relationships 2-3 hops out
   - Impact: "Connect the dots" between related entities
   - Example: Question about "Johnson" also retrieves related people, events, locations

2. **Community-Based Context** (Global Search)
   - Currently: No global context mechanism
   - Needed: Community summaries provide themes and patterns
   - Impact: Answer "What are the main issues in this dataset?"
   - Example: Automatically identify top themes without knowing what to ask

3. **Hybrid Retrieval** (Integrated Approach)
   - Currently: Vector OR graph, not both together
   - Needed: Combine semantic similarity + relational context
   - Impact: 0.96 answer relevancy vs 0.89 (graph only) or 0.91 (vector only)
   - Example: Find semantically similar content AND related entities

4. **Query-Appropriate Strategy** (Router)
   - Currently: One-size-fits-all
   - Needed: Different strategies for different questions
   - Impact: Optimal retrieval for each query type
   - Example: Global questions → community summaries, Specific questions → entity traversal

**Intelligence Enhancement Roadmap:**

```
Phase 1: Foundation (Weeks 1-2)
- Implement entity normalization
- Add multi-hop graph traversal
- Implement basic hybrid retrieval

Phase 2: Community Detection (Weeks 3-4)
- Integrate Neo4j GDS library
- Implement Leiden algorithm
- Build hierarchical community structure

Phase 3: Summaries (Weeks 5-6)
- Generate community summaries (bottom-up)
- Store summaries with metadata
- Build summary retrieval system

Phase 4: Advanced Search (Weeks 7-8)
- Implement global search (map-reduce)
- Enhance local search with communities
- Build query classifier and router

Phase 5: Optimization (Weeks 9-10)
- Performance tuning
- Cost optimization
- Evaluation framework
```

---

## Recommendations

### Critical Priorities (Must Have)

1. **Entity Normalization** - Fix "Officer Johnson" vs "Inspector Johnson"
2. **Community Detection** - Implement Leiden algorithm
3. **Community Summaries** - Generate hierarchical summaries
4. **Global Search** - Implement map-reduce over communities
5. **Hybrid Retrieval** - Integrate vector + graph queries

### Important (Should Have)

6. **Local Search Enhancement** - Multi-hop traversal with community context
7. **Query Classifier** - Route to appropriate search strategy
8. **Schema-Guided Extraction** - Domain-specific entity types
9. **Evaluation Framework** - Measure faithfulness, relevancy, recall

### Nice to Have

10. **Dynamic Community Selection** - Latest Microsoft innovation (2024)
11. **DRIFT Search** - Local + community hybrid
12. **Incremental Updates** - Add new documents without full reindex
13. **Cost Optimization** - Reduce LLM calls with caching

---

## Cost-Benefit Analysis

### Complexity vs Value

**High Value, Medium Complexity:**
- Community Detection (Leiden) - Available in Neo4j GDS
- Hybrid Retrieval - Combine existing queries
- Entity Normalization - Standard NLP

**High Value, High Complexity:**
- Community Summaries - Requires LLM for each community
- Global Search - New query pipeline
- Query Router - Classification logic

**Medium Value, Low Complexity:**
- Multi-hop traversal - Extend existing Cypher
- Schema-guided extraction - Update prompts

### When NOT to Use GraphRAG

Based on research, avoid full GraphRAG if:

1. **Simple documents** - Short, unstructured text without entities
2. **No relational queries** - Users only ask simple factual questions
3. **Limited budget** - Community summaries are expensive to generate
4. **Fast iteration needed** - GraphRAG takes longer to build

### When GraphRAG Excels

Use full GraphRAG when:

1. **Complex documents** - Long documents with many entities across pages
2. **Global questions** - "What are the themes?" "Summarize the dataset"
3. **Relational queries** - "How does X relate to Y?"
4. **Multi-hop reasoning** - Questions requiring connections
5. **Large corpora** - Millions of tokens that need structure

---

## Next Steps

See [GRAPHRAG_IMPLEMENTATION_PLAN.md](./GRAPHRAG_IMPLEMENTATION_PLAN.md) for detailed implementation roadmap.

---

## References

### Papers
- Edge et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." arXiv:2404.16130
- Survey (2025). "Retrieval-Augmented Generation with Graphs (GraphRAG)." arXiv:2501.00309
- Security (2025). "GraphRAG under Fire." arXiv:2501.14050
- Survey (2024). "Graph Retrieval-Augmented Generation: A Survey." arXiv:2408.08921
- HybridRAG (2024). "Integrating Knowledge Graphs and Vector RAG." arXiv:2408.04948

### Implementations
- Microsoft GraphRAG: https://github.com/microsoft/graphrag
- Microsoft Docs: https://microsoft.github.io/graphrag/
- Neo4j GDS Leiden: https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/

### Blog Posts & Guides
- "Do You Really Need GraphRAG?" - Towards Data Science
- "HybridRAG and Why Combine Vector Embeddings with Knowledge Graphs" - Memgraph
- "GraphRAG Field Guide" - Neo4j
- "The Quest for Production-Quality Graph RAG" - Medium
- "Best Chunking Strategies for RAG in 2025" - Firecrawl

---

**Document Version:** 1.0
**Last Updated:** January 2025
**Author:** MIRAGE Development Team
