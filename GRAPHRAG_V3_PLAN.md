# MIRAGE V3: Full GraphRAG Implementation Plan

**Goal**: Transform MIRAGE from a partial GraphRAG implementation (5.25/10) to a complete, state-of-the-art system (9+/10) that surpasses Microsoft GraphRAG in key areas.

**Timeline**: 6 phases, implementable incrementally
**Approach**: Each phase is independently testable and deployable

---

## Architecture Vision: MIRAGE V3

```
                            ┌─────────────────────────────────────────┐
                            │           MIRAGE V3 Architecture         │
                            └─────────────────────────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
              ┌─────▼─────┐            ┌──────▼──────┐           ┌──────▼──────┐
              │  Ingestion │            │  Retrieval  │           │  Generation │
              │  Pipeline  │            │   Engine    │           │   Engine    │
              └─────┬─────┘            └──────┬──────┘           └──────┬──────┘
                    │                         │                         │
     ┌──────────────┼──────────────┐         │          ┌──────────────┼──────────────┐
     │              │              │         │          │              │              │
┌────▼────┐  ┌─────▼─────┐  ┌─────▼─────┐   │    ┌─────▼─────┐  ┌─────▼─────┐  ┌────▼────┐
│ Chunker │  │  Entity   │  │Community  │   │    │  Answer   │  │ Hallucin. │  │ Source  │
│ (Smart) │  │ Extractor │  │ Detector  │   │    │ Generator │  │ Detector  │  │Citation │
└────┬────┘  └─────┬─────┘  └─────┬─────┘   │    └─────┬─────┘  └─────┬─────┘  └────┬────┘
     │              │              │         │          │              │              │
     └──────────────┴──────────────┘         │          └──────────────┴──────────────┘
                    │                         │                         │
              ┌─────▼─────┐                  │                   ┌─────▼─────┐
              │  Storage  │                  │                   │  Quality  │
              │   Layer   │                  │                   │  Control  │
              └─────┬─────┘                  │                   └─────┬─────┘
                    │                         │                         │
     ┌──────────────┼──────────────┐         │          ┌──────────────┼──────────────┐
     │              │              │         │          │              │              │
┌────▼────┐  ┌─────▼─────┐  ┌─────▼─────┐   │    ┌─────▼─────┐  ┌─────▼─────┐  ┌────▼────┐
│ Qdrant  │  │   Neo4j   │  │  Redis    │   │    │  Metrics  │  │ Feedback  │  │ A/B Test│
│ Vectors │  │   Graph   │  │  Cache    │   │    │  Tracker  │  │   Loop    │  │Framework│
└─────────┘  └───────────┘  └───────────┘   │    └───────────┘  └───────────┘  └─────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
              ┌─────▼─────┐           ┌──────▼──────┐          ┌─────▼─────┐
              │   Local   │           │   Global    │          │  Hybrid   │
              │  Search   │           │   Search    │          │  Search   │
              └───────────┘           └─────────────┘          └───────────┘
                    │                        │                        │
                    │                 ┌──────▼──────┐                 │
                    │                 │ Map-Reduce  │                 │
                    │                 │   Engine    │                 │
                    │                 └─────────────┘                 │
                    │                        │                        │
                    └────────────────────────┴────────────────────────┘
                                             │
                                      ┌──────▼──────┐
                                      │   Answer    │
                                      │  + Sources  │
                                      └─────────────┘
```

---

## Phase 1: Global Search (The Critical Missing Piece)

**Priority**: P0 - Critical
**Effort**: 3-4 days
**Impact**: Enables holistic queries (the core GraphRAG innovation)

### 1.1 Map-Reduce Engine

Create a new module for global search over community summaries.

**File**: `mirage/src/core/retrieval/global_search.py`

```python
"""
Global Search Engine - Map-Reduce over Community Summaries

This is THE killer feature of GraphRAG that enables answering:
- "What are the main themes across all documents?"
- "Summarize the key topics in this knowledge base"
- "What patterns emerge from this data?"

Algorithm:
1. MAP: Query each community summary in parallel
   - Each community generates a partial answer
   - Scored by relevance to query

2. FILTER: Keep top-k most relevant partial answers
   - Based on relevance score threshold

3. REDUCE: Combine partial answers into final answer
   - Synthesize coherent response
   - Preserve key insights from each community
"""

@dataclass
class PartialAnswer:
    community_id: str
    level: int
    summary: str
    answer: str
    relevance_score: float
    key_points: List[str]
    sources: List[str]

class GlobalSearchEngine:
    def __init__(
        self,
        neo4j_client,
        llm_client,
        max_communities: int = 20,      # Max communities to query
        min_relevance: float = 0.3,     # Minimum relevance threshold
        parallel_workers: int = 5,      # Parallel LLM calls
        community_level: int = 0        # Which level to search (0=fine, 1+=coarse)
    ):
        ...

    async def search(self, query: str) -> GlobalSearchResult:
        """
        Execute global search using map-reduce pattern.

        Steps:
        1. Get all community summaries at target level
        2. MAP: Generate partial answer for each (parallel)
        3. FILTER: Keep relevant partial answers
        4. REDUCE: Synthesize final answer
        """

        # Step 1: Get communities
        communities = self._get_communities_at_level(self.community_level)

        # Step 2: MAP phase (parallel)
        partial_answers = await self._map_phase(query, communities)

        # Step 3: FILTER phase
        relevant_answers = self._filter_phase(partial_answers)

        # Step 4: REDUCE phase
        final_answer = await self._reduce_phase(query, relevant_answers)

        return final_answer

    async def _map_phase(
        self,
        query: str,
        communities: List[Dict]
    ) -> List[PartialAnswer]:
        """Generate partial answer for each community."""

        tasks = []
        for community in communities:
            task = self._generate_partial_answer(query, community)
            tasks.append(task)

        # Run in parallel with semaphore for rate limiting
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if isinstance(r, PartialAnswer)]

    def _map_prompt(self, query: str, community: Dict) -> str:
        """Prompt for generating partial answer from community summary."""

        return f"""أنت محلل بيانات. استخدم الملخص التالي للإجابة على السؤال.

ملخص المجموعة:
{community['summary']}

الكيانات الرئيسية: {', '.join(community.get('key_entities', [])[:10])}

السؤال: {query}

إذا كان الملخص يحتوي على معلومات ذات صلة بالسؤال، اكتب إجابة موجزة (2-3 جمل).
إذا لم يكن هناك معلومات ذات صلة، اكتب "لا توجد معلومات ذات صلة".

الإجابة:"""

    async def _reduce_phase(
        self,
        query: str,
        partial_answers: List[PartialAnswer]
    ) -> GlobalSearchResult:
        """Combine partial answers into final coherent answer."""

        # Format partial answers for synthesis
        answers_text = self._format_partial_answers(partial_answers)

        prompt = f"""أنت محلل بيانات. اجمع الإجابات الجزئية التالية في إجابة شاملة واحدة.

السؤال: {query}

الإجابات الجزئية من مصادر مختلفة:
{answers_text}

اكتب إجابة شاملة تجمع أهم النقاط من جميع المصادر:
- ابدأ بالنقطة الأهم
- اذكر الأنماط أو الموضوعات المشتركة
- لا تكرر المعلومات
- اذكر المصادر عند الحاجة

الإجابة الشاملة:"""

        response = await self.llm_client.generate(prompt)

        return GlobalSearchResult(
            query=query,
            answer=response,
            partial_answers=partial_answers,
            communities_searched=len(partial_answers),
            total_communities=len(self._get_communities_at_level(self.community_level))
        )
```

### 1.2 Integration with Retrieval Engine

**File**: `mirage/src/core/retrieval/retrieval_engine.py`

Add new `GLOBAL` mode that uses map-reduce:

```python
async def _global_retrieve(self, query: str, top_k: int) -> RetrievalResult:
    """
    Global search using map-reduce over community summaries.

    Best for:
    - "What are the main themes?"
    - "Summarize the key topics"
    - "What patterns emerge?"
    """

    # Initialize global search engine
    global_engine = GlobalSearchEngine(
        neo4j_client=self.neo4j_client,
        llm_client=self.llm_client,
        community_level=0  # Fine-grained communities
    )

    # Execute global search
    result = await global_engine.search(query)

    # Also get supporting chunks from relevant communities
    supporting_chunks = self._get_chunks_from_communities(
        [pa.community_id for pa in result.partial_answers[:5]]
    )

    return RetrievalResult(
        chunks=supporting_chunks,
        mode=RetrievalMode.GLOBAL,
        global_answer=result.answer,
        metadata={
            'communities_searched': result.communities_searched,
            'partial_answers': len(result.partial_answers)
        }
    )
```

### 1.3 Query Router Update

Detect queries that need global search:

```python
GLOBAL_QUERY_PATTERNS = [
    # Arabic patterns
    r'ما هي (?:أهم |رئيسية |)(?:الموضوعات|المواضيع|الأفكار)',
    r'لخص|تلخيص|ملخص',
    r'ما هي (?:الأنماط|الاتجاهات)',
    r'بشكل عام|إجمالاً',
    r'عبر (?:جميع |كل |)(?:الوثائق|المستندات|البيانات)',

    # English patterns
    r'what are the (?:main |key |primary |)(?:themes|topics|ideas)',
    r'summarize|summary|overview',
    r'what (?:patterns|trends)',
    r'across (?:all |the |)(?:documents|data)',
    r'in general|overall',
]

def _is_global_query(self, query: str) -> bool:
    """Detect if query requires global search."""
    query_lower = query.lower()

    for pattern in GLOBAL_QUERY_PATTERNS:
        if re.search(pattern, query_lower):
            return True

    return False
```

---

## Phase 2: Enhanced Entity Extraction

**Priority**: P0 - Critical
**Effort**: 4-5 days
**Impact**: Fixes the garbage-in-garbage-out problem

### 2.1 Hybrid Extraction Pipeline

Create a robust extraction pipeline that combines multiple methods:

**File**: `mirage/src/core/graph_builder/enhanced_entity_extractor.py`

```python
"""
Enhanced Entity Extraction Pipeline

Strategy:
1. Primary: LLM extraction (best quality)
2. Secondary: Rule-based + NER ensemble
3. Validation: Cross-check entities across methods
4. Resolution: Deduplicate and normalize entities
5. Linking: Connect to external knowledge bases

Quality Target: 90%+ F1 on domain entities
"""

class EnhancedEntityExtractor:
    def __init__(
        self,
        llm_client,
        use_camel: bool = True,      # Arabic NER
        use_spacy: bool = True,       # English NER
        use_rules: bool = True,       # Rule-based extraction
        confidence_threshold: float = 0.7,
        enable_linking: bool = True   # Entity linking to Wikidata
    ):
        self.extractors = []

        # Build extractor ensemble
        if llm_client:
            self.extractors.append(LLMExtractor(llm_client, weight=0.5))
        if use_camel:
            self.extractors.append(CAMeLExtractor(weight=0.25))
        if use_spacy:
            self.extractors.append(SpaCyExtractor(weight=0.25))
        if use_rules:
            self.extractors.append(RuleBasedExtractor(weight=0.15))

    def extract(self, text: str, language: str = 'auto') -> ExtractionResult:
        """
        Extract entities using ensemble approach.

        1. Run all extractors in parallel
        2. Merge results with weighted voting
        3. Validate and filter by confidence
        4. Resolve duplicates
        5. Link to external KBs
        """

        # Step 1: Parallel extraction
        all_entities = []
        for extractor in self.extractors:
            entities = extractor.extract(text, language)
            all_entities.extend(entities)

        # Step 2: Weighted voting for entity mentions
        merged = self._merge_with_voting(all_entities)

        # Step 3: Confidence filtering
        confident = [e for e in merged if e.confidence >= self.confidence_threshold]

        # Step 4: Entity resolution
        resolved = self._resolve_entities(confident)

        # Step 5: Entity linking (optional)
        if self.enable_linking:
            resolved = self._link_entities(resolved)

        return ExtractionResult(
            entities=resolved,
            extraction_method='ensemble',
            extractor_contributions=self._get_contributions()
        )

    def _merge_with_voting(self, entities: List[Entity]) -> List[Entity]:
        """Merge entities from multiple extractors using weighted voting."""

        # Group by normalized text
        groups = defaultdict(list)
        for entity in entities:
            key = self._normalize_entity_text(entity.text)
            groups[key].append(entity)

        merged = []
        for key, group in groups.items():
            # Weighted vote for entity type
            type_votes = defaultdict(float)
            total_weight = 0

            for entity in group:
                type_votes[entity.type] += entity.source_weight * entity.confidence
                total_weight += entity.source_weight

            # Select highest-voted type
            best_type = max(type_votes, key=type_votes.get)

            # Calculate merged confidence
            merged_confidence = type_votes[best_type] / total_weight

            # Boost confidence if multiple extractors agree
            agreement_bonus = min(0.2, 0.05 * (len(group) - 1))

            merged.append(Entity(
                text=group[0].text,  # Use original text
                type=best_type,
                confidence=min(1.0, merged_confidence + agreement_bonus),
                sources=[e.source for e in group],
                mentions=len(group)
            ))

        return merged

    def _resolve_entities(self, entities: List[Entity]) -> List[Entity]:
        """Resolve duplicate entities using embedding similarity."""

        if not entities:
            return []

        # Get embeddings for all entities
        texts = [e.text for e in entities]
        embeddings = self.embedder.embed_batch(texts)

        # Cluster similar entities
        clusters = []
        used = set()

        for i, entity in enumerate(entities):
            if i in used:
                continue

            cluster = [entity]
            used.add(i)

            for j, other in enumerate(entities[i+1:], i+1):
                if j in used:
                    continue

                similarity = cosine_similarity(embeddings[i], embeddings[j])

                if similarity > 0.85:  # High similarity threshold
                    cluster.append(other)
                    used.add(j)

            clusters.append(cluster)

        # Merge each cluster into canonical entity
        resolved = []
        for cluster in clusters:
            canonical = self._merge_cluster(cluster)
            resolved.append(canonical)

        return resolved

    def _link_entities(self, entities: List[Entity]) -> List[Entity]:
        """Link entities to Wikidata for disambiguation."""

        for entity in entities:
            # Query Wikidata for matches
            wikidata_id = self._query_wikidata(entity.text, entity.type)

            if wikidata_id:
                entity.wikidata_id = wikidata_id
                entity.linked = True

                # Get additional info from Wikidata
                entity.description = self._get_wikidata_description(wikidata_id)

        return entities
```

### 2.2 Rule-Based Arabic Entity Patterns

**File**: `mirage/src/core/graph_builder/arabic_patterns.py`

```python
"""
Arabic-specific entity extraction patterns.

Covers:
- Saudi government entities (ministries, agencies)
- Organizations and companies
- Persons with titles
- Locations (cities, regions)
- Programs and initiatives
"""

ARABIC_ENTITY_PATTERNS = {
    'Organization': [
        # Ministries
        r'وزارة\s+[\u0600-\u06FF\s]+',
        # Agencies and authorities
        r'هيئة\s+[\u0600-\u06FF\s]+',
        r'مؤسسة\s+[\u0600-\u06FF\s]+',
        # Companies
        r'شركة\s+[\u0600-\u06FF\s]+',
        # Centers
        r'مركز\s+[\u0600-\u06FF\s]+',
        # Universities
        r'جامعة\s+[\u0600-\u06FF\s]+',
    ],

    'Person': [
        # With titles
        r'(?:الأمير|الأميرة|الشيخ|الدكتور|المهندس|الأستاذ)\s+[\u0600-\u06FF\s]+',
        # Minister pattern
        r'وزير\s+[\u0600-\u06FF]+\s+[\u0600-\u06FF\s]+',
    ],

    'Location': [
        # Saudi cities
        r'(?:مدينة|منطقة|محافظة)\s+[\u0600-\u06FF\s]+',
        # Known Saudi cities (explicit list)
        r'(?:الرياض|جدة|مكة|المدينة|الدمام|الخبر|أبها|تبوك|نجران|جازان)',
    ],

    'Program': [
        # Initiatives and programs
        r'(?:برنامج|مبادرة|مشروع|رؤية)\s+[\u0600-\u06FF\s\d]+',
        # Awards
        r'جائزة\s+[\u0600-\u06FF\s]+',
    ],

    'Event': [
        # Conferences and events
        r'(?:مؤتمر|منتدى|قمة|ملتقى)\s+[\u0600-\u06FF\s]+',
    ]
}

class RuleBasedExtractor:
    def extract(self, text: str, language: str = 'ar') -> List[Entity]:
        entities = []

        for entity_type, patterns in ARABIC_ENTITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entities.append(Entity(
                        text=match.group().strip(),
                        type=entity_type,
                        confidence=0.75,  # Rule-based confidence
                        source='rules',
                        start=match.start(),
                        end=match.end()
                    ))

        return entities
```

### 2.3 Typed Relationship Extraction

**File**: `mirage/src/core/graph_builder/relationship_extractor.py`

```python
"""
Semantic Relationship Extraction

Instead of generic RELATED_TO or COOCCURS_WITH, extract typed relationships:
- FOUNDED_BY, WORKS_AT, LOCATED_IN, PART_OF, etc.

This enables reasoning like:
"Find all organizations founded by people from Riyadh"
"""

RELATIONSHIP_TYPES = [
    'FOUNDED_BY',      # Organization <- Person
    'LED_BY',          # Organization <- Person
    'WORKS_AT',        # Person -> Organization
    'LOCATED_IN',      # Entity -> Location
    'PART_OF',         # Entity -> Entity (hierarchy)
    'PARTNERED_WITH',  # Organization <-> Organization
    'FUNDED_BY',       # Project <- Organization
    'LAUNCHED',        # Organization -> Program
    'AWARDED_TO',      # Award -> Person/Organization
    'COLLABORATES_WITH', # Entity <-> Entity
]

class RelationshipExtractor:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def extract(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """Extract typed relationships between entities."""

        if len(entities) < 2:
            return []

        # Create entity context for LLM
        entity_list = "\n".join([
            f"- {e.text} ({e.type})" for e in entities
        ])

        prompt = f"""حلل النص التالي واستخرج العلاقات بين الكيانات.

النص:
{text}

الكيانات المعروفة:
{entity_list}

أنواع العلاقات المتاحة:
- FOUNDED_BY: أسسه/أنشأه
- LED_BY: يقوده/يرأسه
- WORKS_AT: يعمل في
- LOCATED_IN: يقع في
- PART_OF: جزء من
- PARTNERED_WITH: شريك مع
- FUNDED_BY: ممول من
- LAUNCHED: أطلق
- AWARDED_TO: مُنح لـ
- COLLABORATES_WITH: يتعاون مع

استخرج العلاقات بالتنسيق التالي (JSON):
[
  {{"source": "كيان1", "relationship": "RELATIONSHIP_TYPE", "target": "كيان2", "confidence": 0.9}}
]

العلاقات:"""

        response = self.llm_client.generate(prompt)
        relationships = self._parse_relationships(response, entities)

        return relationships
```

---

## Phase 3: Retrieval Quality & Metrics

**Priority**: P0 - Critical
**Effort**: 3-4 days
**Impact**: Enables measurement and improvement

### 3.1 Metrics Tracker

**File**: `mirage/src/core/evaluation/metrics.py`

```python
"""
Retrieval Quality Metrics

Tracks:
- MRR (Mean Reciprocal Rank): How high is the first relevant result?
- NDCG (Normalized Discounted Cumulative Gain): Ranking quality
- Precision@K: % of top-K that are relevant
- Recall@K: % of relevant docs in top-K
- Answer Relevancy: How relevant is the answer to the query?
- Faithfulness: Is the answer grounded in sources?
"""

@dataclass
class RetrievalMetrics:
    mrr: float                    # Mean Reciprocal Rank
    ndcg: float                   # Normalized DCG
    precision_at_k: Dict[int, float]  # Precision@1, @3, @5, @10
    recall_at_k: Dict[int, float]     # Recall@1, @3, @5, @10
    latency_ms: float             # Query latency
    mode_used: str                # Which retrieval mode

@dataclass
class AnswerMetrics:
    relevancy: float              # 0-1: Answer relevance to query
    faithfulness: float           # 0-1: Grounded in sources
    completeness: float           # 0-1: Covers all aspects of query
    coherence: float              # 0-1: Well-structured answer

class MetricsTracker:
    def __init__(self, storage_path: str = "metrics.db"):
        self.db = sqlite3.connect(storage_path)
        self._init_tables()

    def log_retrieval(
        self,
        query: str,
        retrieved_ids: List[str],
        relevant_ids: List[str],  # Ground truth
        mode: str,
        latency_ms: float
    ) -> RetrievalMetrics:
        """Calculate and log retrieval metrics."""

        metrics = RetrievalMetrics(
            mrr=self._calculate_mrr(retrieved_ids, relevant_ids),
            ndcg=self._calculate_ndcg(retrieved_ids, relevant_ids),
            precision_at_k={
                1: self._precision_at_k(retrieved_ids, relevant_ids, 1),
                3: self._precision_at_k(retrieved_ids, relevant_ids, 3),
                5: self._precision_at_k(retrieved_ids, relevant_ids, 5),
                10: self._precision_at_k(retrieved_ids, relevant_ids, 10),
            },
            recall_at_k={
                1: self._recall_at_k(retrieved_ids, relevant_ids, 1),
                3: self._recall_at_k(retrieved_ids, relevant_ids, 3),
                5: self._recall_at_k(retrieved_ids, relevant_ids, 5),
                10: self._recall_at_k(retrieved_ids, relevant_ids, 10),
            },
            latency_ms=latency_ms,
            mode_used=mode
        )

        self._store_metrics(query, metrics)
        return metrics

    def _calculate_mrr(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Mean Reciprocal Rank - position of first relevant result."""
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_ndcg(self, retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
        """Normalized Discounted Cumulative Gain."""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 2)

        # Ideal DCG
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def get_aggregate_metrics(self, time_range: str = "24h") -> Dict:
        """Get aggregate metrics for dashboard."""

        return {
            'avg_mrr': self._query_avg('mrr', time_range),
            'avg_ndcg': self._query_avg('ndcg', time_range),
            'avg_latency_ms': self._query_avg('latency_ms', time_range),
            'mode_distribution': self._query_mode_distribution(time_range),
            'p95_latency': self._query_percentile('latency_ms', 95, time_range),
            'queries_count': self._query_count(time_range),
        }
```

### 3.2 Benchmark Dataset

**File**: `mirage/src/core/evaluation/benchmark.py`

```python
"""
Benchmark Dataset for MIRAGE Evaluation

Contains:
- 100+ query-answer pairs with relevance judgments
- Coverage of all query types (L1-L5)
- Ground truth document IDs for each query
"""

BENCHMARK_QUERIES = [
    # L1: Direct Factual
    {
        "query": "ما هي جائزة الحكومة الرقمية؟",
        "type": "L1_FACTUAL",
        "expected_mode": "NAIVE",
        "relevant_chunk_ids": ["chunk_001", "chunk_002"],
        "expected_entities": ["جائزة الحكومة الرقمية"],
        "ground_truth_answer": "جائزة الحكومة الرقمية هي...",
    },

    # L2: Entity-Centric
    {
        "query": "من هم شركاء مبادرة التحول الرقمي؟",
        "type": "L2_ENTITY",
        "expected_mode": "LOCAL",
        "relevant_chunk_ids": ["chunk_010", "chunk_011", "chunk_012"],
        "expected_entities": ["مبادرة التحول الرقمي", "شركة X", "شركة Y"],
    },

    # L3: Relationship
    {
        "query": "كيف ترتبط وزارة الاتصالات بجائزة الحكومة الرقمية؟",
        "type": "L3_RELATIONSHIP",
        "expected_mode": "GLOBAL",
        "relevant_chunk_ids": ["chunk_020", "chunk_021"],
    },

    # L4: Multi-hop
    {
        "query": "ما المشاريع التي أطلقتها الجهات الفائزة بجائزة الحكومة الرقمية؟",
        "type": "L4_MULTIHOP",
        "expected_mode": "HYBRID",
        "relevant_chunk_ids": ["chunk_030", "chunk_031", "chunk_032", "chunk_033"],
    },

    # L5: Holistic/Global
    {
        "query": "ما هي الموضوعات الرئيسية في قاعدة البيانات؟",
        "type": "L5_GLOBAL",
        "expected_mode": "GLOBAL",
        "relevant_chunk_ids": [],  # All documents relevant
        "requires_global_search": True,
    },
]

class BenchmarkRunner:
    def __init__(self, retrieval_engine, metrics_tracker):
        self.engine = retrieval_engine
        self.metrics = metrics_tracker

    def run_benchmark(self) -> BenchmarkReport:
        """Run full benchmark and generate report."""

        results = []

        for query_data in BENCHMARK_QUERIES:
            result = self._evaluate_query(query_data)
            results.append(result)

        return BenchmarkReport(
            total_queries=len(results),
            avg_mrr=np.mean([r.mrr for r in results]),
            avg_ndcg=np.mean([r.ndcg for r in results]),
            by_query_type=self._aggregate_by_type(results),
            by_mode=self._aggregate_by_mode(results),
            failures=[r for r in results if r.mrr == 0],
        )
```

---

## Phase 4: Self-Correction & Quality Control

**Priority**: P1 - High
**Effort**: 4-5 days
**Impact**: Improves reliability and catches failures

### 4.1 Retrieval Validator (CRAG-style)

**File**: `mirage/src/core/retrieval/validator.py`

```python
"""
Retrieval Validation and Correction

Inspired by CRAG (Corrective RAG):
1. Assess if retrieved documents are relevant
2. If not relevant, try corrective strategies
3. Return validated results or fallback

This prevents garbage answers from garbage retrieval.
"""

class RetrievalValidator:
    def __init__(
        self,
        llm_client,
        reranker,
        relevance_threshold: float = 0.5
    ):
        self.llm_client = llm_client
        self.reranker = reranker
        self.threshold = relevance_threshold

    async def validate_and_correct(
        self,
        query: str,
        results: RetrievalResult,
        max_corrections: int = 2
    ) -> ValidatedResult:
        """
        Validate retrieval results and apply corrections if needed.

        Steps:
        1. Score relevance of each retrieved chunk
        2. If too many irrelevant, apply correction strategy
        3. Return validated results with confidence
        """

        # Step 1: Score relevance
        relevance_scores = await self._score_relevance(query, results.chunks)

        # Step 2: Check if results are acceptable
        relevant_count = sum(1 for s in relevance_scores if s >= self.threshold)
        relevance_ratio = relevant_count / len(results.chunks) if results.chunks else 0

        if relevance_ratio >= 0.5:
            # Results are acceptable
            return ValidatedResult(
                chunks=self._filter_by_relevance(results.chunks, relevance_scores),
                status='VALIDATED',
                confidence=relevance_ratio,
                corrections_applied=0
            )

        # Step 3: Apply corrections
        corrected = await self._apply_corrections(
            query,
            results,
            relevance_scores,
            max_corrections
        )

        return corrected

    async def _score_relevance(
        self,
        query: str,
        chunks: List[Dict]
    ) -> List[float]:
        """Score relevance of each chunk to query."""

        # Use cross-encoder for accurate relevance scoring
        scores = self.reranker.score_pairs(
            query,
            [c['text'] for c in chunks]
        )

        return scores

    async def _apply_corrections(
        self,
        query: str,
        results: RetrievalResult,
        relevance_scores: List[float],
        max_corrections: int
    ) -> ValidatedResult:
        """Apply corrective strategies."""

        corrections = [
            self._try_query_expansion,
            self._try_alternative_mode,
            self._try_increased_depth,
        ]

        for i, correction in enumerate(corrections[:max_corrections]):
            new_results = await correction(query, results)
            new_scores = await self._score_relevance(query, new_results.chunks)

            relevant_count = sum(1 for s in new_scores if s >= self.threshold)
            relevance_ratio = relevant_count / len(new_results.chunks) if new_results.chunks else 0

            if relevance_ratio >= 0.5:
                return ValidatedResult(
                    chunks=self._filter_by_relevance(new_results.chunks, new_scores),
                    status='CORRECTED',
                    confidence=relevance_ratio,
                    corrections_applied=i + 1,
                    correction_method=correction.__name__
                )

        # All corrections failed
        return ValidatedResult(
            chunks=results.chunks[:3],  # Return top 3 anyway
            status='FAILED_CORRECTION',
            confidence=0.3,
            corrections_applied=max_corrections
        )

    async def _try_query_expansion(
        self,
        query: str,
        original: RetrievalResult
    ) -> RetrievalResult:
        """Expand query with synonyms and related terms."""

        expansion_prompt = f"""وسّع هذا الاستعلام بإضافة مصطلحات مرادفة وذات صلة:

الاستعلام الأصلي: {query}

اكتب 3 استعلامات موسعة (واحد في كل سطر):"""

        expanded = await self.llm_client.generate(expansion_prompt)
        expanded_queries = expanded.strip().split('\n')[:3]

        # Search with expanded queries
        all_results = []
        for eq in expanded_queries:
            results = await self.engine.search(eq, top_k=5)
            all_results.extend(results.chunks)

        # Deduplicate and re-rank
        unique_results = self._deduplicate(all_results)
        reranked = self.reranker.rerank(query, unique_results)

        return RetrievalResult(chunks=reranked[:10])

    async def _try_alternative_mode(
        self,
        query: str,
        original: RetrievalResult
    ) -> RetrievalResult:
        """Try different retrieval mode."""

        # If original was NAIVE, try LOCAL
        # If original was LOCAL, try GLOBAL
        # etc.

        alternative_modes = {
            'NAIVE': 'LOCAL',
            'LOCAL': 'HYBRID',
            'GLOBAL': 'LOCAL',
            'HYBRID': 'MIX',
        }

        current_mode = original.mode.value
        new_mode = alternative_modes.get(current_mode, 'HYBRID')

        return await self.engine.search(query, mode=new_mode)
```

### 4.2 Answer Validator

**File**: `mirage/src/core/generation/answer_validator.py`

```python
"""
Answer Quality Validation

Checks:
1. Faithfulness: Is the answer grounded in sources?
2. Relevancy: Does it answer the question?
3. Completeness: Does it cover all aspects?
4. Hallucination detection: Contains unsupported claims?
"""

class AnswerValidator:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def validate(
        self,
        query: str,
        answer: str,
        sources: List[Dict]
    ) -> AnswerValidation:
        """Validate answer quality."""

        # Check faithfulness
        faithfulness = await self._check_faithfulness(answer, sources)

        # Check relevancy
        relevancy = await self._check_relevancy(query, answer)

        # Detect potential hallucinations
        hallucinations = await self._detect_hallucinations(answer, sources)

        return AnswerValidation(
            is_valid=faithfulness > 0.7 and relevancy > 0.7,
            faithfulness=faithfulness,
            relevancy=relevancy,
            hallucination_risk=len(hallucinations) > 0,
            hallucinated_claims=hallucinations,
            confidence=min(faithfulness, relevancy)
        )

    async def _check_faithfulness(
        self,
        answer: str,
        sources: List[Dict]
    ) -> float:
        """Check if answer is grounded in sources."""

        sources_text = "\n---\n".join([s['text'] for s in sources])

        prompt = f"""قيّم مدى دعم المصادر للإجابة.

الإجابة:
{answer}

المصادر:
{sources_text}

على مقياس من 0 إلى 1:
- 0: الإجابة غير مدعومة بالمصادر
- 0.5: الإجابة مدعومة جزئياً
- 1: الإجابة مدعومة بالكامل

الدرجة (رقم فقط):"""

        response = await self.llm_client.generate(prompt)

        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5  # Default if parsing fails

    async def _detect_hallucinations(
        self,
        answer: str,
        sources: List[Dict]
    ) -> List[str]:
        """Detect claims not supported by sources."""

        sources_text = "\n".join([s['text'] for s in sources])

        prompt = f"""حدد أي ادعاءات في الإجابة غير مدعومة بالمصادر.

الإجابة:
{answer}

المصادر:
{sources_text}

إذا وجدت ادعاءات غير مدعومة، اذكرها (واحد في كل سطر).
إذا كانت جميع الادعاءات مدعومة، اكتب "لا توجد".

الادعاءات غير المدعومة:"""

        response = await self.llm_client.generate(prompt)

        if "لا توجد" in response or response.strip() == "":
            return []

        claims = [c.strip() for c in response.strip().split('\n') if c.strip()]
        return claims
```

---

## Phase 5: Intelligence Enhancements

**Priority**: P1 - High
**Effort**: 5-6 days
**Impact**: Makes the system smarter

### 5.1 Query Understanding & Expansion

**File**: `mirage/src/core/retrieval/query_processor.py`

```python
"""
Advanced Query Processing

Features:
1. Query expansion: Add synonyms, related terms
2. Query decomposition: Break complex queries into sub-queries
3. Intent detection: Understand what user really wants
4. Language handling: Normalize Arabic diacritics, handle mixed language
"""

class QueryProcessor:
    def __init__(self, llm_client, embedder):
        self.llm_client = llm_client
        self.embedder = embedder

    async def process(self, query: str) -> ProcessedQuery:
        """Full query processing pipeline."""

        # Step 1: Normalize
        normalized = self._normalize(query)

        # Step 2: Detect language and intent
        language = self._detect_language(normalized)
        intent = await self._detect_intent(normalized)

        # Step 3: Extract key terms
        key_terms = self._extract_key_terms(normalized)

        # Step 4: Expand query
        expansions = await self._expand_query(normalized, language)

        # Step 5: Decompose if complex
        sub_queries = await self._decompose_if_needed(normalized, intent)

        return ProcessedQuery(
            original=query,
            normalized=normalized,
            language=language,
            intent=intent,
            key_terms=key_terms,
            expansions=expansions,
            sub_queries=sub_queries
        )

    def _normalize(self, query: str) -> str:
        """Normalize query text."""

        # Remove Arabic diacritics (tashkeel)
        query = self._remove_diacritics(query)

        # Normalize Arabic letters (alef, yaa, etc.)
        query = self._normalize_arabic(query)

        # Lowercase for consistency
        query = query.lower()

        return query.strip()

    async def _expand_query(
        self,
        query: str,
        language: str
    ) -> List[str]:
        """Generate query expansions with synonyms."""

        prompt = f"""أنت مساعد للبحث. أضف مصطلحات مرادفة وذات صلة لتحسين نتائج البحث.

الاستعلام: {query}

اكتب 3-5 مصطلحات إضافية (مفصولة بفواصل):"""

        response = await self.llm_client.generate(prompt)
        expansions = [t.strip() for t in response.split(',')]

        return expansions[:5]

    async def _decompose_if_needed(
        self,
        query: str,
        intent: QueryIntent
    ) -> List[str]:
        """Decompose complex queries into sub-queries."""

        if intent.complexity < 2:
            return [query]  # Simple query, no decomposition

        prompt = f"""هذا استعلام معقد. قسّمه إلى أسئلة فرعية بسيطة.

الاستعلام: {query}

الأسئلة الفرعية (واحد في كل سطر):"""

        response = await self.llm_client.generate(prompt)
        sub_queries = [q.strip() for q in response.strip().split('\n') if q.strip()]

        return sub_queries[:4]  # Max 4 sub-queries
```

### 5.2 Result Diversification (MMR)

**File**: `mirage/src/core/retrieval/diversifier.py`

```python
"""
Result Diversification using Maximal Marginal Relevance (MMR)

Problem: Top-K results are often semantically similar
Solution: Balance relevance with diversity

MMR = λ * Similarity(query, doc) - (1-λ) * max(Similarity(doc, selected_docs))
"""

class ResultDiversifier:
    def __init__(self, embedder, lambda_param: float = 0.7):
        self.embedder = embedder
        self.lambda_param = lambda_param  # Higher = more relevance, lower = more diversity

    def diversify(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """Select diverse subset using MMR."""

        if len(candidates) <= top_k:
            return candidates

        # Get embeddings
        query_embedding = self.embedder.embed(query)
        doc_embeddings = self.embedder.embed_batch([c['text'] for c in candidates])

        # Calculate query-document similarities
        query_similarities = [
            cosine_similarity(query_embedding, doc_emb)
            for doc_emb in doc_embeddings
        ]

        # MMR selection
        selected = []
        selected_indices = []
        remaining = list(range(len(candidates)))

        for _ in range(top_k):
            if not remaining:
                break

            # Calculate MMR score for each remaining document
            mmr_scores = []

            for idx in remaining:
                relevance = query_similarities[idx]

                # Max similarity to already selected
                if selected_indices:
                    max_sim = max(
                        cosine_similarity(doc_embeddings[idx], doc_embeddings[sel_idx])
                        for sel_idx in selected_indices
                    )
                else:
                    max_sim = 0

                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_scores.append((idx, mmr))

            # Select document with highest MMR
            best_idx, _ = max(mmr_scores, key=lambda x: x[1])

            selected.append(candidates[best_idx])
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return selected
```

### 5.3 Multi-Hop Reasoning

**File**: `mirage/src/core/retrieval/multihop.py`

```python
"""
Multi-Hop Reasoning for Complex Queries

For queries like:
"ما المشاريع التي أطلقتها الجهات الفائزة بجائزة الحكومة الرقمية؟"

Steps:
1. Find winners of the award (hop 1)
2. For each winner, find their projects (hop 2)
3. Combine results
"""

class MultiHopReasoner:
    def __init__(self, retrieval_engine, graph_client, max_hops: int = 3):
        self.engine = retrieval_engine
        self.graph = graph_client
        self.max_hops = max_hops

    async def reason(self, query: str) -> MultiHopResult:
        """Execute multi-hop reasoning."""

        # Step 1: Decompose query into hops
        hops = await self._decompose_to_hops(query)

        if len(hops) == 1:
            # Single hop, use normal retrieval
            result = await self.engine.search(query)
            return MultiHopResult(
                final_answer=result,
                reasoning_chain=[],
                hops_executed=1
            )

        # Step 2: Execute hops sequentially
        reasoning_chain = []
        context = {}

        for i, hop in enumerate(hops[:self.max_hops]):
            hop_result = await self._execute_hop(hop, context)
            reasoning_chain.append(hop_result)

            # Update context for next hop
            context = self._update_context(context, hop_result)

        # Step 3: Synthesize final answer
        final_answer = await self._synthesize(query, reasoning_chain)

        return MultiHopResult(
            final_answer=final_answer,
            reasoning_chain=reasoning_chain,
            hops_executed=len(reasoning_chain)
        )

    async def _decompose_to_hops(self, query: str) -> List[HopQuery]:
        """Decompose query into sequential hops."""

        prompt = f"""حلل هذا الاستعلام المعقد إلى خطوات متسلسلة.

الاستعلام: {query}

مثال:
- الاستعلام: "ما المشاريع التي أطلقتها الجهات الفائزة بالجائزة؟"
- الخطوات:
  1. ابحث عن الجهات الفائزة بالجائزة
  2. لكل جهة، ابحث عن المشاريع التي أطلقتها

اكتب الخطوات (واحدة في كل سطر):"""

        response = await self.llm_client.generate(prompt)

        hops = []
        for line in response.strip().split('\n'):
            if line.strip():
                hops.append(HopQuery(
                    query=line.strip(),
                    depends_on=len(hops) - 1 if hops else None
                ))

        return hops

    async def _execute_hop(
        self,
        hop: HopQuery,
        context: Dict
    ) -> HopResult:
        """Execute a single hop."""

        # Inject context from previous hops
        query_with_context = self._inject_context(hop.query, context)

        # Try graph traversal first
        if context.get('entities'):
            graph_results = await self._graph_hop(
                context['entities'],
                hop.query
            )
            if graph_results:
                return HopResult(
                    query=hop.query,
                    method='graph',
                    results=graph_results,
                    entities_found=self._extract_entities(graph_results)
                )

        # Fallback to vector search
        vector_results = await self.engine.search(query_with_context)

        return HopResult(
            query=hop.query,
            method='vector',
            results=vector_results.chunks,
            entities_found=self._extract_entities(vector_results.chunks)
        )

    async def _graph_hop(
        self,
        source_entities: List[str],
        hop_query: str
    ) -> List[Dict]:
        """Execute hop via graph traversal."""

        results = []

        for entity in source_entities[:5]:  # Limit to top 5 entities
            # Get related entities
            related = self.graph.get_related_entities(
                entity_name=entity,
                relationship_types=['LAUNCHED', 'CREATED', 'FOUNDED', 'WORKS_AT'],
                max_depth=1
            )

            # Get chunks mentioning related entities
            for rel in related:
                chunks = self.graph.get_chunks_for_entity(rel['name'])
                results.extend(chunks)

        return results
```

---

## Phase 6: Production Hardening

**Priority**: P2 - Medium
**Effort**: 4-5 days
**Impact**: Makes system production-ready

### 6.1 Caching Layer

**File**: `mirage/src/core/cache/redis_cache.py`

```python
"""
Redis-based Caching Layer

Caches:
- Query embeddings (avoid recomputation)
- Retrieval results (for repeated queries)
- Community summaries (expensive to regenerate)
- Entity extraction results
"""

class CacheManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)

        # TTL configurations
        self.ttls = {
            'embedding': 3600 * 24,      # 24 hours
            'retrieval': 3600,           # 1 hour
            'community_summary': 3600 * 24 * 7,  # 1 week
            'entity_extraction': 3600 * 24,  # 24 hours
        }

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
        cached = self.redis.get(key)
        if cached:
            return np.frombuffer(cached, dtype=np.float32)
        return None

    def set_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding."""
        key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
        self.redis.setex(key, self.ttls['embedding'], embedding.tobytes())

    def get_retrieval(self, query: str, mode: str) -> Optional[RetrievalResult]:
        """Get cached retrieval result."""
        key = f"ret:{mode}:{hashlib.md5(query.encode()).hexdigest()}"
        cached = self.redis.get(key)
        if cached:
            return pickle.loads(cached)
        return None

    def set_retrieval(self, query: str, mode: str, result: RetrievalResult):
        """Cache retrieval result."""
        key = f"ret:{mode}:{hashlib.md5(query.encode()).hexdigest()}"
        self.redis.setex(key, self.ttls['retrieval'], pickle.dumps(result))
```

### 6.2 Error Handling & Monitoring

**File**: `mirage/src/core/monitoring/health.py`

```python
"""
Health Checks and Monitoring

Monitors:
- Component health (Neo4j, Qdrant, TGI, Redis)
- Query latencies and error rates
- Resource utilization
- Alert thresholds
"""

class HealthMonitor:
    def __init__(self):
        self.components = {
            'neo4j': Neo4jHealthCheck(),
            'qdrant': QdrantHealthCheck(),
            'tgi': TGIHealthCheck(),
            'redis': RedisHealthCheck(),
        }

        self.metrics = MetricsCollector()

    async def check_all(self) -> HealthReport:
        """Check health of all components."""

        results = {}
        for name, checker in self.components.items():
            try:
                result = await checker.check()
                results[name] = result
            except Exception as e:
                results[name] = HealthResult(
                    status='ERROR',
                    message=str(e),
                    latency_ms=-1
                )

        overall = 'HEALTHY' if all(r.status == 'HEALTHY' for r in results.values()) else 'DEGRADED'

        return HealthReport(
            status=overall,
            components=results,
            timestamp=datetime.now()
        )

    def record_query(
        self,
        query: str,
        mode: str,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Record query metrics."""

        self.metrics.record({
            'query_count': 1,
            'query_latency': latency_ms,
            'query_mode': mode,
            'query_success': 1 if success else 0,
            'query_error': error or '',
        })

        # Check for alerts
        if latency_ms > 10000:  # 10 second threshold
            self._alert('HIGH_LATENCY', f"Query took {latency_ms}ms: {query[:50]}...")

        if not success:
            self._alert('QUERY_ERROR', f"Query failed: {error}")
```

### 6.3 Incremental Indexing

**File**: `mirage/src/core/indexing/incremental.py`

```python
"""
Incremental Indexing

Add new documents without full reindex:
1. Process new document
2. Extract entities and relationships
3. Update graph (merge with existing)
4. Update vector index
5. Update affected communities
"""

class IncrementalIndexer:
    def __init__(
        self,
        entity_extractor,
        graph_client,
        vector_client,
        community_detector
    ):
        self.entity_extractor = entity_extractor
        self.graph = graph_client
        self.vector = vector_client
        self.community = community_detector

    async def add_document(self, document: Document) -> IndexingResult:
        """Add single document incrementally."""

        # Step 1: Chunk document
        chunks = self.chunker.chunk(document.text)

        # Step 2: Extract entities from each chunk
        all_entities = []
        all_relationships = []

        for chunk in chunks:
            extraction = await self.entity_extractor.extract(chunk.text)
            all_entities.extend(extraction.entities)
            all_relationships.extend(extraction.relationships)

        # Step 3: Add to vector index
        await self.vector.add_chunks(chunks)

        # Step 4: Update graph (merge entities)
        await self._update_graph(all_entities, all_relationships, document.id)

        # Step 5: Update affected communities
        affected_communities = await self._update_communities(all_entities)

        return IndexingResult(
            document_id=document.id,
            chunks_added=len(chunks),
            entities_added=len(all_entities),
            relationships_added=len(all_relationships),
            communities_updated=len(affected_communities)
        )

    async def _update_graph(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
        document_id: str
    ):
        """Update graph with new entities, merging with existing."""

        for entity in entities:
            # Check if entity exists
            existing = self.graph.find_entity_by_name(entity.text)

            if existing:
                # Merge: add document reference
                self.graph.add_entity_document(existing['id'], document_id)
            else:
                # Create new entity
                self.graph.create_entity(entity, document_id)

        for relationship in relationships:
            # Check if relationship exists
            existing = self.graph.find_relationship(
                relationship.source,
                relationship.target,
                relationship.type
            )

            if not existing:
                self.graph.create_relationship(relationship)

    async def _update_communities(
        self,
        new_entities: List[Entity]
    ) -> List[str]:
        """Update communities affected by new entities."""

        # Find communities containing related entities
        affected = set()

        for entity in new_entities:
            # Find existing similar entities
            similar = self.graph.find_similar_entities(entity.text)

            for sim_entity in similar:
                community = self.graph.get_entity_community(sim_entity['id'])
                if community:
                    affected.add(community['id'])

        # Regenerate summaries for affected communities
        for community_id in affected:
            await self.community.regenerate_summary(community_id)

        return list(affected)
```

---

## Implementation Roadmap

### Week 1: Global Search (Phase 1)
- [ ] Implement `GlobalSearchEngine` with map-reduce
- [ ] Add global search to retrieval engine
- [ ] Update query router for global queries
- [ ] Test with L5 queries
- **Deliverable**: Can answer "What are the main themes?"

### Week 2: Entity Extraction (Phase 2)
- [ ] Implement enhanced ensemble extractor
- [ ] Add Arabic rule-based patterns
- [ ] Implement typed relationship extraction
- [ ] Add entity resolution
- **Deliverable**: 90%+ entity extraction F1

### Week 3: Metrics & Validation (Phases 3-4)
- [ ] Implement metrics tracker
- [ ] Create benchmark dataset
- [ ] Implement retrieval validator (CRAG-style)
- [ ] Add answer validation
- **Deliverable**: Measurable system with quality gates

### Week 4: Intelligence (Phase 5)
- [ ] Query expansion and decomposition
- [ ] Result diversification (MMR)
- [ ] Multi-hop reasoning
- **Deliverable**: Smarter query handling

### Week 5: Production (Phase 6)
- [ ] Redis caching layer
- [ ] Health monitoring
- [ ] Incremental indexing
- **Deliverable**: Production-ready system

### Week 6: Evaluation & Polish
- [ ] Full benchmark evaluation
- [ ] Performance optimization
- [ ] Documentation
- [ ] Final testing
- **Deliverable**: MIRAGE V3 release

---

## Success Metrics

| Metric | Current (V2) | Target (V3) | Measurement |
|--------|--------------|-------------|-------------|
| Answer Relevancy | ~0.70 | 0.90+ | RAGAS evaluation |
| Faithfulness | ~0.75 | 0.92+ | RAGAS evaluation |
| MRR | Unknown | 0.75+ | Benchmark |
| NDCG@10 | Unknown | 0.80+ | Benchmark |
| Global Query Support | 0% | 95%+ | L5 query success |
| Entity Extraction F1 | ~70% | 90%+ | Labeled dataset |
| Query Latency (p50) | 5-15s | 3-8s | Monitoring |
| Query Latency (p99) | 30-60s | 15-25s | Monitoring |

---

## File Structure After Implementation

```
mirage/src/core/
├── retrieval/
│   ├── retrieval_engine.py      # Updated with global search
│   ├── global_search.py         # NEW: Map-reduce engine
│   ├── query_processor.py       # NEW: Query expansion/decomposition
│   ├── validator.py             # NEW: CRAG-style validation
│   ├── diversifier.py           # NEW: MMR diversification
│   ├── multihop.py              # NEW: Multi-hop reasoning
│   ├── reranker.py              # Existing
│   └── fusion.py                # Existing
│
├── graph_builder/
│   ├── enhanced_entity_extractor.py  # NEW: Ensemble extraction
│   ├── relationship_extractor.py     # NEW: Typed relationships
│   ├── arabic_patterns.py            # NEW: Rule-based Arabic NER
│   ├── entity_resolver.py            # NEW: Deduplication
│   ├── community_summarizer.py       # Existing (improved)
│   └── neo4j_client.py               # Existing
│
├── evaluation/
│   ├── metrics.py               # NEW: MRR, NDCG, etc.
│   ├── benchmark.py             # NEW: Benchmark dataset
│   └── ragas_evaluator.py       # NEW: RAGAS integration
│
├── generation/
│   ├── answer_generator.py      # Existing
│   └── answer_validator.py      # NEW: Hallucination detection
│
├── cache/
│   └── redis_cache.py           # NEW: Caching layer
│
├── monitoring/
│   └── health.py                # NEW: Health checks
│
└── indexing/
    └── incremental.py           # NEW: Incremental indexing
```

---

## Conclusion

This plan transforms MIRAGE from a 5.25/10 partial implementation to a 9+/10 full GraphRAG system. The key changes are:

1. **Global Search**: The critical missing piece that enables holistic queries
2. **Better Extraction**: Ensemble approach for 90%+ entity quality
3. **Quality Control**: Metrics, validation, and self-correction
4. **Intelligence**: Query expansion, diversification, multi-hop
5. **Production**: Caching, monitoring, incremental updates

Each phase is independently deployable and testable. The system will surpass Microsoft GraphRAG in Arabic support while matching it in core capabilities.
