# MIRAGE V2 RAG Evaluation Plan

## Evaluation Criteria

### 1. Entity Extraction Quality
| Criteria | Metric | Target |
|----------|--------|--------|
| **Accuracy** | Correct entity type assignment | >80% |
| **Completeness** | Important entities captured | >75% |
| **Meaningfulness** | Entities are useful (not noise) | >70% |
| **Arabic Support** | Arabic entities properly extracted | >80% |

### 2. Relationship Extraction Quality
| Criteria | Metric | Target |
|----------|--------|--------|
| **Semantic Validity** | Relationships make logical sense | >75% |
| **Type Accuracy** | Correct relationship types | >70% |
| **Connectivity** | Entities properly connected | Graph density >0.1 |

### 3. Retrieval Quality (by complexity level)
| Level | Query Type | Success Criteria |
|-------|------------|------------------|
| L1 | Direct Factual | Top-3 contains answer |
| L2 | Entity Lookup | Correct entities retrieved |
| L3 | Relationship | Connected context found |
| L4 | Multi-hop | Chain of reasoning possible |
| L5 | Synthesis | Multiple relevant topics combined |

### 4. Answer Quality
| Criteria | Metric | Target |
|----------|--------|--------|
| **Relevance** | Answer addresses query | >85% |
| **Grounding** | Answer based on sources | >90% |
| **Citations** | Sources properly cited | Present |
| **Language** | Correct language response | Match query |

## Test Cases

### Level 1: Direct Factual
- Query: "ما هي جائزة الحكومة الرقمية؟"
- Expected: Definition from source material
- Mode: Naive

### Level 2: Entity Lookup
- Query: "من هم المرشحون لجائزة أفضل شريك رقمي؟"
- Expected: List of nominee entities
- Mode: Local (entity-focused)

### Level 3: Relationship Query
- Query: "ما العلاقة بين الذكاء الاصطناعي والخدمات الحكومية؟"
- Expected: Connections between AI and services
- Mode: Global (relationship-focused)

### Level 4: Multi-hop Reasoning
- Query: "كيف ساهمت الشراكة بين القطاعين في تحسين الخدمات وما النتائج؟"
- Expected: Partnership → Improvements → Results chain
- Mode: Hybrid

### Level 5: Synthesis
- Query: "ما هي الإنجازات الرئيسية للتحول الرقمي السعودي؟"
- Expected: Aggregated achievements across domains
- Mode: Mix

## Execution Order
1. Analyze extracted entities for quality
2. Analyze relationships for meaningfulness
3. Run L1-L5 test queries
4. Evaluate answer quality
5. Generate summary report
