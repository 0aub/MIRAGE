# Manual Review Template

## Model: [MODEL_NAME]
## Reviewer: Claude
## Date: [DATE]

---

## Summary Scores

| Metric | Score (1-5) | Notes |
|--------|-------------|-------|
| Entity Precision | | |
| Entity Naming Quality | | |
| Entity Type Accuracy | | |
| Relationship Precision | | |
| Relationship Meaningfulness | | |
| Cross-Language Linking | | |
| **Overall Quality** | | |

**Score Guide:**
- 5: Excellent - Production ready
- 4: Good - Minor issues
- 3: Acceptable - Some corrections needed
- 2: Poor - Many errors
- 1: Unusable - Fundamental problems

---

## Entity Review (Sample of 20)

### English Entities (10 samples)

| # | Extracted Entity | Type | Correct? | Name Accurate? | Type Correct? | Notes |
|---|------------------|------|----------|----------------|---------------|-------|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |
| 8 | | | | | | |
| 9 | | | | | | |
| 10 | | | | | | |

**English Entity Summary:**
- Correct: /10
- Name Accurate: /10
- Type Correct: /10

### Arabic Entities (10 samples)

| # | Extracted Entity | Type | Correct? | Name Accurate? | Type Correct? | Notes |
|---|------------------|------|----------|----------------|---------------|-------|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |
| 8 | | | | | | |
| 9 | | | | | | |
| 10 | | | | | | |

**Arabic Entity Summary:**
- Correct: /10
- Name Accurate: /10
- Type Correct: /10

---

## Relationship Review (Sample of 20)

### English Relationships (10 samples)

| # | Source | Relationship | Target | Exists? | Type Correct? | Notes |
|---|--------|--------------|--------|---------|---------------|-------|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |
| 8 | | | | | | |
| 9 | | | | | | |
| 10 | | | | | | |

**English Relationship Summary:**
- Exists in document: /10
- Type Correct: /10

### Arabic Relationships (10 samples)

| # | Source | Relationship | Target | Exists? | Type Correct? | Notes |
|---|--------|--------------|--------|---------|---------------|-------|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |
| 8 | | | | | | |
| 9 | | | | | | |
| 10 | | | | | | |

**Arabic Relationship Summary:**
- Exists in document: /10
- Type Correct: /10

---

## Cross-Language Analysis

### Linked Entities Found
| English | Arabic | Link Correct? |
|---------|--------|---------------|
| | | |
| | | |
| | | |

### Missing Links (Should have been linked)
| English | Arabic | Notes |
|---------|--------|-------|
| | | |

---

## Common Issues Observed

### Entity Issues
1.
2.
3.

### Relationship Issues
1.
2.
3.

---

## Recommendations

1.
2.
3.

---

## Raw Statistics

```json
{
  "total_entities": 0,
  "english_entities": 0,
  "arabic_entities": 0,
  "total_relationships": 0,
  "english_relationships": 0,
  "arabic_relationships": 0,
  "entity_types": {},
  "relationship_types": {},
  "extraction_time_seconds": 0,
  "tokens_per_second": 0
}
```
