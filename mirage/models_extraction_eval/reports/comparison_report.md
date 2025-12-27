# Model Extraction Evaluation Report

**Date:** 2025-12-26 (Updated: 2025-12-27)
**Documents:** PoliciesAr.pdf (125 pages, Arabic), PoliciesEn.pdf (57 pages, English)
**Total Tokens:** ~336K (313,755 Arabic + 22,168 English)

## Summary Table

| Model | Arabic Entities | Arabic Rels | Arabic Time | Ar Speed | English Entities | English Rels | English Time | En Speed |
|-------|----------------|-------------|-------------|----------|-----------------|--------------|--------------|----------|
| gemma2-9b | **4,869** | **3,794** | 58.5 min | 89.4 tok/s | **788** | 623 | 9.0 min | 40.8 tok/s |
| qwen2.5-7b-instruct | 4,624 | 3,853 | 205.1 min | 25.5 tok/s | 649 | **628** | 5.8 min | 64.2 tok/s |
| gemma3:4b | 4,470 | 3,371 | 31.8 min | 164.6 tok/s | 769 | **662** | 5.2 min | 70.7 tok/s |
| qwen2.5-coder-7b | 4,101 | 3,368 | 34.0 min | **153.7 tok/s** | 577 | 528 | 4.0 min | **91.4 tok/s** |
| mistral-7b | 3,969 | 3,193 | 46.7 min | 111.9 tok/s | 710 | 638 | 5.2 min | 70.4 tok/s |
| llama3-8b | 939 | 816 | 31.7 min | **165.2 tok/s** | 183 | 166 | 3.5 min | 106.0 tok/s |
| qwen3:4b (thinking) | 42 | 27 | 527.5 min | 9.2 tok/s | 13 | 8 | 26.6 min | 13.9 tok/s |
| allam-7b (iKhalid) | 3,244 | 2,445 | 24.1 min | **216.7 tok/s** | 396 | 383 | 3.6 min | 103.7 tok/s |

## Key Findings

### Entity Extraction Volume
1. **gemma2-9b** extracted the most entities (4,869 Arabic + 788 English = **5,657 total**)
2. **qwen2.5-7b-instruct** was second (4,624 + 649 = **5,273 total**)
3. **gemma3:4b** third (4,470 + 769 = **5,239 total**)
4. **llama3-8b** extracted significantly fewer entities (~1,122 total) - **5x less than top models**
5. **qwen3:4b (thinking)** extracted only 55 entities total - **99% content loss** compared to top models

### Relationship Extraction
1. **qwen2.5-7b-instruct** extracted most relationships (4,481 total)
2. **gemma2-9b** second (4,417 total)
3. **gemma3:4b** third (4,033 total)

### Speed Performance (tok/sec)
1. **llama3-8b**: 165.2 tok/s (Arabic) - fastest but lowest quality
2. **gemma3:4b**: 164.6 tok/s (Arabic) - fast with good quality
3. **qwen2.5-coder-7b**: 153.7 tok/s (Arabic) - **best speed/quality balance**
4. **qwen2.5-7b-instruct**: 25.5 tok/s (Arabic) - **6x slower** than others

### Total Processing Time
| Model | Total Time | Notes |
|-------|-----------|-------|
| llama3-8b | 35 min | Fast but low quality |
| gemma3:4b | 37 min | Good balance |
| qwen2.5-coder-7b | 38 min | Best speed/quality |
| mistral-7b | 52 min | Moderate |
| gemma2-9b | 68 min | Highest quality but slower |
| qwen2.5-7b-instruct | 211 min | Very slow (3.5 hours) |
| qwen3:4b (thinking) | 554 min | **CRITICAL** - 9+ hours, unusable |
| allam-7b (iKhalid) | 28 min | **Fastest** - but has fabricated entities |

## Vision Mode Results

Both gemma3:4b and gemma2:9b were tested in vision mode (processing PDF page images instead of extracted text).

**Result: FAILED** - 0 entities and 0 relationships extracted in both cases. This indicates:
- Ollama's vision API may not work correctly for document extraction tasks
- These models may not properly support image-to-JSON structured output
- Text extraction remains the reliable approach for document processing

## Recommendations

### Best Overall: **gemma3:4b (text mode)**
- High entity count (5,239 total)
- Fast processing (37 min total)
- Good relationship extraction
- Small model size (4B parameters)

### Best Quality: **gemma2-9b**
- Highest entity extraction (5,657 total)
- High relationship count
- Slower processing (68 min)
- Larger model (9B parameters)

### Best Speed/Quality: **qwen2.5-coder-7b**
- Good entity count (4,678 total)
- Very fast (38 min total, 153.7 tok/s)
- JSON-focused training helps with structured output

### Avoid: **qwen2.5-7b-instruct**
- Similar quality to qwen2.5-coder-7b
- **6x slower** processing time
- No advantage over coder variant for this task

### Avoid: **llama3-8b**
- Significantly lower entity extraction (5x fewer entities)
- Fast but quality tradeoff is too severe
- Not recommended for knowledge graph extraction

## Arabic vs English Performance

All models extracted significantly more from the Arabic document despite it having only ~2x more pages:
- Arabic: ~14x more tokens processed
- Arabic entities: 4-7x more than English per model
- This suggests the Arabic policy document is denser with extractable information

---

# Part 2: Manual Quality Evaluation

**Evaluation Date:** 2025-12-26
**Method:** Deep manual inspection of entity and relationship samples from multiple sections of each document

## Quality Score Summary

| Model | Entity Quality | Relationship Quality | Overall Quality | Critical Issues |
|-------|---------------|---------------------|-----------------|-----------------|
| **qwen2.5-coder-7b** | 72/100 | 65/100 | **68/100** | Fabricated entity ("AML Basics") |
| **gemma2-9b** | 70/100 | 68/100 | **69/100** | Date semantics, TOC noise |
| **gemma3:4b** | 68/100 | 62/100 | **65/100** | Minor Arabic leak, weight=0 issues |
| **qwen2.5-7b-instruct** | 67/100 | 55/100 | **61/100** | 52% RELATED_TO overuse, contradictions |
| **mistral-7b** | 55/100 | 45/100 | **50/100** | Wrong types (Event for Principles), reversed hierarchies |
| **llama3-8b** | 25/100 | 20/100 | **22/100** | **UNUSABLE** - cross-language, dual types, 5x lower count |
| **qwen3:4b (thinking)** | 10/100 | 8/100 | **9/100** | **CRITICAL FAILURE** - 99% content loss, 9+ hours, timeouts |
| **allam-7b (iKhalid)** | 55/100 | 55/100 | **55/100** | Fabricated entities ("John Doe"), fastest speed (216 tok/s) |

---

## Detailed Quality Analysis by Model

### 1. qwen2.5-coder-7b (Best Quality)

**Entity Quality: 72/100**

✅ Strengths:
- Full page coverage (57/57 pages)
- No Arabic contamination in English doc
- No dual-type entities
- Good role extraction ("Business Data Executive", "Data Classification Reviewer")
- Clean type distribution (Concept: 313, Document: 91, Organization: 74)

❌ Issues:
- **Fabricated entity**: "AML Basics" extracted as entity - not in document content
- 12 generic/structural entities ("Definitions", "Scope", "Access")
- Type inconsistency: Same entity typed as "Person" and "Role" on different pages
- Extra non-standard types (Attribute, Process, Data, Section, etc.)

**Relationship Quality: 65/100**

✅ Strengths:
- Good CONTAINS structure (182 relationships, 34%)
- Custom useful types (SUBMITS, REGULATES, SUPPORTS)
- Correct: "Data Requestor" SUBMITS "Data Sharing Request"
- Correct: "Regulatory Authority" REGULATES "Public Entity"

❌ Issues:
- **Fabricated relationship**: "AML Basics" CONTAINS "National Data Governance..."
- Contradictory: Both "NDMO CREATED_BY Regulations" AND "Regulations CREATED_BY NDMO"
- "Public Entity" CREATED_BY "Data Sharing Agreement" - semantically wrong
- 37 unique relationship types - too fragmented

**Sample False Positives:**
```
Entity: "AML Basics" [Concept] - Page 1 (not in document)
Relation: "AML Basics" CONTAINS "National Data Governance Interim Regulations"
```

---

### 2. gemma2-9b (Highest Volume)

**Entity Quality: 70/100**

✅ Strengths:
- Most entities extracted (788 English, 4,869 Arabic)
- Good hierarchical structure captured
- Correct: "Royal Decree No. 59766" as Document
- Correct: "National Cybersecurity Authority" as Organization
- Clean single-type assignments

❌ Issues:
- "Table of Content" extracted as entity (structural noise)
- "Introduction", "Objectives" typed as Event (wrong type)
- "Version 1" as entity (metadata, not knowledge)
- Repeated header "National Data Governance Interim Regulations" on every page

**Relationship Quality: 68/100**

✅ Strengths:
- Good PART_OF, CONTAINS hierarchy
- IS_A relationships for acronyms: "NDMO" IS_A "National Data Management Office"
- Varied weights (0.6-0.9 range)

❌ Issues:
- "Document" CREATED_BY "June 1st, 2020" - semantically inverted (date doesn't create document)
- "GIS data" USES "Health sector" - wrong direction (sector uses data)
- 45% RELATED_TO (285/623) - too generic

**Sample False Positives:**
```
Entity: "Table of Content" [Concept] - structural element
Entity: "Introduction" [Event] - wrong type, should be Concept or excluded
Relation: "National Data Governance..." CREATED_BY "June 1st, 2020" - inverted
```

---

### 3. gemma3:4b (gemma-3n-e4b-text) - Best Balance

**Entity Quality: 68/100**

✅ Strengths:
- Full page coverage (57/57)
- Only 1 Arabic leak in English doc ("البيانات الوطنية")
- No dual-type entities
- Good: "Royal Decree No. 59766", "National Cybersecurity Authority"

❌ Issues:
- 1 Arabic text leak from header/logo
- "organization" extracted as standalone entity (too generic)
- 13 structural/TOC entities
- Duplicates: "Data Subject" appears twice on same page

**Relationship Quality: 62/100**

✅ Strengths:
- Good PART_OF usage (152 relationships, 23%)
- Reasonable type distribution

❌ Issues:
- **90 relationships with weight 0** - meaningless weights
- Duplicate type: "RELATES_TO" (23) vs "RELATED_TO" (218) - inconsistent
- Typos: "RESPONSES_TO" (should be RESPONDS_TO), "SUPERSEES" (should be SUPERSEDES)
- 37 unique relationship types - too many

---

### 4. qwen2.5-7b-instruct (Slow, Mediocre)

**Entity Quality: 67/100**

✅ Strengths:
- Full page coverage (57/57)
- No dual-type entities
- Good role extraction
- Real products: "Government Service Bus", "National Information Center Network"

❌ Issues:
- 3 Arabic entities in English doc
- Unusual types: "Question" (3), "Number" (1)
- "NDMO" and "NDMO (National Data Management Office)" as separate entities

**Relationship Quality: 55/100**

✅ Strengths:
- No date CREATED_BY issues
- Good weight distribution (0.3-1.0)

❌ Issues:
- **52% RELATED_TO (327/628)** - lazy extraction, lacks specificity
- Contradictory: Both "NDMO CREATED_BY Regulations" AND "Regulations CREATED_BY NDMO"
- "Business Data Steward" PART_OF "Data Sharing Agreement" - person isn't part of agreement
- Typo: "DEPENDES_ON"
- Weak targets: "appropriate balance", "potential risks" as entities

---

### 5. mistral-7b (Significant Issues)

**Entity Quality: 55/100**

✅ Strengths:
- No Arabic contamination
- No dual-type entities
- Reasonable count (710 entities)

❌ Issues:
- **Only 52/57 pages covered** - 5 pages skipped
- **20+ Principles typed as "Event"** instead of Concept
- "National Data Governance Interim Regulations" typed as Event (should be Document!)
- **Relationship types used as entity types**: 8 entities typed as "RELATED_TO", 4 as "AFFECTS"
- Type case inconsistency: "CONCEPT" vs "Concept"
- Non-standard types: "Person/Entity", "Classification"

**Relationship Quality: 45/100**

✅ Strengths:
- Custom types: DEFINES, PRINCIPLE_OF, RESPONSIBLE_FOR

❌ Issues:
- **98 relationships with weight 0** - meaningless
- **359 relationships with weight 1.0 (56%)** - no differentiation
- **Reversed hierarchies**:
  - "Data Classification Interim Regulations" CONTAINS "National Data Governance Interim Regulations" (WRONG)
  - "Kingdom" LOCATED_IN "National Open data Portal" (REVERSED)
- "Data Requestor" BELONGS_TO "Data Sharing Request" - semantically wrong
- "NDMO" PART_OF "develop" - "develop" is a verb, not an entity!

**Sample False Positives:**
```
Entity: "Principle 1: Data Sharing Culture" [Event] - should be Concept
Entity: "National Data Governance Interim Regulations" [Event] - should be Document
Relation: "Kingdom" LOCATED_IN "National Open data Portal" - reversed direction
Relation: "Data Classification..." CONTAINS "National Data Governance..." - reversed hierarchy
```

---

### 6. llama3-8b (UNUSABLE - Critical Failures)

**Entity Quality: 25/100**

❌ Critical Issues:
- **Only 183 entities** (5x less than other models)
- **Missing 29 out of 57 pages** - massive content loss
- **Arabic text in English document**: "املبادئ", "ألاساسية", "للبيانات الوطنية"
- **16 dual-type entities (8.7%)**: "Organization, Concept", "Event, Document", etc.
- Bizarre types: "Organization (Concept)", "Person/Role"

**Relationship Quality: 20/100**

❌ Critical Issues:
- **Arabic relationships in English doc**: "املبادئ" -> "ألاساسية"
- **Semantically inverted**:
  - "Data Sharing Agreement" CAUSES "Data Sharing Request" (BACKWARD)
  - "للبيانات الوطنية" LOCATED_IN "National Data Governance..." (nonsense)
- **Fragmented entities**: "National Data Governance" -> "Interim Regulations" CREATED_BY (splits document title)
- **Non-entities used**: "Long-term effect on" as source entity
- **Arbitrary weight sequencing**: Page 10 weights are 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 in order

**Sample Critical Failures:**
```
Entity: "املبادئ" [Concept] - Arabic in English doc
Entity: "National Data Governance Interim Regulations" [Organization, Concept] - dual type
Relation: "Data Sharing Agreement" CAUSES "Data Sharing Request" - semantically backward
Relation: "Long-term effect on" -> "KSA economy" - "Long-term effect on" is not an entity
```

---

### 7. qwen3:4b "thinking" variant (CRITICAL FAILURE - Worst Performance)

**Evaluation Date:** 2025-12-27
**Processing Time:** 8.8 hours (Arabic) + 27 min (English) = **9+ hours total**
**Timeout Errors:** 13 pages failed (Arabic)

**Entity Quality: 10/100**

❌ Critical Issues:
- **Only 42 Arabic entities** (vs 4,000+ from other models) = **99% content loss**
- **Only 13 English entities** (vs 600+ from other models) = **98% content loss**
- **91% of English pages produced no output** (only pages 14, 26, 28, 34, 49 extracted)
- **Page numbers extracted as entities**: "4", "12", "22", "14", "68", "106"
- **"JSON" extracted as entity** - complete noise
- **Severe duplication**: "National Data Governance Interim Regulations" extracted 5 times in English
- **Inconsistent schema**: Some entities use "text" field, others use "name" field
- **Missing all Organizations**: No NDMO, Royal Decree, or any organization extracted in English
- **Missing all Persons/Roles**: No Business Data Steward, Data Requestor, etc.

**Relationship Quality: 8/100**

❌ Critical Issues:
- **Only 27 Arabic relationships** (vs 3,000+ from other models)
- **Only 8 English relationships** (vs 500+ from other models)
- **All English relationships have same source**: "National Data Governance Interim Regulations"
- **3 duplicate relationships**: BELONGS_TO "Interim Regulations" repeated 3 times
- **Page number relationships**: "المحتويات" -> "4" (CONTAINS)
- **No variety**: Missing CREATES, DEFINES, RESPONSIBLE_FOR, REGULATES

**Root Cause Analysis:**

The "thinking" mode in qwen3:4b causes the model to generate extended internal reasoning (Chain of Thought) before producing JSON output. This leads to:

1. **Timeout cascade**: Model spends 2+ minutes per page on reasoning, triggering 120-second timeout
2. **13 pages lost completely** to timeouts in Arabic (10% of document)
3. **Most pages produce no JSON**: Model's thinking output doesn't parse as valid entities/relationships
4. **Speed degradation**: 9.2 tok/s vs 165 tok/s for llama3-8b (18x slower)

**Sample Failures:**

Arabic entities (42 total - most are noise):
```
"4" [Concept] - page number
"12" [Document] - page number
"22" [Concept] - page number
"JSON" [Concept] - technical noise
"68" [Document] - page number
"106" [Concept] - page number
```

English entities (13 total - severe duplication):
```
"National Data Governance Interim Regulations" [Document] - page 14
"National Data Governance Interim Regulations" [Document] - page 26 (duplicate)
"National Data Governance Interim Regulations" [Document] - page 28 (duplicate)
"National Data Governance Interim Regulations" [Document] - page 34 (duplicate)
"National Data Governance Interim Regulations" [Document] - page 49 (duplicate)
"14" [Concept] - page number
"Interim Regulations" [Concept] - repeated 3 times
```

**Conclusion:**

qwen3:4b with "thinking" mode is **completely unsuitable** for knowledge graph extraction:
- 99% content loss makes output unusable
- 9+ hour processing time is impractical
- Even successfully extracted entities are mostly noise
- **Worse than llama3-8b** despite taking 15x longer

**Recommendation:** Use the non-thinking variant of qwen3 or any other model from this evaluation.

---

### 8. allam-7b (iKhalid Community Version) - Fastest but Fabricates

**Evaluation Date:** 2025-12-27
**Processing Time:** 24.1 min (Arabic) + 3.6 min (English) = **27.7 min total**
**Speed:** 216.7 tok/s Arabic, 103.7 tok/s English - **FASTEST MODEL TESTED**

**Entity Quality: 55/100**

✅ Strengths:
- **Fastest processing** of all models (216.7 tok/s)
- Full page coverage (125/125 Arabic, 57/57 English)
- Good entity volume: 3,640 total (3,244 Arabic + 396 English)
- Low cross-language contamination (only 1 Arabic leak in English)
- Proper Date type usage: "June 1st, 2020" as Date
- Good type variety: Concept (1,714), Organization (534), Document (376)

❌ Critical Issues:
- **FABRICATED ENTITIES**: "John Doe" and "Jane Smith" appear 4 times (pages 3, 7 of Arabic doc)
- Type inconsistency: "National Data Governance..." typed as Product (page 1) vs Document (page 2)
- Structural entities: "Introduction", "Definitions", "Objectives", "Scope"
- Non-standard types: "Level of Impact", "Impact Level", "Classification Level" (redundant)

**Relationship Quality: 55/100**

✅ Strengths:
- Good CONTAINS structure: 161 relationships (42%)
- Custom useful types: DATE_OF, CREATED_BY, USES
- Proper hierarchical structure
- Reasonable type distribution

❌ Issues:
- **Typo**: "CONTANS" instead of "CONTAINS"
- **6 "Unknown" type relationships**
- **No weight field** on any relationships
- Missing PART_OF relationships (only 1)

**Sample Fabricated Entities:**
```
"John Doe" [Person] - Page 3 (Arabic doc) - NOT IN DOCUMENT
"Jane Smith" [Person] - Page 3 (Arabic doc) - NOT IN DOCUMENT
"John Doe" [Person] - Page 7 (Arabic doc) - NOT IN DOCUMENT
"Jane Smith" [Person] - Page 7 (Arabic doc) - NOT IN DOCUMENT
```

**Conclusion:**

allam-7b shows promise as the **fastest model** tested, but the fabricated entities are a critical issue that requires post-processing. The model appears to use placeholder names when it cannot confidently extract real person names.

**Recommendation:**
- Use for speed-critical applications with strict post-processing to remove "John Doe" and "Jane Smith"
- Not recommended for production without entity validation pipeline
- Arabic understanding appears good given its Arabic-focused training

---

## Arabic Extraction Quality

| Model | Arabic Entities | English Leaks | Contamination Rate |
|-------|----------------|---------------|-------------------|
| gemma2-9b | 4,869 | 60 | 1.2% |
| gemma3:4b | 4,470 | 49 | 1.1% |
| qwen2.5-coder-7b | 4,101 | 55 | 1.3% |
| allam-7b (iKhalid) | 3,244 | 4 (fabricated) | 0.1% + fabricated |
| llama3-8b | 939 | 85 | **9.1%** |

**Key Findings:**
- All models except llama3-8b have <1.5% English contamination in Arabic docs
- llama3-8b has 9% contamination and 5x fewer entities - not usable for Arabic
- Mixed text issues found: "المزجMasking", "Data( مشاركة البيانات...)"

---

## Revised Recommendations

### Best for Production: **qwen2.5-coder-7b**
- Highest quality score (68/100)
- Fast processing (38 min)
- Clean output format
- ⚠️ Caveat: Remove fabricated "AML Basics" entity manually

### Best for High Volume: **gemma2-9b**
- Most entities extracted
- Good quality (69/100)
- Slower processing (68 min)
- ⚠️ Caveat: Filter out TOC/structural entities

### Best Balance: **gemma3:4b**
- Good quality (65/100)
- Fastest processing (37 min)
- ⚠️ Caveat: Fix weight=0 relationships

### Acceptable: **qwen2.5-7b-instruct**
- Mediocre quality (61/100)
- **6x slower** than coder variant
- No benefit over qwen2.5-coder-7b

### Avoid: **mistral-7b**
- Poor quality (50/100)
- Wrong entity types
- Reversed relationship hierarchies

### DO NOT USE: **llama3-8b**
- Critical failures (22/100)
- Cross-language contamination
- Missing most content
- Unusable for knowledge graph extraction

---

## Post-Processing Recommendations

For all models, apply these filters:

1. **Remove structural entities**: "Introduction", "Definitions", "Objectives", "Scope", "Table of Content", "Key Principles"
2. **Remove generic entities**: "Data", "Information", "System" (when standalone)
3. **Fix date relationships**: Change "X CREATED_BY Date" to "X PUBLISHED_ON Date"
4. **Deduplicate**: Remove repeated entities across pages (e.g., document headers)
5. **Validate relationship directions**: Check CONTAINS/PART_OF hierarchies
6. **Filter zero-weight relationships**: Remove relationships with weight=0

---

## Next Steps

1. ✅ Manual quality evaluation complete (8 models)
2. ✅ qwen3:4b tested - **CRITICAL FAILURE** (9/100 quality, 99% content loss)
3. ✅ allam-7b tested - **Fastest** (216 tok/s) but fabricates entities (55/100 quality)
4. ⏳ Post-processing pipeline implementation
5. ⏳ Cross-language entity linking analysis

## Final Model Ranking (by Quality)

| Rank | Model | Quality Score | Speed | Recommendation |
|------|-------|---------------|-------|----------------|
| 1 | gemma2-9b | 69/100 | 89 tok/s | Best for high-volume extraction |
| 2 | qwen2.5-coder-7b | 68/100 | 154 tok/s | **Best for production** (speed+quality) |
| 3 | gemma3:4b | 65/100 | 165 tok/s | Best balance (small, fast, good quality) |
| 4 | qwen2.5-7b-instruct | 61/100 | 26 tok/s | Acceptable but slow |
| 5 | allam-7b (iKhalid) | 55/100 | **217 tok/s** | Fastest, but fabricates entities |
| 6 | mistral-7b | 50/100 | 112 tok/s | Avoid - wrong types, reversed hierarchies |
| 7 | llama3-8b | 22/100 | 165 tok/s | DO NOT USE - critical failures |
| 8 | qwen3:4b (thinking) | 9/100 | 9 tok/s | DO NOT USE - 99% content loss |
