# MIRAGE Codebase Cleanup Report

**Audit Date:** 2024-12-24
**Cleanup Status:** PHASE 1 COMPLETE
**Total Files Audited:** 375
**Files Removed/Moved:** ~70+

---

## CLEANUP COMPLETED

### Deleted:
- `/docs/archives/` - 30+ archived docs
- `/mirage/config/` - duplicate config
- `/mirage/model_comparison/` - test data
- `/evaluation_results/` - old results
- 4 root-level test files
- 11 scattered test files in mirage/
- 8 duplicate tool scripts

### Moved:
- 12 utility scripts → `/tools/scripts/`
- 11 unique tools → `/tools/`
- Data files → `/data/`

### Requires Manual Cleanup (sudo):
```bash
sudo rm -rf mirage/benchmark_results/
sudo rm -rf mirage/data/sdaia_policies/
sudo rm -rf mirage/docs_to_ingest/
sudo rm -rf mirage/tools/
sudo rm -rf mirage/.pytest_cache/
sudo rm -rf mirage/__pycache__/
```

---

## ORIGINAL AUDIT REPORT

**Severity Levels:** CRITICAL | HIGH | MEDIUM | LOW

---

## EXECUTIVE SUMMARY

The codebase has significant technical debt requiring immediate attention:
- **5 duplicate directories** with identical files
- **22 scattered test files** across multiple locations
- **25+ unused/orphan files** in root directories
- **8+ duplicate tool scripts** between /tools and /mirage/tools
- **30+ archived docs** that could be removed
- **Several 1000+ line monster files** needing refactoring

---

## 1. DUPLICATE DIRECTORIES (CRITICAL)

### 1.1 benchmark_results (3 copies!)
```
/benchmark_results/
/mirage/benchmark_results/
/mirage/tools/benchmark_results/
```
**ACTION:** Keep only `/benchmark_results/`, delete others

### 1.2 sdaia_policies (2 copies)
```
/data/sdaia_policies/
/mirage/data/sdaia_policies/
```
**ACTION:** Keep only `/data/sdaia_policies/`, delete duplicate

### 1.3 tools directories (2 copies with 8 duplicate files!)
```
/tools/
/mirage/tools/
```
**Duplicate files:**
- comprehensive_evaluation.py
- entity_extraction_benchmark.py
- evaluation_test_cases.py
- hybrid_pipeline_benchmark.py
- process_bilingual_pdfs.py
- process_policies.py
- ragas_evaluation.py
- reembed_chunks.py

**ACTION:** Consolidate into `/tools/`, delete `/mirage/tools/`

---

## 2. SCATTERED TEST FILES (HIGH)

### Test files in wrong locations:
```
/test_ollama_integration.py          <- ROOT (wrong!)
/test_models.py                      <- ROOT (wrong!)
/test_model_qwen.py                  <- ROOT (wrong!) + DUPLICATE
/run_model_test.py                   <- ROOT (wrong!)
/mirage/test_*.py (10 files)         <- mirage root (wrong!)
/mirage/src/test_model_comparison.py <- src (wrong!)
```

### Proper test location:
```
/mirage/tests/  <- Only 7 tests here
```

**ACTION:** Move all test_*.py files to `/mirage/tests/`

---

## 3. ROOT-LEVEL ORPHAN FILES (HIGH)

### Files at /mirage/ root that should be organized:
```
audit_db.py
delete_broken_doc.py
ingest_all.py
inspect_ar_chunks.py
inspect_broken_doc.py
inspect_data.py
inspect_docs.py
inspect_qdrant.py
model_comparison_test.py
process_with_quality.py
repair_db.py
reproduce_encoding.py
reset_system.py
test_fix_encoding.py
```

**ACTION:** Move to `/tools/scripts/` or `/tools/maintenance/`

---

## 4. OUTDATED DOCUMENTATION (MEDIUM)

### Docs to DELETE (/docs/archives/):
```
docs/archives/API_UPDATE_SUMMARY.md
docs/archives/QUICK_FIX_SUMMARY.md
docs/archives/URL_YOUTUBE_INTEGRATION_COMPLETE.md
docs/archives/INTEGRATION_COMPLETE.md
docs/archives/SETUP_COMPLETE.md
docs/archives/CHAT_INTEGRATION_COMPLETE.md
docs/archives/CONTENT_TYPE_CHUNKING_PLAN.md
docs/archives/LLM_EXTRACTION_COMPLETE.md
docs/archives/planning/PLANNNNN.md              <- Typo in name!
docs/archives/planning/MIRAGE_V2_COMPLETE_PLAN.md
docs/archives/planning/IMPLEMENTATION_STATUS.md
docs/archives/evaluation/PHASE1_SUMMARY.md
docs/archives/evaluation/TESTING_CHECKLIST.md
docs/archives/old_plans/MIRAGE_V4_IMPROVEMENT_PLAN.md
docs/archives/old_plans/GRAPHRAG_V3_PLAN.md
docs/archives/old_plans/CRITICAL_EVALUATION.md
docs/archives/old_plans/CRITICAL_EVALUATION_V2.md
docs/archives/old_plans/EVALUATION_REPORT.md
docs/archives/old_plans/MIRAGE_10_OUT_OF_10_ROADMAP.md
```

**ACTION:** Delete entire `/docs/archives/` directory (30+ files)

---

## 5. MONSTER FILES NEEDING REFACTORING (MEDIUM)

### Files over 1000 lines:
| File | Lines | Issue |
|------|-------|-------|
| neo4j_client.py | 1480 | Too many responsibilities |
| retrieval_engine.py | 1393 | God class - split into modules |
| benchmark_service.py | 1305 | Evaluation logic mixed with API |
| chat_service.py | 1279 | Business logic in API layer |
| llm_entity_extractor.py | 1145 | Multiple extraction strategies in one file |
| community_detector.py | 1040 | Algorithm should be separate |
| url_service.py | 1019 | Processing mixed with routing |

**ACTION:** Refactor each file into smaller, focused modules

---

## 6. POTENTIALLY UNUSED MODULES (MEDIUM)

### Retrieval modules to verify usage:
```
/mirage/src/core/retrieval/
├── hippocampal_retrieval.py  <- Is this used?
├── dual_level_retrieval.py   <- Is this used?
├── v5_engine.py              <- Duplicate of retrieval_engine?
├── hyde.py                   <- HyDE implementation used?
├── observability.py          <- Monitoring used?
├── drift_search.py           <- DRIFT algorithm used?
```

### Graph builder modules to verify:
```
/mirage/src/core/graph_builder/
├── coreference_resolver.py   <- Is this used?
├── entity_disambiguator.py   <- Is this used?
├── incremental_updater.py    <- Is this used?
├── relationship_enricher.py  <- Is this used?
├── relationship_normalizer.py <- Is this used?
├── community_visualizer.py   <- Is this used?
```

**ACTION:** Trace imports and remove unused modules

---

## 7. CONFIG DUPLICATION (LOW)

### Duplicate config locations:
```
/mirage/config/
/mirage/src/config/
```

**ACTION:** Consolidate to `/mirage/src/config/`

---

## 8. EMPTY/MINIMAL FILES (LOW)

### Files to check for removal:
```
/mirage/src/core/evaluation/  <- What's here?
/mirage/src/models/           <- What's here?
/mirage/src/utils/            <- What's here?
```

---

## RECOMMENDED CLEANUP ORDER

### Phase 1: Quick Wins (1 hour)
1. Delete duplicate directories
2. Delete /docs/archives/ entirely
3. Delete duplicate test files

### Phase 2: Reorganization (2 hours)
1. Consolidate /tools and /mirage/tools
2. Move test files to /mirage/tests/
3. Move utility scripts to /tools/scripts/

### Phase 3: Dead Code Removal (2 hours)
1. Trace imports for unused modules
2. Remove unused retrieval engines
3. Remove unused graph builders

### Phase 4: Refactoring (4+ hours)
1. Split neo4j_client.py
2. Split retrieval_engine.py
3. Split chat_service.py
4. Refactor other monster files

---

## PROPOSED NEW STRUCTURE

```
MIRAGE/
├── docs/                    # Active documentation only
│   ├── README.md
│   ├── SETUP.md
│   └── API.md
├── data/                    # Data files
│   └── sdaia_policies/
├── tools/                   # All tools consolidated
│   ├── scripts/             # Utility scripts
│   ├── evaluation/          # Evaluation tools
│   └── benchmark/           # Benchmark tools
├── benchmark_results/       # Single location for results
├── mirage/
│   ├── main.py
│   ├── src/
│   │   ├── api/            # Clean API layer
│   │   ├── core/           # Business logic
│   │   ├── config/         # All config here
│   │   └── evaluation/     # Evaluation code
│   └── tests/              # All tests here
└── ui/
```

---

## FILES TO DELETE IMMEDIATELY

```bash
# Duplicate benchmark directories
rm -rf mirage/benchmark_results/
rm -rf mirage/tools/benchmark_results/

# Duplicate data
rm -rf mirage/data/sdaia_policies/

# Archived docs (entire directory)
rm -rf docs/archives/

# Root test files (after moving to tests/)
rm test_ollama_integration.py
rm test_models.py
rm test_model_qwen.py
rm run_model_test.py

# Duplicate in mirage/
rm mirage/test_model_qwen.py  # duplicate of root
```

---

**Total Estimated Cleanup Time:** 8-10 hours
**Files to Delete:** ~50+
**Lines of Code to Remove:** ~5000+
