# MIRAGE Pipeline Redesign - Implementation Plan

## Executive Summary

This plan addresses **11 critical design flaws** in the current MIRAGE pipeline, focusing on:
1. Fixing prompts (compression → enrichment)
2. Resolving token limit issues
3. Adding quality validation
4. Optimizing performance
5. Fixing embeddings consistency
6. Centralizing configuration

**Priority:** Critical - Current pipeline produces 0 entities due to flawed design

---

## Phase 1: Foundation (CRITICAL - Do First)

### 1.1 Centralized Prompt System ✅ DONE
**Status:** Completed
**File:** `mirage/src/config/prompts.yaml`

**What was done:**
- Created YAML configuration with enrichment prompts (not compression!)
- Added content-type-specific prompts (YouTube, webpage, files)
- Defined model constraints for all supported providers
- Included Arabic and English versions for all prompts

### 1.2 Prompt Loader Utility
**Priority:** CRITICAL
**Estimated Time:** 1 hour
**Files to create:** `mirage/src/config/prompt_loader.py`

**Implementation:**
```python
import yaml
from pathlib import Path
from typing import Dict, Any

class PromptManager:
    """Centralized prompt management"""

    def __init__(self):
        config_path = Path(__file__).parent / "prompts.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def get_prompt(self, prompt_type: str, language: str,
                   content_type: str = None) -> Dict[str, str]:
        """Get system and user prompt templates

        Args:
            prompt_type: 'content_enrichment', 'youtube_cleanup', etc.
            language: 'english' or 'arabic'
            content_type: 'youtube', 'webpage', 'file' (optional)

        Returns:
            {'system': '...', 'user_template': '...'}
        """
        # Handle content-type-specific prompt selection
        if content_type:
            config = self.config.get('content_type_config', {}).get(content_type, {})
            if config.get('use_cleanup') and 'cleanup_prompt' in config:
                prompt_type = config['cleanup_prompt']

        prompts = self.config[prompt_type][language]
        return {
            'system': prompts['system'],
            'user_template': prompts['user_template']
        }

    def get_model_constraints(self, provider: str) -> Dict[str, Any]:
        """Get token limits and constraints for a model provider"""
        constraints_map = {
            'tgi': 'tgi_allam_7b',
            'openai': 'openai_gpt4',
            'anthropic': 'anthropic_claude',
            'google': 'google_gemini'
        }
        key = constraints_map.get(provider, 'tgi_allam_7b')
        return self.config['model_constraints'][key]
```

**Testing:**
```python
# Test loading prompts
pm = PromptManager()
enrichment = pm.get_prompt('content_enrichment', 'english')
youtube = pm.get_prompt('youtube_cleanup', 'english', content_type='youtube')
constraints = pm.get_model_constraints('tgi')
```

---

## Phase 2: Fix Content Rewriter (CRITICAL)

### 2.1 Update ContentRewriter to Use Centralized Prompts
**Priority:** CRITICAL
**Estimated Time:** 2 hours
**Files to modify:** `mirage/src/core/graph_builder/content_rewriter.py`

**Changes needed:**

1. **Add prompt manager:**
```python
from ...config.prompt_loader import PromptManager

class ContentRewriter:
    def __init__(self):
        # ... existing code ...
        self.prompt_manager = PromptManager()
```

2. **Replace hardcoded prompts:**
```python
def rewrite_chunk(self, chunk_text: str, language: str = "auto",
                  content_type: str = "file") -> str:
    # Detect language
    if language == "auto":
        arabic_ratio = len([c for c in chunk_text if '\u0600' <= c <= '\u06FF']) / max(len(chunk_text), 1)
        language = "arabic" if arabic_ratio > 0.3 else "english"

    # Get appropriate prompts based on content type
    prompts = self.prompt_manager.get_prompt(
        prompt_type='content_enrichment',  # Will be overridden for youtube
        language=language,
        content_type=content_type
    )

    system_prompt = prompts['system']
    user_prompt = prompts['user_template'].format(text=chunk_text)

    # Get model-specific constraints
    if self.provider:
        constraints = self.prompt_manager.get_model_constraints(self.provider)
        # Use constraints for token calculations
```

3. **Add content_type parameter throughout call chain:**
```python
def rewrite_chunks(self, chunks: List[Dict[str, Any]],
                   language: Optional[str] = None,
                   document_id: Optional[str] = None,
                   content_type: str = "file") -> List[Dict[str, Any]]:
    # Pass content_type to rewrite_chunk
    rewritten_text = self.rewrite_chunk(original_text, language or "auto", content_type)
```

### 2.2 Update URL Service to Pass Content Type
**Files to modify:** `mirage/src/api/url_service.py`

**Changes:**
```python
# Around line 585 where rewriter.rewrite_chunks is called
rewritten_chunks = rewriter.rewrite_chunks(
    chunks,
    language=language,
    document_id=document_id,
    content_type=content_type  # ADD THIS - 'youtube' or 'webpage'
)
```

---

## Phase 3: Fix Entity Extraction Token Limits (CRITICAL)

### 3.1 Problem Analysis
**Current Issue:** Line 667 in `llm_entity_extractor.py`:
```python
full_text = " ".join([chunk.get("text", "") for chunk in chunks])
```

This **concatenates all chunks** into one massive string, which:
- Exceeds ALLaM-7B's 2048 token limit on any document > 8000 chars
- Causes extraction to fail silently
- Is why you get 0 entities

### 3.2 Solution: Per-Chunk Extraction with Deduplication
**Priority:** CRITICAL
**Estimated Time:** 3 hours
**Files to modify:** `mirage/src/core/graph_builder/llm_entity_extractor.py`

**Implementation:**

```python
def extract_from_chunks(
    self,
    chunks: List[Dict[str, Any]],
    language: Optional[str] = None,
    document_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract entities and relationships from chunks with per-chunk processing
    and intelligent merging/deduplication
    """
    total_chunks = len(chunks)
    logger.info(f"Extracting entities from {total_chunks} chunks (per-chunk mode)")

    all_entities = []
    all_relationships = []
    entity_name_map = {}  # For deduplication: variant → canonical name

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        logger.info(f"Extracting from chunk {current_chunk}/{total_chunks}")

        # Update progress
        self._update_chunk_progress(document_id, current_chunk, total_chunks)

        # Extract from this chunk only
        chunk_text = chunk.get("text", "")

        # Check token limit for this chunk
        if self.provider == "tgi":
            token_count = len(chunk_text) // 4  # Rough estimate
            if token_count > 1500:  # Leave room for prompt and response
                logger.warning(f"Chunk {current_chunk} too large ({token_count} tokens), splitting further...")
                # Split into smaller pieces
                # (implementation similar to content_rewriter sub-chunking)
                continue

        result = self.extract_entities_and_relationships(
            chunk_text,
            language or "auto",
            document_id=None  # Don't update progress in inner call
        )

        # Merge entities with deduplication
        for entity in result.get("entities", []):
            canonical_name = self._get_canonical_entity_name(
                entity["name"],
                entity.get("type", ""),
                entity_name_map
            )
            entity["name"] = canonical_name

            # Check if entity already exists
            existing = next((e for e in all_entities if e["name"] == canonical_name), None)
            if existing:
                # Merge properties
                if "description" in entity and "description" not in existing:
                    existing["description"] = entity["description"]
            else:
                all_entities.append(entity)

        # Merge relationships with deduplication
        for rel in result.get("relationships", []):
            # Normalize entity names in relationship
            rel["source"] = self._get_canonical_entity_name(
                rel["source"], "", entity_name_map
            )
            rel["target"] = self._get_canonical_entity_name(
                rel["target"], "", entity_name_map
            )

            # Check for duplicate relationships
            rel_key = f"{rel['source']}|{rel['type']}|{rel['target']}"
            if not any(f"{r['source']}|{r['type']}|{r['target']}" == rel_key
                      for r in all_relationships):
                all_relationships.append(rel)

    logger.info(f"Extraction complete: {len(all_entities)} unique entities, "
                f"{len(all_relationships)} unique relationships")

    return {
        "entities": all_entities,
        "relationships": all_relationships
    }

def _get_canonical_entity_name(self, name: str, entity_type: str,
                                 name_map: Dict[str, str]) -> str:
    """
    Normalize entity names to handle variations
    E.g., "Steve Jobs", "Jobs", "S. Jobs" all map to "Steve Jobs"
    """
    name_lower = name.lower().strip()

    # Check if we've seen this variant before
    if name_lower in name_map:
        return name_map[name_lower]

    # Check if this is a variant of an existing entity
    for existing_variant, canonical in name_map.items():
        # Simple heuristic: if one is substring of other (for names)
        if entity_type == "PERSON":
            if name_lower in existing_variant or existing_variant in name_lower:
                # Use the longer name as canonical
                if len(name) > len(canonical):
                    name_map[existing_variant] = name
                    name_map[name_lower] = name
                    return name
                else:
                    name_map[name_lower] = canonical
                    return canonical

    # New entity - use this as canonical
    name_map[name_lower] = name
    return name
```

**Benefits:**
- ✅ No token limit issues - each chunk processed independently
- ✅ Better error handling - one chunk failing doesn't kill entire extraction
- ✅ Intelligent entity deduplication
- ✅ Progress tracking per chunk
- ✅ Scalable to documents of any size

---

## Phase 4: Add Quality Validation

### 4.1 Content Rewriting Quality Checks
**Files to modify:** `mirage/src/core/graph_builder/content_rewriter.py`

**Add validation after rewriting:**

```python
def rewrite_chunks(self, chunks: List[Dict[str, Any]], ...) -> List[Dict[str, Any]]:
    # ... existing code ...

    for i, chunk in enumerate(chunks):
        original_text = chunk.get("text", "")
        rewritten_text = self.rewrite_chunk(original_text, language or "auto", content_type)

        # QUALITY VALIDATION
        validation = self._validate_rewrite_quality(original_text, rewritten_text)

        if not validation["passed"]:
            logger.warning(f"Chunk {current_chunk} rewriting quality check failed: "
                          f"{validation['reason']}. Using original text.")
            rewritten_text = original_text

        # ... rest of code ...

def _validate_rewrite_quality(self, original: str, rewritten: str) -> Dict[str, Any]:
    """
    Validate that rewriting improved (not degraded) the text

    Returns:
        {'passed': bool, 'reason': str, 'metrics': dict}
    """
    # Check 1: Not identical
    if original == rewritten:
        return {
            'passed': False,
            'reason': 'Rewritten text identical to original',
            'metrics': {}
        }

    # Check 2: Not too short (shouldn't remove > 50% of content)
    len_ratio = len(rewritten) / max(len(original), 1)
    if len_ratio < 0.5:
        return {
            'passed': False,
            'reason': f'Rewritten text too short ({len_ratio:.1%} of original)',
            'metrics': {'length_ratio': len_ratio}
        }

    # Check 3: Quick entity count (should not decrease significantly)
    orig_entities = self._quick_entity_count(original)
    new_entities = self._quick_entity_count(rewritten)

    if new_entities < orig_entities * 0.6:
        return {
            'passed': False,
            'reason': f'Too many entities lost (orig: {orig_entities}, new: {new_entities})',
            'metrics': {'orig_entities': orig_entities, 'new_entities': new_entities}
        }

    return {
        'passed': True,
        'reason': 'Quality checks passed',
        'metrics': {
            'length_ratio': len_ratio,
            'orig_entities': orig_entities,
            'new_entities': new_entities
        }
    }

def _quick_entity_count(self, text: str) -> int:
    """
    Quick heuristic entity count without full extraction
    Counts capitalized words and quoted terms
    """
    import re
    # Find capitalized words (potential entity names)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    # Find quoted terms
    quoted = re.findall(r'["\']([^"\']+)["\']', text)
    return len(set(capitalized)) + len(set(quoted))
```

---

## Phase 5: Performance Optimization

### 5.1 Parallel Chunk Rewriting
**Files to modify:** `mirage/src/core/graph_builder/content_rewriter.py`

**Add concurrent processing:**

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def rewrite_chunks(self, chunks: List[Dict[str, Any]], ...) -> List[Dict[str, Any]]:
    """
    Rewrite chunks with parallel processing for speed
    """
    total_chunks = len(chunks)
    logger.info(f"Starting parallel rewriting of {total_chunks} chunks")

    rewritten_chunks = [None] * total_chunks  # Pre-allocate with order preservation

    # Determine parallelism based on provider
    max_workers = 3 if self.provider == "tgi" else 5  # TGI has single endpoint

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {}
        for i, chunk in enumerate(chunks):
            future = executor.submit(
                self._rewrite_chunk_with_index,
                i,
                chunk,
                language or "auto",
                content_type,
                document_id,
                total_chunks
            )
            future_to_index[future] = i

        # Collect results as they complete
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                result = future.result()
                rewritten_chunks[i] = result
            except Exception as e:
                logger.error(f"Error rewriting chunk {i+1}: {e}")
                # Use original on failure
                rewritten_chunks[i] = {
                    "text": chunks[i].get("text", ""),
                    "metadata": chunks[i].get("metadata", {})
                }

    return rewritten_chunks

def _rewrite_chunk_with_index(self, index: int, chunk: Dict, language: str,
                                content_type: str, document_id: str, total: int):
    """Helper for parallel processing"""
    current_chunk = index + 1
    logger.info(f"Rewriting chunk {current_chunk}/{total}")

    # Update progress
    self._update_progress(document_id, current_chunk, total, phase="rewriting")

    # Rewrite
    original_text = chunk.get("text", "")
    rewritten_text = self.rewrite_chunk(original_text, language, content_type)

    return {
        "text": rewritten_text,
        "metadata": chunk.get("metadata", {})
    }
```

**Performance Impact:**
- Current: 50 chunks × 3s = **150 seconds**
- Parallel (3 workers): 50 chunks ÷ 3 × 3s = **50 seconds**
- **3x speedup**

---

## Phase 6: Fix Embeddings Consistency

### 6.1 Problem Analysis
**Current Issue:** Unclear which text gets embedded:
- Original text stored in Neo4j as `full_text`
- Rewritten text stored as `processed_text`
- Which one goes to Qdrant for vector search?

### 6.2 Solution: Embed Processed Text
**Files to modify:** `mirage/src/core/graph_builder/neo4j_client.py` and related files

**Decision:** Embed the **processed (enriched) text** because:
- It has better entity resolution
- More explicit information
- Better for semantic search
- Consistent with what's in the graph

**Implementation:**

In `url_service.py` or wherever embeddings are created:
```python
# After rewriting
rewritten_chunks = rewriter.rewrite_chunks(...)

# Create embeddings from REWRITTEN chunks (not original)
embeddings_for_qdrant = create_embeddings([chunk["text"] for chunk in rewritten_chunks])

# Store in Qdrant with metadata linking to document_id
store_in_qdrant(embeddings_for_qdrant, document_id, processed_text=full_processed_text)
```

---

## Phase 7: Error Handling & Robustness

### 7.1 Add Checkpointing for Long Documents
**Files to create:** `mirage/src/core/graph_builder/checkpoint_manager.py`

```python
import json
from pathlib import Path
from typing import Dict, List, Any

class CheckpointManager:
    """Save progress during long-running operations"""

    def __init__(self, checkpoint_dir: str = "/tmp/mirage_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, document_id: str, phase: str, data: Dict[str, Any]):
        """Save checkpoint for resuming later"""
        checkpoint_file = self.checkpoint_dir / f"{document_id}_{phase}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f)

    def load_checkpoint(self, document_id: str, phase: str) -> Dict[str, Any]:
        """Load checkpoint if exists"""
        checkpoint_file = self.checkpoint_dir / f"{document_id}_{phase}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                return json.load(f)
        return None

    def clear_checkpoint(self, document_id: str, phase: str):
        """Remove checkpoint after successful completion"""
        checkpoint_file = self.checkpoint_dir / f"{document_id}_{phase}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
```

**Usage in content_rewriter.py:**
```python
def rewrite_chunks(self, chunks, ..., enable_checkpointing=True):
    checkpoint_mgr = CheckpointManager() if enable_checkpointing else None

    # Try to resume from checkpoint
    if checkpoint_mgr:
        checkpoint = checkpoint_mgr.load_checkpoint(document_id, "rewriting")
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint['completed_chunks']} chunks done")
            rewritten_chunks = checkpoint['rewritten_chunks']
            start_index = checkpoint['completed_chunks']
        else:
            rewritten_chunks = []
            start_index = 0

    # Process from start_index
    for i in range(start_index, len(chunks)):
        # ... rewrite chunk ...

        # Save checkpoint every 10 chunks
        if checkpoint_mgr and (i + 1) % 10 == 0:
            checkpoint_mgr.save_checkpoint(document_id, "rewriting", {
                'completed_chunks': i + 1,
                'rewritten_chunks': rewritten_chunks
            })

    # Clear checkpoint on success
    if checkpoint_mgr:
        checkpoint_mgr.clear_checkpoint(document_id, "rewriting")
```

---

## Phase 8: Configuration & Monitoring

### 8.1 Add Configuration Flag for Rewriting
**Files to modify:** `mirage/src/config/settings.py`

```python
# Add to settings
enable_content_rewriting: bool = True  # Can be disabled for testing
rewriting_parallelism: int = 3  # Number of concurrent rewriting jobs
```

### 8.2 Add Metrics Collection
**Files to create:** `mirage/src/core/metrics.py`

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class ProcessingMetrics:
    document_id: str
    start_time: datetime
    end_time: datetime
    total_chunks: int
    chunks_rewritten: int
    chunks_failed: int
    entities_extracted: int
    relationships_extracted: int
    rewriting_time_seconds: float
    extraction_time_seconds: float

    def to_dict(self):
        return {
            'document_id': self.document_id,
            'processing_time_total': (self.end_time - self.start_time).total_seconds(),
            'rewriting_time': self.rewriting_time_seconds,
            'extraction_time': self.extraction_time_seconds,
            'chunks_processed': self.total_chunks,
            'entities': self.entities_extracted,
            'relationships': self.relationships_extracted,
            'quality_metrics': {
                'rewrite_success_rate': self.chunks_rewritten / self.total_chunks,
                'entities_per_chunk': self.entities_extracted / self.total_chunks
            }
        }
```

---

## Implementation Priority & Timeline

### Priority 1: CRITICAL (Do First - Week 1)
- [ ] 1.2 Prompt Loader Utility (1 hour)
- [ ] 2.1 Update ContentRewriter to use centralized prompts (2 hours)
- [ ] 2.2 Update URL Service to pass content type (30 min)
- [ ] 3.2 Fix entity extraction token limits (3 hours)
- [ ] Test with YouTube video - verify entities extracted (1 hour)

**Total: ~8 hours**

### Priority 2: HIGH (Week 2)
- [ ] 4.1 Add quality validation (2 hours)
- [ ] 5.1 Parallel chunk rewriting (3 hours)
- [ ] 6.2 Fix embeddings consistency (2 hours)
- [ ] Test entire pipeline end-to-end (2 hours)

**Total: ~9 hours**

### Priority 3: MEDIUM (Week 3)
- [ ] 7.1 Add checkpointing (3 hours)
- [ ] 8.1 Configuration flags (1 hour)
- [ ] 8.2 Metrics collection (2 hours)
- [ ] Documentation updates (2 hours)

**Total: ~8 hours**

---

## Testing Strategy

### Unit Tests
```python
# tests/test_prompt_loader.py
def test_prompt_manager_loads_config():
    pm = PromptManager()
    prompt = pm.get_prompt('content_enrichment', 'english')
    assert 'system' in prompt
    assert 'ENRICH' in prompt['system']  # Verify it's enrichment not compression

# tests/test_content_rewriter.py
def test_rewriter_uses_enrichment_prompts():
    rewriter = ContentRewriter()
    # Mock the LLM call
    result = rewriter.rewrite_chunk("Steve Jobs founded Apple.", content_type="file")
    # Should expand, not compress
    assert len(result) >= len("Steve Jobs founded Apple.")

# tests/test_entity_extractor.py
def test_per_chunk_extraction():
    extractor = EntityExtractor()
    chunks = [{"text": "Steve Jobs founded Apple Inc."}, {"text": "Apple is in Cupertino."}]
    result = extractor.extract_from_chunks(chunks)
    assert len(result['entities']) >= 2  # At least Apple and Steve Jobs
```

### Integration Tests
```python
def test_full_youtube_pipeline():
    """Test complete pipeline with real YouTube video"""
    url = "https://www.youtube.com/watch?v=TEST_VIDEO"
    result = process_url(url)

    assert result['entities_extracted'] > 0  # Must extract SOME entities
    assert result['processed_text'] != result['full_text']  # Must be rewritten
    assert len(result['processed_text']) >= len(result['full_text']) * 0.8  # Not too short
```

---

## Success Metrics

After implementation, we should see:

1. **Entity Extraction:** > 0 entities from YouTube videos (currently 0)
2. **Processing Time:**
   - Rewriting: ~50 seconds for 50 chunks (down from 150s)
   - Total: < 2 minutes for typical video
3. **Quality:**
   - Enriched text is longer or similar length to original
   - No token limit errors
   - Consistent entity names across chunks
4. **Reliability:**
   - No silent failures
   - Checkpointing allows resume after errors
   - Clear error messages and validation

---

## Rollback Plan

If issues arise:

1. **Disable rewriting entirely:**
   ```python
   settings.enable_content_rewriting = False
   ```
   Test if entity extraction works better on raw text

2. **Revert to old prompts temporarily:**
   Keep old prompts in `prompts.yaml` under `legacy_prompts` section

3. **Fall back to sequential processing:**
   Set `settings.rewriting_parallelism = 1`

---

## Additional Recommendations

### Consider: Skip Rewriting Entirely for Testing
Before implementing all of this, **test entity extraction on original (non-rewritten) text**.

Modern LLMs are robust to:
- Typos
- Poor formatting
- Messy transcripts

You might find that **rewriting makes things worse**, not better. If so, disable it and save:
- 2-3 minutes processing time per document
- Complexity
- Cost
- Risk of information loss

Test this hypothesis first before investing weeks in fixing the rewriting pipeline.

---

## Questions to Resolve

1. **Should rewriting be optional per content-type?**
   - Maybe files don't need rewriting but YouTube does?

2. **What's the target entity extraction rate?**
   - Baseline metric to track improvement

3. **Should we use a different model for rewriting vs extraction?**
   - Maybe use faster model (GPT-4o-mini) for rewriting, better model for extraction

4. **Embedding strategy:**
   - Current: Embed processed text
   - Alternative: Embed both and use hybrid search?
