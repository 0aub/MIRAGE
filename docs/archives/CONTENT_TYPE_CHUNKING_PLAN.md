# Content-Type-Specific Chunking - Implementation Plan

## Objective
Implement content-type-specific chunking strategies to optimize entity extraction for different data sources (YouTube, web pages, PDFs).

## Phase 1: YouTube (CURRENT FOCUS)
**Goal**: Validate semantic chunking works for YouTube transcripts

### Tasks:
1. ✅ Create `chunking_strategies.yaml` config
2. Create `ChunkerFactory` to select strategy by content type
3. Refactor document processor to use factory
4. Test with factual YouTube videos
5. Tune parameters (chunk_size, similarity_threshold)

**Timeline**: 4-6 hours
**Success Criteria**: Extract >50 entities from educational YouTube video

## Phase 2: Web Pages (DEFERRED)
**Goal**: Implement HTML-aware structural chunking

### Tasks:
1. Create `StructuralChunker` class
2. Parse HTML with BeautifulSoup
3. Respect `<h2>`, `<section>`, `<article>` boundaries
4. Fallback to semantic if no structure
5. Test with news articles and documentation

**Timeline**: 1-2 days
**Success Criteria**: Chunks align with article sections

## Phase 3: PDFs (DEFERRED)
**Goal**: Implement page-based chunking with heading detection

### Tasks:
1. Create `PageBasedChunker` class
2. Extract page boundaries from PDF
3. Detect headings from PDF outline/bookmarks
4. Split oversized pages intelligently
5. Test with research papers and reports

**Timeline**: 1-2 days
**Success Criteria**: Each page/section is a chunk

## Architecture

```
ChunkerFactory
    ├── SemanticChunker (YouTube, plain text)
    ├── StructuralChunker (Web pages)
    └── PageBasedChunker (PDFs)
```

## Current Status
- Phase 1: IN PROGRESS
- Phase 2: NOT STARTED
- Phase 3: NOT STARTED
