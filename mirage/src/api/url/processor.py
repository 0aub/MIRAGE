"""
URL Service - Background Processor
Handles async URL processing pipeline
"""

import time
from typing import Dict, Any
from loguru import logger

from ...core.document_processor import get_chunker_factory, TextChunker
from ...core.jobs import JobManager, ProcessingPhase
from ...core.graph_builder import (
    EntityExtractor,
    RelationshipExtractor,
    Neo4jClient,
    CooccurrenceExtractor,
    SemanticSimilarityExtractor,
)
from ...core.graph_builder.enhanced_neo4j_client import EnhancedNeo4jClient
from ...core.graph_builder.content_rewriter import ContentRewriter
from ...core.embeddings import JinaEmbedder
from ...core.refrag import REFRAGCompressor, CompressionPolicy, CompressionCache
from ...core.vector_store import QdrantVectorStore
from ...core.orchestration.claude_client import ClaudeClient
from ...config import settings

from .fetchers import extract_youtube_video_id, fetch_youtube_transcript, fetch_webpage_content


# Lazy-initialized components
_components_initialized = False
_text_chunker = None
_jina_embedder = None
_chunker_factory = None
_content_rewriter = None
_entity_extractor = None
_relationship_extractor = None
_cooccurrence_extractor = None
_semantic_similarity_extractor = None
_neo4j_client = None
_refrag_compressor = None
_vector_store = None
_enhanced_neo4j_client = None


def _init_components():
    """Initialize processing components lazily"""
    global _components_initialized
    global _text_chunker, _jina_embedder, _chunker_factory, _content_rewriter
    global _entity_extractor, _relationship_extractor, _cooccurrence_extractor
    global _semantic_similarity_extractor, _neo4j_client, _refrag_compressor
    global _vector_store, _enhanced_neo4j_client

    if _components_initialized:
        return

    _text_chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    _jina_embedder = JinaEmbedder()
    _chunker_factory = get_chunker_factory()
    _content_rewriter = ContentRewriter()
    _entity_extractor = EntityExtractor(embedder=_jina_embedder)
    _relationship_extractor = RelationshipExtractor()
    _cooccurrence_extractor = CooccurrenceExtractor(min_frequency=2)
    _semantic_similarity_extractor = SemanticSimilarityExtractor(similarity_threshold=0.75, top_k=5)
    _neo4j_client = Neo4jClient()

    compression_policy = CompressionPolicy()
    compression_cache = CompressionCache()
    _refrag_compressor = REFRAGCompressor(
        policy=compression_policy,
        cache=compression_cache,
        strategy="hybrid",
    )
    _vector_store = QdrantVectorStore()

    # Initialize enhanced Neo4j client
    try:
        claude_client = ClaudeClient()
        _enhanced_neo4j_client = EnhancedNeo4jClient(
            embedder=_jina_embedder,
            llm_client=claude_client
        )
        logger.info("Initialized EnhancedNeo4jClient")
    except Exception as e:
        logger.warning(f"Could not initialize EnhancedNeo4jClient: {e}")
        _enhanced_neo4j_client = None

    _components_initialized = True


def process_url_background(job_id: str, url: str, use_semantic_chunking: bool = True, job_manager: JobManager = None) -> Dict[str, Any]:
    """
    Process URL in background thread with job status updates
    """
    _init_components()

    start_time = time.time()

    try:
        # Update: Starting
        if job_manager:
            job_manager.update_job(job_id, progress=5, current_phase=ProcessingPhase.FETCHING)

        # Fetch content
        video_id = extract_youtube_video_id(url)

        if video_id:
            content_data = fetch_youtube_transcript(video_id)
            content_type = "youtube"
            document_id = f"yt_{video_id}"
        else:
            content_data = fetch_webpage_content(url)
            content_type = "webpage"
            document_id = "web_" + str(abs(hash(url)))[:12]

        if job_manager:
            job_manager.update_job(
                job_id,
                document_id=document_id,
                title=content_data.get("title"),
                content_type=content_type
            )

        full_text = content_data["text"]
        metadata = {
            "url": url,
            "title": content_data.get("title"),
            "description": content_data.get("description"),
            "content_type": content_type,
            **content_data.get("metadata", {}),
        }

        # Phase 1: Chunking
        if job_manager:
            job_manager.update_job(job_id, progress=15, current_phase=ProcessingPhase.CHUNKING)
        logger.info(f"[Job {job_id}] Phase 1: Chunking content")

        if use_semantic_chunking:
            chunker = _chunker_factory.get_chunker(content_type, embedder=_jina_embedder)
            chunks = chunker.chunk_text(full_text, metadata, document_id)
            chunking_method = _chunker_factory.get_strategy_info(content_type).get("strategy", "semantic")
        else:
            document = {"text": full_text, "metadata": metadata, "document_id": document_id}
            chunked_doc = _text_chunker.chunk_document(document)
            chunks = chunked_doc.get("chunks", [])
            chunking_method = "fixed"

        logger.info(f"[Job {job_id}] Created {len(chunks)} chunks")
        if job_manager:
            job_manager.update_job(job_id, total_chunks=len(chunks))

        # Create document in Neo4j
        try:
            initial_metadata = content_data.get("metadata", {}).copy()
            initial_metadata.update({
                "total_chars": len(full_text),
                "total_words": len(full_text.split()),
                "full_text": full_text,
            })
            _neo4j_client.store_document_metadata(
                document_id=document_id,
                title=content_data.get("title"),
                content_type=content_type,
                metadata=initial_metadata
            )
        except Exception as e:
            logger.warning(f"[Job {job_id}] Failed to create document: {e}")

        # Phase 2: Rewrite (optional)
        if settings.enable_content_rewriting:
            if job_manager:
                job_manager.update_job(job_id, progress=25, current_phase=ProcessingPhase.REWRITING)
            rewritten_chunks = _content_rewriter.rewrite_chunks(chunks, document_id=document_id)
            processed_text = " ".join([chunk.get("text", "") for chunk in rewritten_chunks])
            extraction_chunks = rewritten_chunks
        else:
            extraction_chunks = chunks
            processed_text = full_text

        # Phase 3: Extract entities
        if job_manager:
            job_manager.update_job(job_id, progress=40, current_phase=ProcessingPhase.EXTRACTING)
        logger.info(f"[Job {job_id}] Phase 3: Extracting entities")

        entity_result = _entity_extractor.extract_with_chunk_references(extraction_chunks, document_id=document_id)

        extraction_status = entity_result.get("status", "success")
        if extraction_status == "unavailable":
            raise Exception("LLM entity extraction unavailable")
        elif extraction_status == "rate_limited":
            raise Exception(entity_result.get("message", "Rate limit exceeded"))
        elif extraction_status == "error":
            raise Exception(entity_result.get("message", "Extraction failed"))

        entities_with_chunks = entity_result.get("entities_with_chunks", [])
        all_entities = entity_result.get("all_entities", [])
        relationships = entity_result.get("relationships", [])

        if not relationships and all_entities:
            relationships = _relationship_extractor.extract_relationships(all_entities, full_text)

        # Extract additional relationships
        cooccurrence_relationships = _cooccurrence_extractor.extract_cooccurrences(entities_with_chunks)
        semantic_relationships = _semantic_similarity_extractor.extract_similarities(entities_with_chunks)
        all_relationships = relationships + cooccurrence_relationships + semantic_relationships

        logger.info(f"[Job {job_id}] Extracted {len(entities_with_chunks)} entities, {len(all_relationships)} relationships")

        # Phase 4: Store in Neo4j
        if job_manager:
            job_manager.update_job(job_id, progress=60, current_phase=ProcessingPhase.STORING_GRAPH)

        try:
            processing_time_seconds = int(time.time() - start_time)
            enhanced_metadata = content_data.get("metadata", {}).copy()
            enhanced_metadata.update({
                "total_chars": len(full_text),
                "total_words": len(full_text.split()),
                "full_text": full_text,
                "processed_text": processed_text,
                "processing_time_seconds": processing_time_seconds,
            })

            graph_stats = _neo4j_client.store_chunks_with_entities(
                chunks=chunks,
                entities=entities_with_chunks,
                relationships=all_relationships,
                document_id=document_id,
                document_title=content_data.get("title"),
                document_type=content_type,
                document_metadata=enhanced_metadata,
                enhanced_neo4j_client=_enhanced_neo4j_client
            )
        except Exception as e:
            logger.error(f"[Job {job_id}] Error storing graph: {e}")
            graph_stats = {"error": str(e)}

        # Phase 5: REFRAG compression
        if job_manager:
            job_manager.update_job(job_id, progress=75, current_phase=ProcessingPhase.COMPRESSING)

        compression_result = _refrag_compressor.compress(chunks, query_context=None)
        speedup_factor = compression_result["original_length"] / max(1, compression_result["compressed_length"])

        # Phase 6: Store vectors
        if job_manager:
            job_manager.update_job(job_id, progress=85, current_phase=ProcessingPhase.STORING_VECTOR)

        try:
            vector_stats = _vector_store.add_chunks(chunks, document_id)
        except Exception as e:
            logger.error(f"[Job {job_id}] Error storing vectors: {e}")
            vector_stats = {"error": str(e)}

        # Finalize
        if job_manager:
            job_manager.update_job(job_id, progress=95, current_phase=ProcessingPhase.FINALIZING)

        final_processing_time = int(time.time() - start_time)

        return {
            "document_id": document_id,
            "url": url,
            "title": content_data.get("title"),
            "content_type": content_type,
            "processing_time_seconds": final_processing_time,
            "phase1": {
                "total_chars": len(full_text),
                "total_words": len(full_text.split()),
                "chunk_count": len(chunks),
                "chunking_method": chunking_method,
            },
            "phase2": {
                "entities_extracted": len(entities_with_chunks),
                "entity_mentions": len(all_entities),
                "relationships_extracted": len(relationships),
                "chunks_stored": len(chunks),
                "graph_storage": graph_stats,
            },
            "phase3": {
                "original_length": compression_result["original_length"],
                "compressed_length": compression_result["compressed_length"],
                "compression_ratio": compression_result["compression_ratio"],
                "chunks_compressed": compression_result["chunks_compressed"],
                "speedup_factor": speedup_factor,
                "processing_time": compression_result["processing_time"],
                "strategy": compression_result["strategy"],
            },
            "phase5": {
                "vector_storage": vector_stats,
            },
        }

    except Exception as e:
        logger.error(f"[Job {job_id}] Processing failed: {e}")
        raise
