"""
URL Processing Service API
Handles web page URLs and YouTube video URLs
Extracts content and processes through the pipeline
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, HttpUrl
from loguru import logger
import re
import time
import asyncio
import json
from urllib.parse import urlparse, parse_qs

import trafilatura
import requests
import redis
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

from ..core.document_processor import TextChunker, SemanticChunker, get_chunker_factory
from ..core.jobs import JobManager, BackgroundWorker, JobStatus, ProcessingPhase
from ..core.document_processor.content_cleaner import ContentCleaner
from ..core.document_processor.punctuation_restorer import PunctuationRestorer
from ..core.media_processor import WhisperTranscriber
from ..core.graph_builder import (
    EntityExtractor,
    RelationshipExtractor,
    Neo4jClient,
    CooccurrenceExtractor,
    SemanticSimilarityExtractor,
)
from ..core.graph_builder.enhanced_neo4j_client import EnhancedNeo4jClient
from ..core.graph_builder.content_rewriter import ContentRewriter
from ..core.embeddings import JinaEmbedder
from ..core.refrag import REFRAGCompressor, CompressionPolicy, CompressionCache
from ..core.vector_store import QdrantVectorStore
from ..core.orchestration.claude_client import ClaudeClient
from ..config import settings

router = APIRouter()

# Initialize components
text_chunker = TextChunker(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)
jina_embedder = JinaEmbedder()
# Get global ChunkerFactory instance (uses content-type-specific strategies)
chunker_factory = get_chunker_factory()
content_rewriter = ContentRewriter()
content_cleaner = ContentCleaner()  # YouTube transcript denoiser
punctuation_restorer = PunctuationRestorer()  # BERT punctuation restoration for ASR transcripts
whisper_transcriber = WhisperTranscriber(model_size="large-v3")  # Whisper large-v3 for maximum quality
entity_extractor = EntityExtractor(embedder=jina_embedder)  # Pass embedder for entity embeddings
relationship_extractor = RelationshipExtractor()
cooccurrence_extractor = CooccurrenceExtractor(min_frequency=2)  # Co-occurrence relationships
semantic_similarity_extractor = SemanticSimilarityExtractor(similarity_threshold=0.75, top_k=5)  # Semantic similarity
neo4j_client = Neo4jClient()
compression_policy = CompressionPolicy()
compression_cache = CompressionCache()
refrag_compressor = REFRAGCompressor(
    policy=compression_policy,
    cache=compression_cache,
    strategy="hybrid",
)
vector_store = QdrantVectorStore()

# Initialize enhanced Neo4j client for enrichment
try:
    claude_client = ClaudeClient()
    enhanced_neo4j_client = EnhancedNeo4jClient(
        embedder=jina_embedder,
        llm_client=claude_client
    )
    logger.info("Initialized EnhancedNeo4jClient for document ingestion with enrichment capabilities")
except Exception as e:
    logger.warning(f"Could not initialize EnhancedNeo4jClient: {e}. Entity enrichment will be disabled.")
    enhanced_neo4j_client = None

# Initialize Redis client for processing status tracking
try:
    redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    redis_client.ping()  # Test connection
    logger.info("Redis client initialized for processing status tracking")
except Exception as e:
    logger.error(f"Failed to initialize Redis: {e}")
    redis_client = None

# Initialize Job Manager and Background Worker for async processing
try:
    job_manager = JobManager(redis_host="redis", redis_port=6379, ttl=86400)
    background_worker = BackgroundWorker(job_manager)
    logger.info("Job manager and background worker initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize job manager: {e}")
    job_manager = None
    background_worker = None

# Helper functions for Redis-based processing status tracking
def set_processing_status(document_id: str, title: str, url: str, content_type: str, start_time: int, current_chunk: int = 0, total_chunks: int = 0, phase: str = "starting"):
    """Store processing status in Redis"""
    if not redis_client:
        return
    try:
        status_data = {
            "document_id": document_id,
            "title": title or "Untitled",
            "url": url,
            "content_type": content_type,
            "status": "processing",
            "start_time": start_time,
            "current_chunk": current_chunk,
            "total_chunks": total_chunks,
            "phase": phase,
        }
        # Store with 1 hour TTL
        redis_client.setex(
            f"processing:{document_id}",
            3600,
            json.dumps(status_data)
        )
        logger.info(f"Set processing status for {document_id} - phase={phase}, chunk={current_chunk}/{total_chunks}")
    except Exception as e:
        logger.error(f"Failed to set processing status: {e}")

def update_processing_progress(document_id: str, current_chunk: int, total_chunks: int):
    """Update chunk progress for a processing document"""
    if not redis_client:
        return
    try:
        # Get existing status
        key = f"processing:{document_id}"
        data = redis_client.get(key)
        if data:
            status_data = json.loads(data)
            status_data["current_chunk"] = current_chunk
            status_data["total_chunks"] = total_chunks
            # Update with same TTL
            redis_client.setex(key, 3600, json.dumps(status_data))
            logger.debug(f"Updated progress for {document_id}: {current_chunk}/{total_chunks}")
    except Exception as e:
        logger.error(f"Failed to update processing progress: {e}")

def clear_processing_status(document_id: str):
    """Remove processing status from Redis"""
    if not redis_client:
        return
    try:
        redis_client.delete(f"processing:{document_id}")
        logger.info(f"Cleared processing status for {document_id}")
    except Exception as e:
        logger.error(f"Failed to clear processing status: {e}")

def get_all_processing() -> List[Dict[str, Any]]:
    """Get all currently processing documents"""
    if not redis_client:
        return []
    try:
        keys = redis_client.keys("processing:*")
        results = []
        for key in keys:
            data = redis_client.get(key)
            if data:
                status_info = json.loads(data)
                # Calculate elapsed time
                elapsed = int(time.time() - status_info["start_time"])
                status_info["elapsed_seconds"] = elapsed
                results.append(status_info)
        return results
    except Exception as e:
        logger.error(f"Failed to get processing statuses: {e}")
        return []


# Models
class URLRequest(BaseModel):
    url: HttpUrl
    use_semantic_chunking: bool = True


class URLPreviewResponse(BaseModel):
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    content_preview: str
    estimated_words: int
    content_type: str  # "webpage" or "youtube"


class URLProcessResponse(BaseModel):
    document_id: str
    url: str
    title: Optional[str] = None
    content_type: str
    processing_time_seconds: int  # Total processing time in seconds
    phase1: Dict[str, Any]
    phase2: Dict[str, Any]
    phase3: Dict[str, Any]
    phase5: Dict[str, Any]


# Helper Functions
def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats
    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    """
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*?v=([a-zA-Z0-9_-]{11})',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def fetch_webpage_content(url: str) -> Dict[str, Any]:
    """
    Fetch and extract main content from a web page using trafilatura
    """
    logger.info(f"Fetching web page: {url}")

    try:
        # Download the page
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise ValueError("Failed to download the web page")

        # Extract main content
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )

        if not text:
            raise ValueError("Failed to extract content from the web page")

        # Extract metadata
        metadata = trafilatura.extract_metadata(downloaded)

        title = metadata.title if metadata and metadata.title else "Untitled"
        description = metadata.description if metadata and metadata.description else ""

        return {
            "text": text,
            "title": title,
            "description": description,
            "url": url,
            "metadata": {
                "author": metadata.author if metadata else None,
                "date": metadata.date if metadata else None,
                "sitename": metadata.sitename if metadata else None,
            }
        }

    except Exception as e:
        logger.error(f"Error fetching web page {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch web page: {str(e)}")


def fetch_youtube_transcript(video_id: str) -> Dict[str, Any]:
    """
    Fetch transcript from YouTube video using DUAL-PATH approach
    Priority: YouTube API + BERT (speed) → fallback to Whisper large-v3 (quality)
    """
    logger.info(f"Fetching transcript for video: {video_id}")

    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    full_text = None
    transcript_method = None
    detected_language = "unknown"

    # PATH 1: Try YouTube API + BERT first (SPEED PRIORITY - 10-30 seconds)
    try:
        logger.info("PATH 1: Attempting YouTube API + BERT punctuation (fast mode)...")

        # Create YouTubeTranscriptApi instance
        api = YouTubeTranscriptApi()

        transcript = None
        last_error = None

        # Try to get transcript - priority: any available, then specific languages
        try:
            # First, try English (most common)
            logger.debug(f"Attempting to fetch English transcript for {video_id}")
            transcript = api.fetch(video_id, languages=['en'])
            detected_language = transcript.language_code
            logger.info(f"Successfully fetched {detected_language} transcript for {video_id}")
        except Exception as e:
            logger.debug(f"English transcript failed for {video_id}: {type(e).__name__}: {str(e)}")
            last_error = e
            # If that fails, try Arabic
            try:
                logger.debug(f"Attempting to fetch Arabic transcript for {video_id}")
                transcript = api.fetch(video_id, languages=['ar'])
                detected_language = transcript.language_code
                logger.info(f"Successfully fetched {detected_language} transcript for {video_id}")
            except Exception as e:
                logger.debug(f"Arabic transcript failed for {video_id}: {type(e).__name__}: {str(e)}")
                last_error = e
                # Try any available language
                try:
                    logger.debug(f"Attempting to list and fetch any available transcript for {video_id}")
                    transcript_list = api.list(video_id)
                    # Get first available transcript
                    available_transcripts = list(transcript_list)
                    if available_transcripts:
                        transcript = available_transcripts[0].fetch()
                        detected_language = transcript.language_code
                        logger.info(f"Successfully fetched {detected_language} transcript for {video_id}")
                except Exception as e:
                    logger.debug(f"Listing transcripts failed for {video_id}: {type(e).__name__}: {str(e)}")
                    last_error = e

        if not transcript:
            error_detail = f"No transcript found for this video"
            if last_error:
                logger.debug(f"YouTube API transcript failed: {type(last_error).__name__}: {str(last_error)}")
                error_detail += f" ({type(last_error).__name__})"
            raise Exception(error_detail)

        # Combine all transcript snippets into text
        raw_text = " ".join([snippet.text for snippet in transcript.snippets])

        # Step 1: Clean YouTube transcript noise (music markers, applause, foreign chars)
        cleaning_result = content_cleaner.clean_youtube_transcript(raw_text)
        cleaned_text = cleaning_result["cleaned_text"]

        if cleaning_result["noise_percentage"] > 5.0:
            logger.info(
                f"Removed {cleaning_result['noise_percentage']:.1f}% noise from transcript "
                f"({cleaning_result['original_length']} → {cleaning_result['cleaned_length']} chars)"
            )

        # Step 2: Restore punctuation (CRITICAL for entity extraction)
        # YouTube transcripts lack punctuation, making it hard for LLM to identify entity boundaries
        if punctuation_restorer.should_restore(cleaned_text):
            logger.info("YouTube transcript lacks punctuation - restoring with BERT model")
            punctuation_result = punctuation_restorer.restore_punctuation(cleaned_text)
            full_text = punctuation_result["punctuated_text"]

            logger.info(
                f"Punctuation restored: {punctuation_result['punctuation_marks_added']} marks added "
                f"({punctuation_result['original_length']} → {punctuation_result['punctuated_length']} chars)"
            )
            transcript_method = "youtube_api_bert"
        else:
            full_text = cleaned_text
            transcript_method = "youtube_api"
            logger.info("Transcript already has punctuation - skipping restoration")

        logger.info(f"✅ YouTube API + BERT successful: {len(full_text)} chars")

    except Exception as youtube_error:
        logger.warning(f"YouTube API failed: {youtube_error}")
        logger.info("PATH 2: Falling back to Whisper large-v3 transcription (slow but works without captions)...")

        # PATH 2: Fallback to Whisper (slow but works for any video)
        try:
            whisper_result = whisper_transcriber.transcribe_youtube(
                youtube_url,
                language=None  # Auto-detect (handles both Arabic and English)
            )

            full_text = whisper_result["text"]
            detected_language = whisper_result["language"]
            transcript_method = "whisper_large-v3"

            logger.info(
                f"✅ Whisper fallback successful: {len(full_text)} chars, "
                f"language={detected_language}, duration={whisper_result.get('duration', 0):.1f}s, "
                f"segments={whisper_result.get('segment_count', 0)}"
            )

        except Exception as whisper_error:
            logger.error(f"Both YouTube API and Whisper failed: {whisper_error}")
            raise HTTPException(
                status_code=400,
                detail="Failed to fetch transcript. This video has no captions and Whisper transcription failed."
            )

    # Get video metadata using requests to YouTube's oEmbed API
    # This executes AFTER either YouTube API (PATH 1) or Whisper (PATH 2) succeeds
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url, timeout=10)
        if response.status_code == 200:
            video_info = response.json()
            title = video_info.get('title', 'YouTube Video')
            author = video_info.get('author_name', 'Unknown')
        else:
            title = f"YouTube Video {video_id}"
            author = "Unknown"
    except Exception as e:
        logger.warning(f"Failed to fetch video metadata: {e}")
        title = f"YouTube Video {video_id}"
        author = "Unknown"

    return {
        "text": full_text,
        "title": title,
        "description": f"Transcript from YouTube video by {author}",
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "metadata": {
            "video_id": video_id,
            "author": author,
            "language": detected_language,
            "transcript_length": len(full_text),
        }
    }


def fetch_youtube_metadata_only(video_id: str) -> Dict[str, Any]:
    """
    Fetch ONLY metadata from YouTube video (INSTANT - < 1 second)
    Does NOT fetch transcript - used for preview

    Args:
        video_id: YouTube video ID

    Returns:
        Dict with video metadata (title, author, thumbnail, duration estimate)
    """
    logger.info(f"Fetching metadata (instant preview) for video: {video_id}")

    try:
        # Use YouTube oEmbed API for instant metadata (no API key needed)
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url, timeout=5)

        if response.status_code == 200:
            video_info = response.json()
            title = video_info.get('title', 'YouTube Video')
            author = video_info.get('author_name', 'Unknown')
            thumbnail = video_info.get('thumbnail_url', '')

            logger.info(f"✅ Fetched metadata: {title} by {author}")

            return {
                "title": title,
                "description": f"YouTube video by {author}",
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "author": author,
                "thumbnail": thumbnail,
                "video_id": video_id,
                "content_type": "youtube"
            }
        else:
            logger.warning(f"oEmbed API returned {response.status_code} for video {video_id}")
            raise Exception(f"oEmbed API failed with status {response.status_code}")

    except Exception as e:
        logger.error(f"Failed to fetch video metadata: {e}")
        # Return minimal fallback metadata
        return {
            "title": f"YouTube Video {video_id}",
            "description": "YouTube video (metadata unavailable)",
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "author": "Unknown",
            "thumbnail": "",
            "video_id": video_id,
            "content_type": "youtube"
        }


# Endpoints
@router.post("/preview", response_model=URLPreviewResponse)
async def preview_url_content(request: URLRequest):
    """
    INSTANT preview of content from a URL without fetching full transcript
    For YouTube: Returns metadata only (< 1 second)
    For web pages: Returns page title and description

    This allows users to verify the URL before clicking "Process"
    """
    url_str = str(request.url)
    logger.info(f"INSTANT preview request for URL: {url_str}")

    # Check if it's a YouTube URL
    video_id = extract_youtube_video_id(url_str)

    if video_id:
        # INSTANT metadata fetch (< 1 second) - NO transcript fetching
        metadata = fetch_youtube_metadata_only(video_id)

        return URLPreviewResponse(
            url=url_str,
            title=metadata.get("title"),
            description=metadata.get("description"),
            content_preview=f"YouTube video by {metadata.get('author', 'Unknown')}",
            estimated_words=0,  # Not estimated until processing
            content_type="youtube",
        )
    else:
        # Fetch web page content (still fast for web pages)
        content_data = fetch_webpage_content(url_str)
        text = content_data["text"]
        preview = text[:500] + ("..." if len(text) > 500 else "")

        return URLPreviewResponse(
            url=url_str,
            title=content_data.get("title"),
            description=content_data.get("description"),
            content_preview=preview,
            estimated_words=len(text.split()),
            content_type="webpage",
        )


# Background processing function for async job execution
def process_url_background(job_id: str, url: str, use_semantic_chunking: bool = True) -> Dict[str, Any]:
    """
    Process URL in background thread with job status updates

    Args:
        job_id: Job ID for tracking
        url: URL to process
        use_semantic_chunking: Whether to use semantic chunking

    Returns:
        Processing result dictionary
    """
    start_time = time.time()

    try:
        # Update: Starting
        job_manager.update_job(job_id, progress=5, current_phase=ProcessingPhase.FETCHING)

        # Check if it's a YouTube URL
        video_id = extract_youtube_video_id(url)

        if video_id:
            # Fetch YouTube transcript
            content_data = fetch_youtube_transcript(video_id)
            content_type = "youtube"
            document_id = f"yt_{video_id}"
        else:
            # Fetch web page content
            content_data = fetch_webpage_content(url)
            content_type = "webpage"
            document_id = "web_" + str(abs(hash(url)))[:12]

        # Update job with document info
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
        job_manager.update_job(job_id, progress=15, current_phase=ProcessingPhase.CHUNKING)
        logger.info(f"[Job {job_id}] Phase 1: Chunking content (content_type={content_type})")

        if use_semantic_chunking:
            chunker = chunker_factory.get_chunker(content_type, embedder=jina_embedder)
            chunks = chunker.chunk_text(full_text, metadata, document_id)  # Pass document_id for chunk IDs
            chunking_method = chunker_factory.get_strategy_info(content_type).get("strategy", "semantic")
        else:
            document = {"text": full_text, "metadata": metadata, "document_id": document_id}
            chunked_doc = text_chunker.chunk_document(document)
            chunks = chunked_doc.get("chunks", [])
            chunking_method = "fixed"

        logger.info(f"[Job {job_id}] Created {len(chunks)} chunks")
        job_manager.update_job(job_id, total_chunks=len(chunks))

        # Create document placeholder in Neo4j
        try:
            initial_metadata = content_data.get("metadata", {}).copy()
            initial_metadata.update({
                "total_chars": len(full_text),
                "total_words": len(full_text.split()),
                "full_text": full_text,
            })
            neo4j_client.store_document_metadata(
                document_id=document_id,
                title=content_data.get("title"),
                content_type=content_type,
                metadata=initial_metadata
            )
            logger.info(f"[Job {job_id}] Created document placeholder in Neo4j")
        except Exception as e:
            logger.warning(f"[Job {job_id}] Failed to create document placeholder: {e}")

        # Phase 2: Rewrite (optional)
        if settings.enable_content_rewriting:
            job_manager.update_job(job_id, progress=25, current_phase=ProcessingPhase.REWRITING)
            logger.info(f"[Job {job_id}] Phase 2: Rewriting content")

            rewritten_chunks = content_rewriter.rewrite_chunks(chunks, document_id=document_id)
            processed_text = " ".join([chunk.get("text", "") for chunk in rewritten_chunks])
            extraction_chunks = rewritten_chunks
        else:
            extraction_chunks = chunks
            processed_text = full_text

        # Phase 3: Extract entities with chunk references (Hybrid Vector-Graph Architecture)
        job_manager.update_job(job_id, progress=40, current_phase=ProcessingPhase.EXTRACTING)
        logger.info(f"[Job {job_id}] Phase 3: Extracting entities with chunk references (hybrid architecture)")

        entity_result = entity_extractor.extract_with_chunk_references(extraction_chunks, document_id=document_id)

        # Check extraction status
        extraction_status = entity_result.get("status", "success")
        if extraction_status == "unavailable":
            raise Exception("LLM entity extraction is currently unavailable. TGI may still be loading.")
        elif extraction_status == "rate_limited":
            raise Exception(entity_result.get("message", "API rate limit exceeded"))
        elif extraction_status == "error":
            raise Exception(entity_result.get("message", "Entity extraction failed"))

        # Extract entities and relationships from hybrid result
        entities_with_chunks = entity_result.get("entities_with_chunks", [])
        all_entities = entity_result.get("all_entities", [])
        relationships = entity_result.get("relationships", [])
        extraction_method = entity_result.get("extraction_method", "")

        # Fallback to co-occurrence relationships if LLM didn't extract them
        if not relationships and all_entities:
            relationships = relationship_extractor.extract_relationships(all_entities, full_text)

        logger.info(
            f"[Job {job_id}] Extracted {len(entities_with_chunks)} unique entities "
            f"({len(all_entities)} total mentions), {len(relationships)} LLM relationships"
        )

        # Extract co-occurrence relationships (entities appearing together in chunks)
        logger.info(f"[Job {job_id}] Extracting co-occurrence relationships...")
        cooccurrence_relationships = cooccurrence_extractor.extract_cooccurrences(entities_with_chunks)
        logger.info(f"[Job {job_id}] Extracted {len(cooccurrence_relationships)} co-occurrence relationships")

        # Extract semantic similarity relationships (based on entity embeddings)
        logger.info(f"[Job {job_id}] Extracting semantic similarity relationships...")
        semantic_relationships = semantic_similarity_extractor.extract_similarities(entities_with_chunks)
        logger.info(f"[Job {job_id}] Extracted {len(semantic_relationships)} semantic similarity relationships")

        # Combine all relationship types
        all_relationships = relationships + cooccurrence_relationships + semantic_relationships
        logger.info(
            f"[Job {job_id}] Total relationships: {len(all_relationships)} "
            f"(LLM: {len(relationships)}, co-occurrence: {len(cooccurrence_relationships)}, semantic: {len(semantic_relationships)})"
        )

        # Phase 4: Store in Neo4j (Hybrid Vector-Graph Architecture)
        job_manager.update_job(job_id, progress=60, current_phase=ProcessingPhase.STORING_GRAPH)
        logger.info(f"[Job {job_id}] Phase 4: Storing chunks + entities in Neo4j (hybrid architecture)")

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

            # Store chunks as nodes with MENTIONS relationships to entities
            # Include all relationship types: LLM, co-occurrence, and semantic similarity
            graph_stats = neo4j_client.store_chunks_with_entities(
                chunks=chunks,
                entities=entities_with_chunks,
                relationships=all_relationships,  # Use combined relationships
                document_id=document_id,
                document_title=content_data.get("title"),
                document_type=content_type,
                document_metadata=enhanced_metadata,
                enhanced_neo4j_client=enhanced_neo4j_client
            )
            logger.info(f"[Job {job_id}] Stored graph (hybrid): {graph_stats}")
        except Exception as e:
            logger.error(f"[Job {job_id}] Error storing graph: {e}")
            graph_stats = {"error": str(e)}

        # Phase 5: REFRAG compression
        job_manager.update_job(job_id, progress=75, current_phase=ProcessingPhase.COMPRESSING)
        logger.info(f"[Job {job_id}] Phase 5: REFRAG compression")

        compression_result = refrag_compressor.compress(chunks, query_context=None)
        speedup_factor = compression_result["original_length"] / max(1, compression_result["compressed_length"])

        # Phase 6: Store in vector database
        job_manager.update_job(job_id, progress=85, current_phase=ProcessingPhase.STORING_VECTOR)
        logger.info(f"[Job {job_id}] Phase 6: Storing in vector database")

        try:
            vector_stats = vector_store.add_chunks(chunks, document_id)
            logger.info(f"[Job {job_id}] Stored vectors: {vector_stats}")
        except Exception as e:
            logger.error(f"[Job {job_id}] Error storing vectors: {e}")
            vector_stats = {"error": str(e)}

        # Finalize
        job_manager.update_job(job_id, progress=95, current_phase=ProcessingPhase.FINALIZING)
        final_processing_time = int(time.time() - start_time)

        # Build result
        result = {
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

        logger.info(f"[Job {job_id}] Processing completed in {final_processing_time}s")
        return result

    except Exception as e:
        logger.error(f"[Job {job_id}] Processing failed: {e}")
        raise


class JobSubmitResponse(BaseModel):
    """Response for job submission"""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response for job status"""
    job_id: str
    status: str
    progress: int
    current_phase: Optional[str] = None
    document_id: Optional[str] = None
    title: Optional[str] = None
    content_type: Optional[str] = None
    url: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    elapsed_seconds: int
    error: Optional[str] = None


@router.post("/process-async", response_model=JobSubmitResponse)
async def process_url_async(request: URLRequest):
    """
    Submit URL for asynchronous processing (NEW ENDPOINT - Recommended)

    Returns immediately with job_id for status polling.
    Use /jobs/{job_id}/status to check progress.
    Use /jobs/{job_id}/result to get final result when complete.

    This endpoint allows concurrent processing and doesn't block the UI.
    """
    if not job_manager or not background_worker:
        raise HTTPException(
            status_code=503,
            detail="Job management system is not available"
        )

    url_str = str(request.url)
    logger.info(f"Submitting URL for async processing: {url_str}")

    try:
        # Create job
        job = job_manager.create_job(
            job_type="url_processing",
            url=url_str,
            metadata={"use_semantic_chunking": request.use_semantic_chunking}
        )

        # Submit to background worker
        background_worker.submit_job(
            job.job_id,
            process_url_background,
            url_str,
            request.use_semantic_chunking
        )

        logger.info(f"Job {job.job_id} submitted successfully for {url_str}")

        return JobSubmitResponse(
            job_id=job.job_id,
            status="queued",
            message=f"Processing started for {url_str}. Poll /jobs/{job.job_id}/status for progress."
        )

    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process", response_model=URLProcessResponse)
async def process_url(request: URLRequest):
    """
    Process a URL synchronously (DEPRECATED - Blocks until complete)

    WARNING: This endpoint blocks the request until processing completes.
    For better UX, use /process-async instead.

    Kept for backward compatibility.
    """
    logger.warning("Using deprecated synchronous /process endpoint. Consider migrating to /process-async for better UX.")

    url_str = str(request.url)
    logger.info(f"Processing URL synchronously: {url_str}")

    try:
        # Process directly using the background function (but synchronously)
        result = process_url_background("sync", url_str, request.use_semantic_chunking)

        # Convert result dict to URLProcessResponse
        return URLProcessResponse(**result)

    except ConnectionError as e:
        logger.error(f"TGI connection error: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "tgi_unavailable",
                "message": "TGI container is not running or not reachable",
                "suggestion": "Start TGI with: docker compose up -d tgi",
                "technical_details": str(e)
            }
        )
    except TimeoutError as e:
        logger.error(f"TGI timeout error: {e}")
        raise HTTPException(
            status_code=504,
            detail={
                "error": "tgi_timeout",
                "message": "TGI request timed out after 30 seconds",
                "suggestion": "The ALLaM model may be loading. Check TGI logs: docker logs mirage-tgi",
                "technical_details": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        # Check if it's a rate limit error
        error_str = str(e).lower()
        if any(term in error_str for term in ["rate limit", "quota", "429", "resource exhausted", "requests per"]):
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Please wait before trying again. Error: {str(e)}"
            )
        raise HTTPException(status_code=500, detail=str(e))




# Job status and result endpoints

@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a processing job

    Returns current progress, phase, and status.
    Poll this endpoint to track job progress.
    """
    if not job_manager:
        raise HTTPException(
            status_code=503,
            detail="Job management system is not available"
        )

    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    return JobStatusResponse(**job.to_dict())


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Get the final result of a completed job

    Returns the full processing result when job status is 'completed'.
    Returns error if job is still processing or failed.
    """
    if not job_manager:
        raise HTTPException(
            status_code=503,
            detail="Job management system is not available"
        )

    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    if job.status == JobStatus.PROCESSING or job.status == JobStatus.QUEUED:
        raise HTTPException(
            status_code=202,  # Accepted but not ready
            detail={
                "message": "Job is still processing",
                "status": job.status,
                "progress": job.progress
            }
        )

    if job.status == JobStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Job failed",
                "error": job.error
            }
        )

    if job.status == JobStatus.CANCELLED:
        raise HTTPException(
            status_code=410,  # Gone
            detail="Job was cancelled"
        )

    # Job completed successfully
    return job.result


@router.get("/jobs/active")
async def get_active_jobs():
    """
    Get all currently active jobs (queued or processing)

    Returns list of jobs with their current status and progress.
    """
    if not job_manager:
        raise HTTPException(
            status_code=503,
            detail="Job management system is not available"
        )

    active_jobs = job_manager.get_active_jobs()

    return {
        "jobs": [job.to_dict() for job in active_jobs],
        "count": len(active_jobs)
    }


# Legacy processing status endpoint (kept for compatibility)
@router.get("/processing-status")
async def get_processing_status():
    """
    Get all currently processing documents from Redis
    Returns list of processing documents with their status and elapsed time
    """
    try:
        processing_items = get_all_processing()
        return {
            "processing": processing_items,
            "count": len(processing_items)
        }
    except Exception as e:
        logger.error(f"Error getting processing status: {e}")
        return {
            "processing": [],
            "count": 0
        }
