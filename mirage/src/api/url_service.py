"""
URL Processing Service API
Handles web page URLs and YouTube video URLs
Extracts content and processes through the pipeline
"""

from fastapi import APIRouter, HTTPException
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

# Shared variable to track if processing is currently active
_is_processing = False

from ..core.document_processor import TextChunker, SemanticChunker, get_chunker_factory
from ..core.document_processor.content_cleaner import ContentCleaner
from ..core.document_processor.punctuation_restorer import PunctuationRestorer
from ..core.media_processor import WhisperTranscriber
from ..core.graph_builder import EntityExtractor, RelationshipExtractor, Neo4jClient
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
entity_extractor = EntityExtractor()
relationship_extractor = RelationshipExtractor()
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


@router.post("/process", response_model=URLProcessResponse)
async def process_url(request: URLRequest):
    """
    Process a URL (web page or YouTube) through the full pipeline
    Phase 1: Content extraction & chunking
    Phase 2: Entity extraction & graph building
    Phase 3: REFRAG compression
    Phase 5: Vector storage
    """
    global _is_processing

    # Check if already processing
    if _is_processing:
        raise HTTPException(
            status_code=409,
            detail="Another document is currently being processed. Please wait for it to complete before processing a new document."
        )

    url_str = str(request.url)
    logger.info(f"Processing URL: {url_str}")

    # Mark as processing
    _is_processing = True

    try:
        # Start timing the entire process
        start_time = time.time()

        # Check if it's a YouTube URL
        video_id = extract_youtube_video_id(url_str)

        if video_id:
            # Fetch YouTube transcript
            content_data = fetch_youtube_transcript(video_id)
            content_type = "youtube"
            document_id = f"yt_{video_id}"
        else:
            # Fetch web page content
            content_data = fetch_webpage_content(url_str)
            content_type = "webpage"
            # Generate document ID from URL
            document_id = "web_" + str(abs(hash(url_str)))[:12]

        full_text = content_data["text"]
        metadata = {
            "url": url_str,
            "title": content_data.get("title"),
            "description": content_data.get("description"),
            "content_type": content_type,
            **content_data.get("metadata", {}),
        }

        # Phase 1: Chunking
        logger.info(f"Phase 1: Chunking content (content_type={content_type})")
        if request.use_semantic_chunking:
            # Get appropriate chunker for content type from factory
            # Factory will select semantic chunker for YouTube, structural for webpages (future), etc.
            chunker = chunker_factory.get_chunker(content_type, embedder=jina_embedder)
            chunks = chunker.chunk_text(full_text, metadata)
            chunking_method = chunker_factory.get_strategy_info(content_type).get("strategy", "semantic")
        else:
            # Use basic text chunker
            document = {"text": full_text, "metadata": metadata}
            chunked_doc = text_chunker.chunk_document(document)
            chunks = chunked_doc.get("chunks", [])
            chunking_method = "fixed"

        logger.info(f"Created {len(chunks)} chunks")

        # Set processing status to "rewriting" phase (Phase 1: Preparing Content)
        set_processing_status(
            document_id=document_id,
            title=content_data.get("title", "Untitled"),
            url=url_str,
            content_type=content_type,
            start_time=int(start_time),
            current_chunk=0,
            total_chunks=len(chunks),
            phase="rewriting"
        )

        # Create document placeholder in Neo4j IMMEDIATELY (before extraction)
        # This ensures the document appears in the list during processing and on page reload
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
            logger.info(f"Created document placeholder in Neo4j: {document_id}")
        except Exception as e:
            logger.warning(f"Failed to create document placeholder: {e}")

        # Phase 2: Rewrite and clean content chunks (OPTIONAL - configurable)
        if settings.enable_content_rewriting:
            logger.info("Phase 2: Rewriting and cleaning content (ENABLED)")

            # Update status to rewriting phase
            set_processing_status(
                document_id=document_id,
                title=content_data.get("title", "Untitled"),
                url=url_str,
                content_type=content_type,
                start_time=int(start_time),
                current_chunk=0,
                total_chunks=len(chunks),
                phase="rewriting"
            )

            rewritten_chunks = content_rewriter.rewrite_chunks(chunks, document_id=document_id)
            logger.info(f"Completed rewriting {len(rewritten_chunks)} chunks")

            # Combine rewritten chunks into processed text
            processed_text = " ".join([chunk.get("text", "") for chunk in rewritten_chunks])

            # Use rewritten chunks for extraction
            extraction_chunks = rewritten_chunks
        else:
            logger.info("Phase 2: Content rewriting DISABLED - using raw text for extraction")
            # Skip rewriting, use original chunks directly
            extraction_chunks = chunks
            processed_text = full_text  # No rewriting, processed = raw

        # Update status for extraction phase
        set_processing_status(
            document_id=document_id,
            title=content_data.get("title", "Untitled"),
            url=url_str,
            content_type=content_type,
            start_time=int(start_time),
            current_chunk=0,
            total_chunks=len(extraction_chunks),
            phase="extraction"
        )

        # Phase 3: Extract entities and relationships from RAW text (rewriting disabled by default)
        logger.info(f"Phase 3: Extracting entities and relationships from {'processed' if settings.enable_content_rewriting else 'raw'} text")
        entity_result = entity_extractor.extract_from_chunks(extraction_chunks, document_id=document_id)

        # Check extraction status
        extraction_status = entity_result.get("status", "success")
        if extraction_status == "unavailable":
            logger.warning("LLM extraction unavailable - TGI may still be loading")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_unavailable",
                    "message": entity_result.get("message", "LLM entity extraction is currently unavailable"),
                    "suggestion": "The Llama 3.1 70B model may still be downloading. Please wait a few minutes and try again."
                }
            )
        elif extraction_status == "rate_limited":
            logger.warning(f"Rate limit hit: {entity_result.get('message')}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limited",
                    "message": entity_result.get("message", "API rate limit exceeded"),
                    "suggestion": "Consider using local TGI for unlimited processing, or wait before trying again."
                }
            )
        elif extraction_status == "error":
            logger.error(f"Extraction error: {entity_result.get('message')}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "extraction_failed",
                    "message": entity_result.get("message", "Entity extraction failed"),
                    "suggestion": "Please check the logs and try again later."
                }
            )

        entities = entity_result.get("entities", [])
        logger.info(f"Extracted {len(entities)} unique entities")

        # Check if LLM already extracted relationships
        extraction_method = entity_result.get("extraction_method", "")
        if "relationships" in entity_result and extraction_method.startswith("llm"):
            relationships = entity_result["relationships"]
            logger.info(f"LLM extracted {len(relationships)} relationships (method: {extraction_method})")
        else:
            # Fall back to traditional relationship extraction
            logger.info(f"Phase 2: Extracting relationships using co-occurrence (LLM method was: {extraction_method})")
            relationships = relationship_extractor.extract_relationships(entities, full_text)
            logger.info(f"Extracted {len(relationships)} relationships")

        # Phase 2: Store in Neo4j
        logger.info("Phase 2: Storing in Neo4j")
        try:
            # Calculate processing time
            processing_time_seconds = int(time.time() - start_time)

            # Prepare enhanced metadata with text statistics and full content
            enhanced_metadata = content_data.get("metadata", {}).copy()
            enhanced_metadata.update({
                "total_chars": len(full_text),
                "total_words": len(full_text.split()),
                "full_text": full_text,  # Store the complete raw transcript/content
                "processed_text": processed_text,  # Store the rewritten/cleaned content
                "processing_time_seconds": processing_time_seconds,  # Track processing time in seconds
            })

            graph_stats = neo4j_client.store_graph(
                entities=entities,
                relationships=relationships,
                document_id=document_id,
                document_title=content_data.get("title"),
                document_type=content_type,
                document_metadata=enhanced_metadata,
                enhanced_neo4j_client=enhanced_neo4j_client  # Pass enhanced client for enrichment
            )
            logger.info(f"Stored graph: {graph_stats}")
        except Exception as e:
            logger.error(f"Error storing graph in Neo4j: {e}")
            graph_stats = {"error": str(e)}

        # Phase 3: REFRAG compression
        logger.info("Phase 3: Applying REFRAG compression")
        compression_result = refrag_compressor.compress(chunks, query_context=None)

        speedup_factor = (
            compression_result["original_length"] /
            max(1, compression_result["compressed_length"])
        )

        logger.info(
            f"REFRAG compression: {compression_result['compression_ratio']:.2f} ratio, "
            f"{speedup_factor:.2f}x speedup"
        )

        # Phase 5: Store in vector database
        logger.info("Phase 5: Storing in vector database")
        try:
            vector_stats = vector_store.add_chunks(chunks, document_id)
            logger.info(f"Stored vectors: {vector_stats}")
        except Exception as e:
            logger.error(f"Error storing in vector database: {e}")
            vector_stats = {"error": str(e)}

        # Return complete result with final processing time
        final_processing_time = int(time.time() - start_time)
        return URLProcessResponse(
            document_id=document_id,
            url=url_str,
            title=content_data.get("title"),
            content_type=content_type,
            processing_time_seconds=final_processing_time,
            phase1={
                "total_chars": len(full_text),
                "total_words": len(full_text.split()),
                "chunk_count": len(chunks),
                "chunking_method": chunking_method,
            },
            phase2={
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "graph_storage": graph_stats,
            },
            phase3={
                "original_length": compression_result["original_length"],
                "compressed_length": compression_result["compressed_length"],
                "compression_ratio": compression_result["compression_ratio"],
                "chunks_compressed": compression_result["chunks_compressed"],
                "speedup_factor": speedup_factor,
                "processing_time": compression_result["processing_time"],
                "strategy": compression_result["strategy"],
            },
            phase5={
                "vector_storage": vector_stats,
            },
        )

    except ConnectionError as e:
        logger.error(f"TGI connection error: {e}")
        _is_processing = False
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
        _is_processing = False
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
        # Always reset the processing flag on error
        _is_processing = False

        # Check if it's a rate limit error
        error_str = str(e).lower()
        if any(term in error_str for term in ["rate limit", "quota", "429", "resource exhausted", "requests per"]):
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Please wait before trying again. Error: {str(e)}"
            )

        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always reset the processing flag when done (success or failure)
        _is_processing = False

        # Clear processing status from Redis if document_id was created
        if 'document_id' in locals():
            clear_processing_status(document_id)


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
