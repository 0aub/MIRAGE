"""
Document Registry Service API
Lists all processed documents (files, URLs, YouTube videos)
from Neo4j graph database
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from loguru import logger
from datetime import datetime
import time

from ..core.graph_builder import Neo4jClient, EntityExtractor, RelationshipExtractor
from ..core.document_processor import TextChunker, get_chunker_factory
from ..core.embeddings import JinaEmbedder
from ..core.vector_store import QdrantVectorStore
from ..config import settings

router = APIRouter()

# Initialize Neo4j client
neo4j_client = Neo4jClient()

# Initialize components for reprocessing
entity_extractor = EntityExtractor()
relationship_extractor = RelationshipExtractor()
text_chunker = TextChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
chunker_factory = get_chunker_factory()
jina_embedder = JinaEmbedder()
vector_store = QdrantVectorStore()


# Models
class DocumentMetadata(BaseModel):
    document_id: str
    title: str
    content_type: str  # "file", "webpage", "youtube"
    url: Optional[str] = None
    author: Optional[str] = None
    language: Optional[str] = None
    entity_count: int
    relationship_count: int
    chunk_count: Optional[int] = None  # Number of chunks in vector store
    created_at: Optional[int] = None  # Unix timestamp in milliseconds
    total_chars: Optional[int] = None
    total_words: Optional[int] = None
    video_id: Optional[str] = None
    transcript_length: Optional[int] = None
    processing_time_seconds: Optional[int] = None  # Total processing time in seconds


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentMetadata]


# Endpoints
@router.get("/documents", response_model=DocumentListResponse)
async def list_all_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    content_type: Optional[str] = Query(None, description="Filter by type: file, webpage, or youtube"),
):
    """
    List all processed documents from Neo4j
    Includes files, web pages, and YouTube videos
    """
    logger.info(f"Listing documents: skip={skip}, limit={limit}, type={content_type}")

    try:
        # Get all documents from Neo4j
        all_docs = neo4j_client.get_all_documents()

        # Filter out documents with empty document_id (orphaned data from failed processing)
        all_docs = [doc for doc in all_docs if doc.get("document_id", "").strip()]

        # Filter by content type if specified
        if content_type:
            all_docs = [doc for doc in all_docs if doc.get("content_type") == content_type]

        # Apply pagination
        total = len(all_docs)
        paginated_docs = all_docs[skip : skip + limit]

        # Convert to response format
        documents = []
        for doc in paginated_docs:
            # Convert Neo4j timestamp (milliseconds) to readable format if needed
            created_at = doc.get("created_at")
            document_id = doc.get("document_id", "")

            # Note: Removed chunk_count query from list endpoint for performance
            # It was making individual Qdrant queries per document (slow)
            # Chunk count can be fetched on-demand in detail view if needed

            documents.append(DocumentMetadata(
                document_id=document_id,
                title=doc.get("title", "Untitled"),
                content_type=doc.get("content_type", "file"),
                url=doc.get("url"),
                author=doc.get("author"),
                language=doc.get("language"),
                entity_count=doc.get("entity_count", 0),
                relationship_count=doc.get("relationship_count", 0),
                chunk_count=None,  # Removed for performance - use total_words instead
                created_at=created_at,
                total_chars=doc.get("total_chars"),
                total_words=doc.get("total_words"),
                video_id=doc.get("video_id"),
                transcript_length=doc.get("transcript_length"),
                processing_time_seconds=doc.get("processing_time_seconds"),
            ))

        logger.info(f"Found {total} documents, returning {len(documents)}")

        return DocumentListResponse(
            total=total,
            documents=documents,
        )

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/documents/{document_id}/content")
async def get_document_content(document_id: str):
    """
    Get the full text content of a document
    """
    logger.info(f"Fetching content for document: {document_id}")

    try:
        # Ensure Neo4j is connected
        if not neo4j_client._connected:
            neo4j_client.connect()

        # Query for document content
        query = """
        MATCH (d:Document {document_id: $document_id})
        RETURN d.full_text as full_text, d.processed_text as processed_text, d.title as title
        """

        with neo4j_client.driver.session() as session:
            result = session.run(query, {"document_id": document_id})
            record = result.single()

            if not record or not record["full_text"]:
                raise HTTPException(
                    status_code=404,
                    detail="Document content not found or not available"
                )

            return {
                "document_id": document_id,
                "title": record["title"],
                "full_text": record["full_text"],
                "processed_text": record.get("processed_text"),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching document content {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch document content: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its associated data from both graph and vector stores
    """
    # Validate document_id
    if not document_id or not document_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID: document ID cannot be empty"
        )

    logger.info(f"Deleting document: {document_id}")

    try:
        # Delete from Neo4j
        graph_stats = neo4j_client.delete_by_document(document_id)
        logger.info(f"Deleted from Neo4j: {graph_stats}")

        # Delete from Qdrant vector store
        vector_stats = vector_store.delete_by_document(document_id)
        logger.info(f"Deleted from vector store: {vector_stats}")

        logger.info(f"Fully deleted document {document_id} from both databases")

        return {
            "document_id": document_id,
            "message": "Document deleted successfully from both databases",
            "stats": {
                "graph": graph_stats,
                "vector": vector_stats,
            },
        }

    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.post("/documents/cleanup-orphaned")
async def cleanup_orphaned_documents():
    """
    Clean up orphaned documents with empty or invalid document_ids
    This removes documents created from failed processing attempts
    """
    logger.info("Cleaning up orphaned documents")

    try:
        # Ensure Neo4j is connected
        if not neo4j_client._connected:
            neo4j_client.connect()

        # Query to find and delete documents with empty document_id
        query = """
        MATCH (d:Document)
        WHERE d.document_id IS NULL OR d.document_id = ''
        WITH d
        OPTIONAL MATCH (d)-[r]-()
        DELETE r, d
        RETURN count(DISTINCT d) as deleted_count
        """

        with neo4j_client.driver.session() as session:
            result = session.run(query)
            record = result.single()
            deleted_count = record["deleted_count"] if record else 0

        logger.info(f"Cleaned up {deleted_count} orphaned documents")

        return {
            "message": "Cleanup completed successfully",
            "deleted_count": deleted_count,
        }

    except Exception as e:
        logger.error(f"Error cleaning up orphaned documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup orphaned documents: {str(e)}")


@router.post("/documents/{document_id}/reprocess")
async def reprocess_document(document_id: str):
    """
    Reprocess a document with entity extraction using TGI
    - Fetches existing document content from Neo4j
    - Re-runs entity and relationship extraction
    - Updates the graph with new entities/relationships
    - Keeps vector store chunks unchanged

    Useful when:
    - Initial processing failed (0 entities/relationships)
    - TGI was down during original processing
    - You want to update extraction with better prompts
    """
    logger.info(f"Reprocessing document: {document_id}")
    start_time = time.time()

    try:
        # 1. Fetch document content and metadata from Neo4j
        if not neo4j_client._connected:
            neo4j_client.connect()

        query = """
        MATCH (d:Document {document_id: $document_id})
        RETURN d.full_text as full_text,
               d.title as title,
               d.content_type as content_type,
               d.total_chars as total_chars,
               d.total_words as total_words,
               d
        """

        with neo4j_client.driver.session() as session:
            result = session.run(query, {"document_id": document_id})
            record = result.single()

            if not record:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document {document_id} not found"
                )

            full_text = record["full_text"]
            if not full_text:
                raise HTTPException(
                    status_code=400,
                    detail=f"Document {document_id} has no text content to reprocess"
                )

            title = record["title"]
            content_type = record["content_type"]
            doc_props = dict(record["d"])

        logger.info(f"Found document: {title} ({len(full_text)} chars, type={content_type})")

        # 2. Re-chunk the text (use same chunking as original)
        logger.info("Re-chunking text...")
        chunker = chunker_factory.get_chunker(content_type, embedder=jina_embedder)
        chunks = chunker.chunk_text(full_text, {"title": title, "content_type": content_type})
        logger.info(f"Created {len(chunks)} chunks")

        # 3. Extract entities and relationships with TGI
        logger.info("Extracting entities and relationships (using TGI)...")
        entity_result = entity_extractor.extract_from_chunks(chunks, document_id=document_id)

        # Check extraction status
        extraction_status = entity_result.get("status", "success")
        if extraction_status == "unavailable":
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_unavailable",
                    "message": entity_result.get("message", "LLM entity extraction is currently unavailable"),
                    "suggestion": "TGI may be loading. Please wait and try again."
                }
            )

        entities = entity_result.get("entities", [])
        logger.info(f"Extracted {len(entities)} unique entities")

        # Get relationships
        extraction_method = entity_result.get("extraction_method", "")
        if "relationships" in entity_result and extraction_method.startswith("llm"):
            relationships = entity_result["relationships"]
            logger.info(f"LLM extracted {len(relationships)} relationships (method: {extraction_method})")
        else:
            logger.info(f"Extracting relationships using co-occurrence (LLM method was: {extraction_method})")
            relationships = relationship_extractor.extract_relationships(entities, full_text)
            logger.info(f"Extracted {len(relationships)} relationships")

        # 4. Delete old entities and relationships for this document
        logger.info("Deleting old entities and relationships...")
        delete_query = """
        MATCH (e:Entity {document_id: $document_id})
        DETACH DELETE e
        """
        with neo4j_client.driver.session() as session:
            session.run(delete_query, {"document_id": document_id})

        # 5. Store new graph
        logger.info("Storing new entities and relationships...")
        graph_stats = neo4j_client.store_graph(
            entities=entities,
            relationships=relationships,
            document_id=document_id,
            document_title=title,
            document_type=content_type,
            document_metadata={
                **doc_props,
                "reprocessed_at": int(time.time() * 1000),  # Unix timestamp in ms
                "reprocessing_time_seconds": int(time.time() - start_time),
            }
        )

        processing_time = int(time.time() - start_time)
        logger.info(f"Reprocessing complete in {processing_time}s: {graph_stats}")

        return {
            "document_id": document_id,
            "title": title,
            "message": "Document reprocessed successfully",
            "stats": {
                "chunks": len(chunks),
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "graph_storage": graph_stats,
                "processing_time_seconds": processing_time,
            }
        }

    except ConnectionError as e:
        logger.error(f"TGI connection error during reprocessing: {e}")
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
        logger.error(f"TGI timeout error during reprocessing: {e}")
        raise HTTPException(
            status_code=504,
            detail={
                "error": "tgi_timeout",
                "message": "TGI request timed out after 30 seconds",
                "suggestion": "The ALLaM model may be loading. Check TGI logs: docker logs mirage-tgi",
                "technical_details": str(e)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reprocess document: {str(e)}")
