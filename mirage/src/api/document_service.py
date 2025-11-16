"""
Document Registry Service API
Lists all processed documents (files, URLs, YouTube videos)
from Neo4j graph database
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from loguru import logger
from datetime import datetime

from ..core.graph_builder import Neo4jClient

router = APIRouter()

# Initialize Neo4j client
neo4j_client = Neo4jClient()


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

            documents.append(DocumentMetadata(
                document_id=doc.get("document_id", ""),
                title=doc.get("title", "Untitled"),
                content_type=doc.get("content_type", "file"),
                url=doc.get("url"),
                author=doc.get("author"),
                language=doc.get("language"),
                entity_count=doc.get("entity_count", 0),
                relationship_count=doc.get("relationship_count", 0),
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
    Delete a document and all its associated data from the graph
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
        stats = neo4j_client.delete_by_document(document_id)

        logger.info(f"Deleted document {document_id}: {stats}")

        return {
            "document_id": document_id,
            "message": "Document deleted successfully",
            "stats": stats,
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
