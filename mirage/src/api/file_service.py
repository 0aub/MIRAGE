"""
File Management Service API
Handles file storage, listing, deletion, and metadata management
Provides UI integration for file operations
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from loguru import logger
from pathlib import Path
import os
import shutil
from datetime import datetime
import mimetypes

from ..config import settings
from ..core.graph_builder import Neo4jClient
from ..core.vector_store import QdrantVectorStore
from ..core.document_processor.pdf_processor import PDFProcessor
from ..core.document_processor.text_chunker import TextChunker
from ..core.document_processor.chunker_factory import ChunkerFactory
from ..core.graph_builder.entity_extractor import EntityExtractor
from ..core.graph_builder.relationship_extractor import RelationshipExtractor
from ..core.embeddings.jina_embedder import JinaEmbedder
import time

router = APIRouter()

# Initialize processors (lightweight, can be done at import time)
pdf_processor = PDFProcessor()
text_chunker = TextChunker()

# These will be lazily imported from url_service to avoid circular imports
_jina_embedder = None
_chunker_factory = None
_entity_extractor = None
_relationship_extractor = None
_job_manager = None
_background_worker = None


def _get_url_service_components():
    """Lazily import components from url_service to avoid circular imports"""
    global _jina_embedder, _chunker_factory, _entity_extractor, _relationship_extractor
    global _job_manager, _background_worker

    if _jina_embedder is None:
        from . import url_service
        _jina_embedder = url_service.jina_embedder
        _chunker_factory = url_service.chunker_factory
        _entity_extractor = url_service.entity_extractor
        _relationship_extractor = url_service.relationship_extractor
        _job_manager = url_service.job_manager
        _background_worker = url_service.background_worker

    return {
        'jina_embedder': _jina_embedder,
        'chunker_factory': _chunker_factory,
        'entity_extractor': _entity_extractor,
        'relationship_extractor': _relationship_extractor,
        'job_manager': _job_manager,
        'background_worker': _background_worker,
    }

# Initialize database clients for cleanup
neo4j_client = Neo4jClient()
vector_store = QdrantVectorStore()

# File storage configuration
UPLOAD_DIR = Path(settings.upload_dir if hasattr(settings, 'upload_dir') else "/data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Models
class FileMetadata(BaseModel):
    file_id: str
    filename: str
    file_type: str
    size: int
    uploaded_at: str
    path: str
    mime_type: Optional[str] = None


class FileListResponse(BaseModel):
    total: int
    files: List[FileMetadata]


class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    size: int
    message: str


class FileDeleteResponse(BaseModel):
    file_id: str
    filename: str
    message: str


# Endpoints
@router.get("/files", response_model=FileListResponse)
async def list_files(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    file_type: Optional[str] = None,
):
    """
    List all uploaded files with pagination
    Optionally filter by file type (pdf, html, json, etc.)
    """
    logger.info(f"Listing files: skip={skip}, limit={limit}, type={file_type}")

    try:
        # Get all files in upload directory
        all_files = []

        if UPLOAD_DIR.exists():
            for file_path in UPLOAD_DIR.rglob("*"):
                if file_path.is_file():
                    # Get file stats
                    stat = file_path.stat()
                    file_ext = file_path.suffix.lower().lstrip('.')

                    # Apply type filter if specified
                    if file_type and file_ext != file_type:
                        continue

                    # Get MIME type
                    mime_type, _ = mimetypes.guess_type(str(file_path))

                    file_meta = FileMetadata(
                        file_id=file_path.stem,
                        filename=file_path.name,
                        file_type=file_ext,
                        size=stat.st_size,
                        uploaded_at=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        path=str(file_path.relative_to(UPLOAD_DIR)),
                        mime_type=mime_type,
                    )
                    all_files.append(file_meta)

        # Sort by upload time (newest first)
        all_files.sort(key=lambda x: x.uploaded_at, reverse=True)

        # Apply pagination
        paginated_files = all_files[skip:skip + limit]

        return FileListResponse(
            total=len(all_files),
            files=paginated_files,
        )

    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{file_id}")
async def get_file_metadata(file_id: str):
    """
    Get metadata for a specific file
    """
    logger.info(f"Getting metadata for file: {file_id}")

    try:
        # Find file with matching ID
        for file_path in UPLOAD_DIR.rglob("*"):
            if file_path.is_file() and file_path.stem == file_id:
                stat = file_path.stat()
                mime_type, _ = mimetypes.guess_type(str(file_path))

                return {
                    "file_id": file_id,
                    "filename": file_path.name,
                    "file_type": file_path.suffix.lower().lstrip('.'),
                    "size": stat.st_size,
                    "uploaded_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": str(file_path.relative_to(UPLOAD_DIR)),
                    "mime_type": mime_type,
                    "absolute_path": str(file_path),
                }

        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/files/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a new file
    Supports PDF, HTML, JSON, TXT, and other document formats
    """
    logger.info(f"Uploading file: {file.filename}")

    try:
        # Generate unique file ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = f"{timestamp}_{Path(file.filename).stem}"
        file_ext = Path(file.filename).suffix

        # Create target path
        target_path = UPLOAD_DIR / f"{file_id}{file_ext}"

        # Save file
        with target_path.open("wb") as buffer:
            content = await file.read()
            buffer.write(content)

        file_size = len(content)

        logger.info(f"File uploaded: {target_path} ({file_size} bytes)")

        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            size=file_size,
            message=f"File {file.filename} uploaded successfully",
        )

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/files/upload-multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple files at once
    """
    logger.info(f"Uploading {len(files)} files")

    results = []
    errors = []

    for file in files:
        try:
            # Generate unique file ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_id = f"{timestamp}_{Path(file.filename).stem}"
            file_ext = Path(file.filename).suffix

            # Create target path
            target_path = UPLOAD_DIR / f"{file_id}{file_ext}"

            # Save file
            with target_path.open("wb") as buffer:
                content = await file.read()
                buffer.write(content)

            file_size = len(content)

            results.append({
                "file_id": file_id,
                "filename": file.filename,
                "size": file_size,
                "status": "success",
            })

        except Exception as e:
            logger.error(f"Error uploading file {file.filename}: {e}")
            errors.append({
                "filename": file.filename,
                "error": str(e),
            })

    return {
        "total": len(files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


@router.delete("/files/{file_id}", response_model=FileDeleteResponse)
async def delete_file(file_id: str):
    """
    Delete a file by its ID
    Also removes it from vector store and graph database
    """
    logger.info(f"Deleting file: {file_id}")

    try:
        # Find and delete file
        deleted = False
        deleted_filename = None

        for file_path in UPLOAD_DIR.rglob("*"):
            if file_path.is_file() and file_path.stem == file_id:
                deleted_filename = file_path.name
                file_path.unlink()
                deleted = True
                logger.info(f"Deleted file: {file_path}")
                break

        if not deleted:
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")

        # Delete from vector store and graph database
        try:
            neo4j_client.delete_by_document(file_id)
            vector_store.delete_by_document(file_id)
            logger.info(f"Deleted {file_id} from vector store and graph database")
        except Exception as e:
            logger.warning(f"Error cleaning up databases for {file_id}: {e}")

        return FileDeleteResponse(
            file_id=file_id,
            filename=deleted_filename,
            message=f"File {deleted_filename} deleted successfully from all systems",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/batch")
async def delete_multiple_files(file_ids: List[str]):
    """
    Delete multiple files at once
    """
    logger.info(f"Deleting {len(file_ids)} files")

    results = []
    errors = []

    for file_id in file_ids:
        try:
            deleted = False
            deleted_filename = None

            for file_path in UPLOAD_DIR.rglob("*"):
                if file_path.is_file() and file_path.stem == file_id:
                    deleted_filename = file_path.name
                    file_path.unlink()
                    deleted = True
                    break

            if deleted:
                # Delete from vector store and graph database
                try:
                    neo4j_client.delete_by_document(file_id)
                    vector_store.delete_by_document(file_id)
                    logger.info(f"Deleted {file_id} from databases")
                except Exception as e:
                    logger.warning(f"Error cleaning up databases for {file_id}: {e}")

                results.append({
                    "file_id": file_id,
                    "filename": deleted_filename,
                    "status": "deleted",
                })
            else:
                errors.append({
                    "file_id": file_id,
                    "error": "File not found",
                })

        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {e}")
            errors.append({
                "file_id": file_id,
                "error": str(e),
            })

    return {
        "total": len(file_ids),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


@router.get("/files/stats")
async def get_file_statistics():
    """
    Get statistics about uploaded files
    Total count, size, types, etc.
    """
    logger.info("Getting file statistics")

    try:
        total_files = 0
        total_size = 0
        file_types = {}

        if UPLOAD_DIR.exists():
            for file_path in UPLOAD_DIR.rglob("*"):
                if file_path.is_file():
                    total_files += 1
                    stat = file_path.stat()
                    total_size += stat.st_size

                    # Count by type
                    file_ext = file_path.suffix.lower().lstrip('.')
                    if file_ext:
                        file_types[file_ext] = file_types.get(file_ext, 0) + 1

        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_types": file_types,
            "upload_directory": str(UPLOAD_DIR),
        }

    except Exception as e:
        logger.error(f"Error getting file statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background file processing function
def process_file_background(job_id: str, file_path: str, file_id: str, file_type: str) -> Dict[str, Any]:
    """
    Process uploaded file in background thread with job status updates.
    Supports PDF, TXT, and other document formats.
    """
    start_time = time.time()

    # Get components lazily
    components = _get_url_service_components()
    jina_embedder = components['jina_embedder']
    chunker_factory = components['chunker_factory']
    entity_extractor = components['entity_extractor']
    relationship_extractor = components['relationship_extractor']
    job_manager = components['job_manager']

    try:
        logger.info(f"[Job {job_id}] Starting file processing: {file_path}")

        if job_manager:
            job_manager.update_job(job_id, progress=5, current_phase="reading")

        # Extract content based on file type
        if file_type == "pdf":
            result = pdf_processor.process(file_path)
            full_text = result["text"]
            title = result["metadata"].get("title") or Path(file_path).stem
            content_type = "pdf"
            file_metadata = result["metadata"]
        elif file_type in ["txt", "text"]:
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
            title = Path(file_path).stem
            content_type = "text"
            file_metadata = {"file_type": "text"}
        elif file_type in ["html", "htm"]:
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
            # Could add HTML parsing here
            title = Path(file_path).stem
            content_type = "webpage"
            file_metadata = {"file_type": "html"}
        else:
            # Try to read as text
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()
            title = Path(file_path).stem
            content_type = "document"
            file_metadata = {"file_type": file_type}

        if not full_text or len(full_text.strip()) < 50:
            raise Exception(f"Insufficient content extracted from file: {len(full_text)} chars")

        document_id = f"file_{file_id}"

        if job_manager:
            job_manager.update_job(
                job_id,
                document_id=document_id,
                title=title,
                content_type=content_type,
                progress=15,
                current_phase="chunking"
            )

        logger.info(f"[Job {job_id}] Extracted {len(full_text)} chars from {file_type}")

        # Phase 1: Chunking
        metadata = {
            "file_path": file_path,
            "file_id": file_id,
            "title": title,
            "content_type": content_type,
            **file_metadata,
        }

        chunker = chunker_factory.get_chunker(content_type, embedder=jina_embedder)
        chunks = chunker.chunk_text(full_text, metadata, document_id)

        logger.info(f"[Job {job_id}] Created {len(chunks)} chunks")

        if job_manager:
            job_manager.update_job(job_id, total_chunks=len(chunks), progress=25)

        # Create document in Neo4j
        try:
            initial_metadata = {
                "total_chars": len(full_text),
                "total_words": len(full_text.split()),
                "source_type": "file",
                "file_type": file_type,
                **file_metadata,
            }
            neo4j_client.store_document_metadata(
                document_id=document_id,
                title=title,
                content_type=content_type,
                metadata=initial_metadata
            )
            logger.info(f"[Job {job_id}] Created document in Neo4j: {document_id}")
        except Exception as e:
            logger.warning(f"[Job {job_id}] Failed to create document in Neo4j: {e}")

        # Phase 2: Entity extraction
        if job_manager:
            job_manager.update_job(job_id, progress=40, current_phase="extracting")

        logger.info(f"[Job {job_id}] Extracting entities from {len(chunks)} chunks")

        entity_result = entity_extractor.extract_with_chunk_references(chunks, document_id=document_id)

        extraction_status = entity_result.get("status", "success")
        if extraction_status == "unavailable":
            raise Exception("LLM entity extraction unavailable. TGI may still be loading.")
        elif extraction_status in ["rate_limited", "error"]:
            raise Exception(entity_result.get("message", "Entity extraction failed"))

        entities_with_chunks = entity_result.get("entities_with_chunks", [])
        all_entities = entity_result.get("all_entities", [])
        relationships = entity_result.get("relationships", [])

        if not relationships and all_entities:
            relationships = relationship_extractor.extract_relationships(all_entities, full_text)

        logger.info(f"[Job {job_id}] Extracted {len(entities_with_chunks)} entities, {len(relationships)} relationships")

        # Phase 3: Store in vector DB
        if job_manager:
            job_manager.update_job(job_id, progress=70, current_phase="storing_vectors")

        logger.info(f"[Job {job_id}] Storing {len(chunks)} chunks in vector DB")

        for i, chunk in enumerate(chunks):
            try:
                embedding = jina_embedder.embed_text(chunk.get("text", ""))
                vector_store.add_chunk(
                    chunk_id=chunk.get("chunk_id", f"{document_id}_chunk_{i}"),
                    document_id=document_id,
                    text=chunk.get("text", ""),
                    embedding=embedding,
                    metadata={
                        "chunk_index": i,
                        "title": title,
                        "content_type": content_type,
                        **chunk.get("metadata", {}),
                    }
                )

                if job_manager and i % 5 == 0:
                    job_manager.update_job(job_id, current_chunk=i)

            except Exception as e:
                logger.warning(f"[Job {job_id}] Failed to store chunk {i}: {e}")

        # Phase 4: Store in graph DB (using same method as url_service)
        if job_manager:
            job_manager.update_job(job_id, progress=85, current_phase="storing_graph")

        logger.info(f"[Job {job_id}] Storing entities and relationships in graph")

        try:
            graph_stats = neo4j_client.store_chunks_with_entities(
                chunks=chunks,
                entities=entities_with_chunks,
                relationships=relationships,
                document_id=document_id,
                document_title=title,
                document_type=content_type,
                document_metadata=metadata,
            )
            logger.info(f"[Job {job_id}] Stored graph: {graph_stats}")
        except Exception as e:
            logger.warning(f"[Job {job_id}] Failed to store in graph: {e}")

        # Done
        elapsed = time.time() - start_time

        if job_manager:
            job_manager.update_job(job_id, progress=100, current_phase="complete")

        result = {
            "document_id": document_id,
            "title": title,
            "content_type": content_type,
            "file_type": file_type,
            "chunks_count": len(chunks),
            "entities_count": len(entities_with_chunks),
            "relationships_count": len(relationships),
            "total_chars": len(full_text),
            "processing_time_seconds": round(elapsed, 2),
            "status": "success",
        }

        logger.info(f"[Job {job_id}] File processing complete: {result}")
        return result

    except Exception as e:
        logger.error(f"[Job {job_id}] File processing failed: {e}")
        # Let background_worker handle the failure status update
        raise


class FileProcessRequest(BaseModel):
    extract_entities: bool = True


class FileProcessResponse(BaseModel):
    job_id: str
    file_id: str
    status: str
    message: str


@router.post("/files/{file_id}/process", response_model=FileProcessResponse)
async def process_uploaded_file(file_id: str, request: FileProcessRequest = FileProcessRequest()):
    """
    Process an uploaded file for RAG ingestion.

    This extracts text, creates chunks, extracts entities and relationships,
    and stores everything in the vector and graph databases.

    Supports: PDF, TXT, HTML
    """
    logger.info(f"Processing file: {file_id}")

    # Find the file
    file_path = None
    file_type = None

    for fp in UPLOAD_DIR.rglob("*"):
        if fp.is_file() and fp.stem == file_id:
            file_path = str(fp)
            file_type = fp.suffix.lower().lstrip('.')
            break

    if not file_path:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    # Get components lazily
    components = _get_url_service_components()
    job_mgr = components['job_manager']
    bg_worker = components['background_worker']

    if not job_mgr or not bg_worker:
        raise HTTPException(status_code=503, detail="Job manager not available. Redis may be down.")

    # Create job
    job = job_mgr.create_job(
        job_type="file_processing",
        url=file_path,
        metadata={"file_id": file_id, "file_type": file_type}
    )

    # Submit to background worker
    bg_worker.submit_job(
        job.job_id,
        process_file_background,
        file_path,
        file_id,
        file_type
    )

    logger.info(f"Job {job.job_id} submitted for file {file_id}")

    return FileProcessResponse(
        job_id=job.job_id,
        file_id=file_id,
        status="queued",
        message=f"File {file_id} queued for processing. Check /url/jobs/{job.job_id}/status for progress."
    )
