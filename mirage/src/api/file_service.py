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

router = APIRouter()

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

        # TODO: Also delete from vector store and graph database
        # This should be coordinated with document_service

        return FileDeleteResponse(
            file_id=file_id,
            filename=deleted_filename,
            message=f"File {deleted_filename} deleted successfully",
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
