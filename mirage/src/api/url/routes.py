"""
URL Service - API Routes
FastAPI endpoints for URL processing
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
import json
import time
from typing import List, Dict, Any
import redis

from ...core.jobs import JobManager, BackgroundWorker, JobStatus
from ...config import settings

from .models import (
    URLRequest,
    URLPreviewResponse,
    URLProcessResponse,
    JobSubmitResponse,
    JobStatusResponse,
)
from .fetchers import extract_youtube_video_id, fetch_webpage_content, fetch_youtube_metadata_only
from .processor import process_url_background

router = APIRouter()

# Initialize Redis and Job Manager
try:
    redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    redis_client.ping()
    logger.info("Redis client initialized")
except Exception as e:
    logger.error(f"Failed to initialize Redis: {e}")
    redis_client = None

try:
    job_manager = JobManager(redis_host="redis", redis_port=6379, ttl=86400)
    background_worker = BackgroundWorker(job_manager)
    logger.info("Job manager initialized")
except Exception as e:
    logger.error(f"Failed to initialize job manager: {e}")
    job_manager = None
    background_worker = None


# Redis status helpers
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
        redis_client.setex(f"processing:{document_id}", 3600, json.dumps(status_data))
    except Exception as e:
        logger.error(f"Failed to set processing status: {e}")


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
                elapsed = int(time.time() - status_info["start_time"])
                status_info["elapsed_seconds"] = elapsed
                results.append(status_info)
        return results
    except Exception as e:
        logger.error(f"Failed to get processing statuses: {e}")
        return []


@router.post("/preview", response_model=URLPreviewResponse)
async def preview_url_content(request: URLRequest):
    """INSTANT preview of content from a URL"""
    url_str = str(request.url)
    logger.info(f"Preview request for URL: {url_str}")

    video_id = extract_youtube_video_id(url_str)

    if video_id:
        metadata = fetch_youtube_metadata_only(video_id)
        return URLPreviewResponse(
            url=url_str,
            title=metadata.get("title"),
            description=metadata.get("description"),
            content_preview=f"YouTube video by {metadata.get('author', 'Unknown')}",
            estimated_words=0,
            content_type="youtube",
        )
    else:
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


def _process_url_with_job_manager(job_id: str, url: str, use_semantic_chunking: bool):
    """Wrapper to pass job_manager to processor"""
    return process_url_background(job_id, url, use_semantic_chunking, job_manager)


@router.post("/process-async", response_model=JobSubmitResponse)
async def process_url_async(request: URLRequest):
    """Submit URL for asynchronous processing"""
    if not job_manager or not background_worker:
        raise HTTPException(status_code=503, detail="Job management system unavailable")

    url_str = str(request.url)
    logger.info(f"Submitting URL for async processing: {url_str}")

    try:
        job = job_manager.create_job(
            job_type="url_processing",
            url=url_str,
            metadata={"use_semantic_chunking": request.use_semantic_chunking}
        )

        background_worker.submit_job(
            job.job_id,
            _process_url_with_job_manager,
            url_str,
            request.use_semantic_chunking
        )

        return JobSubmitResponse(
            job_id=job.job_id,
            status="queued",
            message=f"Processing started. Poll /jobs/{job.job_id}/status for progress."
        )

    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process", response_model=URLProcessResponse)
async def process_url(request: URLRequest):
    """Process a URL synchronously (DEPRECATED)"""
    logger.warning("Using deprecated synchronous /process endpoint")

    url_str = str(request.url)

    try:
        result = process_url_background("sync", url_str, request.use_semantic_chunking)
        return URLProcessResponse(**result)

    except ConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail={"error": "tgi_unavailable", "message": str(e)}
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail={"error": "tgi_timeout", "message": str(e)}
        )
    except Exception as e:
        error_str = str(e).lower()
        if any(term in error_str for term in ["rate limit", "429"]):
            raise HTTPException(status_code=429, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    if not job_manager:
        raise HTTPException(status_code=503, detail="Job management unavailable")

    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job_dict = job.to_dict()
    job_dict["progress"] = job_dict["progress"] / 100.0

    return JobStatusResponse(**job_dict)


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get the final result of a completed job"""
    if not job_manager:
        raise HTTPException(status_code=503, detail="Job management unavailable")

    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status in [JobStatus.PROCESSING, JobStatus.QUEUED]:
        raise HTTPException(
            status_code=202,
            detail={"message": "Job still processing", "status": job.status, "progress": job.progress}
        )

    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail={"message": "Job failed", "error": job.error})

    if job.status == JobStatus.CANCELLED:
        raise HTTPException(status_code=410, detail="Job was cancelled")

    return job.result


@router.get("/jobs/active")
async def get_active_jobs():
    """Get all currently active jobs"""
    if not job_manager:
        raise HTTPException(status_code=503, detail="Job management unavailable")

    active_jobs = job_manager.get_active_jobs()

    return {
        "jobs": [job.to_dict() for job in active_jobs],
        "count": len(active_jobs)
    }


@router.get("/processing-status")
async def get_processing_status():
    """Get all currently processing documents from Redis (legacy)"""
    try:
        processing_items = get_all_processing()
        return {"processing": processing_items, "count": len(processing_items)}
    except Exception as e:
        logger.error(f"Error getting processing status: {e}")
        return {"processing": [], "count": 0}
