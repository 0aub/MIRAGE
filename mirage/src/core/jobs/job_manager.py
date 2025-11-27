"""
Job Manager for Async Background Processing
Uses Redis for job state tracking and coordination
"""

import json
import time
import uuid
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger
import redis


class JobStatus(str, Enum):
    """Job status states"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingPhase(str, Enum):
    """Processing phases for progress tracking"""
    FETCHING = "fetching"
    CHUNKING = "chunking"
    REWRITING = "rewriting"
    EXTRACTING = "extracting"
    STORING_GRAPH = "storing_graph"
    COMPRESSING = "compressing"
    STORING_VECTOR = "storing_vector"
    FINALIZING = "finalizing"


class ProcessingJob:
    """Job data model"""

    def __init__(
        self,
        job_id: str,
        job_type: str,
        status: JobStatus,
        url: Optional[str] = None,
        document_id: Optional[str] = None,
        title: Optional[str] = None,
        content_type: Optional[str] = None,
        progress: int = 0,
        current_phase: Optional[str] = None,
        current_chunk: int = 0,
        total_chunks: int = 0,
        created_at: float = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.job_id = job_id
        self.job_type = job_type
        self.status = status
        self.url = url
        self.document_id = document_id
        self.title = title
        self.content_type = content_type
        self.progress = progress
        self.current_phase = current_phase
        self.current_chunk = current_chunk
        self.total_chunks = total_chunks
        self.created_at = created_at or time.time()
        self.started_at = started_at
        self.completed_at = completed_at
        self.error = error
        self.result = result
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "url": self.url,
            "document_id": self.document_id,
            "title": self.title,
            "content_type": self.content_type,
            "progress": self.progress,
            "current_phase": self.current_phase,
            "current_chunk": self.current_chunk,
            "total_chunks": self.total_chunks,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": int(time.time() - self.started_at) if self.started_at else 0,
            "error": self.error,
            "result": self.result,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingJob":
        """Create job from dictionary"""
        # Remove computed fields that aren't constructor parameters
        data_copy = data.copy()
        data_copy.pop("elapsed_seconds", None)
        return cls(**data_copy)


class JobManager:
    """
    Manages background jobs using Redis for state persistence
    """

    def __init__(self, redis_host: str = "redis", redis_port: int = 6379, ttl: int = 86400):
        """
        Initialize job manager

        Args:
            redis_host: Redis host
            redis_port: Redis port
            ttl: Time-to-live for job data in seconds (default 24 hours)
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
        )
        self.ttl = ttl
        logger.info(f"JobManager initialized with Redis at {redis_host}:{redis_port}")

    def create_job(
        self,
        job_type: str,
        url: Optional[str] = None,
        document_id: Optional[str] = None,
        title: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProcessingJob:
        """
        Create a new job

        Args:
            job_type: Type of job (e.g., "url_processing", "file_processing")
            url: URL being processed (for URL jobs)
            document_id: Document ID
            title: Document title
            content_type: Content type (webpage, youtube, etc.)
            metadata: Additional metadata

        Returns:
            ProcessingJob instance
        """
        job_id = str(uuid.uuid4())

        job = ProcessingJob(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.QUEUED,
            url=url,
            document_id=document_id,
            title=title,
            content_type=content_type,
            metadata=metadata,
        )

        self._save_job(job)
        logger.info(f"Created job {job_id} of type {job_type}")

        return job

    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job by ID"""
        key = f"job:{job_id}"
        data = self.redis_client.get(key)

        if not data:
            return None

        try:
            job_dict = json.loads(data)
            return ProcessingJob.from_dict(job_dict)
        except Exception as e:
            logger.error(f"Error parsing job {job_id}: {e}")
            return None

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[int] = None,
        current_phase: Optional[str] = None,
        current_chunk: Optional[int] = None,
        total_chunks: Optional[int] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        Update job status and progress

        Args:
            job_id: Job ID
            status: New status
            progress: Progress percentage (0-100)
            current_phase: Current processing phase
            current_chunk: Current chunk being processed
            total_chunks: Total number of chunks
            error: Error message (for failed jobs)
            result: Final result (for completed jobs)
            **kwargs: Additional fields to update

        Returns:
            True if updated successfully
        """
        job = self.get_job(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found for update")
            return False

        # Update fields
        if status is not None:
            job.status = status

            # Update timestamps based on status
            if status == JobStatus.PROCESSING and not job.started_at:
                job.started_at = time.time()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                job.completed_at = time.time()

        if progress is not None:
            job.progress = min(100, max(0, progress))

        if current_phase is not None:
            job.current_phase = current_phase

        if current_chunk is not None:
            job.current_chunk = current_chunk

        if total_chunks is not None:
            job.total_chunks = total_chunks

        if error is not None:
            job.error = error

        if result is not None:
            job.result = result

        # Update any additional fields
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)

        self._save_job(job)
        return True

    def get_active_jobs(self) -> List[ProcessingJob]:
        """Get all active jobs (queued or processing)"""
        jobs = []

        # Scan for all job keys
        for key in self.redis_client.scan_iter("job:*"):
            job = self.get_job(key.replace("job:", ""))
            if job and job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                jobs.append(job)

        return jobs

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        return self.update_job(job_id, status=JobStatus.CANCELLED)

    def cleanup_old_jobs(self, max_age_seconds: int = 86400) -> int:
        """
        Clean up jobs older than max_age_seconds

        Args:
            max_age_seconds: Maximum age in seconds (default 24 hours)

        Returns:
            Number of jobs cleaned up
        """
        cleaned = 0
        current_time = time.time()

        for key in self.redis_client.scan_iter("job:*"):
            job = self.get_job(key.replace("job:", ""))
            if job and (current_time - job.created_at) > max_age_seconds:
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                    self.redis_client.delete(key)
                    cleaned += 1

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old jobs")

        return cleaned

    def _save_job(self, job: ProcessingJob):
        """Save job to Redis"""
        key = f"job:{job.job_id}"
        value = json.dumps(job.to_dict())
        self.redis_client.setex(key, self.ttl, value)
