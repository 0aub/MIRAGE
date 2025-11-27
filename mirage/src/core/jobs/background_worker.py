"""
Background Worker for Async Job Processing
Processes jobs in separate threads to avoid blocking HTTP requests
"""

import threading
import time
from typing import Callable, Dict, Any
from loguru import logger

from .job_manager import JobManager, JobStatus


class BackgroundWorker:
    """
    Background worker that processes jobs in separate threads
    """

    def __init__(self, job_manager: JobManager):
        """
        Initialize background worker

        Args:
            job_manager: JobManager instance for job tracking
        """
        self.job_manager = job_manager
        self.active_threads: Dict[str, threading.Thread] = {}
        logger.info("BackgroundWorker initialized")

    def submit_job(
        self,
        job_id: str,
        processing_function: Callable,
        *args,
        **kwargs
    ) -> bool:
        """
        Submit a job for background processing

        Args:
            job_id: Job ID
            processing_function: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            True if job submitted successfully
        """
        job = self.job_manager.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return False

        # Create and start background thread
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, processing_function, args, kwargs),
            daemon=True,
        )

        self.active_threads[job_id] = thread
        thread.start()

        logger.info(f"Job {job_id} submitted to background worker (thread: {thread.name})")
        return True

    def _run_job(
        self,
        job_id: str,
        processing_function: Callable,
        args: tuple,
        kwargs: dict
    ):
        """
        Run job in background thread

        Args:
            job_id: Job ID
            processing_function: Function to execute
            args: Function args
            kwargs: Function kwargs
        """
        try:
            # Update status to processing
            self.job_manager.update_job(
                job_id,
                status=JobStatus.PROCESSING,
                progress=0
            )

            logger.info(f"Starting job {job_id} in thread {threading.current_thread().name}")

            # Execute the processing function
            result = processing_function(job_id, *args, **kwargs)

            # Mark as completed
            self.job_manager.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress=100,
                result=result
            )

            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            # Mark as failed
            error_message = str(e)
            logger.error(f"Job {job_id} failed: {error_message}")

            self.job_manager.update_job(
                job_id,
                status=JobStatus.FAILED,
                error=error_message
            )

        finally:
            # Clean up thread reference
            if job_id in self.active_threads:
                del self.active_threads[job_id]

    def get_active_job_count(self) -> int:
        """Get number of actively running jobs"""
        # Clean up finished threads
        self.active_threads = {
            job_id: thread
            for job_id, thread in self.active_threads.items()
            if thread.is_alive()
        }

        return len(self.active_threads)

    def is_job_running(self, job_id: str) -> bool:
        """Check if a job is currently running"""
        thread = self.active_threads.get(job_id)
        return thread is not None and thread.is_alive()
