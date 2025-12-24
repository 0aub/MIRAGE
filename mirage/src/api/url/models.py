"""
URL Service - Pydantic Models
Request/response models for URL processing API
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, HttpUrl


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
    processing_time_seconds: int
    phase1: Dict[str, Any]
    phase2: Dict[str, Any]
    phase3: Dict[str, Any]
    phase5: Dict[str, Any]


class JobSubmitResponse(BaseModel):
    """Response for job submission"""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response for job status"""
    job_id: str
    status: str
    progress: float  # 0.0-1.0
    current_phase: Optional[str] = None
    current_chunk: Optional[int] = None
    total_chunks: Optional[int] = None
    document_id: Optional[str] = None
    title: Optional[str] = None
    content_type: Optional[str] = None
    url: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    elapsed_seconds: int
    error: Optional[str] = None
