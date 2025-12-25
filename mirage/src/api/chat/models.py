"""
Chat Service - Pydantic Models
Request/response models for chat API endpoints
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime


class Message(BaseModel):
    role: str  # user or assistant
    content: str
    sources: Optional[List[str]] = None
    timestamp: datetime = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    use_graph: bool = True
    language: Optional[str] = None
    retrieval_mode: Optional[str] = None  # auto, vector, local, global, hybrid, mix, semantic
    top_k: Optional[int] = 10
    use_hyde: bool = False
    enable_tracing: bool = False
    use_decomposition: bool = False


class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    sources: List[str]
    retrieved_nodes: int
    compression_rate: Optional[float] = None
    response_time_ms: int


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime


class V5Request(BaseModel):
    """Request for V5 unified retrieval."""
    message: str
    use_hyde: bool = True
    use_ppr: bool = True
    use_community_selection: bool = True
    use_dual_level: bool = True
    top_k: int = 10
