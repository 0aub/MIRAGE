"""
Workflow State
Defines the state passed between LangGraph nodes
"""

from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
from operator import add


class WorkflowState(TypedDict):
    """State passed through the RAG workflow"""
    
    # Input
    query: str
    session_id: Optional[str]
    
    # Retrieval phase
    entities_found: List[Dict[str, Any]]
    subgraph: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    
    # Compression phase
    compressed_chunks: List[Dict[str, Any]]
    compression_stats: Dict[str, Any]
    
    # Generation phase
    context: str
    response: str
    citations: List[Dict[str, Any]]
    
    # Metadata
    workflow_step: str
    error: Optional[str]
    latency_ms: Dict[str, float]
    
    # Messages (for LangGraph message passing)
    messages: Annotated[List[Dict[str, Any]], add]
