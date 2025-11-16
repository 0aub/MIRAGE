"""
Chat Service API
Handles chat interactions with WebSocket support
Integrates Graph-RAG + REFRAG + Claude API
Phase 4: LangGraph Orchestration
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from loguru import logger
from datetime import datetime
import time

from ..core.orchestration import RAGWorkflow

router = APIRouter()

# Initialize RAG workflow
rag_workflow = RAGWorkflow()


# Models
class Message(BaseModel):
    role: str  # user or assistant
    content: str
    sources: Optional[List[str]] = None
    timestamp: datetime = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    use_graph: bool = True
    use_refrag: bool = True
    language: Optional[str] = None  # auto-detect if not provided


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


# Endpoints
@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Send a message and get AI response
    Pipeline: Query -> Graph Retrieval -> REFRAG Compression -> Claude API
    Phase 4: Full LangGraph orchestration
    """
    start_time = time.time()
    logger.info(f"Received message: {request.message[:50]}...")

    try:
        # Run RAG workflow
        result = rag_workflow.invoke(
            query=request.message,
            session_id=request.conversation_id,
        )

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Build sources from citations
        sources = []
        for citation in result.get("citations", []):
            citation_text = citation.get("text", "")
            if citation_text:
                sources.append(citation_text[:100])  # Truncate long sources

        # Get metadata
        metadata = result.get("metadata", {})
        compression_stats = metadata.get("compression_stats", {})

        return ChatResponse(
            message=result.get("response", "No response generated"),
            conversation_id=request.conversation_id or f"conv_{hash(request.message)}",
            sources=sources,
            retrieved_nodes=metadata.get("nodes_retrieved", 0),
            compression_rate=compression_stats.get("compression_ratio"),
            response_time_ms=response_time_ms,
        )

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat responses
    Allows real-time interaction with the system
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            logger.info(f"WebSocket received: {data[:50]}...")

            # TODO: Process message through pipeline
            # Stream response back in chunks

            # Mock streaming response
            response = f"Echo: {data}"
            await websocket.send_text(response)

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")


@router.get("/history/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_history(conversation_id: str):
    """Retrieve conversation history"""
    logger.info(f"Fetching history for conversation: {conversation_id}")

    # TODO: Query from database

    return ConversationHistory(
        conversation_id=conversation_id,
        messages=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@router.delete("/history/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and its history"""
    logger.info(f"Deleting conversation: {conversation_id}")

    # TODO: Implement deletion

    return {"message": f"Conversation {conversation_id} deleted successfully"}


@router.get("/conversations")
async def list_conversations(skip: int = 0, limit: int = 50):
    """List all conversations"""
    # TODO: Implement database query

    return {
        "total": 0,
        "conversations": [],
    }


@router.get("/workflow/stats")
async def get_workflow_stats():
    """Get RAG workflow statistics"""
    try:
        stats = rag_workflow.get_stats()
        return {
            "workflow_status": "operational",
            "statistics": stats,
        }
    except Exception as e:
        logger.error(f"Error getting workflow stats: {e}")
        return {
            "workflow_status": "error",
            "error": str(e),
        }


@router.post("/query-detailed", response_model=Dict[str, Any])
async def query_with_details(request: ChatRequest):
    """
    Query with detailed workflow information
    Returns full workflow state including all intermediate steps
    Plus graph visualization data and compression comparison
    """
    logger.info(f"Detailed query request: {request.message[:50]}...")

    try:
        # Run RAG workflow
        result = rag_workflow.invoke(
            query=request.message,
            session_id=request.conversation_id,
        )

        # Get workflow data including graph and compression
        metadata = result.get("metadata", {})
        graph_visualization = result.get("graph_visualization", {"nodes": [], "edges": []})
        compression_comparison = result.get("compression_comparison", {
            "original_chunks": [],
            "compressed_chunks": [],
            "stats": {},
        })

        return {
            "query": request.message,
            "response": result.get("response", ""),
            "citations": result.get("citations", []),
            "workflow_metadata": metadata,
            "graph_visualization": graph_visualization,
            "compression_comparison": compression_comparison,
            "success": result.get("success", False),
            "error": result.get("error"),
        }

    except Exception as e:
        logger.error(f"Error in detailed query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
