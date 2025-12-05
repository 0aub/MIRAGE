"""
Chat Service API
Handles chat interactions with WebSocket support
V2: Uses unified RetrievalEngine with 7 modes and automatic routing
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from loguru import logger
from datetime import datetime
import time
import asyncio

from ..core.orchestration import RAGWorkflow
from ..core.retrieval import get_retrieval_engine, RetrievalMode
from ..core.generation import get_prompt_manager, get_response_generator

router = APIRouter()

# Initialize RAG workflow (for backward compatibility)
rag_workflow = RAGWorkflow()

# V2: Initialize retrieval engine with auto-routing
retrieval_engine = get_retrieval_engine()
prompt_manager = get_prompt_manager()
response_generator = get_response_generator()


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
    # V2: Retrieval mode selection
    retrieval_mode: Optional[str] = None  # auto, naive, local, global, hybrid, mix, semantic
    top_k: Optional[int] = 10


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
    V2 WebSocket endpoint for streaming RAG responses

    Protocol:
    - Client sends: {"query": "...", "mode": "auto|naive|hybrid|..."}
    - Server streams: {"type": "retrieval|chunk|done|error", "data": ...}

    Stream sequence:
    1. {"type": "retrieval", "data": {"mode": "...", "count": N}}
    2. {"type": "chunk", "data": {"text": "...", "index": N}}
    3. {"type": "done", "data": {"total_time_ms": N, "tokens": N}}
    """
    import json

    await websocket.accept()
    logger.info("V2 WebSocket connection established")

    try:
        while True:
            # Receive message from client
            raw_data = await websocket.receive_text()
            logger.info(f"WebSocket received: {raw_data[:100]}...")

            try:
                data = json.loads(raw_data)
                query = data.get("query", raw_data)
                mode_str = data.get("mode", "auto")
            except json.JSONDecodeError:
                query = raw_data
                mode_str = "auto"

            start_time = time.time()

            try:
                # 1. Parse retrieval mode
                mode = None
                if mode_str and mode_str != "auto":
                    try:
                        mode = RetrievalMode(mode_str)
                    except ValueError:
                        await websocket.send_json({
                            "type": "error",
                            "data": {"message": f"Invalid mode: {mode_str}"}
                        })
                        continue

                # 2. Retrieval step
                response = retrieval_engine.retrieve(query, mode=mode, top_k=5)

                await websocket.send_json({
                    "type": "retrieval",
                    "data": {
                        "mode": response.mode.value if response.mode else "unknown",
                        "count": len(response.results),
                        "time_ms": response.retrieval_time_ms
                    }
                })

                # 3. Build context and prompt
                context = [
                    {"text": r.text, "document_id": r.document_id, "chunk_id": r.chunk_id}
                    for r in response.results
                ]
                prompt = prompt_manager.create_qa_prompt(question=query, context=context)

                # 4. Stream response chunks
                # For now, simulate streaming with the full response
                # In production, this would use generator.generate_stream()
                full_response = f"Based on {len(context)} retrieved chunks:\n\n"
                for i, ctx in enumerate(context[:3]):
                    chunk_text = ctx["text"][:200] + "..." if len(ctx["text"]) > 200 else ctx["text"]
                    full_response += f"[{i+1}] {chunk_text}\n\n"

                # Stream in chunks
                words = full_response.split()
                chunk_size = 5
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i+chunk_size]
                    await websocket.send_json({
                        "type": "chunk",
                        "data": {
                            "text": " ".join(chunk_words) + " ",
                            "index": i // chunk_size
                        }
                    })
                    await asyncio.sleep(0.05)  # Small delay for streaming effect

                # 5. Send completion
                total_time = (time.time() - start_time) * 1000
                await websocket.send_json({
                    "type": "done",
                    "data": {
                        "total_time_ms": total_time,
                        "chunks_used": len(context),
                        "mode": response.mode.value if response.mode else "unknown"
                    }
                })

            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": str(e)}
                })

    except WebSocketDisconnect:
        logger.info("V2 WebSocket connection closed")


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
    """Get RAG workflow statistics including V2 components"""
    try:
        workflow_stats = rag_workflow.get_stats()

        # V2 component stats
        v2_stats = {
            "retrieval_engine": {
                "default_mode": retrieval_engine.config.default_mode.value,
                "auto_route": retrieval_engine.config.auto_route,
                "available_modes": [m.value for m in RetrievalMode],
            },
            "prompt_manager": {
                "template_count": len(prompt_manager.list_templates()),
            },
            "response_generator": {
                "mode": response_generator.config.mode.value,
                "temperature": response_generator.config.temperature,
            }
        }

        return {
            "workflow_status": "operational",
            "v2_components": v2_stats,
            "legacy_workflow": workflow_stats,
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


@router.post("/retrieve", response_model=Dict[str, Any])
async def retrieve_chunks(request: ChatRequest):
    """
    V2: Unified retrieval endpoint with 7 modes and automatic routing

    Modes:
    - auto (default): Automatically selects best mode based on query
    - naive: Simple vector search
    - local: Entity-focused (entity → chunks)
    - global: Relationship-focused (relationship → entity → chunks)
    - hybrid: Combines local + global
    - mix: All modes with RRF fusion
    - semantic: Deep semantic matching

    Returns detailed explanation of retrieval process
    """
    logger.info(f"V2 retrieval for query: {request.message[:100]}...")

    try:
        # Parse retrieval mode
        mode = None
        if request.retrieval_mode and request.retrieval_mode != "auto":
            try:
                mode = RetrievalMode(request.retrieval_mode)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid retrieval_mode: {request.retrieval_mode}. "
                           f"Valid modes: auto, naive, local, global, hybrid, mix, semantic"
                )

        # Get detailed explanation
        explanation = retrieval_engine.explain_retrieval(
            query=request.message,
            mode=mode,
            top_k=request.top_k or 10
        )

        # Format response
        return {
            "query": request.message,
            "routing": explanation["routing"],
            "mode_used": explanation["actual_mode"],
            "results": explanation["results"],
            "metadata": explanation["metadata"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in V2 retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrieval/modes")
async def list_retrieval_modes():
    """List available retrieval modes with descriptions"""
    return {
        "modes": [
            {"name": "auto", "description": "Automatic mode selection based on query analysis"},
            {"name": "naive", "description": "Simple vector similarity search"},
            {"name": "local", "description": "Entity-focused: query → entities → chunks"},
            {"name": "global", "description": "Relationship-focused: query → relationships → entities → chunks"},
            {"name": "hybrid", "description": "Combines local + global with RRF fusion"},
            {"name": "mix", "description": "All modes combined with weighted RRF"},
            {"name": "semantic", "description": "Deep semantic matching with re-ranking"},
        ],
        "default": retrieval_engine.config.default_mode.value,
        "auto_route": retrieval_engine.config.auto_route
    }


@router.post("/ask", response_model=Dict[str, Any])
async def ask_v2(request: ChatRequest):
    """
    V2 RAG endpoint: Full pipeline with retrieval + LLM generation

    Uses:
    - V2 RetrievalEngine with automatic mode routing
    - V2 PromptManager for context formatting
    - TGI (local LLM) for response generation

    Returns complete response with sources and timing.
    """
    import httpx

    start_time = time.time()
    logger.info(f"V2 RAG query: {request.message[:100]}...")

    try:
        # 1. Parse retrieval mode
        mode = None
        if request.retrieval_mode and request.retrieval_mode != "auto":
            try:
                mode = RetrievalMode(request.retrieval_mode)
            except ValueError:
                pass  # Use auto mode

        # 2. Retrieve relevant chunks
        retrieval_start = time.time()
        response = retrieval_engine.retrieve(
            query=request.message,
            mode=mode,
            top_k=request.top_k or 5
        )
        retrieval_time = (time.time() - retrieval_start) * 1000

        # 3. Build context
        context = [
            {"text": r.text, "document_id": r.document_id, "chunk_id": r.chunk_id}
            for r in response.results
        ]

        if not context:
            return {
                "query": request.message,
                "answer": "لم يتم العثور على معلومات ذات صلة في قاعدة البيانات.",
                "sources": [],
                "retrieval_mode": response.mode.value if response.mode else "unknown",
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": 0,
                "total_time_ms": (time.time() - start_time) * 1000
            }

        # 4. Check for global search (GraphRAG) - uses pre-generated answer
        if response.metadata and response.metadata.get("is_global_search"):
            global_answer = response.metadata.get("global_answer", "")
            if global_answer:
                generation_time = 0  # Already generated in global search
                return {
                    "query": request.message,
                    "answer": global_answer,
                    "sources": [
                        {
                            "document_id": r.document_id,
                            "chunk_id": r.chunk_id,
                            "text": r.text[:200] + "..." if len(r.text) > 200 else r.text
                        }
                        for r in response.results[:5]
                    ],
                    "retrieval_mode": "global_search",
                    "retrieval_time_ms": retrieval_time,
                    "generation_time_ms": generation_time,
                    "total_time_ms": (time.time() - start_time) * 1000,
                    "metadata": {
                        "communities_searched": response.metadata.get("communities_searched", 0),
                        "total_communities": response.metadata.get("total_communities", 0),
                        "themes": response.metadata.get("themes", []),
                        "confidence": response.metadata.get("confidence", 0)
                    }
                }

        # 5. Create prompt (with entity names for LOCAL mode)
        # Extract entity names from retrieval metadata for L2 entity lookup queries
        entity_names = response.metadata.get("entity_names", []) if response.metadata else []

        # If we have entity names and using LOCAL mode, inject them into context
        if entity_names and response.mode == RetrievalMode.LOCAL:
            # Add entity list as a special context entry
            entity_context = {
                "text": f"الكيانات المستخرجة (Extracted Entities): {', '.join(entity_names[:15])}",
                "document_id": "entity_list",
                "chunk_id": "entities"
            }
            context = [entity_context] + context

        prompt = prompt_manager.create_qa_prompt(
            question=request.message,
            context=context
        )

        # 6. Call TGI for generation
        generation_start = time.time()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                tgi_response = await client.post(
                    "http://tgi:80/generate",
                    json={
                        "inputs": f"{prompt.system_message}\n\n{prompt.user_message}",
                        "parameters": {
                            "max_new_tokens": 256,
                            "temperature": 0.3,
                            "do_sample": True,
                            "top_p": 0.9,
                            "return_full_text": False,
                            "stop": ["\n\n\n", "Question:", "Context:", "---"]
                        }
                    }
                )
                tgi_result = tgi_response.json()
                answer = tgi_result.get("generated_text", "").strip()
                # Clean up repetitions
                if "I cannot find" in answer:
                    parts = answer.split("I cannot find")
                    answer = parts[0].strip() + (" لا أجد هذه المعلومات في السياق المقدم." if not parts[0].strip() else "")
        except Exception as e:
            logger.error(f"TGI generation error: {e}")
            # Fallback to showing retrieved context
            answer = f"خطأ في الإنشاء. السياق المسترد:\n\n"
            for i, ctx in enumerate(context[:3]):
                answer += f"[{i+1}] {ctx['text'][:300]}...\n\n"

        generation_time = (time.time() - generation_start) * 1000
        total_time = (time.time() - start_time) * 1000

        # 6. Build sources
        sources = [
            {
                "chunk_id": ctx["chunk_id"],
                "document_id": ctx["document_id"],
                "text_preview": ctx["text"][:200] + "..." if len(ctx["text"]) > 200 else ctx["text"],
                "score": response.results[i].score if i < len(response.results) else 0
            }
            for i, ctx in enumerate(context[:5])
        ]

        result = {
            "query": request.message,
            "answer": answer,
            "sources": sources,
            "retrieval_mode": response.mode.value if response.mode else "unknown",
            "chunks_retrieved": len(context),
            "retrieval_time_ms": round(retrieval_time, 1),
            "generation_time_ms": round(generation_time, 1),
            "total_time_ms": round(total_time, 1)
        }

        # Add entity names for LOCAL mode
        if entity_names:
            result["entities_found"] = entity_names[:20]
            result["entity_count"] = len(entity_names)

        return result

    except Exception as e:
        logger.error(f"V2 RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
