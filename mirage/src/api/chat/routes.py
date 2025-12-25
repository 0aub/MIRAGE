"""
Chat Service - API Routes
FastAPI endpoints for chat interactions with WebSocket support
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, Any
from loguru import logger
from datetime import datetime
import time
import asyncio
import json
import httpx

from ...core.orchestration import RAGWorkflow
from ...core.retrieval import get_retrieval_engine, RetrievalMode
from ...core.retrieval import get_v5_engine, get_observability
from ...core.retrieval import get_drift_search_engine
from ...core.retrieval import get_query_decomposer
from ...core.generation import get_prompt_manager, get_response_generator

from .models import (
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    V5Request,
)
from .helpers import (
    clean_llm_response,
    is_no_info_response,
    create_fallback_answer,
    rerank_chunks_by_keywords,
)

router = APIRouter()

# Initialize components
rag_workflow = RAGWorkflow()
retrieval_engine = get_retrieval_engine()
prompt_manager = get_prompt_manager()
response_generator = get_response_generator()
observability = get_observability()

# Lazy-initialized engines
v5_engine = None
drift_engine = None
query_decomposer = None


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """Send a message and get AI response via RAG workflow"""
    start_time = time.time()
    logger.info(f"Received message: {request.message[:50]}...")

    try:
        result = rag_workflow.invoke(
            query=request.message,
            session_id=request.conversation_id,
        )

        response_time_ms = int((time.time() - start_time) * 1000)

        sources = []
        for citation in result.get("citations", []):
            citation_text = citation.get("text", "")
            if citation_text:
                sources.append(citation_text[:100])

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
    """WebSocket endpoint for streaming RAG responses"""
    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        while True:
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

                response = retrieval_engine.retrieve(query, mode=mode, top_k=5)

                await websocket.send_json({
                    "type": "retrieval",
                    "data": {
                        "mode": response.mode.value if response.mode else "unknown",
                        "count": len(response.results),
                        "time_ms": response.retrieval_time_ms
                    }
                })

                context = [
                    {"text": r.text, "document_id": r.document_id, "chunk_id": r.chunk_id}
                    for r in response.results
                ]

                full_response = f"Based on {len(context)} retrieved chunks:\n\n"
                for i, ctx in enumerate(context[:3]):
                    chunk_text = ctx["text"][:200] + "..." if len(ctx["text"]) > 200 else ctx["text"]
                    full_response += f"[{i+1}] {chunk_text}\n\n"

                words = full_response.split()
                chunk_size = 5
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i+chunk_size]
                    await websocket.send_json({
                        "type": "chunk",
                        "data": {"text": " ".join(chunk_words) + " ", "index": i // chunk_size}
                    })
                    await asyncio.sleep(0.05)

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
                await websocket.send_json({"type": "error", "data": {"message": str(e)}})

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")


@router.get("/history/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_history(conversation_id: str):
    """Retrieve conversation history"""
    return ConversationHistory(
        conversation_id=conversation_id,
        messages=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@router.delete("/history/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and its history"""
    return {"message": f"Conversation {conversation_id} deleted successfully"}


@router.get("/conversations")
async def list_conversations(skip: int = 0, limit: int = 50):
    """List all conversations"""
    return {"total": 0, "conversations": []}


@router.get("/workflow/stats")
async def get_workflow_stats():
    """Get RAG workflow statistics"""
    try:
        workflow_stats = rag_workflow.get_stats()
        v2_stats = {
            "retrieval_engine": {
                "default_mode": retrieval_engine.config.default_mode.value,
                "auto_route": retrieval_engine.config.auto_route,
                "available_modes": [m.value for m in RetrievalMode],
            },
            "prompt_manager": {"template_count": len(prompt_manager.list_templates())},
            "response_generator": {
                "mode": response_generator.config.mode.value,
                "temperature": response_generator.config.temperature,
            }
        }
        return {"workflow_status": "operational", "v2_components": v2_stats, "legacy_workflow": workflow_stats}
    except Exception as e:
        logger.error(f"Error getting workflow stats: {e}")
        return {"workflow_status": "error", "error": str(e)}


@router.post("/query-detailed", response_model=Dict[str, Any])
async def query_with_details(request: ChatRequest):
    """Query with detailed workflow information"""
    try:
        result = rag_workflow.invoke(query=request.message, session_id=request.conversation_id)
        metadata = result.get("metadata", {})
        return {
            "query": request.message,
            "response": result.get("response", ""),
            "citations": result.get("citations", []),
            "workflow_metadata": metadata,
            "graph_visualization": result.get("graph_visualization", {"nodes": [], "edges": []}),
            "compression_comparison": result.get("compression_comparison", {}),
            "success": result.get("success", False),
            "error": result.get("error"),
        }
    except Exception as e:
        logger.error(f"Error in detailed query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve", response_model=Dict[str, Any])
async def retrieve_chunks(request: ChatRequest):
    """Unified retrieval endpoint with 7 modes and automatic routing"""
    try:
        mode = None
        if request.retrieval_mode and request.retrieval_mode != "auto":
            try:
                mode = RetrievalMode(request.retrieval_mode)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid retrieval_mode: {request.retrieval_mode}")

        explanation = retrieval_engine.explain_retrieval(
            query=request.message, mode=mode, top_k=request.top_k or 10
        )

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
        logger.error(f"Error in retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrieval/modes")
async def list_retrieval_modes():
    """List available retrieval modes"""
    return {
        "modes": [
            {"name": "auto", "description": "Automatic mode selection"},
            {"name": "vector", "description": "Simple vector similarity search"},
            {"name": "local", "description": "Entity-focused retrieval"},
            {"name": "global", "description": "Relationship-focused retrieval"},
            {"name": "global_search", "description": "GraphRAG map-reduce"},
            {"name": "hybrid", "description": "Local + global with RRF"},
            {"name": "mix", "description": "All modes combined"},
            {"name": "semantic", "description": "Deep semantic matching"},
            {"name": "drift", "description": "GraphRAG drift search"},
        ],
        "default": retrieval_engine.config.default_mode.value,
        "auto_route": retrieval_engine.config.auto_route
    }


@router.post("/ask", response_model=Dict[str, Any])
async def ask_v2(request: ChatRequest):
    """
    Unified RAG endpoint: Full pipeline with retrieval + LLM generation
    """
    global drift_engine, query_decomposer

    start_time = time.time()
    logger.info(f"RAG query: {request.message[:100]}...")

    trace = None
    trace_id = None
    if request.enable_tracing:
        trace = observability.start_trace(request.message)
        trace_id = trace.trace_id

    try:
        # Parse retrieval mode
        mode = None
        global_search_warning = None
        if request.retrieval_mode and request.retrieval_mode != "auto":
            try:
                mode = RetrievalMode(request.retrieval_mode)
                if mode == RetrievalMode.GLOBAL_SEARCH:
                    global_search_warning = "global_search mode is slow (~40s)"
            except ValueError:
                pass

        # Handle DRIFT mode
        if mode == RetrievalMode.DRIFT:
            if drift_engine is None:
                drift_engine = get_drift_search_engine()

            drift_start = time.time()
            drift_result = drift_engine.search(request.message)
            drift_time = (time.time() - drift_start) * 1000

            return {
                "query": request.message,
                "answer": drift_result.answer,
                "chunks": [{"chunk_id": f"drift_{i}", "text": c.get("text", "")[:500], "score": 0.8}
                           for i, c in enumerate(drift_result.local_context[:10])],
                "retrieval_mode": "drift",
                "retrieval_time_ms": round(drift_time, 1),
                "total_time_ms": round((time.time() - start_time) * 1000, 1),
            }

        # Handle query decomposition
        decomposition_info = None
        if request.use_decomposition:
            if query_decomposer is None:
                query_decomposer = get_query_decomposer()

            decomp_result = query_decomposer.decompose(request.message)
            if decomp_result.needs_decomposition:
                decomposition_info = {
                    "decomposed": True,
                    "sub_queries": [sq.query for sq in decomp_result.sub_queries],
                }

        # HyDE enhancement
        enhanced_query = None
        if request.use_hyde:
            try:
                from ...core.retrieval import get_hyde_enhancer
                hyde = get_hyde_enhancer()
                enhanced = hyde.enhance(request.message, mode="hypothetical")
                enhanced_query = enhanced.hypothetical_answer
            except Exception as e:
                logger.warning(f"HyDE failed: {e}")

        # Retrieve
        retrieval_start = time.time()
        response = retrieval_engine.retrieve(query=request.message, mode=mode, top_k=request.top_k or 5)
        retrieval_time = (time.time() - retrieval_start) * 1000

        reranked_results = rerank_chunks_by_keywords(request.message, response.results)

        context = [{"text": r.text, "document_id": r.document_id, "chunk_id": r.chunk_id}
                   for r in reranked_results]

        if not context:
            return {
                "query": request.message,
                "answer": "لم يتم العثور على معلومات ذات صلة.",
                "chunks": [],
                "retrieval_mode": response.mode.value if response.mode else "unknown",
                "chunks_retrieved": 0,
                "retrieval_time_ms": retrieval_time,
                "total_time_ms": (time.time() - start_time) * 1000
            }

        # Handle global search pre-generated answer
        if response.metadata and response.metadata.get("is_global_search"):
            global_answer = response.metadata.get("global_answer", "")
            if global_answer:
                return {
                    "query": request.message,
                    "answer": global_answer,
                    "chunks": [{"chunk_id": r.chunk_id, "text": r.text[:200]} for r in response.results[:10]],
                    "retrieval_mode": "global_search",
                    "retrieval_time_ms": retrieval_time,
                    "total_time_ms": (time.time() - start_time) * 1000,
                }

        # Build prompt
        entity_names = response.metadata.get("entity_names", []) if response.metadata else []
        if entity_names and response.mode == RetrievalMode.LOCAL:
            entity_context = {"text": f"الكيانات: {', '.join(entity_names[:15])}", "document_id": "entity_list", "chunk_id": "entities"}
            context = [entity_context] + context

        prompt = prompt_manager.create_qa_prompt(question=request.message, context=context)

        # LLM generation
        generation_start = time.time()
        try:
            timeout_config = httpx.Timeout(timeout=90.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                tgi_response = await client.post(
                    "http://tgi:80/generate",
                    json={
                        "inputs": f"{prompt.system_message}\n\n{prompt.user_message}",
                        "parameters": {
                            "max_new_tokens": 200,
                            "temperature": 0.3,
                            "do_sample": True,
                            "top_p": 0.9,
                            "repetition_penalty": 1.15,
                            "return_full_text": False,
                            "stop": ["\n\n\n", "Human:", "---", "Question:"]
                        }
                    }
                )
                raw_answer = tgi_response.json().get("generated_text", "").strip()
                answer = clean_llm_response(raw_answer)

                if is_no_info_response(answer) and context:
                    answer = create_fallback_answer(request.message, context)

        except httpx.TimeoutException:
            answer = create_fallback_answer(request.message, context) if context else "انتهت مهلة الطلب."
        except Exception as e:
            logger.error(f"TGI error: {e}")
            answer = create_fallback_answer(request.message, context) if context else "خطأ في المعالجة."

        generation_time = (time.time() - generation_start) * 1000
        total_time = (time.time() - start_time) * 1000

        # Build chunks with source type
        chunks = []
        vector_count = graph_1hop = graph_2hop = 0
        for r in reranked_results[:10]:
            hop = r.hop_distance if hasattr(r, 'hop_distance') else 0
            if hop == 0:
                source_type = "vector"
                vector_count += 1
            elif hop == 1:
                source_type = "graph_1hop"
                graph_1hop += 1
            else:
                source_type = "graph_2hop"
                graph_2hop += 1

            chunks.append({
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "text": r.text,
                "score": r.score,
                "source_type": source_type,
                "via_entity": r.via_entity if hasattr(r, 'via_entity') else None,
            })

        result = {
            "query": request.message,
            "answer": answer,
            "chunks": chunks,
            "retrieval_mode": response.mode.value if response.mode else "unknown",
            "chunks_retrieved": len(context),
            "retrieval_time_ms": round(retrieval_time, 1),
            "generation_time_ms": round(generation_time, 1),
            "total_time_ms": round(total_time, 1),
            "retrieval_stats": {
                "vector_chunks": vector_count,
                "graph_1hop_chunks": graph_1hop,
                "graph_2hop_chunks": graph_2hop,
            },
        }

        if entity_names:
            result["entities_found"] = entity_names[:20]
        if enhanced_query:
            result["enhanced_query"] = enhanced_query
        if trace_id:
            result["trace_id"] = trace_id
        if global_search_warning:
            result["warning"] = global_search_warning
        if decomposition_info:
            result["decomposition"] = decomposition_info

        if trace:
            observability.end_trace(trace, success=True)

        return result

    except Exception as e:
        if trace:
            observability.end_trace(trace, success=False, error=str(e))
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v5/ask", response_model=Dict[str, Any])
async def ask_v5(request: V5Request):
    """MIRAGE V5 RAG endpoint with all SOTA innovations"""
    global v5_engine

    start_time = time.time()
    trace = observability.start_trace(request.message)

    try:
        if v5_engine is None:
            v5_engine = get_v5_engine()

        v5_engine.config.use_hyde = request.use_hyde
        v5_engine.config.use_hippocampal = request.use_ppr
        v5_engine.config.use_dynamic_community_selection = request.use_community_selection
        v5_engine.config.use_dual_level = request.use_dual_level
        v5_engine.config.max_chunks = request.top_k

        with observability.record_step(trace, "v5_retrieval") as step:
            result = v5_engine.retrieve(request.message)
            step.metadata["chunks"] = len(result.chunks)

        with observability.record_step(trace, "generation") as step:
            context = [{"text": c.get("text", ""), "document_id": c.get("document_id", ""), "chunk_id": c.get("chunk_id", "")}
                       for c in result.chunks]

            if context:
                prompt = prompt_manager.create_qa_prompt(question=request.message, context=context)
                timeout_config = httpx.Timeout(timeout=90.0, connect=10.0)
                async with httpx.AsyncClient(timeout=timeout_config) as client:
                    llm_response = await client.post(
                        "http://tgi:80/generate",
                        json={
                            "inputs": f"{prompt.system_message}\n\n{prompt.user_message}",
                            "parameters": {"max_new_tokens": 200, "temperature": 0.3, "do_sample": True}
                        }
                    )
                    answer = clean_llm_response(llm_response.json().get("generated_text", "").strip())
            else:
                answer = "لم يتم العثور على معلومات."

        total_time = (time.time() - start_time) * 1000
        observability.end_trace(trace, success=True)

        return {
            "query": request.message,
            "answer": answer,
            "chunks": [{"chunk_id": c.get("chunk_id", ""), "text": c.get("text", "")[:300]} for c in result.chunks[:5]],
            "metadata": {
                "enhanced_query": result.enhanced_query,
                "entities_found": result.entities_found,
                "communities_searched": result.communities_searched,
            },
            "timing": {"retrieval_ms": result.retrieval_time_ms, "total_ms": total_time},
            "trace_id": trace.trace_id,
        }

    except Exception as e:
        observability.end_trace(trace, success=False, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v5/metrics")
async def v5_metrics():
    """Get V5 engine performance metrics"""
    metrics = observability.get_metrics()
    dashboard = observability.get_dashboard_data()

    return {
        "metrics": {
            "total_queries": metrics.total_queries,
            "avg_latency_ms": round(metrics.avg_latency_ms, 2),
            "p95_latency_ms": round(metrics.p95_latency_ms, 2),
            "error_rate": round(metrics.error_rate * 100, 2),
        },
        "mode_distribution": metrics.mode_distribution,
        "recent_traces": dashboard.get("recent_traces", []),
    }
