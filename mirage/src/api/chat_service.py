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
from ..core.retrieval import get_v5_engine, V5Config, get_observability
from ..core.retrieval import get_drift_search_engine, DriftConfig
from ..core.retrieval import get_query_decomposer, DecomposeAndRetrieve
from ..core.generation import get_prompt_manager, get_response_generator
from ..core.refrag.compressor import REFRAGCompressor

router = APIRouter()

# Initialize RefRAG compressor
refrag_compressor = REFRAGCompressor(strategy="hybrid")

# Initialize RAG workflow (for backward compatibility)
rag_workflow = RAGWorkflow()

# V2: Initialize retrieval engine with auto-routing
retrieval_engine = get_retrieval_engine()
prompt_manager = get_prompt_manager()
response_generator = get_response_generator()

# V5: Initialize unified engine with all SOTA innovations
v5_engine = None  # Lazy init to avoid startup issues
observability = get_observability()

# GraphRAG: Drift search engine (lazy init)
drift_engine = None

# GraphRAG Phase 7: Query decomposer (lazy init)
query_decomposer = None


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
    # Retrieval mode selection
    retrieval_mode: Optional[str] = None  # auto, naive, local, global, hybrid, mix, semantic
    top_k: Optional[int] = 10
    # V5 features (optional enhancements)
    use_hyde: bool = False  # HyDE query enhancement
    enable_tracing: bool = False  # Observability tracing
    # GraphRAG Phase 7: Query decomposition for complex queries
    use_decomposition: bool = False  # Decompose multi-hop queries into sub-queries


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
            {"name": "global_search", "description": "GraphRAG map-reduce over community summaries"},
            {"name": "hybrid", "description": "Combines local + global with RRF fusion"},
            {"name": "mix", "description": "All modes combined with weighted RRF"},
            {"name": "semantic", "description": "Deep semantic matching with re-ranking"},
            {"name": "drift", "description": "GraphRAG drift search: dynamic global→local with claims"},
        ],
        "default": retrieval_engine.config.default_mode.value,
        "auto_route": retrieval_engine.config.auto_route
    }


def _clean_llm_response(answer: str) -> str:
    """
    Clean up LLM response artifacts:
    - Remove citation markers like [1], [2], etc.
    - Remove repeated content (question echo, etc.)
    - Clean up "I cannot find" patterns
    """
    import re

    if not answer:
        return answer

    # Remove citation markers like [1], [2], etc.
    answer = re.sub(r'\[\d+\]', '', answer)

    # Truncate at prompt echo patterns (model repeating the conversation)
    for pattern in ["Human:", "Question:", "Context:"]:
        if pattern in answer:
            answer = answer.split(pattern)[0].strip()

    # Remove "I cannot find" patterns
    if "I cannot find" in answer:
        parts = answer.split("I cannot find")
        answer = parts[0].strip()
        if not answer:
            return "لا أجد هذه المعلومات في السياق المقدم."

    # Clean up multiple spaces
    answer = re.sub(r'[ \t]+', ' ', answer).strip()

    # Remove trailing incomplete sentences (ends with comma or nothing)
    if answer and not answer[-1] in '.؟!':
        # Find last complete sentence
        for end_marker in ['. ', '؟ ', '! ']:
            if end_marker in answer:
                last_idx = answer.rfind(end_marker)
                if last_idx > len(answer) * 0.5:  # Only if we keep >50% of content
                    answer = answer[:last_idx + 1]
                    break

    return answer


def _is_no_info_response(answer: str) -> bool:
    """
    Detect if the LLM response indicates "no information found" or is too generic.
    This triggers fallback to chunk summarization.

    Phase 3: More nuanced detection - only flag if answer is PURELY negative,
    not if it leads with negative but then provides useful content.
    """
    if not answer or len(answer) < 30:
        return True

    # Pure "no info" patterns - these indicate a complete failure to answer
    pure_no_info_patterns = [
        # Arabic patterns - these indicate complete failure
        "لا أجد أي معلومات",
        "لم أجد معلومات",
        "لا توجد معلومات متاحة",
        "لا تتوفر معلومات",
        "لم يتم العثور على أي",
        "لا يمكنني الإجابة",
        "لا أستطيع الإجابة",
        "لا يوجد في السياق المقدم",
        # English patterns
        "i cannot find any",
        "no information available",
        "unable to find any",
        "cannot answer this",
    ]

    answer_lower = answer.lower()

    # Check if answer is PURELY a "no info" statement (short + negative)
    if len(answer.strip()) < 80:
        if any(pattern in answer_lower for pattern in pure_no_info_patterns):
            return True

    # Phase 3: If answer is longer (>100 chars), it likely has useful content
    # even if it starts with "لا يوجد ذكر مباشر" etc.
    if len(answer.strip()) > 100:
        return False

    # Short answers (<50 chars) that aren't useful
    if len(answer.strip()) < 50:
        return True

    # Detect if answer is just repeating the question
    if answer.count("؟") > 1:
        return True

    return False


def _extract_key_terms(question: str) -> list:
    """
    Extract key terms (entities/nouns) from Arabic/English question.
    Used to create better fallback answers.
    """
    import re

    # Remove question words and stopwords
    arabic_stopwords = [
        "ما", "من", "هل", "كيف", "لماذا", "أين", "متى", "هي", "هو", "هذا", "هذه",
        "التي", "الذي", "في", "على", "إلى", "عن", "مع", "بين", "أو", "و", "ال",
        "أن", "إن", "كان", "يكون", "له", "لها", "هم", "هن", "نحن", "أنت", "أنا"
    ]

    # Clean and tokenize
    words = re.findall(r'[\u0600-\u06FF]+|[a-zA-Z]+', question)

    # Filter out stopwords and short words
    key_terms = []
    for word in words:
        # Skip short words and stopwords
        if len(word) < 3:
            continue
        if word in arabic_stopwords:
            continue
        # Remove common Arabic prefixes (ال، ب، و، ف، ك، ل)
        clean_word = word
        if clean_word.startswith('ال'):
            clean_word = clean_word[2:]
        if clean_word.startswith(('ب', 'و', 'ف', 'ك', 'ل')) and len(clean_word) > 3:
            clean_word = clean_word[1:]
        if len(clean_word) >= 2:
            key_terms.append(clean_word)

    return key_terms


def _normalize_arabic(text: str) -> str:
    """Normalize Arabic text for matching."""
    import re
    if not text:
        return ""
    # Normalize ة to ه
    text = text.replace('ة', 'ه')
    # Normalize alef variants
    text = re.sub(r'[أإآ]', 'ا', text)
    return text


def _create_fallback_answer(question: str, chunks: list) -> str:
    """
    Create a fallback answer by summarizing the top chunks.
    Improved: Extracts key terms and finds relevant snippets.
    Used when LLM fails to synthesize an answer.
    """
    if not chunks:
        return "لم يتم العثور على معلومات ذات صلة."

    # Extract key terms from question
    key_terms = _extract_key_terms(question)
    normalized_terms = [_normalize_arabic(t) for t in key_terms]

    # Take top 3 chunks
    top_chunks = chunks[:3]

    # Find the most relevant snippet from each chunk
    answer_parts = []

    # Create context-aware intro if we found key terms
    if key_terms:
        main_entity = key_terms[0] if key_terms else ""
        answer_parts.append(f"بخصوص {main_entity}، إليك المعلومات المتوفرة:")
    else:
        answer_parts.append("بناءً على المعلومات المتوفرة:")

    for i, chunk in enumerate(top_chunks, 1):
        text = chunk.get("text", "")
        if not text:
            continue

        normalized_text = _normalize_arabic(text)

        # Try to find a snippet containing a key term
        best_snippet = None
        best_score = 0

        # Split into sentences (Arabic uses . and ،)
        sentences = text.replace('،', '.').replace('؟', '.').split('.')

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            norm_sentence = _normalize_arabic(sentence)
            score = sum(1 for t in normalized_terms if t in norm_sentence)

            if score > best_score:
                best_score = score
                best_snippet = sentence

        # Use best snippet or first 200 chars
        if best_snippet:
            snippet = best_snippet[:250].strip()
        else:
            snippet = text[:200].strip()

        # Clean up snippet ending
        if len(snippet) > 200:
            last_space = snippet.rfind(' ')
            if last_space > 100:
                snippet = snippet[:last_space]
            snippet += "..."

        answer_parts.append(f"\n• {snippet}")

    return "".join(answer_parts)


def _rerank_chunks_by_keywords(question: str, results: list, boost_factor: float = 0.3) -> list:
    """
    Re-rank retrieval results by boosting chunks that contain query keywords.
    This helps when vector similarity misses exact keyword matches.

    Args:
        question: User's question
        results: List of retrieval results (with .text and .score attributes)
        boost_factor: Score boost per keyword match (0.0-1.0)

    Returns:
        Re-ranked results list
    """
    if not results:
        return results

    # Extract key terms from question
    key_terms = _extract_key_terms(question)
    if not key_terms:
        return results

    normalized_terms = [_normalize_arabic(t) for t in key_terms]

    # Calculate keyword boost for each result
    boosted_results = []
    for r in results:
        text = r.text if hasattr(r, 'text') else ""
        normalized_text = _normalize_arabic(text)

        # Count keyword matches
        match_count = sum(1 for t in normalized_terms if t in normalized_text)

        # Calculate boost (cap at 0.5 to avoid overwhelming vector score)
        keyword_boost = min(match_count * boost_factor, 0.5)

        # Create a tuple of (result, original_score, boosted_score)
        original_score = r.score if hasattr(r, 'score') else 0.5
        boosted_score = original_score + keyword_boost

        boosted_results.append((r, boosted_score, match_count))

    # Sort by boosted score (descending)
    boosted_results.sort(key=lambda x: x[1], reverse=True)

    # Log re-ranking if it changed order
    if boosted_results:
        reranked_count = sum(1 for i, (r, _, mc) in enumerate(boosted_results) if mc > 0)
        if reranked_count > 0:
            logger.debug(f"Re-ranked {reranked_count} chunks by keyword match ({key_terms[:3]})")

    # Return re-ordered results
    return [r for r, _, _ in boosted_results]


@router.post("/ask", response_model=Dict[str, Any])
async def ask_v2(request: ChatRequest):
    """
    Unified RAG endpoint: Full pipeline with retrieval + LLM generation

    Uses:
    - RetrievalEngine with automatic mode routing
    - PromptManager for context formatting
    - TGI (local LLM) for response generation

    Optional V5 Features:
    - use_hyde: HyDE query enhancement for better semantic matching
    - enable_tracing: Full observability with trace IDs

    Returns complete response with sources and timing.
    """
    import httpx

    start_time = time.time()
    logger.info(f"RAG query: {request.message[:100]}... (hyde={request.use_hyde})")

    # Start trace if enabled
    trace = None
    trace_id = None
    if request.enable_tracing:
        trace = observability.start_trace(request.message)
        trace_id = trace.trace_id

    try:
        # 1. Parse retrieval mode
        mode = None
        global_search_warning = None
        if request.retrieval_mode and request.retrieval_mode != "auto":
            try:
                mode = RetrievalMode(request.retrieval_mode)
                # WARN about global_search latency
                if mode == RetrievalMode.GLOBAL_SEARCH:
                    global_search_warning = "global_search mode is experimental and slow (~40s). Consider using 'local' or 'hybrid' for better latency."
                    logger.warning(f"global_search requested - {global_search_warning}")
            except ValueError:
                pass  # Use auto mode

        # 1b. Handle DRIFT mode specially (GraphRAG drift search)
        if mode == RetrievalMode.DRIFT:
            global drift_engine
            if drift_engine is None:
                logger.info("Initializing Drift Search Engine...")
                drift_engine = get_drift_search_engine()

            drift_start = time.time()
            drift_result = drift_engine.search(request.message)
            drift_time = (time.time() - drift_start) * 1000
            total_time = (time.time() - start_time) * 1000

            return {
                "query": request.message,
                "answer": drift_result.answer,
                "chunks": [
                    {
                        "chunk_id": c.get("chunk_id", f"drift_{i}"),
                        "document_id": c.get("document_id", "unknown"),
                        "text": c.get("text", "")[:500],
                        "score": 0.8,
                        "source_type": "drift_local",
                        "via_entity": c.get("entity_name")
                    }
                    for i, c in enumerate(drift_result.local_context[:10])
                ],
                "sources": [
                    {"text": s[:300], "type": "community_summary"}
                    for s in drift_result.global_context[:5]
                ],
                "retrieval_mode": "drift",
                "retrieval_time_ms": round(drift_time, 1),
                "generation_time_ms": 0,  # Included in drift search
                "total_time_ms": round(total_time, 1),
                "metadata": {
                    "drift_phases": [p.value for p in drift_result.phases_executed],
                    "drift_path": drift_result.drift_path,
                    "communities_visited": len(drift_result.communities_visited),
                    "entities_traversed": drift_result.entities_traversed[:10],
                    "claims_used": len(drift_result.claims_used),
                    "confidence": drift_result.confidence
                }
            }

        # 1c. Handle query decomposition for complex queries (GraphRAG Phase 7)
        decomposition_info = None
        if request.use_decomposition:
            global query_decomposer
            if query_decomposer is None:
                logger.info("Initializing Query Decomposer...")
                query_decomposer = get_query_decomposer()

            decomposition_start = time.time()
            decomposition_result = query_decomposer.decompose(request.message)
            decomposition_time = (time.time() - decomposition_start) * 1000

            if decomposition_result.needs_decomposition:
                logger.info(f"Query decomposed into {len(decomposition_result.sub_queries)} sub-queries")
                decomposition_info = {
                    "decomposed": True,
                    "query_type": decomposition_result.query_type.value,
                    "sub_queries": [
                        {
                            "query": sq.query,
                            "type": sq.query_type,
                            "target_entities": sq.target_entities,
                            "reasoning": sq.reasoning
                        }
                        for sq in decomposition_result.sub_queries
                    ],
                    "combination_strategy": decomposition_result.combination_strategy,
                    "reasoning": decomposition_result.reasoning,
                    "decomposition_time_ms": round(decomposition_time, 1)
                }
            else:
                decomposition_info = {
                    "decomposed": False,
                    "query_type": "simple",
                    "reasoning": "Query is simple enough to answer directly"
                }

        # 2. Optional: HyDE query enhancement
        enhanced_query = None
        if request.use_hyde:
            try:
                from ..core.retrieval import get_hyde_enhancer
                hyde = get_hyde_enhancer()
                enhanced = hyde.enhance(request.message, mode="hypothetical")
                enhanced_query = enhanced.hypothetical_answer
                logger.info(f"HyDE enhanced query: {enhanced_query[:100]}...")
            except Exception as e:
                logger.warning(f"HyDE enhancement failed: {e}")

        # 3. Retrieve relevant chunks
        retrieval_start = time.time()
        response = retrieval_engine.retrieve(
            query=request.message,
            mode=mode,
            top_k=request.top_k or 5
        )
        retrieval_time = (time.time() - retrieval_start) * 1000

        # 3b. Re-rank results by keyword match (Phase 2.2)
        # This boosts chunks containing exact query keywords
        reranked_results = _rerank_chunks_by_keywords(request.message, response.results)

        # 4. Build context from re-ranked results
        context = [
            {"text": r.text, "document_id": r.document_id, "chunk_id": r.chunk_id}
            for r in reranked_results
        ]

        if not context:
            return {
                "query": request.message,
                "answer": "لم يتم العثور على معلومات ذات صلة في قاعدة البيانات.",
                "chunks": [],
                "sources": [],
                "retrieval_mode": response.mode.value if response.mode else "unknown",
                "chunks_retrieved": 0,
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
                    "chunks": [
                        {
                            "chunk_id": r.chunk_id,
                            "document_id": r.document_id,
                            "text": r.text,
                            "score": r.score,
                            "metadata": r.metadata if hasattr(r, 'metadata') else {}
                        }
                        for r in response.results[:10]
                    ],
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
        # Phase 4: Improved timeout handling with separate connect/read timeouts
        generation_start = time.time()
        try:
            # Use proper timeout config: 10s connect, 90s read (LLM can be slow)
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
                tgi_result = tgi_response.json()
                raw_answer = tgi_result.get("generated_text", "").strip()

                # Clean up LLM artifacts (citation markers, prompt echoes, etc.)
                answer = _clean_llm_response(raw_answer)

                # FALLBACK: If LLM returned "no info" despite having chunks, summarize chunks
                if _is_no_info_response(answer) and context:
                    logger.info(f"LLM returned no-info response, using chunk fallback. Original: {answer[:100]}")
                    answer = _create_fallback_answer(request.message, context)

        except httpx.TimeoutException as e:
            logger.warning(f"TGI generation timeout after {time.time() - generation_start:.1f}s: {e}")
            # For timeout, use chunk fallback with a timeout note
            answer = _create_fallback_answer(request.message, context) if context else "انتهت مهلة معالجة الطلب. يرجى المحاولة مرة أخرى."

        except httpx.ConnectError as e:
            logger.error(f"TGI connection error: {e}")
            answer = _create_fallback_answer(request.message, context) if context else "خطأ في الاتصال بخدمة المعالجة."

        except Exception as e:
            logger.error(f"TGI generation error: {e}")
            # Fallback to showing retrieved context
            answer = _create_fallback_answer(request.message, context) if context else "خطأ في معالجة الطلب."

        generation_time = (time.time() - generation_start) * 1000
        total_time = (time.time() - start_time) * 1000

        # 6. Build sources (from re-ranked results)
        sources = [
            {
                "chunk_id": ctx["chunk_id"],
                "document_id": ctx["document_id"],
                "text_preview": ctx["text"][:200] + "..." if len(ctx["text"]) > 200 else ctx["text"],
                "score": reranked_results[i].score if i < len(reranked_results) else 0
            }
            for i, ctx in enumerate(context[:5])
        ]

        # Build full chunks for transparency (full text, not truncated)
        # Include source type (graph vs vector) based on hop_distance
        chunks = []
        vector_count = 0
        graph_1hop_count = 0
        graph_2hop_count = 0

        for r in reranked_results[:10]:  # Top 10 chunks (re-ranked)
            hop = r.hop_distance if hasattr(r, 'hop_distance') else 0
            via = r.via_entity if hasattr(r, 'via_entity') else None

            # Determine source type
            if hop == 0:
                source_type = "vector"
                vector_count += 1
            elif hop == 1:
                source_type = "graph_1hop"
                graph_1hop_count += 1
            else:
                source_type = "graph_2hop"
                graph_2hop_count += 1

            chunks.append({
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "text": r.text,
                "score": r.score,
                "source_type": source_type,
                "hop_distance": hop,
                "via_entity": via,
                "metadata": r.metadata if hasattr(r, 'metadata') else {}
            })

        # Calculate retrieval stats
        retrieval_stats = {
            "total_chunks": len(chunks),
            "vector_chunks": vector_count,
            "graph_1hop_chunks": graph_1hop_count,
            "graph_2hop_chunks": graph_2hop_count,
            "graph_total": graph_1hop_count + graph_2hop_count,
            "entities_used": list(set(c["via_entity"] for c in chunks if c["via_entity"]))
        }

        # Apply RefRAG compression if enabled
        compression_stats = None
        if request.use_refrag and chunks:
            compression_start = time.time()
            compression_input = [{"text": c["text"], "chunk_id": c["chunk_id"]} for c in chunks]
            compression_result = refrag_compressor.compress(compression_input, query_context=request.message)
            compression_time = (time.time() - compression_start) * 1000

            compression_stats = {
                "enabled": True,
                "original_length": compression_result["original_length"],
                "compressed_length": compression_result["compressed_length"],
                "compression_ratio": round(compression_result["compression_ratio"], 3),
                "chunks_compressed": compression_result["chunks_compressed"],
                "compression_time_ms": round(compression_time, 1),
                "strategy": compression_result.get("strategy", "hybrid")
            }
        else:
            compression_stats = {
                "enabled": False,
                "original_length": sum(len(c["text"]) for c in chunks),
                "compressed_length": sum(len(c["text"]) for c in chunks),
                "compression_ratio": 1.0
            }

        # Build metadata from retrieval response
        # Convert any numpy types to Python types for JSON serialization
        disambiguated = []
        if response.metadata and response.metadata.get("disambiguated_entities"):
            for ent in response.metadata.get("disambiguated_entities", []):
                clean_ent = {}
                for k, v in ent.items():
                    if hasattr(v, 'item'):  # numpy types have .item()
                        clean_ent[k] = v.item()
                    else:
                        clean_ent[k] = v
                disambiguated.append(clean_ent)

        metadata = {
            "entities_found": entity_names[:20] if entity_names else [],
            "entity_count": len(entity_names) if entity_names else 0,
            "graph_chunks_count": graph_1hop_count + graph_2hop_count,
            "vector_chunks_count": vector_count,
            "disambiguated_entities": disambiguated,
        }

        result = {
            "query": request.message,
            "answer": answer,
            "chunks": chunks,  # Full chunk data with source metadata
            "sources": sources,  # Truncated preview for display
            "retrieval_mode": response.mode.value if response.mode else "unknown",
            "chunks_retrieved": len(context),
            "retrieval_time_ms": round(retrieval_time, 1),
            "generation_time_ms": round(generation_time, 1),
            "total_time_ms": round(total_time, 1),
            # New detailed metadata
            "retrieval_stats": retrieval_stats,
            "compression_stats": compression_stats,
            "metadata": metadata
        }

        # Also add entity names at top level for backwards compatibility
        if entity_names:
            result["entities_found"] = entity_names[:20]
            result["entity_count"] = len(entity_names)

        # Add V5 features if used
        if enhanced_query:
            result["enhanced_query"] = enhanced_query
        if trace_id:
            result["trace_id"] = trace_id
        if global_search_warning:
            result["warning"] = global_search_warning
        if decomposition_info:
            result["decomposition"] = decomposition_info

        # End trace if enabled
        if trace:
            observability.end_trace(trace, success=True)

        return result

    except Exception as e:
        # End trace on error
        if trace:
            observability.end_trace(trace, success=False, error=str(e))
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MIRAGE V5 Endpoint - All SOTA Innovations
# =============================================================================

class V5Request(BaseModel):
    """Request for V5 unified retrieval."""
    message: str
    use_hyde: bool = True
    use_ppr: bool = True
    use_community_selection: bool = True
    use_dual_level: bool = True
    top_k: int = 10


@router.post("/v5/ask", response_model=Dict[str, Any])
async def ask_v5(request: V5Request):
    """
    MIRAGE V5 RAG endpoint with all SOTA innovations:

    - HyDE: Hypothetical Document Embeddings for query enhancement
    - PPR: Personalized PageRank (HippoRAG) for graph traversal
    - Community Selection: O(log C) hierarchical pruning
    - Dual-Level: LightRAG-style low + high level retrieval
    - Observability: Full tracing and metrics

    This is the production-ready 10/10 retrieval engine.
    """
    import httpx
    global v5_engine

    start_time = time.time()
    logger.info(f"V5 RAG query: {request.message[:100]}...")

    # Start trace
    trace = observability.start_trace(request.message)

    try:
        # Lazy initialize V5 engine
        if v5_engine is None:
            logger.info("Initializing V5 engine...")
            v5_engine = get_v5_engine()
            logger.info("V5 engine initialized")

        # Configure based on request
        v5_engine.config.use_hyde = request.use_hyde
        v5_engine.config.use_hippocampal = request.use_ppr
        v5_engine.config.use_dynamic_community_selection = request.use_community_selection
        v5_engine.config.use_dual_level = request.use_dual_level
        v5_engine.config.max_chunks = request.top_k

        # Retrieve with V5 engine
        with observability.record_step(trace, "v5_retrieval") as step:
            result = v5_engine.retrieve(request.message)
            step.metadata["chunks"] = len(result.chunks)
            step.metadata["communities_searched"] = result.communities_searched

        # Generate answer with LLM
        with observability.record_step(trace, "generation") as step:
            context = [
                {"text": c.get("text", ""), "document_id": c.get("document_id", ""), "chunk_id": c.get("chunk_id", "")}
                for c in result.chunks
            ]

            if context:
                prompt = prompt_manager.create_qa_prompt(
                    question=request.message,
                    context=context
                )

                # Phase 4: Improved timeout handling
                timeout_config = httpx.Timeout(timeout=90.0, connect=10.0)
                async with httpx.AsyncClient(timeout=timeout_config) as client:
                    llm_response = await client.post(
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
                    llm_result = llm_response.json()
                    raw_answer = llm_result.get("generated_text", "لم يتم إنشاء إجابة").strip()
                    # Clean up LLM artifacts (same as V2)
                    answer = _clean_llm_response(raw_answer)
            else:
                answer = "لم يتم العثور على معلومات ذات صلة في قاعدة البيانات."

            step.metadata["answer_length"] = len(answer)

        # End trace
        total_time = (time.time() - start_time) * 1000
        observability.end_trace(trace, success=True)

        return {
            "query": request.message,
            "answer": answer,
            "chunks": [
                {
                    "chunk_id": c.get("chunk_id", ""),
                    "text": c.get("text", "")[:300],
                    "document_id": c.get("document_id", ""),
                    "score": c.get("fused_score", c.get("score", 0.5))
                }
                for c in result.chunks[:5]
            ],
            "metadata": {
                "enhanced_query": result.enhanced_query,
                "entities_found": result.entities_found,
                "communities_searched": result.communities_searched,
                "communities_pruned": result.communities_pruned,
                "ppr_activated_entities": result.ppr_activated_entities,
                "query_type": result.query_type,
                "confidence": result.confidence,
                "low_level_chunks": result.low_level_chunks,
                "high_level_chunks": result.high_level_chunks
            },
            "timing": {
                "retrieval_ms": result.retrieval_time_ms,
                "total_ms": total_time
            },
            "trace_id": trace.trace_id,
            "v5_features": {
                "hyde": request.use_hyde,
                "ppr": request.use_ppr,
                "community_selection": request.use_community_selection,
                "dual_level": request.use_dual_level
            }
        }

    except Exception as e:
        observability.end_trace(trace, success=False, error=str(e))
        logger.error(f"V5 RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v5/metrics")
async def v5_metrics():
    """Get V5 engine performance metrics."""
    metrics = observability.get_metrics()
    dashboard = observability.get_dashboard_data()

    return {
        "metrics": {
            "total_queries": metrics.total_queries,
            "avg_latency_ms": round(metrics.avg_latency_ms, 2),
            "p50_latency_ms": round(metrics.p50_latency_ms, 2),
            "p95_latency_ms": round(metrics.p95_latency_ms, 2),
            "p99_latency_ms": round(metrics.p99_latency_ms, 2),
            "qps": round(metrics.queries_per_second, 2),
            "error_rate": round(metrics.error_rate * 100, 2),
            "cache_hit_rate": round(metrics.cache_hit_rate * 100, 2)
        },
        "mode_distribution": metrics.mode_distribution,
        "recent_traces": dashboard.get("recent_traces", []),
        "slow_traces": dashboard.get("slow_traces", [])
    }
