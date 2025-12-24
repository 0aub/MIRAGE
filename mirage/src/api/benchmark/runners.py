"""
Benchmark Service - Execution Logic
Core benchmark running functionality
"""

import time
import httpx
import json
from typing import List, Dict, Any, Optional
from loguru import logger

from ...core.retrieval import get_retrieval_engine, RetrievalMode
from ...core.refrag.compressor import REFRAGCompressor
from ...core.generation import get_prompt_manager

from .models import (
    DetailedBenchmarkResult,
    RetrievalStats,
    TimingMetrics,
    CompressionMetrics,
)

# Lazy-initialized components
_retrieval_engine = None
_prompt_manager = None
_refrag_compressor = None


def get_components():
    """Get or initialize benchmark components"""
    global _retrieval_engine, _prompt_manager, _refrag_compressor

    if _retrieval_engine is None:
        _retrieval_engine = get_retrieval_engine()
    if _prompt_manager is None:
        _prompt_manager = get_prompt_manager()
    if _refrag_compressor is None:
        _refrag_compressor = REFRAGCompressor(strategy="hybrid")

    return _retrieval_engine, _prompt_manager, _refrag_compressor


async def run_detailed_benchmark(
    query: str,
    mode_str: str,
    use_refrag: bool,
    measure_ttft: bool,
    top_k: int
) -> DetailedBenchmarkResult:
    """Run a single detailed benchmark"""
    retrieval_engine, prompt_manager, refrag_compressor = get_components()

    start_time = time.time()
    config_name = f"{mode_str.upper()}" + (" + RefRAG" if use_refrag else "")

    try:
        # Parse mode
        mode = None
        if mode_str and mode_str != "auto":
            try:
                mode = RetrievalMode(mode_str)
            except ValueError:
                mode = RetrievalMode.NAIVE

        # 1. Retrieval
        retrieval_start = time.time()
        response = retrieval_engine.retrieve(query, mode=mode, top_k=top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000

        # 2. Calculate retrieval stats
        vector_count = 0
        graph_1hop = 0
        graph_2hop = 0
        entities_used = set()

        for r in response.results:
            hop = r.hop_distance if hasattr(r, 'hop_distance') else 0
            if hop == 0:
                vector_count += 1
            elif hop == 1:
                graph_1hop += 1
            else:
                graph_2hop += 1

            if hasattr(r, 'via_entity') and r.via_entity:
                entities_used.add(r.via_entity)

        retrieval_stats = RetrievalStats(
            total_chunks=len(response.results),
            vector_chunks=vector_count,
            graph_1hop_chunks=graph_1hop,
            graph_2hop_chunks=graph_2hop,
            graph_total=graph_1hop + graph_2hop,
            entities_used=list(entities_used)[:10]
        )

        # 3. Optional RefRAG compression
        compression_time = 0
        if use_refrag and response.results:
            compression_start = time.time()
            compression_input = [{"text": r.text, "chunk_id": r.chunk_id} for r in response.results]
            compression_result = refrag_compressor.compress(compression_input, query_context=query)
            compression_time = (time.time() - compression_start) * 1000

            compression = CompressionMetrics(
                enabled=True,
                original_length=compression_result["original_length"],
                compressed_length=compression_result["compressed_length"],
                compression_ratio=round(compression_result["compression_ratio"], 3),
                chunks_compressed=compression_result["chunks_compressed"],
                strategy=compression_result.get("strategy", "hybrid")
            )
        else:
            total_len = sum(len(r.text) for r in response.results) if response.results else 0
            compression = CompressionMetrics(
                enabled=False,
                original_length=total_len,
                compressed_length=total_len,
                compression_ratio=1.0
            )

        # 4. Build context and prompt
        context = [
            {"text": r.text, "document_id": r.document_id, "chunk_id": r.chunk_id}
            for r in response.results
        ]
        prompt = prompt_manager.create_qa_prompt(question=query, context=context)

        # 5. LLM generation with optional TTFT measurement
        generation_start = time.time()
        ttft_ms = None
        answer = ""

        try:
            timeout_config = httpx.Timeout(timeout=60.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                if measure_ttft:
                    async with client.stream(
                        "POST",
                        "http://tgi:80/generate_stream",
                        json={
                            "inputs": f"{prompt.system_message}\n\n{prompt.user_message}",
                            "parameters": {
                                "max_new_tokens": 200,
                                "temperature": 0.3,
                                "do_sample": True,
                                "top_p": 0.9,
                            }
                        }
                    ) as stream_response:
                        first_token_received = False
                        async for line in stream_response.aiter_lines():
                            if line.startswith("data:"):
                                if not first_token_received:
                                    ttft_ms = (time.time() - generation_start) * 1000
                                    first_token_received = True
                                try:
                                    data = json.loads(line[5:])
                                    token = data.get("token", {}).get("text", "")
                                    answer += token
                                except:
                                    pass
                else:
                    llm_response = await client.post(
                        "http://tgi:80/generate",
                        json={
                            "inputs": f"{prompt.system_message}\n\n{prompt.user_message}",
                            "parameters": {
                                "max_new_tokens": 200,
                                "temperature": 0.3,
                                "do_sample": True,
                                "top_p": 0.9,
                                "return_full_text": False
                            }
                        }
                    )
                    llm_result = llm_response.json()
                    answer = llm_result.get("generated_text", "").strip()

        except Exception as e:
            logger.warning(f"LLM generation error in benchmark: {e}")
            answer = f"[Generation error: {str(e)[:50]}]"

        generation_time = (time.time() - generation_start) * 1000
        total_time = (time.time() - start_time) * 1000

        return DetailedBenchmarkResult(
            config_name=config_name,
            retrieval_mode=mode_str,
            use_refrag=use_refrag,
            timing=TimingMetrics(
                total_ms=round(total_time, 1),
                retrieval_ms=round(retrieval_time, 1),
                compression_ms=round(compression_time, 1) if use_refrag else None,
                generation_ms=round(generation_time, 1),
                ttft_ms=round(ttft_ms, 1) if ttft_ms else None
            ),
            retrieval_stats=retrieval_stats,
            compression=compression,
            response=answer[:500],
            response_length=len(answer)
        )

    except Exception as e:
        logger.error(f"Detailed benchmark failed for {config_name}: {e}")
        return DetailedBenchmarkResult(
            config_name=config_name,
            retrieval_mode=mode_str,
            use_refrag=use_refrag,
            timing=TimingMetrics(
                total_ms=(time.time() - start_time) * 1000,
                retrieval_ms=0,
                generation_ms=0
            ),
            retrieval_stats=RetrievalStats(
                total_chunks=0, vector_chunks=0, graph_1hop_chunks=0,
                graph_2hop_chunks=0, graph_total=0, entities_used=[]
            ),
            compression=CompressionMetrics(
                enabled=False, original_length=0, compressed_length=0, compression_ratio=1.0
            ),
            response=f"[Error: {str(e)[:100]}]",
            response_length=0
        )
