"""
Benchmark Service - RAGAS Evaluation System
Ground-truth based evaluation with semantic similarity scoring
"""

import time
import httpx
import sys
from typing import List, Dict, Any
from loguru import logger

from ...core.retrieval import get_retrieval_engine, RetrievalMode
from ...core.generation import get_prompt_manager

from .models import RagasScores

# Import test cases
sys.path.insert(0, "/app/tools")
try:
    from evaluation_test_cases import TEST_CASES, TestCase, compute_semantic_overlap
    RAGAS_AVAILABLE = True
except ImportError:
    TEST_CASES = []
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS evaluation not available - evaluation_test_cases.py not found")

# Lazy-initialized components
_retrieval_engine = None
_prompt_manager = None


def get_components():
    """Get or initialize benchmark components"""
    global _retrieval_engine, _prompt_manager

    if _retrieval_engine is None:
        _retrieval_engine = get_retrieval_engine()
    if _prompt_manager is None:
        _prompt_manager = get_prompt_manager()

    return _retrieval_engine, _prompt_manager


def is_available() -> bool:
    """Check if RAGAS evaluation is available"""
    return RAGAS_AVAILABLE


def get_test_cases():
    """Get available test cases"""
    return TEST_CASES


async def run_ragas_single(
    test_case: "TestCase",
    mode_str: str,
    use_refrag: bool,  # Kept for API compatibility, but ignored
    top_k: int
) -> Dict[str, Any]:
    """
    Run a single RAGAS evaluation for a test case and mode.
    Returns dict with answer, scores, timing, etc.
    """
    retrieval_engine, prompt_manager = get_components()

    start_time = time.time()

    try:
        # Parse mode
        mode = None
        if mode_str and mode_str != "auto":
            try:
                mode = RetrievalMode(mode_str)
            except ValueError:
                mode = RetrievalMode.VECTOR

        # 1. Retrieval
        response = retrieval_engine.retrieve(test_case.query, mode=mode, top_k=top_k)
        chunks = response.results

        # Extract entities
        entities_found = set()
        for r in chunks:
            if hasattr(r, 'via_entity') and r.via_entity:
                entities_found.add(r.via_entity)

        # 2. Build context and prompt
        context = [
            {"text": r.text, "document_id": r.document_id, "chunk_id": r.chunk_id}
            for r in chunks
        ]
        prompt = prompt_manager.create_qa_prompt(question=test_case.query, context=context)

        # 3. LLM generation
        answer = ""
        try:
            timeout_config = httpx.Timeout(timeout=60.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                llm_response = await client.post(
                    "http://tgi:80/generate",
                    json={
                        "inputs": f"{prompt.system_message}\n\n{prompt.user_message}",
                        "parameters": {
                            "max_new_tokens": 300,
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
            logger.warning(f"LLM generation error: {e}")
            answer = "[Generation error]"

        total_time = (time.time() - start_time) * 1000

        # 4. Calculate RAGAS scores
        scores = calculate_ragas_scores(test_case, chunks, answer, list(entities_found))

        return {
            "answer": answer[:1000],
            "scores": scores,
            "timing_ms": round(total_time, 1),
            "chunks_used": len(chunks),
            "entities_found": list(entities_found)[:10],
            "token_savings": 0  # No compression
        }

    except Exception as e:
        logger.error(f"RAGAS single evaluation error: {e}")
        return {
            "answer": f"[Error: {str(e)[:100]}]",
            "scores": {
                "chunk_count": 0,
                "entity_match": 0,
                "chunk_content": 0,
                "answer_keywords": 0,
                "ground_truth_similarity": 0,
                "weighted_score": 0,
                "passed": False
            },
            "timing_ms": 0,
            "chunks_used": 0,
            "entities_found": [],
            "token_savings": 0
        }


def calculate_ragas_scores(test_case: "TestCase", chunks: List, answer: str, entities_found: List[str]) -> Dict[str, Any]:
    """
    Calculate RAGAS evaluation scores.

    Scoring weights:
    - ground_truth_similarity: 35% (most important)
    - answer_keywords: 20%
    - chunk_content: 20%
    - entity_match: 15%
    - chunk_count: 10%
    """
    # Check 1: Chunk count
    chunk_count = len(chunks)
    chunk_count_score = 1.0 if chunk_count >= test_case.min_chunk_count else 0.0

    # Check 2: Entity match
    entity_match_score = 0.0
    if test_case.expected_entities:
        found_entities = set()
        entity_names_lower = [e.lower() for e in entities_found]
        for expected in test_case.expected_entities:
            expected_lower = expected.lower()
            if any(expected_lower in name or name in expected_lower for name in entity_names_lower):
                found_entities.add(expected)
        entity_match_score = len(found_entities) / len(test_case.expected_entities)
    else:
        entity_match_score = 1.0

    # Check 3: Chunk content match
    chunk_content_score = 0.0
    if test_case.expected_chunks:
        all_chunk_text = " ".join([getattr(c, 'text', '') for c in chunks]).lower()
        matches = 0
        for exp_chunk in test_case.expected_chunks:
            if all(pattern.lower() in all_chunk_text for pattern in exp_chunk.content_contains):
                matches += 1
        chunk_content_score = matches / len(test_case.expected_chunks)
    else:
        chunk_content_score = 1.0

    # Check 4: Answer keywords
    answer_keywords_score = 0.0
    if test_case.expected_answer_contains:
        answer_lower = answer.lower()
        keywords_found = sum(1 for kw in test_case.expected_answer_contains if kw.lower() in answer_lower)
        answer_keywords_score = keywords_found / len(test_case.expected_answer_contains)
    else:
        answer_keywords_score = 1.0

    # Check 5: Ground truth semantic similarity
    ground_truth_score = 0.0
    if test_case.expected_answer and RAGAS_AVAILABLE:
        ground_truth_score = compute_semantic_overlap(answer, test_case.expected_answer)

    # Calculate weighted score
    weights = {
        "chunk_count": 0.10,
        "entity_match": 0.15,
        "chunk_content": 0.20,
        "answer_keywords": 0.20,
        "ground_truth_similarity": 0.35
    }

    weighted_score = (
        weights["chunk_count"] * chunk_count_score +
        weights["entity_match"] * entity_match_score +
        weights["chunk_content"] * chunk_content_score +
        weights["answer_keywords"] * answer_keywords_score +
        weights["ground_truth_similarity"] * (1.0 if ground_truth_score >= 0.3 else 0.0)
    )

    passed = weighted_score >= 0.6

    return {
        "chunk_count": round(chunk_count_score, 3),
        "entity_match": round(entity_match_score, 3),
        "chunk_content": round(chunk_content_score, 3),
        "answer_keywords": round(answer_keywords_score, 3),
        "ground_truth_similarity": round(ground_truth_score, 3),
        "weighted_score": round(weighted_score, 3),
        "passed": passed
    }
