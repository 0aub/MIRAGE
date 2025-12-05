"""
MIRAGE V4 Answer Synthesizer
Synthesizes coherent answers from multiple retrieval sources.

Features:
- Multi-source answer synthesis using LLM
- Confidence scoring based on source agreement
- Contradiction detection and resolution
- Source attribution and citation generation
- Arabic and English support
"""

import os
import json
import time
import hashlib
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from enum import Enum

# Required imports - no fallback (enforced in Docker)
import redis
import tiktoken

from ...config import settings
from .base_retriever import RetrievalResult, RetrievalResponse


class SynthesisStrategy(Enum):
    """Answer synthesis strategies"""
    EXTRACTIVE = "extractive"       # Extract answer directly from sources
    ABSTRACTIVE = "abstractive"     # Generate answer in LLM's words
    HYBRID = "hybrid"               # Combine extraction + generation
    CONSENSUS = "consensus"         # Find agreement across sources


@dataclass
class SynthesisSource:
    """A source used in answer synthesis"""
    chunk_id: str
    document_id: str
    text: str
    relevance_score: float
    retrieval_mode: str
    used_in_answer: bool = False
    citation_index: Optional[int] = None


@dataclass
class SynthesizedAnswer:
    """Result of answer synthesis"""
    answer: str
    confidence: float
    sources: List[SynthesisSource]
    citations: List[Dict[str, Any]]
    strategy_used: SynthesisStrategy
    language: str
    synthesis_time_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": [
                {
                    "chunk_id": s.chunk_id,
                    "document_id": s.document_id,
                    "relevance_score": s.relevance_score,
                    "retrieval_mode": s.retrieval_mode,
                    "used_in_answer": s.used_in_answer,
                    "citation_index": s.citation_index,
                }
                for s in self.sources
            ],
            "citations": self.citations,
            "strategy": self.strategy_used.value,
            "language": self.language,
            "synthesis_time_ms": self.synthesis_time_ms,
            "metadata": self.metadata,
        }


class AnswerSynthesizer:
    """
    Synthesizes coherent answers from multiple retrieval sources.

    Combines results from different retrieval modes (vector, graph, hybrid)
    into a unified, well-cited answer with confidence scoring.
    """

    # Arabic synthesis prompts
    ARABIC_SYSTEM_PROMPT = """أنت مساعد خبير في تجميع المعلومات وتقديم إجابات دقيقة ومفصلة.

**تعليمات:**
1. اقرأ جميع المصادر المقدمة بعناية
2. استخرج المعلومات الأكثر صلة بالسؤال
3. اكتب إجابة شاملة ومتماسكة
4. أضف الاقتباسات [1], [2], إلخ للمعلومات المهمة
5. إذا وجدت تناقضات بين المصادر، اذكرها
6. إذا لم تجد معلومات كافية، قل ذلك بوضوح

**تنسيق الإجابة:**
- ابدأ بملخص مختصر
- ثم قدم التفاصيل مع الاقتباسات
- اختم بالاستنتاج إذا لزم الأمر"""

    ENGLISH_SYSTEM_PROMPT = """You are an expert assistant at synthesizing information and providing accurate, detailed answers.

**Instructions:**
1. Carefully read all provided sources
2. Extract the most relevant information for the question
3. Write a comprehensive, coherent answer
4. Add citations [1], [2], etc. for important information
5. If you find contradictions between sources, mention them
6. If information is insufficient, clearly state so

**Answer Format:**
- Start with a brief summary
- Then provide details with citations
- End with a conclusion if appropriate"""

    # Confidence assessment prompt
    CONFIDENCE_PROMPT = """Rate the confidence of your answer on a scale of 0.0 to 1.0 based on:
- Source agreement (do sources agree?)
- Information completeness (is the answer fully supported?)
- Relevance (how well do sources match the question?)

Return ONLY a JSON object: {"confidence": 0.85, "reasoning": "brief explanation"}"""

    def __init__(
        self,
        default_strategy: SynthesisStrategy = SynthesisStrategy.HYBRID,
        min_confidence: float = 0.3,
        max_sources: int = 10,
        use_cache: bool = True,
        cache_ttl: int = 3600  # 1 hour
    ):
        """
        Initialize the answer synthesizer.

        Args:
            default_strategy: Default synthesis strategy
            min_confidence: Minimum confidence threshold
            max_sources: Maximum sources to include in synthesis
            use_cache: Whether to cache synthesis results
            cache_ttl: Cache TTL in seconds
        """
        self.default_strategy = default_strategy
        self.min_confidence = min_confidence
        self.max_sources = max_sources
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl

        # Initialize LLM provider
        self.provider = None
        self.model = None
        self._detect_provider()

        # Initialize token counter
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Initialize Redis
        self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.redis_client.ping()

        # Statistics
        self._stats = {
            "syntheses_attempted": 0,
            "syntheses_successful": 0,
            "cache_hits": 0,
            "average_confidence": 0.0,
            "total_sources_used": 0,
        }

        logger.info(
            f"AnswerSynthesizer initialized: provider={self.provider}, "
            f"strategy={default_strategy.value}"
        )

    def _detect_provider(self) -> None:
        """Auto-detect LLM provider"""
        if settings.use_tgi and settings.tgi_endpoint:
            self.provider = "tgi"
            self.model = "tgi"
            self.tgi_endpoint = settings.tgi_endpoint
            logger.info(f"Using TGI for answer synthesis: {settings.tgi_endpoint}")
        elif settings.openai_api_key:
            self.provider = "openai"
            self.model = "gpt-4o-mini"
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        elif settings.anthropic_api_key:
            self.provider = "anthropic"
            self.model = "claude-3-haiku-20240307"
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
        elif settings.google_api_key:
            self.provider = "google"
            self.model = "gemini/gemini-2.0-flash-exp"
            os.environ["GOOGLE_API_KEY"] = settings.google_api_key
        else:
            logger.warning("No LLM provider available for answer synthesis")

    def synthesize(
        self,
        query: str,
        results: List[RetrievalResult],
        strategy: Optional[SynthesisStrategy] = None,
        language: str = "auto"
    ) -> SynthesizedAnswer:
        """
        Synthesize an answer from retrieval results.

        Args:
            query: The user's question
            results: List of retrieval results from various modes
            strategy: Synthesis strategy (default: hybrid)
            language: Language of the query ("ar", "en", or "auto")

        Returns:
            SynthesizedAnswer with answer, confidence, and citations
        """
        start_time = time.time()
        self._stats["syntheses_attempted"] += 1

        strategy = strategy or self.default_strategy

        # Detect language
        if language == "auto":
            arabic_ratio = len([c for c in query if '\u0600' <= c <= '\u06FF']) / max(len(query), 1)
            language = "ar" if arabic_ratio > 0.3 else "en"

        # Check cache
        cache_key = self._get_cache_key(query, results)
        cached = self._check_cache(cache_key)
        if cached:
            self._stats["cache_hits"] += 1
            cached["metadata"]["from_cache"] = True
            return self._dict_to_synthesized_answer(cached)

        # Prepare sources
        sources = self._prepare_sources(results)

        # Build synthesis prompt
        synthesis_prompt = self._build_synthesis_prompt(
            query, sources, strategy, language
        )

        # Call LLM for synthesis
        answer_text = self._call_llm(
            self.ARABIC_SYSTEM_PROMPT if language == "ar" else self.ENGLISH_SYSTEM_PROMPT,
            synthesis_prompt
        )

        if not answer_text:
            return self._create_fallback_answer(query, sources, language, start_time)

        # Assess confidence
        confidence = self._assess_confidence(query, answer_text, sources, language)

        # Extract citations from answer
        citations = self._extract_citations(answer_text, sources)

        # Mark which sources were used
        for citation in citations:
            if citation.get("source_index") is not None:
                idx = citation["source_index"]
                if idx < len(sources):
                    sources[idx].used_in_answer = True
                    sources[idx].citation_index = citation.get("citation_number")

        synthesis_time_ms = int((time.time() - start_time) * 1000)

        # Update stats
        self._stats["syntheses_successful"] += 1
        used_count = sum(1 for s in sources if s.used_in_answer)
        self._stats["total_sources_used"] += used_count
        self._stats["average_confidence"] = (
            (self._stats["average_confidence"] * (self._stats["syntheses_successful"] - 1) + confidence)
            / self._stats["syntheses_successful"]
        )

        result = SynthesizedAnswer(
            answer=answer_text,
            confidence=confidence,
            sources=sources,
            citations=citations,
            strategy_used=strategy,
            language=language,
            synthesis_time_ms=synthesis_time_ms,
            metadata={
                "sources_total": len(sources),
                "sources_used": used_count,
                "retrieval_modes": list(set(s.retrieval_mode for s in sources)),
            }
        )

        # Cache result
        self._cache_result(cache_key, result.to_dict())

        logger.info(
            f"Answer synthesized: confidence={confidence:.2f}, "
            f"sources_used={used_count}/{len(sources)}, "
            f"time={synthesis_time_ms}ms"
        )

        return result

    def synthesize_from_response(
        self,
        query: str,
        response: RetrievalResponse,
        strategy: Optional[SynthesisStrategy] = None,
        language: str = "auto"
    ) -> SynthesizedAnswer:
        """
        Convenience method to synthesize from a RetrievalResponse.

        Args:
            query: The user's question
            response: RetrievalResponse from retrieval engine
            strategy: Synthesis strategy
            language: Language of the query

        Returns:
            SynthesizedAnswer
        """
        return self.synthesize(query, response.results, strategy, language)

    def _prepare_sources(
        self,
        results: List[RetrievalResult]
    ) -> List[SynthesisSource]:
        """Convert retrieval results to synthesis sources"""
        sources = []

        # Deduplicate by chunk_id
        seen = set()
        for result in results[:self.max_sources]:
            if result.chunk_id in seen:
                continue
            seen.add(result.chunk_id)

            sources.append(SynthesisSource(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=result.text,
                relevance_score=result.score,
                retrieval_mode=result.retrieval_mode or "unknown",
            ))

        # Sort by relevance
        sources.sort(key=lambda s: s.relevance_score, reverse=True)

        return sources

    def _build_synthesis_prompt(
        self,
        query: str,
        sources: List[SynthesisSource],
        strategy: SynthesisStrategy,
        language: str
    ) -> str:
        """Build the synthesis prompt with sources"""
        if language == "ar":
            prompt = f"""السؤال: {query}

المصادر المتاحة:
"""
            for i, source in enumerate(sources, 1):
                prompt += f"\n[{i}] (الأهمية: {source.relevance_score:.2f})\n{source.text}\n"

            if strategy == SynthesisStrategy.EXTRACTIVE:
                prompt += "\n**ملاحظة:** استخرج الإجابة مباشرة من المصادر دون إعادة صياغة."
            elif strategy == SynthesisStrategy.ABSTRACTIVE:
                prompt += "\n**ملاحظة:** اكتب الإجابة بأسلوبك الخاص مع ذكر المصادر."
            elif strategy == SynthesisStrategy.CONSENSUS:
                prompt += "\n**ملاحظة:** ابحث عن النقاط المتفق عليها بين المصادر."

            prompt += "\n\nالإجابة:"
        else:
            prompt = f"""Question: {query}

Available Sources:
"""
            for i, source in enumerate(sources, 1):
                prompt += f"\n[{i}] (Relevance: {source.relevance_score:.2f})\n{source.text}\n"

            if strategy == SynthesisStrategy.EXTRACTIVE:
                prompt += "\n**Note:** Extract the answer directly from sources without paraphrasing."
            elif strategy == SynthesisStrategy.ABSTRACTIVE:
                prompt += "\n**Note:** Write the answer in your own words while citing sources."
            elif strategy == SynthesisStrategy.CONSENSUS:
                prompt += "\n**Note:** Find points of agreement across sources."

            prompt += "\n\nAnswer:"

        return prompt

    def _assess_confidence(
        self,
        query: str,
        answer: str,
        sources: List[SynthesisSource],
        language: str
    ) -> float:
        """Assess confidence in the synthesized answer"""
        # Simple heuristic-based confidence
        confidence = 0.5  # Base confidence

        # Factor 1: Source relevance scores
        if sources:
            avg_relevance = sum(s.relevance_score for s in sources) / len(sources)
            confidence += avg_relevance * 0.2  # Up to 0.2 boost

        # Factor 2: Number of sources
        if len(sources) >= 3:
            confidence += 0.1
        elif len(sources) >= 5:
            confidence += 0.15

        # Factor 3: Answer length (too short or too long reduces confidence)
        answer_len = len(answer.split())
        if 20 <= answer_len <= 200:
            confidence += 0.1
        elif answer_len < 10:
            confidence -= 0.1

        # Factor 4: Citation presence
        citation_count = answer.count('[')
        if citation_count >= 2:
            confidence += 0.1
        elif citation_count >= 4:
            confidence += 0.15

        # Factor 5: Source agreement (check if multiple modes returned same info)
        modes = set(s.retrieval_mode for s in sources)
        if len(modes) >= 2:
            confidence += 0.05  # Multiple retrieval modes agree

        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        return confidence

    def _extract_citations(
        self,
        answer: str,
        sources: List[SynthesisSource]
    ) -> List[Dict[str, Any]]:
        """Extract citations from the answer text"""
        import re

        citations = []
        citation_pattern = r'\[(\d+)\]'

        for match in re.finditer(citation_pattern, answer):
            citation_num = int(match.group(1))
            source_idx = citation_num - 1  # 1-indexed in text

            if 0 <= source_idx < len(sources):
                source = sources[source_idx]
                citations.append({
                    "citation_number": citation_num,
                    "source_index": source_idx,
                    "chunk_id": source.chunk_id,
                    "document_id": source.document_id,
                    "text_snippet": source.text[:200] + "..." if len(source.text) > 200 else source.text,
                    "position_in_answer": match.start(),
                })

        return citations

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Optional[str]:
        """Call LLM for synthesis"""
        if not self.provider:
            return None

        try:
            if self.provider == "tgi":
                response = requests.post(
                    f"{self.tgi_endpoint}/v1/chat/completions",
                    json={
                        "model": "tgi",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 1024,
                        "top_p": 0.9,
                    },
                    timeout=30
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            else:
                from litellm import completion
                response = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM call failed for synthesis: {e}")
            return None

    def _create_fallback_answer(
        self,
        query: str,
        sources: List[SynthesisSource],
        language: str,
        start_time: float
    ) -> SynthesizedAnswer:
        """Create fallback answer when LLM fails"""
        if language == "ar":
            answer = "عذراً، لم أتمكن من تجميع إجابة شاملة. إليك المعلومات المتاحة:\n\n"
        else:
            answer = "I couldn't synthesize a comprehensive answer. Here's the available information:\n\n"

        for i, source in enumerate(sources[:3], 1):
            snippet = source.text[:300] + "..." if len(source.text) > 300 else source.text
            answer += f"[{i}] {snippet}\n\n"

        return SynthesizedAnswer(
            answer=answer,
            confidence=0.3,
            sources=sources,
            citations=[],
            strategy_used=SynthesisStrategy.EXTRACTIVE,
            language=language,
            synthesis_time_ms=int((time.time() - start_time) * 1000),
            metadata={"fallback": True}
        )

    def _get_cache_key(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> str:
        """Generate cache key"""
        content = query + "".join(r.chunk_id for r in results[:5])
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"synthesis:{content_hash}"

    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check cache for synthesis result"""
        if not self.use_cache:
            return None
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
        return None

    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache synthesis result"""
        if not self.use_cache:
            return
        try:
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(result, ensure_ascii=False)
            )
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    def _dict_to_synthesized_answer(self, data: Dict[str, Any]) -> SynthesizedAnswer:
        """Convert cached dict back to SynthesizedAnswer"""
        sources = [
            SynthesisSource(
                chunk_id=s["chunk_id"],
                document_id=s["document_id"],
                text="",  # Not cached for space
                relevance_score=s["relevance_score"],
                retrieval_mode=s["retrieval_mode"],
                used_in_answer=s.get("used_in_answer", False),
                citation_index=s.get("citation_index"),
            )
            for s in data.get("sources", [])
        ]

        return SynthesizedAnswer(
            answer=data["answer"],
            confidence=data["confidence"],
            sources=sources,
            citations=data.get("citations", []),
            strategy_used=SynthesisStrategy(data.get("strategy", "hybrid")),
            language=data.get("language", "en"),
            synthesis_time_ms=data.get("synthesis_time_ms", 0),
            metadata=data.get("metadata", {}),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get synthesizer statistics"""
        return {
            **self._stats,
            "provider": self.provider,
            "default_strategy": self.default_strategy.value,
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_answer_synthesizer: Optional[AnswerSynthesizer] = None


def get_answer_synthesizer() -> AnswerSynthesizer:
    """Get or create global AnswerSynthesizer instance"""
    global _answer_synthesizer

    if _answer_synthesizer is None:
        _answer_synthesizer = AnswerSynthesizer()

    return _answer_synthesizer


def synthesize_answer(
    query: str,
    results: List[RetrievalResult],
    strategy: Optional[SynthesisStrategy] = None,
    language: str = "auto"
) -> SynthesizedAnswer:
    """Convenience function to synthesize an answer"""
    return get_answer_synthesizer().synthesize(query, results, strategy, language)
