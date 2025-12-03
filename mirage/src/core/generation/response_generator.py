"""
MIRAGE V2 Response Generator
Generates responses using LLM with prompt templates.
"""

from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import time
import re

from .prompt_manager import (
    PromptManager,
    PromptType,
    FormattedPrompt,
    get_prompt_manager
)


class GenerationMode(Enum):
    """Generation modes"""
    STANDARD = "standard"      # Single-shot generation
    COT = "cot"               # Chain-of-thought
    REFINEMENT = "refinement"  # Generate then refine


@dataclass
class GenerationConfig:
    """Configuration for response generation"""
    # LLM settings
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.95

    # Generation mode
    mode: GenerationMode = GenerationMode.STANDARD

    # Response formatting
    include_citations: bool = True
    citation_format: str = "inline"  # "inline" or "footnote"
    language: str = "auto"  # "en", "ar", "auto"

    # Quality settings
    include_confidence: bool = True
    include_sources: bool = True


@dataclass
class GeneratedResponse:
    """Generated response with metadata"""
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    generation_time_ms: float = 0.0
    prompt_used: str = ""
    model_used: str = ""
    mode: GenerationMode = GenerationMode.STANDARD
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseGenerator:
    """
    Generates responses from retrieved context using LLM.

    Features:
    - Multiple generation modes
    - Automatic prompt selection
    - Citation extraction
    - Confidence estimation
    - Bilingual support
    """

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        prompt_manager: Optional[PromptManager] = None,
        llm_manager = None
    ):
        """
        Initialize response generator.

        Args:
            config: Generation configuration
            prompt_manager: Prompt template manager
            llm_manager: LLM manager for generation
        """
        self.config = config or GenerationConfig()
        self.prompt_manager = prompt_manager or get_prompt_manager()
        self._llm_manager = llm_manager

        logger.info(
            f"ResponseGenerator initialized: mode={self.config.mode.value}, "
            f"citations={self.config.include_citations}"
        )

    @property
    def llm_manager(self):
        """Lazy load LLM manager"""
        if self._llm_manager is None:
            try:
                from ..models.llm_manager import get_llm_manager
                self._llm_manager = get_llm_manager()
            except ImportError:
                logger.warning("Could not load LLM manager")
        return self._llm_manager

    def generate(
        self,
        question: str,
        context: List[Dict[str, Any]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> GeneratedResponse:
        """
        Generate a response for a question using retrieved context.

        Args:
            question: User question
            context: Retrieved context chunks
            config: Override generation config
            **kwargs: Additional parameters

        Returns:
            GeneratedResponse with answer and metadata
        """
        start_time = time.time()
        cfg = config or self.config

        # Detect language
        language = self._detect_language(question) if cfg.language == "auto" else cfg.language

        # Create prompt
        use_cot = cfg.mode == GenerationMode.COT
        formatted = self.prompt_manager.create_qa_prompt(
            question=question,
            context=context,
            use_cot=use_cot,
            max_context_length=kwargs.get("max_context_length", 4000)
        )

        # Generate response
        if self.llm_manager is None:
            # Return mock response for testing
            return self._create_mock_response(question, context, cfg, start_time)

        try:
            response_text = self.llm_manager.generate(
                prompt=formatted.user_message,
                system_message=formatted.system_message,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return GeneratedResponse(
                answer=f"I encountered an error generating the response: {str(e)}",
                generation_time_ms=(time.time() - start_time) * 1000,
                mode=cfg.mode
            )

        # Post-process response
        answer, citations = self._extract_citations(response_text, context)
        confidence = self._estimate_confidence(answer, context)

        return GeneratedResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            generation_time_ms=(time.time() - start_time) * 1000,
            prompt_used=formatted.template_name,
            model_used=getattr(self.llm_manager, 'model_id', 'unknown'),
            mode=cfg.mode,
            metadata={
                "language": language,
                "context_chunks": len(context),
                "question_length": len(question)
            }
        )

    async def generate_stream(
        self,
        question: str,
        context: List[Dict[str, Any]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream response generation.

        Yields chunks of the response as they're generated.
        """
        cfg = config or self.config

        # Create prompt
        use_cot = cfg.mode == GenerationMode.COT
        formatted = self.prompt_manager.create_qa_prompt(
            question=question,
            context=context,
            use_cot=use_cot
        )

        if self.llm_manager is None or not hasattr(self.llm_manager, 'generate_stream'):
            # Yield mock response
            yield "I cannot generate a response without an LLM manager."
            return

        try:
            async for chunk in self.llm_manager.generate_stream(
                prompt=formatted.user_message,
                system_message=formatted.system_message,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            yield f"\n\nError: {str(e)}"

    def _detect_language(self, text: str) -> str:
        """Detect text language (simple heuristic)"""
        # Count Arabic characters
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(text.replace(" ", ""))

        if total_chars == 0:
            return "en"

        arabic_ratio = arabic_chars / total_chars
        return "ar" if arabic_ratio > 0.3 else "en"

    def _extract_citations(
        self,
        response: str,
        context: List[Dict[str, Any]]
    ) -> tuple:
        """Extract citations from response"""
        citations = []

        # Find citation markers [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, response)

        for match in matches:
            idx = int(match) - 1
            if 0 <= idx < len(context):
                chunk = context[idx]
                citations.append({
                    "index": int(match),
                    "document_id": chunk.get("document_id", ""),
                    "text": chunk.get("text", "")[:200],
                    "score": chunk.get("score", 0)
                })

        # Deduplicate
        seen_indices = set()
        unique_citations = []
        for c in citations:
            if c["index"] not in seen_indices:
                seen_indices.add(c["index"])
                unique_citations.append(c)

        return response, unique_citations

    def _estimate_confidence(
        self,
        answer: str,
        context: List[Dict[str, Any]]
    ) -> float:
        """Estimate confidence in the answer"""
        # Simple heuristic based on:
        # 1. Number of citations
        # 2. Context scores
        # 3. Answer length

        # Count citations
        citations = len(re.findall(r'\[(\d+)\]', answer))
        citation_score = min(citations / 3, 1.0) * 0.3

        # Average context score
        if context:
            avg_score = sum(c.get("score", 0) for c in context) / len(context)
            context_score = avg_score * 0.5
        else:
            context_score = 0

        # Answer quality (not too short, not hedging too much)
        answer_lower = answer.lower()
        hedging_words = ["i'm not sure", "i cannot", "i don't know", "uncertain"]
        has_hedging = any(h in answer_lower for h in hedging_words)

        if has_hedging:
            quality_score = 0.1
        elif len(answer) < 50:
            quality_score = 0.1
        else:
            quality_score = 0.2

        return min(citation_score + context_score + quality_score, 1.0)

    def _create_mock_response(
        self,
        question: str,
        context: List[Dict[str, Any]],
        config: GenerationConfig,
        start_time: float
    ) -> GeneratedResponse:
        """Create mock response for testing without LLM"""
        # Build mock answer from context
        if context:
            answer = "Based on the provided context:\n\n"
            for i, chunk in enumerate(context[:3], 1):
                text = chunk.get("text", "")[:200]
                answer += f"- {text}... [{i}]\n"
            answer += "\nThis information was extracted from the retrieved documents."
        else:
            answer = "I could not find relevant information to answer your question."

        citations = [
            {
                "index": i + 1,
                "document_id": c.get("document_id", ""),
                "text": c.get("text", "")[:100]
            }
            for i, c in enumerate(context[:3])
        ]

        return GeneratedResponse(
            answer=answer,
            citations=citations,
            confidence=0.5 if context else 0.1,
            generation_time_ms=(time.time() - start_time) * 1000,
            prompt_used="mock",
            model_used="mock",
            mode=config.mode,
            metadata={"mock": True}
        )

    def refine_response(
        self,
        initial_response: GeneratedResponse,
        feedback: str
    ) -> GeneratedResponse:
        """Refine a response based on feedback"""
        # Build refinement prompt
        prompt = f"""The following answer was generated but needs refinement.

Original Answer:
{initial_response.answer}

Feedback:
{feedback}

Please provide an improved answer that addresses the feedback while maintaining accuracy and citations."""

        if self.llm_manager is None:
            return initial_response

        try:
            refined_text = self.llm_manager.generate(
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=0.1
            )

            return GeneratedResponse(
                answer=refined_text,
                citations=initial_response.citations,
                confidence=initial_response.confidence,
                mode=GenerationMode.REFINEMENT,
                metadata={"refined_from": initial_response.prompt_used}
            )
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return initial_response


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_response_generator: Optional[ResponseGenerator] = None


def get_response_generator(**kwargs) -> ResponseGenerator:
    """Get or create global ResponseGenerator"""
    global _response_generator

    if _response_generator is None:
        _response_generator = ResponseGenerator(**kwargs)

    return _response_generator
