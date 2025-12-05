"""
HyDE - Hypothetical Document Embeddings for Query Enhancement

Key Innovation:
- Instead of embedding the raw query, generate a hypothetical answer first
- Embed the hypothetical answer (which looks more like real documents)
- Use that embedding for retrieval

Why it works:
- Queries: "What is X?" (short, interrogative)
- Documents: "X is a technology that..." (long, declarative)
- Hypothetical answers bridge the semantic gap

Example:
- Query: "ما هي جائزة الحكومة الرقمية؟"
- Hypothetical: "جائزة الحكومة الرقمية هي جائزة سنوية تمنحها..."
- The hypothetical is more similar to actual documents than the query

Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al.)
"""

import numpy as np
import requests
from typing import Optional, List, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class EnhancedQuery:
    """Query enhanced with hypothetical document."""
    original_query: str
    hypothetical_answer: str
    query_embedding: Optional[np.ndarray] = None
    hyde_embedding: Optional[np.ndarray] = None
    combined_embedding: Optional[np.ndarray] = None
    generation_time_ms: float = 0.0


class HyDEQueryEnhancer:
    """
    Hypothetical Document Embeddings for better retrieval.

    Key Insight:
    A hypothetical answer is semantically more similar to real documents
    than the original question.

    Usage:
        enhancer = HyDEQueryEnhancer(embedder)
        enhanced = enhancer.enhance(query)
        # Use enhanced.combined_embedding for retrieval

    Modes:
    - hyde_only: Use only hypothetical embedding
    - combined: Weighted combination of query + hypothetical
    - multi_hyde: Generate multiple hypotheticals, average embeddings
    """

    def __init__(
        self,
        embedder,
        llm_endpoint: str = "http://tgi:80",
        alpha: float = 0.5,
        max_tokens: int = 200,
        temperature: float = 0.7,
        num_hypotheticals: int = 1
    ):
        """
        Initialize HyDE enhancer.

        Args:
            embedder: Embedding manager for generating embeddings
            llm_endpoint: TGI endpoint for hypothetical generation
            alpha: Weight for query embedding in combined (0=HyDE only, 1=query only)
            max_tokens: Max tokens for hypothetical generation
            temperature: LLM temperature (higher = more diverse hypotheticals)
            num_hypotheticals: Number of hypotheticals to generate (>1 for multi-hyde)
        """
        self.embedder = embedder
        self.llm_endpoint = llm_endpoint
        self.alpha = alpha
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_hypotheticals = num_hypotheticals

        logger.info(
            f"HyDEQueryEnhancer initialized: alpha={alpha}, "
            f"num_hypotheticals={num_hypotheticals}"
        )

    def enhance(
        self,
        query: str,
        mode: str = "combined"
    ) -> EnhancedQuery:
        """
        Enhance query with hypothetical document embedding.

        Args:
            query: Original user query
            mode: Enhancement mode
                - "combined": Weighted combination (default)
                - "hyde_only": Only hypothetical embedding
                - "query_only": Only query embedding (no enhancement)
                - "multi_hyde": Average multiple hypotheticals

        Returns:
            EnhancedQuery with embeddings
        """
        import time
        start_time = time.time()

        # Get query embedding
        query_embedding = self._embed(query)

        if mode == "query_only":
            return EnhancedQuery(
                original_query=query,
                hypothetical_answer="",
                query_embedding=query_embedding,
                combined_embedding=query_embedding,
                generation_time_ms=(time.time() - start_time) * 1000
            )

        # Generate hypothetical(s)
        if mode == "multi_hyde" and self.num_hypotheticals > 1:
            hypotheticals = self._generate_multiple_hypotheticals(query)
            hypothetical = hypotheticals[0] if hypotheticals else ""

            # Average embeddings of all hypotheticals
            hyde_embeddings = [self._embed(h) for h in hypotheticals]
            hyde_embeddings = [e for e in hyde_embeddings if e is not None]

            if hyde_embeddings:
                hyde_embedding = np.mean(hyde_embeddings, axis=0)
                # Normalize
                hyde_embedding = hyde_embedding / np.linalg.norm(hyde_embedding)
            else:
                hyde_embedding = query_embedding
        else:
            hypothetical = self._generate_hypothetical(query)
            hyde_embedding = self._embed(hypothetical) if hypothetical else query_embedding

        # Combine embeddings
        if mode == "hyde_only":
            combined_embedding = hyde_embedding
        else:
            combined_embedding = self._combine_embeddings(
                query_embedding, hyde_embedding
            )

        generation_time = (time.time() - start_time) * 1000

        logger.debug(
            f"HyDE enhanced: query='{query[:50]}...' -> "
            f"hypothetical='{hypothetical[:80]}...' "
            f"({generation_time:.1f}ms)"
        )

        return EnhancedQuery(
            original_query=query,
            hypothetical_answer=hypothetical,
            query_embedding=query_embedding,
            hyde_embedding=hyde_embedding,
            combined_embedding=combined_embedding,
            generation_time_ms=generation_time
        )

    def _generate_hypothetical(self, query: str) -> str:
        """
        Generate a hypothetical answer using LLM.

        The prompt is designed to produce document-like text,
        not a conversational response.
        """
        # Bilingual prompt (Arabic-first with English fallback)
        prompt = self._build_hyde_prompt(query)

        try:
            response = requests.post(
                f"{self.llm_endpoint}/generate",
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "top_p": 0.95,
                        "do_sample": True,
                        "return_full_text": False
                    }
                },
                timeout=15
            )

            if response.status_code != 200:
                logger.warning(f"HyDE generation failed: {response.status_code}")
                return ""

            result = response.json()
            hypothetical = result.get('generated_text', '').strip()

            # Clean up the response
            hypothetical = self._clean_hypothetical(hypothetical)

            return hypothetical

        except Exception as e:
            logger.warning(f"HyDE generation error: {e}")
            return ""

    def _generate_multiple_hypotheticals(
        self,
        query: str,
        n: Optional[int] = None
    ) -> List[str]:
        """Generate multiple diverse hypotheticals."""
        n = n or self.num_hypotheticals
        hypotheticals = []

        # Use different temperatures for diversity
        temps = [0.5, 0.7, 0.9][:n]

        for temp in temps:
            original_temp = self.temperature
            self.temperature = temp
            hyp = self._generate_hypothetical(query)
            self.temperature = original_temp

            if hyp:
                hypotheticals.append(hyp)

        return hypotheticals

    def _build_hyde_prompt(self, query: str) -> str:
        """
        Build prompt for hypothetical generation.

        Key: Frame it as document generation, not Q&A.
        """
        # Detect language
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in query)

        if is_arabic:
            prompt = f"""اكتب فقرة قصيرة (3-4 جمل) تحتوي على إجابة تفصيلية للسؤال التالي.
اكتب بأسلوب وثائقي رسمي، كأنها مقتطف من وثيقة رسمية أو تقرير.
لا تبدأ بـ "الإجابة:" أو أي مقدمة، فقط اكتب المحتوى مباشرة.

السؤال: {query}

الفقرة:"""
        else:
            prompt = f"""Write a short paragraph (3-4 sentences) containing a detailed answer to the following question.
Write in a formal documentary style, as if it's an excerpt from an official document or report.
Don't start with "Answer:" or any introduction, just write the content directly.

Question: {query}

Paragraph:"""

        return prompt

    def _clean_hypothetical(self, text: str) -> str:
        """Clean up generated hypothetical."""
        if not text:
            return ""

        # Remove common prefixes
        prefixes_to_remove = [
            "الإجابة:", "Answer:", "الفقرة:", "Paragraph:",
            "الجواب:", "Response:", "الرد:", "المحتوى:",
        ]

        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # Remove quotes
        text = text.strip('"\'')

        # Limit length
        if len(text) > 500:
            # Find last sentence boundary
            last_period = text.rfind('.')
            if last_period > 200:
                text = text[:last_period + 1]
            else:
                text = text[:500] + "..."

        return text.strip()

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text."""
        if not text:
            return None

        try:
            embedding = self.embedder.embed(text)
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            return embedding
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def _combine_embeddings(
        self,
        query_emb: Optional[np.ndarray],
        hyde_emb: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Combine query and HyDE embeddings.

        Uses weighted average with normalization.
        """
        if query_emb is None and hyde_emb is None:
            return None

        if query_emb is None:
            return hyde_emb

        if hyde_emb is None:
            return query_emb

        # Weighted combination
        combined = self.alpha * query_emb + (1 - self.alpha) * hyde_emb

        # Normalize to unit vector
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined


class HyPEQueryEnhancer(HyDEQueryEnhancer):
    """
    HyPE - Hypothetical Passage Embeddings

    Extension of HyDE that generates multiple passage types:
    - Definition passages
    - Explanation passages
    - Example passages

    Then combines all embeddings for richer retrieval.
    """

    def enhance(self, query: str, mode: str = "combined") -> EnhancedQuery:
        """
        Enhanced version with multiple passage types.
        """
        import time
        start_time = time.time()

        # Get query embedding
        query_embedding = self._embed(query)

        # Generate different types of passages
        passage_types = self._detect_passage_types(query)
        passages = []

        for ptype in passage_types[:3]:  # Limit to 3 types
            passage = self._generate_typed_passage(query, ptype)
            if passage:
                passages.append(passage)

        if not passages:
            # Fallback to standard HyDE
            return super().enhance(query, mode)

        # Embed all passages
        passage_embeddings = [self._embed(p) for p in passages]
        passage_embeddings = [e for e in passage_embeddings if e is not None]

        # Average passage embeddings
        if passage_embeddings:
            hyde_embedding = np.mean(passage_embeddings, axis=0)
            hyde_embedding = hyde_embedding / np.linalg.norm(hyde_embedding)
        else:
            hyde_embedding = query_embedding

        # Combine with query
        combined_embedding = self._combine_embeddings(query_embedding, hyde_embedding)

        return EnhancedQuery(
            original_query=query,
            hypothetical_answer=" | ".join(passages),
            query_embedding=query_embedding,
            hyde_embedding=hyde_embedding,
            combined_embedding=combined_embedding,
            generation_time_ms=(time.time() - start_time) * 1000
        )

    def _detect_passage_types(self, query: str) -> List[str]:
        """Detect what types of passages would be helpful."""
        query_lower = query.lower()

        types = []

        # Definition query patterns
        if any(p in query_lower for p in ["ما هي", "ما هو", "what is", "define"]):
            types.append("definition")

        # How-to patterns
        if any(p in query_lower for p in ["كيف", "how to", "طريقة", "steps"]):
            types.append("procedure")

        # Comparison patterns
        if any(p in query_lower for p in ["مقارنة", "compare", "الفرق", "difference"]):
            types.append("comparison")

        # List patterns
        if any(p in query_lower for p in ["أنواع", "types", "قائمة", "list"]):
            types.append("enumeration")

        # Default: explanation
        if not types:
            types.append("explanation")

        return types

    def _generate_typed_passage(self, query: str, passage_type: str) -> str:
        """Generate passage of specific type."""
        type_prompts = {
            "definition": "اكتب تعريفاً رسمياً مختصراً (2-3 جمل) لما يلي:",
            "procedure": "اكتب الخطوات الرئيسية (3-4 خطوات) للقيام بما يلي:",
            "comparison": "اكتب مقارنة موجزة (3-4 نقاط) حول:",
            "enumeration": "اذكر العناصر الرئيسية (3-5 عناصر) المتعلقة بـ:",
            "explanation": "اشرح بإيجاز (3-4 جمل) ما يلي:"
        }

        prompt = f"{type_prompts.get(passage_type, type_prompts['explanation'])}\n{query}\n\n"

        try:
            response = requests.post(
                f"{self.llm_endpoint}/generate",
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 150,
                        "temperature": 0.6,
                        "do_sample": True
                    }
                },
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return self._clean_hypothetical(result.get('generated_text', ''))

        except Exception:
            pass

        return ""


# Singleton instance
_hyde_enhancer: Optional[HyDEQueryEnhancer] = None


def get_hyde_enhancer(
    embedder=None,
    **kwargs
) -> HyDEQueryEnhancer:
    """Get or create HyDE enhancer singleton."""
    global _hyde_enhancer

    if _hyde_enhancer is None:
        if embedder is None:
            from ..models.embedding_manager import get_embedding_manager
            embedder = get_embedding_manager()

        _hyde_enhancer = HyDEQueryEnhancer(
            embedder=embedder,
            **kwargs
        )

    return _hyde_enhancer
