"""
RAG Evaluation Metrics (Optimized for SLMs like Allam)

Implements RAGAS-inspired metrics with heavy emphasis on rule-based
evaluation to minimize LLM calls while maintaining accuracy.

Metrics:
- Faithfulness: How factually accurate is the answer?
- Answer Relevancy: How relevant is the answer to the question?
- Context Recall: How much of the ground truth is in the context?
- Context Precision: How precise is the retrieved context?
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import re
import requests
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for RAG evaluation."""
    faithfulness: float  # 0-1, how factually accurate
    answer_relevancy: float  # 0-1, how relevant to question
    context_recall: float  # 0-1, how much ground truth retrieved
    context_precision: float  # 0-1, precision of retrieved context

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'faithfulness': self.faithfulness,
            'answer_relevancy': self.answer_relevancy,
            'context_recall': self.context_recall,
            'context_precision': self.context_precision,
            'average': (
                self.faithfulness +
                self.answer_relevancy +
                self.context_recall +
                self.context_precision
            ) / 4
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Faithfulness: {self.faithfulness:.2f}, "
            f"Answer Relevancy: {self.answer_relevancy:.2f}, "
            f"Context Recall: {self.context_recall:.2f}, "
            f"Context Precision: {self.context_precision:.2f}, "
            f"Average: {self.to_dict()['average']:.2f}"
        )


class RAGEvaluator:
    """
    Evaluator for RAG systems using hybrid approach.

    Strategy:
    - 70% rule-based metrics (fast, free, reliable)
    - 30% LLM-based validation (only when needed)

    This minimizes LLM calls while maintaining quality.
    """

    def __init__(self, llm_endpoint: Optional[str] = None):
        """
        Initialize evaluator.

        Args:
            llm_endpoint: Optional LLM endpoint for validation (TGI, OpenAI, etc.)
        """
        self.llm_endpoint = llm_endpoint or "http://tgi:8765"
        self.use_llm = False  # Default: pure rule-based

    def enable_llm_validation(self, endpoint: str = None):
        """
        Enable LLM validation for ambiguous cases.

        Args:
            endpoint: LLM endpoint (default: use initialization value)
        """
        if endpoint:
            self.llm_endpoint = endpoint
        self.use_llm = True
        logger.info(f"LLM validation enabled (endpoint: {self.llm_endpoint})")

    def disable_llm_validation(self):
        """Disable LLM validation (use only rule-based metrics)."""
        self.use_llm = False
        logger.info("LLM validation disabled (pure rule-based evaluation)")

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationMetrics:
        """
        Evaluate a RAG answer using hybrid metrics.

        Args:
            question: The user question
            answer: Generated answer
            contexts: Retrieved contexts used
            ground_truth: Optional ground truth answer

        Returns:
            EvaluationMetrics with scores

        Example:
            >>> evaluator = RAGEvaluator()
            >>> metrics = evaluator.evaluate_answer(
            ...     question="What is GraphRAG?",
            ...     answer="GraphRAG is a graph-based RAG system...",
            ...     contexts=["GraphRAG uses knowledge graphs..."]
            ... )
            >>> print(metrics)
        """
        # Faithfulness: hybrid rule-based + optional LLM
        faithfulness = self._evaluate_faithfulness_hybrid(answer, contexts)

        # Answer relevancy: rule-based with optional LLM validation
        answer_relevancy = self._evaluate_answer_relevancy_hybrid(question, answer)

        # Context metrics: pure rule-based
        context_recall = 0.0
        context_precision = 0.0

        if ground_truth:
            context_recall = self._evaluate_context_recall_rules(contexts, ground_truth)

        context_precision = self._evaluate_context_precision_rules(contexts, question)

        return EvaluationMetrics(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_recall=context_recall,
            context_precision=context_precision
        )

    # ========================================================================
    # Faithfulness Evaluation
    # ========================================================================

    def _evaluate_faithfulness_hybrid(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Evaluate if answer is faithful to contexts (not hallucinated).

        Uses hybrid approach:
        1. Extract claims from answer (rule-based)
        2. Check each claim against contexts (keyword matching)
        3. If score is ambiguous (0.3-0.7), optionally use LLM validation

        Args:
            answer: Generated answer
            contexts: Source contexts

        Returns:
            Faithfulness score (0-1)
        """
        if not answer or not contexts:
            return 0.0

        # Step 1: Extract claims from answer (rule-based)
        claims = self._extract_claims_rules(answer)

        if not claims:
            return 1.0  # No claims = nothing to verify = faithful

        # Step 2: Check each claim against contexts (keyword matching)
        supported_count = 0
        for claim in claims:
            if self._claim_supported_by_contexts_rules(claim, contexts):
                supported_count += 1

        rule_score = supported_count / len(claims)

        # Step 3: LLM validation (only if enabled and score is ambiguous)
        if self.use_llm and 0.3 < rule_score < 0.7:
            try:
                llm_score = self._llm_faithfulness_check(answer, contexts)
                # Average rule-based and LLM scores
                return (rule_score + llm_score) / 2
            except Exception as e:
                logger.warning(f"LLM validation failed, using rule score: {e}")

        return rule_score

    def _extract_claims_rules(self, answer: str) -> List[str]:
        """Extract factual claims using sentence splitting."""
        # Split by sentence boundaries
        sentences = re.split(r'[.!?]+|[.!?؟]+', answer)  # English + Arabic
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims

    def _claim_supported_by_contexts_rules(
        self,
        claim: str,
        contexts: List[str]
    ) -> bool:
        """
        Check if claim is supported using keyword matching.

        Args:
            claim: Single claim to verify
            contexts: Source contexts

        Returns:
            True if claim is likely supported
        """
        # Extract key terms from claim
        claim_terms = set(self._extract_key_terms(claim))

        if not claim_terms:
            return True  # No terms to check

        for context in contexts:
            context_terms = set(self._extract_key_terms(context))

            # If >40% of claim terms are in context, consider supported
            overlap = len(claim_terms & context_terms)
            if overlap / len(claim_terms) > 0.4:
                return True

        return False

    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms (nouns, named entities, important words).

        Simple approach: Remove stop words, keep longer words.
        """
        # Common stop words (English + Arabic)
        stop_words = {
            # English
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who',
            'when', 'where', 'why', 'how',
            # Arabic common words
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
            'الذي', 'التي', 'هو', 'هي', 'هم', 'هن', 'أن', 'إن', 'كان', 'كانت',
            'يكون', 'تكون', 'ما', 'من', 'إلى', 'عند', 'لدى', 'حول', 'خلال',
        }

        # Tokenize
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter: remove stop words, keep words > 3 characters
        key_terms = [
            w for w in words
            if w not in stop_words and len(w) > 3
        ]

        return key_terms

    def _llm_faithfulness_check(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        LLM-based faithfulness check (used sparingly).

        Args:
            answer: Generated answer
            contexts: Source contexts

        Returns:
            Faithfulness score (0-1)
        """
        # Limit context length
        context_text = "\n\n".join(contexts[:3])[:1000]

        prompt = f"""هل الإجابة التالية مدعومة بالسياقات المقدمة؟

الإجابة: {answer[:500]}

السياقات:
{context_text}

قيّم من 0.0 (غير مدعومة) إلى 1.0 (مدعومة تماماً).
أعد رقماً فقط.
"""

        try:
            response = requests.post(
                f"{self.llm_endpoint}/generate",
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 10,
                        "temperature": 0.1
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                content = response.json().get('generated_text', '').strip()
                # Extract number from response
                numbers = re.findall(r'0?\.\d+|[01]\.0|[01]', content)
                if numbers:
                    score = float(numbers[0])
                    return max(0.0, min(1.0, score))

        except Exception as e:
            logger.debug(f"LLM faithfulness check failed: {e}")

        return 0.5  # Default: ambiguous

    # ========================================================================
    # Answer Relevancy Evaluation
    # ========================================================================

    def _evaluate_answer_relevancy_hybrid(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Evaluate if answer is relevant to the question.

        Uses hybrid approach:
        1. Rule-based keyword overlap check
        2. If overlap is clear (>80% or <20%), return rule score
        3. If ambiguous, optionally use LLM validation

        Args:
            question: User question
            answer: Generated answer

        Returns:
            Relevancy score (0-1)
        """
        if not answer or not question:
            return 0.0

        # Rule-based quick check
        question_terms = set(self._extract_key_terms(question))
        answer_terms = set(self._extract_key_terms(answer))

        if not question_terms:
            return 1.0  # Can't evaluate without terms

        overlap = len(question_terms & answer_terms)
        overlap_ratio = overlap / len(question_terms)

        # Clear cases: don't need LLM
        if overlap_ratio > 0.8:
            return 1.0  # High overlap = very relevant

        if overlap_ratio < 0.2:
            return 0.2  # Low overlap = not relevant

        # Ambiguous case: use LLM if enabled
        if self.use_llm:
            try:
                return self._llm_relevancy_check(question, answer)
            except Exception as e:
                logger.warning(f"LLM relevancy check failed, using rule score: {e}")

        # Return rule-based score (scaled to 0-1)
        return overlap_ratio

    def _llm_relevancy_check(self, question: str, answer: str) -> float:
        """
        LLM-based relevancy check.

        Args:
            question: User question
            answer: Generated answer

        Returns:
            Relevancy score (0-1)
        """
        prompt = f"""هل الإجابة ذات صلة بالسؤال؟

السؤال: {question}
الإجابة: {answer[:300]}

قيّم من 0.0 إلى 1.0
أعد رقماً فقط.
"""

        try:
            response = requests.post(
                f"{self.llm_endpoint}/generate",
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 10,
                        "temperature": 0.1
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                content = response.json().get('generated_text', '').strip()
                numbers = re.findall(r'0?\.\d+|[01]\.0|[01]', content)
                if numbers:
                    score = float(numbers[0])
                    return max(0.0, min(1.0, score))

        except Exception as e:
            logger.debug(f"LLM relevancy check failed: {e}")

        return 0.5

    # ========================================================================
    # Context Metrics (Pure Rule-Based)
    # ========================================================================

    def _evaluate_context_recall_rules(
        self,
        contexts: List[str],
        ground_truth: str
    ) -> float:
        """
        Rule-based recall calculation.

        Measures: What fraction of ground truth info is in contexts?

        Args:
            contexts: Retrieved contexts
            ground_truth: Ground truth answer

        Returns:
            Recall score (0-1)
        """
        if not ground_truth or not contexts:
            return 0.0

        gt_terms = set(self._extract_key_terms(ground_truth))

        if not gt_terms:
            return 0.0

        context_terms = set()
        for ctx in contexts:
            context_terms.update(self._extract_key_terms(ctx))

        overlap = len(gt_terms & context_terms)
        return overlap / len(gt_terms)

    def _evaluate_context_precision_rules(
        self,
        contexts: List[str],
        question: str
    ) -> float:
        """
        Rule-based precision calculation.

        Measures: What fraction of contexts are relevant to question?

        Args:
            contexts: Retrieved contexts
            question: User question

        Returns:
            Precision score (0-1)
        """
        if not contexts or not question:
            return 0.0

        question_terms = set(self._extract_key_terms(question))

        if not question_terms:
            return 0.0

        relevant_count = 0

        for context in contexts:
            context_terms = set(self._extract_key_terms(context))
            overlap = len(question_terms & context_terms)

            # If >30% of question terms in context, consider relevant
            if overlap / len(question_terms) > 0.3:
                relevant_count += 1

        return relevant_count / len(contexts)

    # ========================================================================
    # Batch Evaluation
    # ========================================================================

    def evaluate_batch(
        self,
        test_cases: List[Dict]
    ) -> Dict:
        """
        Evaluate multiple test cases.

        Args:
            test_cases: List of dicts with 'question', 'answer', 'contexts', optional 'ground_truth'

        Returns:
            Dict with aggregated metrics

        Example:
            >>> test_cases = [
            ...     {
            ...         "question": "What is GraphRAG?",
            ...         "answer": "GraphRAG is...",
            ...         "contexts": ["..."],
            ...         "ground_truth": "..."
            ...     },
            ...     # ... more cases
            ... ]
            >>> results = evaluator.evaluate_batch(test_cases)
            >>> print(f"Average faithfulness: {results['avg_faithfulness']:.2f}")
        """
        all_metrics = []

        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Evaluating test case {i}/{len(test_cases)}")

            metrics = self.evaluate_answer(
                question=test_case.get('question', ''),
                answer=test_case.get('answer', ''),
                contexts=test_case.get('contexts', []),
                ground_truth=test_case.get('ground_truth')
            )

            all_metrics.append(metrics)

        # Aggregate results
        avg_metrics = {
            'avg_faithfulness': sum(m.faithfulness for m in all_metrics) / len(all_metrics),
            'avg_answer_relevancy': sum(m.answer_relevancy for m in all_metrics) / len(all_metrics),
            'avg_context_recall': sum(m.context_recall for m in all_metrics) / len(all_metrics),
            'avg_context_precision': sum(m.context_precision for m in all_metrics) / len(all_metrics),
            'test_count': len(test_cases),
            'individual_results': [m.to_dict() for m in all_metrics]
        }

        avg_metrics['overall_average'] = (
            avg_metrics['avg_faithfulness'] +
            avg_metrics['avg_answer_relevancy'] +
            avg_metrics['avg_context_recall'] +
            avg_metrics['avg_context_precision']
        ) / 4

        return avg_metrics
