"""
MIRAGE V4 Summary Validator
Validates that generated summaries are faithful to source material.

Features:
- Faithfulness checking (is the answer supported by sources?)
- Hallucination detection
- Citation verification
- Factual consistency scoring
- Arabic and English support
"""

import os
import json
import time
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

# Required imports - no fallback (enforced in Docker)
import redis

from ...config import settings


class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"      # All claims must be directly supported
    MODERATE = "moderate"  # Allow minor inferences
    LENIENT = "lenient"    # Allow reasonable paraphrasing


class ValidationStatus(Enum):
    """Validation result status"""
    VALID = "valid"              # Answer fully supported
    PARTIALLY_VALID = "partial"  # Some claims unsupported
    INVALID = "invalid"          # Major unsupported claims
    UNCERTAIN = "uncertain"      # Could not validate


@dataclass
class ValidationIssue:
    """A specific validation issue found"""
    issue_type: str          # "hallucination", "unsupported_claim", "citation_error"
    description: str         # Human-readable description
    text_span: Optional[str] # The problematic text
    severity: str            # "high", "medium", "low"
    suggestion: Optional[str] = None  # How to fix

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.issue_type,
            "description": self.description,
            "text_span": self.text_span,
            "severity": self.severity,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of summary validation"""
    status: ValidationStatus
    faithfulness_score: float      # 0-1: How faithful to sources
    coverage_score: float          # 0-1: How much of relevant info is covered
    citation_accuracy: float       # 0-1: Are citations correctly used
    overall_score: float           # Combined score
    issues: List[ValidationIssue]
    validation_time_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "faithfulness_score": self.faithfulness_score,
            "coverage_score": self.coverage_score,
            "citation_accuracy": self.citation_accuracy,
            "overall_score": self.overall_score,
            "issues": [i.to_dict() for i in self.issues],
            "validation_time_ms": self.validation_time_ms,
            "metadata": self.metadata,
        }

    @property
    def is_valid(self) -> bool:
        return self.status in [ValidationStatus.VALID, ValidationStatus.PARTIALLY_VALID]


class SummaryValidator:
    """
    Validates that generated summaries/answers are faithful to sources.

    Uses NLI-style entailment checking and LLM-based verification.
    """

    # Arabic validation prompt
    ARABIC_VALIDATION_PROMPT = """أنت خبير في التحقق من صحة المعلومات. مهمتك تقييم مدى توافق الإجابة مع المصادر.

**تعليمات:**
1. قارن كل ادعاء في الإجابة مع المصادر المقدمة
2. حدد أي معلومات غير مدعومة بالمصادر
3. تحقق من صحة استخدام الاقتباسات [1], [2], إلخ
4. قيّم مدى أمانة الإجابة للمصادر

**تنسيق الرد (JSON فقط):**
{
    "faithfulness": 0.0-1.0,
    "issues": [
        {"type": "نوع المشكلة", "text": "النص المعني", "severity": "high/medium/low", "suggestion": "اقتراح"}
    ],
    "reasoning": "تفسير مختصر"
}"""

    ENGLISH_VALIDATION_PROMPT = """You are an expert at verifying information accuracy. Your task is to evaluate how well the answer aligns with the sources.

**Instructions:**
1. Compare each claim in the answer against the provided sources
2. Identify any information not supported by sources
3. Verify that citations [1], [2], etc. are correctly used
4. Rate how faithful the answer is to the sources

**Response Format (JSON only):**
{
    "faithfulness": 0.0-1.0,
    "issues": [
        {"type": "issue_type", "text": "problematic text", "severity": "high/medium/low", "suggestion": "how to fix"}
    ],
    "reasoning": "brief explanation"
}"""

    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.MODERATE,
        min_faithfulness: float = 0.7,
        use_cache: bool = True
    ):
        """
        Initialize the summary validator.

        Args:
            level: Validation strictness level
            min_faithfulness: Minimum faithfulness score to pass
            use_cache: Whether to cache validation results
        """
        self.level = level
        self.min_faithfulness = min_faithfulness
        self.use_cache = use_cache

        # Initialize LLM provider
        self.provider = None
        self.model = None
        self._detect_provider()

        # Initialize Redis
        self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.redis_client.ping()

        # Statistics
        self._stats = {
            "validations_total": 0,
            "validations_passed": 0,
            "validations_failed": 0,
            "average_faithfulness": 0.0,
            "issues_found": 0,
        }

        logger.info(
            f"SummaryValidator initialized: level={level.value}, "
            f"min_faithfulness={min_faithfulness}"
        )

    def _detect_provider(self) -> None:
        """Auto-detect LLM provider"""
        if settings.use_tgi and settings.tgi_endpoint:
            self.provider = "tgi"
            self.model = "tgi"
            self.tgi_endpoint = settings.tgi_endpoint
            logger.info(f"Using TGI for validation: {settings.tgi_endpoint}")
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
            logger.warning("No LLM provider available for validation")

    def validate(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        query: Optional[str] = None,
        language: str = "auto"
    ) -> ValidationResult:
        """
        Validate that an answer is faithful to its sources.

        Args:
            answer: The generated answer to validate
            sources: List of source dicts with "text" field
            query: Optional original query for context
            language: Language of the content

        Returns:
            ValidationResult with scores and issues
        """
        start_time = time.time()
        self._stats["validations_total"] += 1

        # Detect language
        if language == "auto":
            arabic_ratio = len([c for c in answer if '\u0600' <= c <= '\u06FF']) / max(len(answer), 1)
            language = "ar" if arabic_ratio > 0.3 else "en"

        # Rule-based checks first
        rule_issues = self._rule_based_validation(answer, sources)

        # LLM-based validation
        if self.provider:
            llm_result = self._llm_validation(answer, sources, query, language)
        else:
            llm_result = None

        # Citation verification
        citation_accuracy = self._verify_citations(answer, sources)

        # Combine results
        if llm_result:
            faithfulness = llm_result.get("faithfulness", 0.5)
            llm_issues = llm_result.get("issues", [])
        else:
            faithfulness = self._heuristic_faithfulness(answer, sources)
            llm_issues = []

        # Convert LLM issues to ValidationIssue objects
        all_issues = list(rule_issues)
        for issue in llm_issues:
            all_issues.append(ValidationIssue(
                issue_type=issue.get("type", "unknown"),
                description=issue.get("description", issue.get("type", "")),
                text_span=issue.get("text"),
                severity=issue.get("severity", "medium"),
                suggestion=issue.get("suggestion"),
            ))

        # Calculate coverage (how much of source info is in answer)
        coverage = self._calculate_coverage(answer, sources)

        # Overall score
        overall = (faithfulness * 0.5 + citation_accuracy * 0.3 + coverage * 0.2)

        # Determine status
        if faithfulness >= self.min_faithfulness and len([i for i in all_issues if i.severity == "high"]) == 0:
            status = ValidationStatus.VALID
            self._stats["validations_passed"] += 1
        elif faithfulness >= 0.5:
            status = ValidationStatus.PARTIALLY_VALID
        elif faithfulness > 0:
            status = ValidationStatus.INVALID
            self._stats["validations_failed"] += 1
        else:
            status = ValidationStatus.UNCERTAIN

        self._stats["issues_found"] += len(all_issues)
        self._stats["average_faithfulness"] = (
            (self._stats["average_faithfulness"] * (self._stats["validations_total"] - 1) + faithfulness)
            / self._stats["validations_total"]
        )

        validation_time_ms = int((time.time() - start_time) * 1000)

        result = ValidationResult(
            status=status,
            faithfulness_score=faithfulness,
            coverage_score=coverage,
            citation_accuracy=citation_accuracy,
            overall_score=overall,
            issues=all_issues,
            validation_time_ms=validation_time_ms,
            metadata={
                "level": self.level.value,
                "language": language,
                "sources_count": len(sources),
            }
        )

        logger.info(
            f"Validation complete: status={status.value}, "
            f"faithfulness={faithfulness:.2f}, overall={overall:.2f}, "
            f"issues={len(all_issues)}"
        )

        return result

    def _rule_based_validation(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> List[ValidationIssue]:
        """Apply rule-based validation checks"""
        issues = []

        # Check 1: Empty answer
        if not answer.strip():
            issues.append(ValidationIssue(
                issue_type="empty_answer",
                description="The answer is empty",
                text_span=None,
                severity="high",
            ))
            return issues

        # Check 2: Answer too short for the sources
        source_text = " ".join(s.get("text", "") for s in sources)
        if len(answer.split()) < 10 and len(source_text.split()) > 100:
            issues.append(ValidationIssue(
                issue_type="too_brief",
                description="Answer seems too brief for the available information",
                text_span=None,
                severity="low",
            ))

        # Check 3: Answer much longer than sources (potential hallucination)
        if len(answer.split()) > len(source_text.split()) * 2:
            issues.append(ValidationIssue(
                issue_type="excessive_length",
                description="Answer is much longer than sources, may contain unsupported claims",
                text_span=None,
                severity="medium",
            ))

        # Check 4: Citations without brackets
        import re
        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, answer)

        if not citations and len(sources) > 0:
            issues.append(ValidationIssue(
                issue_type="missing_citations",
                description="Answer lacks citations to sources",
                text_span=None,
                severity="medium",
                suggestion="Add citations [1], [2], etc. to attribute information",
            ))

        # Check 5: Invalid citation numbers
        for citation in citations:
            citation_num = int(citation)
            if citation_num < 1 or citation_num > len(sources):
                issues.append(ValidationIssue(
                    issue_type="invalid_citation",
                    description=f"Citation [{citation}] refers to non-existent source",
                    text_span=f"[{citation}]",
                    severity="high",
                ))

        return issues

    def _verify_citations(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """Verify that citations are correctly used"""
        import re

        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, answer)

        if not citations:
            return 0.5  # Neutral score if no citations

        valid_citations = 0
        total_citations = len(citations)

        for citation in citations:
            citation_num = int(citation) - 1  # Convert to 0-indexed

            if 0 <= citation_num < len(sources):
                valid_citations += 1

        return valid_citations / max(total_citations, 1)

    def _calculate_coverage(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """Calculate how much of source information is covered"""
        # Simple word overlap approach
        answer_words = set(answer.lower().split())
        source_words = set()

        for source in sources:
            text = source.get("text", "")
            source_words.update(text.lower().split())

        # Filter common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "و", "في", "من", "على", "إلى"}
        answer_words = answer_words - stopwords
        source_words = source_words - stopwords

        if not source_words:
            return 0.5

        # Overlap ratio
        overlap = len(answer_words & source_words)
        coverage = min(1.0, overlap / max(len(source_words), 1) * 2)  # Scale up

        return coverage

    def _heuristic_faithfulness(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """Calculate faithfulness using heuristics when no LLM available"""
        # Word overlap approach
        answer_words = set(answer.lower().split())
        source_text = " ".join(s.get("text", "") for s in sources)
        source_words = set(source_text.lower().split())

        if not source_words:
            return 0.5

        overlap = len(answer_words & source_words)
        ratio = overlap / max(len(answer_words), 1)

        # Adjust based on length ratio
        length_ratio = len(answer.split()) / max(len(source_text.split()), 1)
        if length_ratio > 1.5:
            ratio *= 0.8  # Penalty for being longer than sources

        return min(1.0, ratio)

    def _llm_validation(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        query: Optional[str],
        language: str
    ) -> Optional[Dict[str, Any]]:
        """Use LLM for validation"""
        if not self.provider:
            return None

        # Build prompt
        system_prompt = self.ARABIC_VALIDATION_PROMPT if language == "ar" else self.ENGLISH_VALIDATION_PROMPT

        if language == "ar":
            user_prompt = f"""السؤال: {query or 'غير محدد'}

الإجابة المراد التحقق منها:
{answer}

المصادر:
"""
        else:
            user_prompt = f"""Question: {query or 'Not specified'}

Answer to validate:
{answer}

Sources:
"""

        for i, source in enumerate(sources[:5], 1):  # Limit sources
            text = source.get("text", "")[:500]  # Truncate
            user_prompt += f"\n[{i}] {text}\n"

        if language == "ar":
            user_prompt += "\nJSON:"
        else:
            user_prompt += "\nJSON:"

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
                        "temperature": 0.1,
                        "max_tokens": 512,
                    },
                    timeout=20
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
            else:
                from litellm import completion
                response = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=512,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content

            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())

        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            **self._stats,
            "provider": self.provider,
            "level": self.level.value,
            "pass_rate": (
                self._stats["validations_passed"] /
                max(self._stats["validations_total"], 1)
            ),
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_summary_validator: Optional[SummaryValidator] = None


def get_summary_validator() -> SummaryValidator:
    """Get or create global SummaryValidator instance"""
    global _summary_validator

    if _summary_validator is None:
        _summary_validator = SummaryValidator()

    return _summary_validator


def validate_summary(
    answer: str,
    sources: List[Dict[str, Any]],
    query: Optional[str] = None,
    language: str = "auto"
) -> ValidationResult:
    """Convenience function to validate a summary"""
    return get_summary_validator().validate(answer, sources, query, language)
