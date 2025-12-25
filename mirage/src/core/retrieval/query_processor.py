"""
Advanced Query Processing - Phase 5 GraphRAG Enhancement

Features:
1. Query normalization: Arabic diacritics, letter normalization
2. Query expansion: Add synonyms, related terms
3. Query decomposition: Break complex queries into sub-queries
4. Intent detection: Understand query type
5. Language detection: Handle Arabic/English/mixed

This improves retrieval by understanding what the user really wants.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from ..utils.arabic import normalize_arabic as _normalize_arabic_text


class QueryIntent(Enum):
    """Types of query intent."""
    FACTUAL = "factual"           # What is X?
    COMPARATIVE = "comparative"    # How does X compare to Y?
    PROCEDURAL = "procedural"      # How to do X?
    EXPLORATORY = "exploratory"    # What are the main themes?
    ENTITY_LOOKUP = "entity"       # Who is X? What is organization Y?
    RELATIONAL = "relational"      # What is the relationship between X and Y?
    AGGREGATIVE = "aggregative"    # How many X? List all Y


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = 1           # Single concept
    MODERATE = 2         # Multiple concepts or constraints
    COMPLEX = 3          # Multi-hop reasoning needed


@dataclass
class ProcessedQuery:
    """Result of query processing."""
    original: str
    normalized: str
    language: str                  # 'ar', 'en', 'mixed'
    intent: QueryIntent
    complexity: QueryComplexity
    key_terms: List[str] = field(default_factory=list)
    expansions: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    entities_mentioned: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "language": self.language,
            "intent": self.intent.value,
            "complexity": self.complexity.value,
            "key_terms": self.key_terms,
            "expansions": self.expansions,
            "sub_queries": self.sub_queries,
            "entities_mentioned": self.entities_mentioned,
        }


class QueryProcessor:
    """
    Advanced query processing for improved retrieval.

    Usage:
        processor = QueryProcessor()
        processed = processor.process("ما هي جائزة الحكومة الرقمية؟")

        # Use processed query
        print(processed.intent)       # QueryIntent.FACTUAL
        print(processed.expansions)   # ['جائزة', 'مكافأة', 'تكريم']
        print(processed.sub_queries)  # []
    """

    # NOTE: Arabic normalization now uses shared utils.arabic module

    # Intent detection patterns (Arabic)
    INTENT_PATTERNS = {
        QueryIntent.FACTUAL: [
            r'^ما\s+(?:هو|هي|هم)\s+',     # ما هو/هي
            r'^ماذا\s+(?:يعني|تعني)\s+',   # ماذا يعني
            r'^عرّف\s+',                   # Define
            r'^what\s+is\s+',              # English
        ],
        QueryIntent.COMPARATIVE: [
            r'مقارنة\s+بين',               # Compare between
            r'الفرق\s+بين',                # Difference between
            r'أفضل\s+من',                  # Better than
            r'compare\s+',                  # English
            r'difference\s+between',        # English
        ],
        QueryIntent.PROCEDURAL: [
            r'^كيف\s+',                    # How
            r'^طريقة\s+',                  # Method
            r'^خطوات\s+',                  # Steps
            r'^how\s+to\s+',               # English
        ],
        QueryIntent.EXPLORATORY: [
            r'ما\s+هي\s+(?:أهم|الرئيسية)',  # What are the main
            r'ما\s+هي\s+(?:المواضيع|الأفكار)', # What are the topics/ideas
            r'لخّص',                        # Summarize
            r'overview\s+of',               # English
        ],
        QueryIntent.ENTITY_LOOKUP: [
            r'^من\s+(?:هو|هي|هم)\s+',       # Who is
            r'^أين\s+(?:يقع|تقع)\s+',       # Where is
            r'^who\s+is\s+',                # English
        ],
        QueryIntent.RELATIONAL: [
            r'علاقة\s+بين',                 # Relationship between
            r'كيف\s+(?:يرتبط|ترتبط)',       # How is related
            r'relationship\s+between',       # English
        ],
        QueryIntent.AGGREGATIVE: [
            r'^كم\s+',                      # How many
            r'^عدد\s+',                     # Number of
            r'اذكر\s+(?:جميع|كل)',          # List all
            r'^list\s+(?:all)?',            # English
        ],
    }

    # Complexity indicators
    COMPLEXITY_INDICATORS = {
        QueryComplexity.COMPLEX: [
            r'و(?:أيضاً|كذلك)',             # And also
            r'بينما',                        # While
            r'إذا\s+كان',                   # If
            r'في\s+حالة',                   # In case
            r'multiple',                     # English
        ],
        QueryComplexity.MODERATE: [
            r'\s+و\s+',                     # And
            r'\s+أو\s+',                    # Or
            r'مع\s+',                       # With
        ],
    }

    # Common Arabic synonyms for query expansion
    ARABIC_SYNONYMS = {
        'جائزة': ['مكافأة', 'تكريم', 'وسام'],
        'حكومة': ['جهات حكومية', 'قطاع حكومي', 'قطاع عام'],
        'رقمية': ['تقنية', 'إلكترونية', 'رقمي'],
        'برنامج': ['مبادرة', 'مشروع', 'خطة'],
        'خدمات': ['خدمة', 'منتجات'],
        'شركة': ['مؤسسة', 'جهة', 'منظمة'],
        'تحول': ['تطوير', 'تغيير', 'انتقال'],
        'شريك': ['شراكة', 'تعاون', 'حليف'],
        'وزارة': ['هيئة', 'جهة حكومية', 'مؤسسة'],
        'مشروع': ['مبادرة', 'برنامج'],
        'أهداف': ['غايات', 'رؤية'],
        'نتائج': ['مخرجات', 'إنجازات'],
    }

    def __init__(self, llm_client=None, use_llm_expansion: bool = False):
        """
        Initialize query processor.

        Args:
            llm_client: LLM for advanced expansion (optional)
            use_llm_expansion: Whether to use LLM for expansion
        """
        self.llm_client = llm_client
        self.use_llm_expansion = use_llm_expansion and llm_client is not None
        logger.info(f"QueryProcessor initialized: use_llm_expansion={self.use_llm_expansion}")

    def process(self, query: str) -> ProcessedQuery:
        """
        Full query processing pipeline.

        Args:
            query: Raw user query

        Returns:
            ProcessedQuery with all analysis
        """
        # Step 1: Normalize
        normalized = self._normalize(query)

        # Step 2: Detect language
        language = self._detect_language(normalized)

        # Step 3: Detect intent
        intent = self._detect_intent(normalized)

        # Step 4: Assess complexity
        complexity = self._assess_complexity(normalized)

        # Step 5: Extract key terms
        key_terms = self._extract_key_terms(normalized, language)

        # Step 6: Expand query
        expansions = self._expand_query(normalized, language)

        # Step 7: Decompose if complex
        sub_queries = self._decompose_if_needed(normalized, complexity)

        # Step 8: Extract entities
        entities = self._extract_entities(normalized)

        return ProcessedQuery(
            original=query,
            normalized=normalized,
            language=language,
            intent=intent,
            complexity=complexity,
            key_terms=key_terms,
            expansions=expansions,
            sub_queries=sub_queries,
            entities_mentioned=entities
        )

    def _normalize(self, query: str) -> str:
        """Normalize query text."""
        # Remove extra whitespace
        query = ' '.join(query.split())

        # Normalize Arabic text (diacritics, letter variants)
        query = _normalize_arabic_text(query)

        # Remove punctuation at edges
        query = query.strip('?!.,،؟')

        return query.strip()

    def _detect_language(self, text: str) -> str:
        """Detect query language."""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total = arabic_chars + english_chars

        if total == 0:
            return 'ar'  # Default to Arabic

        arabic_ratio = arabic_chars / total

        if arabic_ratio > 0.8:
            return 'ar'
        elif arabic_ratio < 0.2:
            return 'en'
        else:
            return 'mixed'

    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent from patterns."""
        query_lower = query.lower()

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent

        # Default to factual
        return QueryIntent.FACTUAL

    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity."""
        query_lower = query.lower()

        # Check for complex indicators
        for patterns in self.COMPLEXITY_INDICATORS.get(QueryComplexity.COMPLEX, []):
            if re.search(patterns, query_lower):
                return QueryComplexity.COMPLEX

        for patterns in self.COMPLEXITY_INDICATORS.get(QueryComplexity.MODERATE, []):
            if re.search(patterns, query_lower):
                return QueryComplexity.MODERATE

        # Check word count
        word_count = len(query.split())
        if word_count > 15:
            return QueryComplexity.COMPLEX
        elif word_count > 8:
            return QueryComplexity.MODERATE

        return QueryComplexity.SIMPLE

    def _extract_key_terms(self, query: str, language: str) -> List[str]:
        """Extract key terms from query."""
        # Arabic stopwords
        arabic_stopwords = {
            'ما', 'هو', 'هي', 'هل', 'من', 'في', 'على', 'إلى', 'عن',
            'مع', 'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي', 'و', 'أو',
            'ان', 'أن', 'لم', 'لن', 'كان', 'كانت', 'يكون', 'تكون'
        }

        # English stopwords
        english_stopwords = {
            'what', 'is', 'the', 'a', 'an', 'how', 'who', 'which',
            'where', 'when', 'why', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'to', 'of',
            'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'or', 'and'
        }

        stopwords = arabic_stopwords | english_stopwords

        # Tokenize and filter
        words = query.split()
        key_terms = []

        for word in words:
            # Remove definite article for Arabic
            clean_word = re.sub(r'^ال', '', word)
            if clean_word.lower() not in stopwords and len(clean_word) > 2:
                key_terms.append(word)

        return key_terms[:10]  # Top 10 terms

    def _expand_query(self, query: str, language: str) -> List[str]:
        """Expand query with synonyms."""
        if self.use_llm_expansion:
            return self._llm_expand(query)

        expansions = []

        # Rule-based expansion using synonym dictionary
        for word, synonyms in self.ARABIC_SYNONYMS.items():
            if word in query:
                for syn in synonyms[:2]:
                    expanded = query.replace(word, syn)
                    if expanded not in expansions and expanded != query:
                        expansions.append(expanded)

        return expansions[:5]

    def _llm_expand(self, query: str) -> List[str]:
        """Use LLM for query expansion."""
        if not self.llm_client:
            return []

        try:
            prompt = f"""أضف مصطلحات مرادفة وذات صلة لتحسين نتائج البحث.
الاستعلام: {query}
اكتب 3 مصطلحات إضافية (واحد في كل سطر):"""

            response = self.llm_client.generate(prompt, max_tokens=100)
            lines = response.strip().split('\n')

            expansions = []
            for line in lines[:3]:
                term = line.strip()
                if term and not term.startswith('-'):
                    expansions.append(term)

            return expansions

        except Exception as e:
            logger.warning(f"LLM expansion failed: {e}")
            return []

    def _decompose_if_needed(
        self,
        query: str,
        complexity: QueryComplexity
    ) -> List[str]:
        """Decompose complex queries into sub-queries."""
        if complexity == QueryComplexity.SIMPLE:
            return []

        sub_queries = []

        # Simple rule-based decomposition
        # Split on conjunctions
        parts = re.split(r'\s+و\s+|\s+و(?:كذلك|أيضا)\s+', query)

        if len(parts) > 1:
            for part in parts:
                part = part.strip()
                if len(part) > 5:  # Minimum length
                    sub_queries.append(part)

        # If complex and we have LLM, use it for better decomposition
        if complexity == QueryComplexity.COMPLEX and self.llm_client and not sub_queries:
            try:
                prompt = f"""قسّم هذا الاستعلام إلى أسئلة فرعية بسيطة:
الاستعلام: {query}
الأسئلة الفرعية (واحد في كل سطر):"""

                response = self.llm_client.generate(prompt, max_tokens=150)
                lines = response.strip().split('\n')

                for line in lines[:3]:
                    q = line.strip()
                    if q and len(q) > 5:
                        sub_queries.append(q)

            except Exception as e:
                logger.warning(f"LLM decomposition failed: {e}")

        return sub_queries[:4]

    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entity mentions from query."""
        entities = []

        # Arabic definite noun patterns (potential entities)
        # Words starting with "ال" that are likely entity names
        pattern = r'ال[\u0600-\u06FF]{3,}'
        matches = re.findall(pattern, query)

        for match in matches:
            if match not in ['الذي', 'التي', 'الذين', 'اللاتي', 'اللواتي']:
                entities.append(match)

        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        return list(set(entities))[:10]

    def get_expanded_queries(self, query: str) -> List[str]:
        """
        Get all expanded versions of a query for multi-query retrieval.

        Args:
            query: Original query

        Returns:
            List of query variations including original
        """
        processed = self.process(query)

        queries = [processed.normalized]

        # Add expansions
        queries.extend(processed.expansions)

        # Add sub-queries
        queries.extend(processed.sub_queries)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        return unique[:5]  # Max 5 queries


# Singleton
_query_processor: Optional[QueryProcessor] = None


def get_query_processor(**kwargs) -> QueryProcessor:
    """Get or create query processor singleton."""
    global _query_processor
    if _query_processor is None:
        _query_processor = QueryProcessor(**kwargs)
    return _query_processor
