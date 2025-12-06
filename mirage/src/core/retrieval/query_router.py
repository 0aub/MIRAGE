"""
MIRAGE V2 Query Router
Routes queries to the most appropriate retrieval mode based on query analysis.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import re

from .base_retriever import RetrievalMode


class QueryType(Enum):
    """Types of queries for routing decisions"""
    FACTUAL = "factual"            # Who, what, when, where
    EXPLANATORY = "explanatory"    # How, why
    COMPARATIVE = "comparative"    # Compare X and Y
    EXPLORATORY = "exploratory"    # What are the main themes...
    HOLISTIC = "holistic"          # Global themes, summaries, patterns (GraphRAG global search)
    ENTITY_CENTRIC = "entity"      # About specific entity
    RELATIONSHIP = "relationship"  # About connections
    UNKNOWN = "unknown"


@dataclass
class RoutingDecision:
    """Result of query routing"""
    query: str
    query_type: QueryType
    recommended_mode: RetrievalMode
    confidence: float
    reasoning: str
    alternative_modes: List[Tuple[RetrievalMode, float]] = field(default_factory=list)


class QueryRouter:
    """
    Routes queries to optimal retrieval modes.

    Uses rule-based analysis for efficiency, with optional LLM fallback.

    Routing logic:
    - Entity mentions + factual → LOCAL
    - Relationship questions → GLOBAL
    - Exploratory/theme questions → GLOBAL
    - General questions → HYBRID
    - Comparative → MIX
    - Short/simple → NAIVE
    """

    # Query patterns for classification (improved Arabic support)
    ENTITY_PATTERNS = [
        r"من هو|من هي|من هم",      # Who is/are (Arabic singular/plural)
        r"ما هو|ما هي|ما هم",      # What is/are (Arabic)
        r"المرشح|المرشحون",         # Nominee/nominees (Arabic)
        r"الفائز|الفائزون",         # Winner/winners (Arabic)
        r"who is|who are|who were",
        r"what is|what are",
        r"tell me about",
        r"أخبرني عن",              # Tell me about (Arabic)
        r"معلومات عن",             # Information about (Arabic)
    ]

    RELATIONSHIP_PATTERNS = [
        r"العلاقة بين",            # Relationship between (Arabic)
        r"كيف يرتبط|كيف ترتبط",    # How is related (Arabic)
        r"ما العلاقة",             # What is the relationship (Arabic)
        r"الارتباط بين",           # Connection between (Arabic)
        r"relationship between",
        r"how (?:is|are) .+ related",
        r"connection between",
        r"linked to",
    ]

    EXPLORATORY_PATTERNS = [
        r"أهم",                     # Most important (Arabic)
        r"الإنجازات",              # Achievements (Arabic)
        r"النتائج المحققة",         # Achieved results (Arabic)
        r"overview of",
        r"achievements",
    ]

    # Holistic patterns - for GraphRAG global search (map-reduce over communities)
    HOLISTIC_PATTERNS = [
        # Arabic patterns for holistic queries
        r"ما هي الموضوعات الرئيسية",    # What are the main themes
        r"الموضوعات الرئيسية",           # Main themes
        r"المواضيع الرئيسية",            # Main topics
        r"لخص|تلخيص|ملخص",              # Summarize/summary
        r"ما الأنماط",                   # What patterns
        r"الاتجاهات العامة",             # General trends
        r"بشكل عام|إجمالاً",             # In general/overall
        r"عبر جميع|عبر كل",              # Across all
        r"في قاعدة البيانات",            # In the database
        r"ما هي أبرز",                   # What are the most prominent
        r"ما هي أهم النقاط",             # What are the most important points
        r"استعرض",                       # Review/overview

        # English patterns for holistic queries
        r"what are the main themes",
        r"what are the key topics",
        r"summarize (?:the|all|everything)",
        r"summary of (?:the|all)",
        r"what patterns",
        r"what trends",
        r"across all (?:documents|data)",
        r"in general",
        r"overall",
        r"key takeaways",
        r"main insights",
        r"what can you tell me about",
    ]

    COMPARATIVE_PATTERNS = [
        r"مقارنة بين|مقارنة ب",    # Comparison between (Arabic)
        r"الفرق بين",              # Difference between (Arabic)
        r"تقارن|يقارن",            # Compare (Arabic)
        r"compare",
        r"difference between",
        r"vs\\.?|versus",
        r"how does .+ differ",
    ]

    EXPLANATORY_PATTERNS = [
        r"لماذا",                  # Why (Arabic)
        r"كيف ساهم|كيف ساهمت",     # How contributed (Arabic)
        r"كيف يعمل|كيف تعمل",      # How works (Arabic)
        r"why did|why does|why is",
        r"how did|how does|how is",
        r"how contributed",
        r"explain",
        r"اشرح",                   # Explain (Arabic)
    ]

    def __init__(
        self,
        default_mode: RetrievalMode = RetrievalMode.HYBRID,
        entity_threshold: float = 0.7,
        use_llm_routing: bool = False
    ):
        """
        Initialize query router.

        Args:
            default_mode: Fallback mode when uncertain
            entity_threshold: Confidence threshold for entity detection
            use_llm_routing: Whether to use LLM for complex routing
        """
        self.default_mode = default_mode
        self.entity_threshold = entity_threshold
        self.use_llm_routing = use_llm_routing

        # Compile patterns
        self._entity_re = self._compile_patterns(self.ENTITY_PATTERNS)
        self._relationship_re = self._compile_patterns(self.RELATIONSHIP_PATTERNS)
        self._exploratory_re = self._compile_patterns(self.EXPLORATORY_PATTERNS)
        self._holistic_re = self._compile_patterns(self.HOLISTIC_PATTERNS)
        self._comparative_re = self._compile_patterns(self.COMPARATIVE_PATTERNS)
        self._explanatory_re = self._compile_patterns(self.EXPLANATORY_PATTERNS)

        logger.info(f"QueryRouter initialized: default_mode={default_mode.value}")

    def _compile_patterns(self, patterns: List[str]) -> re.Pattern:
        """Compile list of patterns into single regex"""
        combined = "|".join(f"(?:{p})" for p in patterns)
        return re.compile(combined, re.IGNORECASE | re.UNICODE)

    def route(self, query: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Route a query to the optimal retrieval mode.

        Args:
            query: User query string
            context: Optional context (e.g., conversation history)

        Returns:
            RoutingDecision with recommended mode
        """
        context = context or {}
        query_lower = query.lower().strip()

        # Analyze query type
        query_type, type_confidence = self._classify_query_type(query_lower)

        # Determine mode based on query type and patterns
        mode, confidence, reasoning = self._determine_mode(
            query_lower, query_type, type_confidence, context
        )

        # Calculate alternative modes
        alternatives = self._get_alternatives(mode, confidence)

        return RoutingDecision(
            query=query,
            query_type=query_type,
            recommended_mode=mode,
            confidence=confidence,
            reasoning=reasoning,
            alternative_modes=alternatives
        )

    def _classify_query_type(self, query: str) -> Tuple[QueryType, float]:
        """Classify the query type"""
        # Check patterns in order of specificity
        # HOLISTIC first - these are the global search queries (GraphRAG)
        if self._holistic_re.search(query):
            return QueryType.HOLISTIC, 0.92

        if self._comparative_re.search(query):
            return QueryType.COMPARATIVE, 0.9

        if self._relationship_re.search(query):
            return QueryType.RELATIONSHIP, 0.85

        if self._exploratory_re.search(query):
            return QueryType.EXPLORATORY, 0.85

        if self._explanatory_re.search(query):
            return QueryType.EXPLANATORY, 0.8

        if self._entity_re.search(query):
            return QueryType.ENTITY_CENTRIC, 0.8

        # Check for factual indicators
        if self._is_factual_query(query):
            return QueryType.FACTUAL, 0.7

        return QueryType.UNKNOWN, 0.5

    def _is_factual_query(self, query: str) -> bool:
        """Check if query is seeking factual information"""
        factual_words = [
            "what", "who", "when", "where", "which",
            "ما", "من", "متى", "أين", "أي",  # Arabic
            "how many", "how much", "كم",
        ]
        return any(query.startswith(w) for w in factual_words)

    def _determine_mode(
        self,
        query: str,
        query_type: QueryType,
        type_confidence: float,
        context: Dict[str, Any]
    ) -> Tuple[RetrievalMode, float, str]:
        """Determine the optimal retrieval mode"""

        # Short queries → NAIVE (simple vector search is sufficient)
        # But check patterns first for short Arabic queries
        word_count = len(query.split())

        # Check for multi-hop indicators (even in short queries)
        multi_hop_patterns = [
            # Relationship patterns
            r"العلاقة بين|ما علاقة|الصلة بين|كيف ترتبط|كيف يرتبط",
            # Causal patterns
            r"لماذا|ما سبب|ما أسباب|أدى إلى|ما تأثير",
            r"وما النتائج|وما الأثر",    # And what results/impact
            r"ساهم.*وما|ساهمت.*وما",     # Contributed and what
            # Comparative patterns
            r"الفرق بين|ما الفرق|كيف تختلف|كيف يختلف|مقارنة بين",
            # Multi-entity connectors
            r" و .*و | بين ",
            # English patterns
            r"related to|relationship between|connection between",
            r"why|what caused|led to|resulted in",
            r"difference between|compared to|how different",
            r"and what|and how",
        ]
        is_multi_hop = any(re.search(p, query, re.IGNORECASE | re.UNICODE) for p in multi_hop_patterns)

        if word_count <= 2 and not is_multi_hop:
            return (
                RetrievalMode.NAIVE,
                0.8,
                f"Very short query ({word_count} words) - using simple vector search"
            )

        # Check for multi-hop queries (requires HYBRID)
        if is_multi_hop:
            return (
                RetrievalMode.HYBRID,
                0.9,
                "Multi-hop reasoning query - using hybrid for connected context"
            )

        # Route based on query type
        if query_type == QueryType.HOLISTIC:
            return (
                RetrievalMode.GLOBAL_SEARCH,
                type_confidence,
                "Holistic query - using GraphRAG global search (map-reduce over communities)"
            )

        if query_type == QueryType.COMPARATIVE:
            return (
                RetrievalMode.MIX,
                type_confidence,
                "Comparative query - using all modes for comprehensive results"
            )

        if query_type == QueryType.RELATIONSHIP:
            return (
                RetrievalMode.GLOBAL,
                type_confidence,
                "Relationship-focused query - using global search"
            )

        if query_type == QueryType.EXPLORATORY:
            return (
                RetrievalMode.GLOBAL,
                type_confidence,
                "Exploratory query - using global search for themes"
            )

        if query_type == QueryType.ENTITY_CENTRIC:
            return (
                RetrievalMode.LOCAL,
                type_confidence,
                "Entity-focused query - using local entity search"
            )

        if query_type == QueryType.EXPLANATORY:
            return (
                RetrievalMode.HYBRID,
                type_confidence,
                "Explanatory query - using hybrid for depth"
            )

        if query_type == QueryType.FACTUAL:
            # Factual can go either local or hybrid
            if self._has_named_entity(query):
                return (
                    RetrievalMode.LOCAL,
                    type_confidence,
                    "Factual query with named entity - using local search"
                )
            return (
                RetrievalMode.HYBRID,
                type_confidence * 0.9,
                "Factual query - using hybrid search"
            )

        # Default fallback
        return (
            self.default_mode,
            0.6,
            f"Unknown query type - using default {self.default_mode.value}"
        )

    def _has_named_entity(self, query: str) -> bool:
        """Check if query contains capitalized words (potential named entities)"""
        words = query.split()
        # Check for capitalized words (not at start of sentence)
        for i, word in enumerate(words[1:], 1):
            if word and word[0].isupper():
                return True
        return False

    def _get_alternatives(
        self,
        primary_mode: RetrievalMode,
        confidence: float
    ) -> List[Tuple[RetrievalMode, float]]:
        """Get alternative modes with their confidence scores"""
        alternatives = []

        # Always include HYBRID as fallback
        if primary_mode != RetrievalMode.HYBRID:
            alternatives.append((RetrievalMode.HYBRID, 0.7))

        # Mode-specific alternatives
        if primary_mode == RetrievalMode.LOCAL:
            alternatives.append((RetrievalMode.NAIVE, 0.5))
        elif primary_mode == RetrievalMode.GLOBAL:
            alternatives.append((RetrievalMode.LOCAL, 0.5))
        elif primary_mode == RetrievalMode.GLOBAL_SEARCH:
            alternatives.append((RetrievalMode.GLOBAL, 0.6))  # Fallback to relationship-based
        elif primary_mode == RetrievalMode.NAIVE:
            alternatives.append((RetrievalMode.LOCAL, 0.6))

        return alternatives

    def route_with_fallback(
        self,
        query: str,
        primary_failed: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> RetrievalMode:
        """
        Route with fallback logic.

        Args:
            query: User query
            primary_failed: Whether primary mode failed
            context: Optional context

        Returns:
            RetrievalMode to use
        """
        decision = self.route(query, context)

        if not primary_failed:
            return decision.recommended_mode

        # Use alternative if primary failed
        if decision.alternative_modes:
            logger.warning(
                f"Primary mode {decision.recommended_mode.value} failed, "
                f"falling back to {decision.alternative_modes[0][0].value}"
            )
            return decision.alternative_modes[0][0]

        # Ultimate fallback
        return RetrievalMode.NAIVE


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_query_router: Optional[QueryRouter] = None


def get_query_router(**kwargs) -> QueryRouter:
    """Get or create global QueryRouter"""
    global _query_router

    if _query_router is None:
        _query_router = QueryRouter(**kwargs)

    return _query_router
