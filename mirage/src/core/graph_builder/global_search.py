"""
Global Search Module for GraphRAG

DEPRECATED: This module is maintained for backward compatibility.
For new code, use core.retrieval.global_search which provides:
- Async support
- Better error handling
- More detailed result types (PartialAnswer, GlobalSearchResult)
- Integration with RetrievalEngine

Implements map-reduce search over community summaries to answer
high-level, holistic questions about the knowledge base.

This is THE killer feature that differentiates GraphRAG from traditional RAG:
- Traditional RAG: "Who works at IBM?" → searches specific facts
- GraphRAG Global: "What are the main themes?" → searches summaries

Uses Allam LLM for map-reduce query answering.
"""

import json
import warnings
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import requests

# Import centralized constants
from core.config.constants import (
    MAX_COMMUNITIES_TO_SEARCH,
    GLOBAL_SEARCH_MIN_RELEVANCE,
    LLM_TEMPERATURE_FACTUAL,
    MAX_SUMMARY_TOKENS,
    TGI_ENDPOINT_DEFAULT,
)

logger = logging.getLogger(__name__)

# Emit deprecation warning on import
warnings.warn(
    "core.graph_builder.global_search is deprecated. "
    "Use core.retrieval.global_search for new code.",
    DeprecationWarning,
    stacklevel=2
)


@dataclass
class SearchResult:
    """Result from global search."""
    query: str
    answer: str
    themes: List[str]
    communities_searched: int
    confidence: float
    intermediate_answers: List[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "themes": self.themes,
            "communities_searched": self.communities_searched,
            "confidence": self.confidence,
            "intermediate_answers": self.intermediate_answers or []
        }


class GlobalSearchEngine:
    """
    Global search engine using map-reduce over community summaries.

    Strategy (Map-Reduce):
    1. MAP: Query each community summary independently
    2. REDUCE: Combine all intermediate answers into final answer

    Best for:
    - "What are the main themes?"
    - "Summarize everything about X"
    - "What are the relationships between A and B?"
    - High-level, holistic questions

    Not good for:
    - "Who works at IBM?" (use Local Search instead)
    - Specific fact retrieval
    - Recent information (summaries may be outdated)

    Usage:
        search_engine = GlobalSearchEngine(neo4j_client, llm_endpoint)
        result = search_engine.search(
            query="What are the main themes in the knowledge base?",
            level=2
        )
        print(result.answer)
    """

    def __init__(
        self,
        neo4j_client,
        llm_endpoint: str = None,
        max_tokens_per_answer: int = None,
        temperature: float = None,
        max_communities: int = None
    ):
        """
        Initialize global search engine.

        Args:
            neo4j_client: Neo4j client instance
            llm_endpoint: TGI endpoint for Allam (default: from constants)
            max_tokens_per_answer: Maximum tokens per answer (default: from constants)
            temperature: LLM temperature (default: from constants, 0.3 = more deterministic)
            max_communities: Maximum communities to search (default: from constants)
        """
        self.neo4j_client = neo4j_client
        self.llm_endpoint = llm_endpoint or TGI_ENDPOINT_DEFAULT
        self.max_tokens_per_answer = max_tokens_per_answer or MAX_SUMMARY_TOKENS
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE_FACTUAL
        self.max_communities = max_communities or MAX_COMMUNITIES_TO_SEARCH

    def search(
        self,
        query: str,
        level: int = 1,
        theme_filter: Optional[List[str]] = None
    ) -> SearchResult:
        """
        Execute global search over community summaries.

        Args:
            query: User question (holistic, high-level)
            level: Community level to search (0=finest, higher=coarser)
            theme_filter: Optional list of themes to filter communities

        Returns:
            SearchResult with answer and metadata

        Example:
            >>> engine = GlobalSearchEngine(neo4j_client)
            >>> result = engine.search("What are the main topics?", level=1)
            >>> print(result.answer)
        """
        logger.info(f"Global search: '{query}' at level {level}")

        # Step 1: Retrieve community summaries
        summaries = self._get_community_summaries(
            level=level,
            theme_filter=theme_filter,
            limit=self.max_communities
        )

        if not summaries:
            logger.warning(f"No summaries found at level {level}")
            return SearchResult(
                query=query,
                answer="No information found. Please ensure communities have been summarized.",
                themes=[],
                communities_searched=0,
                confidence=0.0
            )

        logger.info(f"Retrieved {len(summaries)} community summaries for search")

        # Step 2: MAP - Query each community summary
        intermediate_answers = self._map_phase(query, summaries)

        logger.info(f"MAP phase complete: {len(intermediate_answers)} intermediate answers")

        # Step 3: REDUCE - Combine intermediate answers
        final_answer, themes = self._reduce_phase(query, intermediate_answers)

        logger.info(f"REDUCE phase complete: Generated final answer ({len(final_answer)} chars)")

        # Calculate confidence (simple heuristic)
        confidence = min(1.0, len(intermediate_answers) / 5.0)  # Max confidence at 5+ communities

        return SearchResult(
            query=query,
            answer=final_answer,
            themes=themes,
            communities_searched=len(summaries),
            confidence=confidence,
            intermediate_answers=intermediate_answers
        )

    def search_by_theme(
        self,
        theme: str,
        level: int = 1
    ) -> SearchResult:
        """
        Search communities filtered by theme.

        Args:
            theme: Theme to search for (e.g., "technology", "التكنولوجيا")
            level: Community level

        Returns:
            SearchResult with theme-filtered results

        Example:
            >>> result = engine.search_by_theme("AI", level=0)
        """
        query = f"What information is available about {theme}?"

        return self.search(
            query=query,
            level=level,
            theme_filter=[theme]
        )

    def _get_community_summaries(
        self,
        level: int,
        theme_filter: Optional[List[str]],
        limit: int
    ) -> List[Dict]:
        """
        Retrieve community summaries from Neo4j.

        Args:
            level: Community hierarchy level
            theme_filter: Optional themes to filter by
            limit: Maximum summaries to retrieve

        Returns:
            List of summary dicts
        """
        # Build theme filter clause
        theme_clause = ""
        params = {'level': level, 'limit': limit}

        if theme_filter:
            # Case-insensitive theme matching
            theme_conditions = [
                f"ANY(theme IN c.themes WHERE toLower(theme) CONTAINS toLower('{theme}'))"
                for theme in theme_filter
            ]
            theme_clause = f"AND ({' OR '.join(theme_conditions)})"

        query = f"""
        MATCH (c:Community {{level: $level}})
        WHERE c.summary IS NOT NULL
        {theme_clause}
        RETURN
            c.id as community_id,
            c.summary as summary,
            c.themes as themes,
            c.size as size
        ORDER BY c.size DESC
        LIMIT $limit
        """

        try:
            results = self.neo4j_client.execute_query(query, params)

            summaries = []
            for record in results:
                summaries.append({
                    'community_id': record.get('community_id'),
                    'summary': record.get('summary'),
                    'themes': record.get('themes', []),
                    'size': record.get('size', 0)
                })

            return summaries

        except Exception as e:
            logger.error(f"Error retrieving community summaries: {e}")
            return []

    def _map_phase(
        self,
        query: str,
        summaries: List[Dict]
    ) -> List[str]:
        """
        MAP phase: Query each community summary independently.

        Args:
            query: User question
            summaries: List of community summaries

        Returns:
            List of intermediate answers (one per community)
        """
        intermediate_answers = []

        for i, summary_data in enumerate(summaries):
            summary_text = summary_data['summary']
            community_id = summary_data['community_id']

            logger.debug(f"MAP {i+1}/{len(summaries)}: Querying {community_id}")

            # Generate prompt for this community
            answer = self._query_single_summary(query, summary_text, community_id)

            if answer and len(answer.strip()) > 10:  # Filter out trivial answers
                intermediate_answers.append(answer)

        return intermediate_answers

    def _query_single_summary(
        self,
        query: str,
        summary: str,
        community_id: str
    ) -> str:
        """
        Query a single community summary.

        Args:
            query: User question
            summary: Community summary text
            community_id: Community identifier

        Returns:
            Answer based on this summary
        """
        # Build prompt
        prompt = f"""استنادًا إلى الملخص التالي، أجب على السؤال.

الملخص:
{summary}

السؤال: {query}

إذا لم يحتوي الملخص على معلومات ذات صلة، أجب بـ "لا توجد معلومات ذات صلة".

الإجابة (2-3 جمل):
"""

        # Call Allam
        try:
            response = requests.post(
                f"{self.llm_endpoint}/generate",
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": self.max_tokens_per_answer,
                        "temperature": self.temperature,
                        "top_p": 0.95,
                        "do_sample": True,
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get('generated_text', '').strip()

                # Filter out "no information" responses
                if any(phrase in answer.lower() for phrase in [
                    'لا توجد معلومات',
                    'no information',
                    'not relevant',
                    'غير ذات صلة'
                ]):
                    return ""

                return answer

            else:
                logger.error(f"LLM request failed: {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"Error querying summary for {community_id}: {e}")
            return ""

    def _reduce_phase(
        self,
        query: str,
        intermediate_answers: List[str]
    ) -> Tuple[str, List[str]]:
        """
        REDUCE phase: Combine intermediate answers into final answer.

        Args:
            query: Original user question
            intermediate_answers: Answers from MAP phase

        Returns:
            (final_answer, extracted_themes)
        """
        if not intermediate_answers:
            return "No relevant information found.", []

        # Combine intermediate answers (truncate if too long for context)
        combined = "\n\n".join([
            f"{i+1}. {answer}"
            for i, answer in enumerate(intermediate_answers[:10])  # Max 10 for Allam 2K context
        ])

        # Build reduce prompt
        prompt = f"""لديك عدة إجابات جزئية للسؤال التالي. ادمجها في إجابة شاملة واحدة.

السؤال الأصلي: {query}

الإجابات الجزئية:
{combined}

التعليمات:
1. ادمج جميع المعلومات ذات الصلة
2. تجنب التكرار
3. نظّم الإجابة بشكل منطقي
4. اذكر الموضوعات الرئيسية
5. كن موجزًا (3-5 فقرات)

الإجابة النهائية:
"""

        # Call Allam for reduce
        try:
            response = requests.post(
                f"{self.llm_endpoint}/generate",
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 800,  # Longer for final answer
                        "temperature": self.temperature,
                        "top_p": 0.95,
                        "do_sample": True,
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                final_answer = result.get('generated_text', '').strip()

                # Extract themes (simple heuristic)
                themes = self._extract_themes_from_text(final_answer)

                return final_answer, themes

            else:
                logger.error(f"REDUCE phase LLM request failed: {response.status_code}")
                # Fallback: concatenate answers
                return self._fallback_reduce(intermediate_answers), []

        except Exception as e:
            logger.error(f"Error in REDUCE phase: {e}")
            return self._fallback_reduce(intermediate_answers), []

    def _fallback_reduce(self, intermediate_answers: List[str]) -> str:
        """
        Fallback reduce strategy when LLM fails.

        Simply concatenates intermediate answers.
        """
        return "\n\n".join([
            f"• {answer}"
            for answer in intermediate_answers
        ])

    def _extract_themes_from_text(self, text: str) -> List[str]:
        """
        Extract themes from text using keyword matching.

        Args:
            text: Text to extract themes from

        Returns:
            List of themes
        """
        themes = []

        # Arabic keywords
        arabic_keywords = [
            'التكنولوجيا', 'الذكاء الاصطناعي', 'التعليم', 'الصحة',
            'الاقتصاد', 'السياسة', 'البيئة', 'الطاقة', 'النقل',
            'الثقافة', 'العلوم', 'الأمن', 'التجارة', 'الأعمال'
        ]

        # English keywords
        english_keywords = [
            'technology', 'ai', 'artificial intelligence', 'education', 'health',
            'economy', 'politics', 'environment', 'energy', 'transportation',
            'culture', 'science', 'security', 'trade', 'business'
        ]

        text_lower = text.lower()

        for keyword in arabic_keywords + english_keywords:
            if keyword in text_lower and keyword not in themes:
                themes.append(keyword)

        return themes[:5]  # Max 5 themes

    def get_search_statistics(self) -> Dict:
        """
        Get statistics about available summaries for search.

        Returns:
            Dict with statistics
        """
        query = """
        MATCH (c:Community)
        WHERE c.summary IS NOT NULL
        WITH c.level as level,
             COUNT(c) as summary_count,
             SUM(c.size) as total_entities,
             COLLECT(DISTINCT c.themes) as all_themes
        RETURN
            level,
            summary_count,
            total_entities,
            REDUCE(merged = [], themes IN all_themes | merged + themes) as unique_themes
        ORDER BY level
        """

        try:
            results = self.neo4j_client.execute_query(query)

            stats = {
                'levels': [],
                'total_summaries': 0,
                'all_themes': set()
            }

            for record in results:
                level_stats = {
                    'level': record.get('level'),
                    'summary_count': record.get('summary_count', 0),
                    'total_entities': record.get('total_entities', 0),
                    'unique_themes': list(set(record.get('unique_themes', [])))
                }
                stats['levels'].append(level_stats)
                stats['total_summaries'] += level_stats['summary_count']
                stats['all_themes'].update(level_stats['unique_themes'])

            stats['all_themes'] = list(stats['all_themes'])

            return stats

        except Exception as e:
            logger.error(f"Error getting search statistics: {e}")
            return {
                'levels': [],
                'total_summaries': 0,
                'all_themes': []
            }
