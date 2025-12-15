"""
Community Summarization Module for GraphRAG

Generates hierarchical summaries of communities using Allam LLM.
This is THE killer feature of GraphRAG that enables global search
over themes rather than just specific facts.

Optimized for Allam (2K token context) with excellent Arabic support.
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import requests

# Import centralized constants (relative import for package compatibility)
from ..config.constants import (
    MAX_ENTITIES_IN_SUMMARY,
    MAX_RELATIONSHIPS_IN_SUMMARY,
    MAX_SUMMARY_TOKENS,
    SUMMARIZATION_TEMPERATURE,
    MAX_CONTEXT_TOKENS,
    MAX_RESPONSE_TOKENS,
    TGI_ENDPOINT_DEFAULT,
)

logger = logging.getLogger(__name__)


@dataclass
class CommunitySummary:
    """Summary of a community."""
    community_id: str
    level: int
    summary: str
    entity_count: int
    key_entities: List[str]
    themes: List[str]
    token_count: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "community_id": self.community_id,
            "level": self.level,
            "summary": self.summary,
            "entity_count": self.entity_count,
            "key_entities": self.key_entities,
            "themes": self.themes,
            "token_count": self.token_count
        }


class CommunitySummarizer:
    """
    Generate hierarchical summaries of communities using Allam LLM.

    Strategy:
    - Level 0: Summarize entities + relationships
    - Level 1+: Summarize child community summaries
    - Bottom-up: Generate fine summaries first, then coarse

    Optimizations for Allam (2K token context):
    - Limit entities shown: 15-20 max
    - Limit relationships: 10-15 max
    - Limit child summaries: 3-5 max
    - Conservative token estimation (4 chars/token)

    Features:
    - Bilingual prompts (Arabic + English)
    - Theme extraction
    - Key entity identification
    - Hierarchical generation

    Usage:
        summarizer = CommunitySummarizer(neo4j_client, llm_endpoint)
        summaries = summarizer.generate_all_summaries(max_level=3)
        summarizer.store_summaries_in_neo4j(summaries)
    """

    def __init__(
        self,
        neo4j_client,
        llm_endpoint: str = None,
        max_tokens_per_summary: int = None,
        temperature: float = None
    ):
        """
        Initialize community summarizer.

        Args:
            neo4j_client: Neo4j client instance
            llm_endpoint: TGI endpoint for Allam (default: from constants)
            max_tokens_per_summary: Maximum tokens to generate per summary (default: from constants)
            temperature: LLM temperature (default: from constants, 0.3 = more deterministic)
        """
        self.neo4j_client = neo4j_client
        self.llm_endpoint = llm_endpoint or TGI_ENDPOINT_DEFAULT
        self.max_tokens_per_summary = max_tokens_per_summary or MAX_SUMMARY_TOKENS
        self.temperature = temperature if temperature is not None else SUMMARIZATION_TEMPERATURE

        # Token limits for Allam (2K context) - from centralized constants
        self.max_context_tokens = MAX_CONTEXT_TOKENS
        self.reserved_response_tokens = MAX_RESPONSE_TOKENS
        self.max_entities = MAX_ENTITIES_IN_SUMMARY
        self.max_relationships = MAX_RELATIONSHIPS_IN_SUMMARY

    def generate_all_summaries(
        self,
        max_level: Optional[int] = None,
        batch_size: int = 10
    ) -> List[CommunitySummary]:
        """
        Generate summaries for all communities (bottom-up).

        Args:
            max_level: Maximum level to summarize (None = all levels)
            batch_size: Process communities in batches

        Returns:
            List of CommunitySummary objects

        Example:
            >>> summarizer = CommunitySummarizer(neo4j_client)
            >>> summaries = summarizer.generate_all_summaries(max_level=2)
            >>> print(f"Generated {len(summaries)} summaries")
        """
        logger.info("Starting hierarchical community summarization...")

        # Get all community levels
        levels_query = """
        MATCH (c:Community)
        RETURN DISTINCT c.level as level
        ORDER BY c.level
        """

        try:
            results = self.neo4j_client.execute_query(levels_query)
            levels = [r.get('level') for r in results]

            if max_level is not None:
                levels = [l for l in levels if l <= max_level]

            logger.info(f"Found {len(levels)} community levels: {levels}")

        except Exception as e:
            logger.error(f"Error getting community levels: {e}")
            return []

        # Generate summaries bottom-up
        all_summaries = []

        for level in levels:
            logger.info(f"Generating summaries for Level {level}...")

            level_summaries = self._generate_summaries_for_level(
                level,
                batch_size=batch_size
            )

            all_summaries.extend(level_summaries)

            logger.info(
                f"Level {level} complete: {len(level_summaries)} summaries generated"
            )

        logger.info(f"Summarization complete: {len(all_summaries)} total summaries")

        return all_summaries

    def _generate_summaries_for_level(
        self,
        level: int,
        batch_size: int
    ) -> List[CommunitySummary]:
        """Generate summaries for all communities at a specific level."""

        # Get all communities at this level
        query = """
        MATCH (c:Community {level: $level})
        RETURN c.id as community_id
        ORDER BY c.id
        """

        try:
            results = self.neo4j_client.execute_query(query, {'level': level})
            community_ids = [r.get('community_id') for r in results]

        except Exception as e:
            logger.error(f"Error getting communities for level {level}: {e}")
            return []

        logger.info(f"Processing {len(community_ids)} communities at level {level}")

        summaries = []

        # Process in batches
        for i in range(0, len(community_ids), batch_size):
            batch = community_ids[i:i + batch_size]

            for comm_id in batch:
                summary = self.generate_community_summary(comm_id, level)
                if summary:
                    summaries.append(summary)

            logger.info(f"Batch {i//batch_size + 1}: {len(batch)} summaries generated")

        return summaries

    def generate_community_summary(
        self,
        community_id: str,
        level: int
    ) -> Optional[CommunitySummary]:
        """
        Generate summary for a single community.

        Args:
            community_id: Community identifier
            level: Community level

        Returns:
            CommunitySummary or None if generation failed
        """
        logger.debug(f"Generating summary for {community_id}...")

        # Get community data
        if level == 0:
            # Level 0: Use entities and relationships
            context = self._get_level0_context(community_id)
        else:
            # Level 1+: Use child community summaries
            context = self._get_higher_level_context(community_id, level)

        if not context:
            logger.warning(f"No context found for {community_id}")
            return None

        # Generate summary using Allam
        summary_text, themes = self._generate_summary_with_llm(
            context,
            community_id,
            level
        )

        if not summary_text:
            logger.error(f"Failed to generate summary for {community_id}")
            return None

        # Extract key entities (use configured max)
        key_entities = context.get('key_entities', [])[:self.max_entities]

        # Estimate token count (rough: 4 chars per token)
        token_count = len(summary_text) // 4

        summary = CommunitySummary(
            community_id=community_id,
            level=level,
            summary=summary_text,
            entity_count=context.get('entity_count', 0),
            key_entities=key_entities,
            themes=themes,
            token_count=token_count
        )

        logger.debug(
            f"Summary generated for {community_id}: "
            f"{len(summary_text)} chars, {len(themes)} themes"
        )

        return summary

    def _get_level0_context(self, community_id: str) -> Optional[Dict]:
        """
        Get context for Level 0 community (entities + relationships).

        GraphRAG Enhancement: Now includes entity descriptions for richer summaries.
        Returns context optimized for Allam's 2K token limit.
        """
        query = """
        MATCH (c:Community {id: $community_id})
        MATCH (e:Entity)-[:BELONGS_TO]->(c)

        // Get relationships between entities in this community (with descriptions)
        OPTIONAL MATCH (e1:Entity)-[r]->(e2:Entity)
        WHERE (e1)-[:BELONGS_TO]->(c)
          AND (e2)-[:BELONGS_TO]->(c)

        WITH c,
             COLLECT(DISTINCT {
                 name: e.name,
                 type: e.type,
                 description: COALESCE(e.description, ''),
                 importance: COALESCE(e.importance, 'medium')
             }) as entities,
             COLLECT(DISTINCT {
                 source: e1.name,
                 target: e2.name,
                 type: type(r),
                 description: COALESCE(r.description, '')
             }) as relationships

        RETURN
            c.id as community_id,
            c.size as size,
            entities,
            relationships
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'community_id': community_id
            })

            if not results:
                return None

            record = results[0]
            entities = record.get('entities', [])
            relationships = record.get('relationships', [])

            # Filter out empty relationships
            relationships = [
                r for r in relationships
                if r.get('source') and r.get('target')
            ]

            # Limit to fit in context (Allam 2K tokens)
            # Estimate: ~100 tokens per entity, ~50 tokens per relationship
            # Using centralized constants

            if len(entities) > self.max_entities:
                logger.warning(
                    f"Community {community_id} has {len(entities)} entities, "
                    f"limiting to {self.max_entities}"
                )
                entities = entities[:self.max_entities]

            if len(relationships) > self.max_relationships:
                relationships = relationships[:self.max_relationships]

            return {
                'community_id': community_id,
                'entity_count': len(entities),
                'entities': entities,
                'relationships': relationships,
                'key_entities': [e['name'] for e in entities]
            }

        except Exception as e:
            logger.error(f"Error getting Level 0 context for {community_id}: {e}")
            return None

    def _get_higher_level_context(
        self,
        community_id: str,
        level: int
    ) -> Optional[Dict]:
        """
        Get context for Level 1+ community (child summaries).

        Returns context optimized for Allam's 2K token limit.
        """
        query = """
        MATCH (parent:Community {id: $community_id})-[:PARENT_OF]->(child:Community)
        OPTIONAL MATCH (e:Entity)-[:BELONGS_TO]->(parent)
        RETURN
            parent.id as community_id,
            parent.size as size,
            COLLECT(DISTINCT child.id) as child_ids,
            COLLECT(DISTINCT e.name) as entities
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'community_id': community_id
            })

            if not results:
                return None

            record = results[0]
            child_ids = record.get('child_ids', [])
            entities = record.get('entities', [])

            # Get child summaries
            child_summaries = []
            for child_id in child_ids:
                summary = self._get_stored_summary(child_id)
                if summary:
                    child_summaries.append(summary)

            # Limit child summaries to fit in context
            # Estimate: ~300 tokens per child summary
            max_child_summaries = 4  # ~1200 tokens

            if len(child_summaries) > max_child_summaries:
                logger.warning(
                    f"Community {community_id} has {len(child_summaries)} children, "
                    f"limiting to {max_child_summaries}"
                )
                child_summaries = child_summaries[:max_child_summaries]

            return {
                'community_id': community_id,
                'entity_count': len(entities),
                'child_summaries': child_summaries,
                'key_entities': entities[:self.max_entities]
            }

        except Exception as e:
            logger.error(
                f"Error getting higher level context for {community_id}: {e}"
            )
            return None

    def _get_stored_summary(self, community_id: str) -> Optional[str]:
        """Get stored summary for a community."""
        query = """
        MATCH (c:Community {id: $community_id})
        RETURN c.summary as summary
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'community_id': community_id
            })

            if results and results[0].get('summary'):
                return results[0].get('summary')
            return None

        except Exception as e:
            logger.error(f"Error getting stored summary for {community_id}: {e}")
            return None

    def _generate_summary_with_llm(
        self,
        context: Dict,
        community_id: str,
        level: int
    ) -> Tuple[str, List[str]]:
        """
        Generate summary using Allam LLM.

        Returns:
            (summary_text, themes)
        """
        # Build prompt based on level
        if level == 0:
            prompt = self._build_level0_prompt(context)
        else:
            prompt = self._build_higher_level_prompt(context, level)

        # Call Allam via TGI
        try:
            response = requests.post(
                f"{self.llm_endpoint}/generate",
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": self.max_tokens_per_summary,
                        "temperature": self.temperature,
                        "top_p": 0.95,
                        "do_sample": True,
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('generated_text', '').strip()

                # Extract summary and themes
                summary, themes = self._parse_llm_response(generated_text)

                return summary, themes

            else:
                logger.error(
                    f"LLM request failed with status {response.status_code}: "
                    f"{response.text}"
                )
                return "", []

        except Exception as e:
            logger.error(f"Error calling Allam for {community_id}: {e}")
            return "", []

    def _build_level0_prompt(self, context: Dict) -> str:
        """
        Build prompt for Level 0 community (entities + relationships).

        GraphRAG Enhancement: Now includes entity and relationship descriptions
        for richer, more informative community summaries.
        """
        entities = context.get('entities', [])
        relationships = context.get('relationships', [])

        # Format entities with descriptions (GraphRAG style)
        # Group by type, include description if available
        entities_by_type = {}
        for e in entities:
            etype = e.get('type', 'Unknown')
            if etype not in entities_by_type:
                entities_by_type[etype] = []

            # Include description if available
            name = e.get('name', '')
            desc = e.get('description', '').strip()
            if desc and len(desc) > 10:
                # Truncate long descriptions to save tokens
                if len(desc) > 150:
                    desc = desc[:150] + "..."
                entities_by_type[etype].append(f"{name}: {desc}")
            else:
                entities_by_type[etype].append(name)

        entity_lines = []
        for etype, items in entities_by_type.items():
            if any(':' in item for item in items):
                # Has descriptions - format as bullet list
                entity_lines.append(f"## {etype}:")
                for item in items[:5]:  # Limit per type for token budget
                    entity_lines.append(f"  • {item}")
            else:
                # No descriptions - compact format
                entity_lines.append(f"• {etype}: {', '.join(items[:8])}")

        entities_text = "\n".join(entity_lines)

        # Format relationships with descriptions (GraphRAG style)
        rel_lines = []
        for r in relationships[:10]:  # Limit for token budget
            if r.get('source') and r.get('target'):
                rel_type = r.get('type', 'RELATED_TO')
                rel_desc = r.get('description', '').strip()

                if rel_desc and len(rel_desc) > 10:
                    # Truncate long descriptions
                    if len(rel_desc) > 100:
                        rel_desc = rel_desc[:100] + "..."
                    rel_lines.append(
                        f"• {r['source']} → {r['target']} ({rel_type}): {rel_desc}"
                    )
                else:
                    rel_lines.append(f"• {r['source']} → {r['target']} ({rel_type})")

        relationships_text = "\n".join(rel_lines) if rel_lines else "(مترابطة ضمنياً)"

        # Direct task prompt with GraphRAG-style instructions
        prompt = f"""أنت محلل بيانات متخصص. اكتب ملخصاً شاملاً لهذه المجموعة من الكيانات والعلاقات:

الكيانات وأوصافها:
{entities_text}

العلاقات:
{relationships_text}

اكتب فقرة واحدة (4-5 جمل) تصف:
- ما هو الموضوع أو المجال الرئيسي لهذه المجموعة؟
- ما هي الكيانات الأهم وما دورها؟
- كيف ترتبط هذه الكيانات ببعضها وما أهمية هذه العلاقات؟
- ما الأنماط أو الاتجاهات البارزة؟

ابدأ الملخص مباشرة (بدون مقدمات أو عناوين):"""

        return prompt

    def _build_higher_level_prompt(self, context: Dict, level: int) -> str:
        """Build prompt for Level 1+ community (child summaries)."""

        child_summaries = context.get('child_summaries', [])

        # Format child summaries - cleaner format
        summary_lines = []
        for i, summary in enumerate(child_summaries, 1):
            # Truncate if too long and clean up
            summary_clean = summary.strip()[:400]
            if len(summary) > 400:
                summary_clean += "..."
            summary_lines.append(f"[{i}] {summary_clean}")

        summaries_text = "\n\n".join(summary_lines)

        # Direct task prompt
        prompt = f"""أنت محلل بيانات. اقرأ هذه الملخصات واكتب ملخصاً شاملاً يجمعها:

{summaries_text}

اكتب فقرتين (4-6 جمل) تجمع المعلومات أعلاه:
- ما الموضوع الرئيسي المشترك؟
- ما الأنماط أو الاتجاهات البارزة؟
- ما أهم النقاط التي يجب معرفتها؟

ابدأ الملخص مباشرة:"""

        return prompt

    def _parse_llm_response(self, response: str) -> Tuple[str, List[str]]:
        """
        Parse LLM response to extract summary and themes.

        Returns:
            (summary_text, themes)
        """
        # Clean up the response - remove meta-content
        summary = response.strip()

        # Remove common meta-content patterns (LLM explaining what to do)
        meta_patterns = [
            "يجب أن",  # "should be"
            "يجب ألا",  # "should not"
            "لا يحتوي على",  # "does not contain"
            "يركز على",  # "focuses on" (when giving instructions)
            "يُستخدم ل",  # "is used for"
            "الإجابة النهائية",  # "final answer"
            "**الملخص:**",  # markdown headers
            "###",  # markdown headers
            "---",  # dividers
        ]

        # Split into lines and filter out meta-content lines
        lines = summary.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip lines that start with meta-patterns
            is_meta = False
            for pattern in meta_patterns:
                if line.startswith(pattern) or line.startswith(f"- {pattern}"):
                    is_meta = True
                    break
            if not is_meta and len(line) > 10:  # Skip very short lines
                clean_lines.append(line)

        summary = ' '.join(clean_lines)

        # Remove bullet points at start
        if summary.startswith('- '):
            summary = summary[2:]
        if summary.startswith('• '):
            summary = summary[2:]

        # Extract themes using simple heuristics
        themes = []

        # Look for common theme indicators in Arabic
        theme_keywords_ar = [
            'التكنولوجيا', 'الذكاء الاصطناعي', 'التعليم', 'الصحة',
            'الاقتصاد', 'السياسة', 'البيئة', 'الطاقة', 'النقل',
            'الثقافة', 'العلوم', 'الأمن', 'التجارة', 'الحكومة',
            'الرقمي', 'التحول', 'الابتكار', 'الشراكة'
        ]

        # Look for common theme indicators in English
        theme_keywords_en = [
            'technology', 'ai', 'education', 'health', 'economy',
            'politics', 'environment', 'energy', 'transportation',
            'culture', 'science', 'security', 'trade', 'business',
            'government', 'digital', 'innovation', 'partnership'
        ]

        response_lower = summary.lower()

        for keyword in theme_keywords_ar + theme_keywords_en:
            if keyword in response_lower and keyword not in themes:
                themes.append(keyword)

        return summary, themes[:5]  # Max 5 themes

    def store_summaries_in_neo4j(
        self,
        summaries: List[CommunitySummary],
        clear_existing: bool = False
    ) -> Dict:
        """
        Store community summaries in Neo4j.

        Updates Community nodes with summary, themes, and metadata.

        Args:
            summaries: List of CommunitySummary objects
            clear_existing: If True, clear existing summaries first

        Returns:
            Dict with storage statistics
        """
        if clear_existing:
            logger.info("Clearing existing summaries from Neo4j...")
            self._clear_summaries()

        logger.info(f"Storing {len(summaries)} summaries in Neo4j...")

        stats = {
            'summaries_stored': 0,
            'summaries_failed': 0
        }

        for summary in summaries:
            success = self._store_single_summary(summary)
            if success:
                stats['summaries_stored'] += 1
            else:
                stats['summaries_failed'] += 1

        logger.info(
            f"Summary storage complete: "
            f"{stats['summaries_stored']} stored, "
            f"{stats['summaries_failed']} failed"
        )

        return stats

    def _clear_summaries(self):
        """Clear all existing summaries from Community nodes."""
        query = """
        MATCH (c:Community)
        WHERE c.summary IS NOT NULL
        REMOVE c.summary, c.themes, c.key_entities, c.summary_token_count
        """

        try:
            self.neo4j_client.execute_query(query)
            logger.info("Existing summaries cleared")
        except Exception as e:
            logger.error(f"Error clearing summaries: {e}")

    def _store_single_summary(self, summary: CommunitySummary) -> bool:
        """Store a single summary in Neo4j."""
        query = """
        MATCH (c:Community {id: $community_id})
        SET c.summary = $summary,
            c.themes = $themes,
            c.key_entities = $key_entities,
            c.summary_token_count = $token_count,
            c.summary_generated_at = datetime()
        """

        try:
            self.neo4j_client.execute_query(query, {
                'community_id': summary.community_id,
                'summary': summary.summary,
                'themes': summary.themes,
                'key_entities': summary.key_entities,
                'token_count': summary.token_count
            })
            return True

        except Exception as e:
            logger.error(f"Error storing summary for {summary.community_id}: {e}")
            return False

    def get_summary(self, community_id: str) -> Optional[Dict]:
        """
        Get stored summary for a community.

        Args:
            community_id: Community identifier

        Returns:
            Dict with summary data or None
        """
        query = """
        MATCH (c:Community {id: $community_id})
        RETURN
            c.id as community_id,
            c.level as level,
            c.summary as summary,
            c.themes as themes,
            c.key_entities as key_entities,
            c.summary_token_count as token_count
        """

        try:
            results = self.neo4j_client.execute_query(query, {
                'community_id': community_id
            })

            if results and results[0].get('summary'):
                record = results[0]
                return {
                    'community_id': record.get('community_id'),
                    'level': record.get('level'),
                    'summary': record.get('summary'),
                    'themes': record.get('themes', []),
                    'key_entities': record.get('key_entities', []),
                    'token_count': record.get('token_count', 0)
                }
            return None

        except Exception as e:
            logger.error(f"Error getting summary for {community_id}: {e}")
            return None

    def get_all_summaries_at_level(self, level: int) -> List[Dict]:
        """
        Get all summaries at a specific level.

        Args:
            level: Community level

        Returns:
            List of summary dicts
        """
        query = """
        MATCH (c:Community {level: $level})
        WHERE c.summary IS NOT NULL
        RETURN
            c.id as community_id,
            c.level as level,
            c.summary as summary,
            c.themes as themes,
            c.size as size
        ORDER BY c.id
        """

        try:
            results = self.neo4j_client.execute_query(query, {'level': level})

            summaries = []
            for record in results:
                summaries.append({
                    'community_id': record.get('community_id'),
                    'level': record.get('level'),
                    'summary': record.get('summary'),
                    'themes': record.get('themes', []),
                    'size': record.get('size', 0)
                })

            return summaries

        except Exception as e:
            logger.error(f"Error getting summaries at level {level}: {e}")
            return []

    def get_summary_statistics(self) -> Dict:
        """
        Get statistics about generated summaries.

        Returns:
            Dict with statistics
        """
        query = """
        MATCH (c:Community)
        WHERE c.summary IS NOT NULL
        WITH c.level as level,
             COUNT(c) as summary_count,
             AVG(c.summary_token_count) as avg_tokens
        RETURN
            level,
            summary_count,
            avg_tokens
        ORDER BY level
        """

        try:
            results = self.neo4j_client.execute_query(query)

            stats = {
                'levels': [],
                'total_summaries': 0
            }

            for record in results:
                level_stats = {
                    'level': record.get('level'),
                    'summary_count': record.get('summary_count', 0),
                    'avg_tokens': record.get('avg_tokens', 0)
                }
                stats['levels'].append(level_stats)
                stats['total_summaries'] += level_stats['summary_count']

            return stats

        except Exception as e:
            logger.error(f"Error getting summary statistics: {e}")
            return {'levels': [], 'total_summaries': 0}
