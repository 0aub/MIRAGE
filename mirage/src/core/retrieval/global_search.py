"""
Global Search Engine - Map-Reduce over Community Summaries

This is THE killer feature of GraphRAG that enables answering:
- "What are the main themes across all documents?"
- "Summarize the key topics in this knowledge base"
- "What patterns emerge from this data?"

Algorithm (per Microsoft GraphRAG specification):
1. MAP: Query each community summary in parallel
   - Each community generates a partial answer
   - Uses JSON mode for structured output (score, answer, explanation)
   - Scored by relevance to query (0-100)

2. FILTER: Keep top-k most relevant partial answers
   - Based on relevance score threshold

3. REDUCE: Combine partial answers into final answer
   - Synthesize coherent response
   - Preserve key insights from each community

GraphRAG Enhancements:
- JSON mode in map phase for structured scoring
- Entity descriptions for richer context
- Automatic community level selection
- Multi-level search capability

Optimized for Arabic + English bilingual support.
"""

import asyncio
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import requests
from loguru import logger

from ..config.constants import TGI_ENDPOINT_DEFAULT


@dataclass
class PartialAnswer:
    """Partial answer from a single community."""
    community_id: str
    level: int
    summary: str
    answer: str
    relevance_score: float
    key_points: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)


@dataclass
class GlobalSearchResult:
    """Result of global search across all communities."""
    query: str
    answer: str
    partial_answers: List[PartialAnswer]
    communities_searched: int
    total_communities: int
    search_level: int
    confidence: float = 0.0
    themes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "communities_searched": self.communities_searched,
            "total_communities": self.total_communities,
            "search_level": self.search_level,
            "confidence": self.confidence,
            "themes": self.themes,
            "partial_answers": [
                {
                    "community_id": pa.community_id,
                    "relevance_score": pa.relevance_score,
                    "key_points": pa.key_points
                }
                for pa in self.partial_answers
            ]
        }


class GlobalSearchEngine:
    """
    Global search using map-reduce over community summaries.

    This enables answering holistic queries that require understanding
    across the entire knowledge base, not just specific documents.

    Features:
    - Parallel community querying (map phase)
    - Relevance-based filtering
    - Coherent answer synthesis (reduce phase)
    - Multi-level community search
    - Bilingual support (Arabic + English)
    """

    def __init__(
        self,
        neo4j_client,
        llm_endpoint: str = TGI_ENDPOINT_DEFAULT,
        max_communities: int = 30,
        min_relevance: float = 0.3,
        parallel_workers: int = 5,
        community_level: int = 0,
        max_tokens: int = 400,
        temperature: float = 0.3,
        use_json_mode: bool = True
    ):
        """
        Initialize global search engine.

        GraphRAG Enhancement: Now supports JSON mode for structured map phase output.

        Args:
            neo4j_client: Neo4j client for community data
            llm_endpoint: TGI endpoint for LLM calls
            max_communities: Maximum communities to query in map phase
            min_relevance: Minimum relevance score to keep partial answer
            parallel_workers: Number of parallel LLM calls
            community_level: Which hierarchy level to search (0=fine, 1+=coarse)
            max_tokens: Max tokens per LLM response
            temperature: LLM temperature
            use_json_mode: If True, use JSON output format in map phase (GraphRAG spec)
        """
        self.neo4j_client = neo4j_client
        self.llm_endpoint = llm_endpoint
        self.max_communities = max_communities
        self.min_relevance = min_relevance
        self.parallel_workers = parallel_workers
        self.community_level = community_level
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_json_mode = use_json_mode

        self._executor = ThreadPoolExecutor(max_workers=parallel_workers)

        logger.info(
            f"GlobalSearchEngine initialized: level={community_level}, "
            f"max_communities={max_communities}, workers={parallel_workers}, "
            f"json_mode={use_json_mode}"
        )

    def search(self, query: str) -> GlobalSearchResult:
        """
        Execute global search using map-reduce pattern.

        Steps:
        1. Get all community summaries at target level
        2. MAP: Generate partial answer for each (parallel)
        3. FILTER: Keep relevant partial answers
        4. REDUCE: Synthesize final answer

        Args:
            query: User query string

        Returns:
            GlobalSearchResult with synthesized answer
        """
        logger.info(f"Global search: '{query[:50]}...' at level {self.community_level}")

        # Step 1: Get communities at target level
        communities = self._get_communities_at_level(self.community_level)

        if not communities:
            logger.warning("No communities found for global search")
            return GlobalSearchResult(
                query=query,
                answer="لا توجد معلومات متاحة للإجابة على هذا السؤال.",
                partial_answers=[],
                communities_searched=0,
                total_communities=0,
                search_level=self.community_level
            )

        total_communities = len(communities)
        communities = communities[:self.max_communities]

        logger.info(f"Querying {len(communities)} communities (of {total_communities} total)")

        # Step 2: MAP phase - generate partial answers in parallel
        partial_answers = self._map_phase(query, communities)

        logger.info(f"Map phase complete: {len(partial_answers)} partial answers")

        # Step 3: FILTER phase - keep relevant answers
        relevant_answers = self._filter_phase(partial_answers)

        logger.info(f"Filter phase complete: {len(relevant_answers)} relevant answers")

        if not relevant_answers:
            return GlobalSearchResult(
                query=query,
                answer="لم يتم العثور على معلومات ذات صلة بالسؤال في قاعدة البيانات.",
                partial_answers=[],
                communities_searched=len(communities),
                total_communities=total_communities,
                search_level=self.community_level
            )

        # Step 4: REDUCE phase - synthesize final answer
        final_answer, themes, confidence = self._reduce_phase(query, relevant_answers)

        return GlobalSearchResult(
            query=query,
            answer=final_answer,
            partial_answers=relevant_answers,
            communities_searched=len(communities),
            total_communities=total_communities,
            search_level=self.community_level,
            confidence=confidence,
            themes=themes
        )

    def _get_communities_at_level(self, level: int) -> List[Dict]:
        """
        Get all communities with summaries at a specific level.

        GraphRAG Enhancement: Now includes entity descriptions for richer context
        in the map phase, enabling more accurate partial answers.
        """
        query = """
        MATCH (c:Community {level: $level})
        WHERE c.summary IS NOT NULL AND c.summary <> ''

        // GraphRAG: Get entity descriptions for richer context
        OPTIONAL MATCH (e:Entity)-[:BELONGS_TO]->(c)
        WHERE e.description IS NOT NULL AND e.description <> ''

        WITH c,
             COLLECT(DISTINCT {
                 name: e.name,
                 type: e.type,
                 description: e.description
             })[0..5] as entity_details

        RETURN
            c.id as community_id,
            c.level as level,
            c.summary as summary,
            c.size as size,
            c.key_entities as key_entities,
            c.themes as themes,
            entity_details
        ORDER BY c.size DESC
        """

        try:
            results = self.neo4j_client.execute_query(query, {'level': level})
            return results
        except Exception as e:
            logger.error(f"Error getting communities at level {level}: {e}")
            return []

    def _select_optimal_level(self, query: str) -> int:
        """
        GraphRAG Enhancement: Select optimal community level based on query type.

        - Broad/thematic queries → Higher levels (coarser communities)
        - Specific/entity queries → Lower levels (finer communities)
        """
        # Keywords indicating broad queries (use higher levels)
        broad_keywords = [
            'الموضوعات', 'المجالات', 'الاتجاهات', 'الأنماط', 'الملخص',
            'بشكل عام', 'إجمالي', 'كل', 'جميع', 'الرئيسية',
            'themes', 'topics', 'trends', 'patterns', 'summary',
            'overall', 'total', 'all', 'main', 'general', 'overview'
        ]

        # Keywords indicating specific queries (use lower levels)
        specific_keywords = [
            'من', 'ما هو', 'ما هي', 'أين', 'متى', 'كم',
            'بالتحديد', 'محدد', 'معين',
            'who', 'what is', 'where', 'when', 'how many',
            'specific', 'exactly', 'particular'
        ]

        query_lower = query.lower()

        # Count keyword matches
        broad_score = sum(1 for kw in broad_keywords if kw in query_lower)
        specific_score = sum(1 for kw in specific_keywords if kw in query_lower)

        if broad_score > specific_score:
            return 1  # Higher level for broad queries
        elif specific_score > broad_score:
            return 0  # Lower level for specific queries
        else:
            return self.community_level  # Default to configured level

    def _map_phase(
        self,
        query: str,
        communities: List[Dict]
    ) -> List[PartialAnswer]:
        """
        MAP phase: Generate partial answer for each community.

        Uses parallel execution for efficiency.
        """
        partial_answers = []

        # Process in parallel using thread pool
        futures = []
        for community in communities:
            future = self._executor.submit(
                self._generate_partial_answer,
                query,
                community
            )
            futures.append((community, future))

        # Collect results
        for community, future in futures:
            try:
                result = future.result(timeout=30)  # 30s timeout per community
                if result:
                    partial_answers.append(result)
            except Exception as e:
                logger.warning(
                    f"Failed to get partial answer for {community.get('community_id')}: {e}"
                )

        return partial_answers

    def _generate_partial_answer(
        self,
        query: str,
        community: Dict
    ) -> Optional[PartialAnswer]:
        """
        Generate partial answer from a single community.

        GraphRAG Enhancement: Now uses entity descriptions for richer context.
        """
        community_id = community.get('community_id', '')
        summary = community.get('summary', '')
        key_entities = community.get('key_entities', []) or []
        level = community.get('level', 0)
        entity_details = community.get('entity_details', []) or []

        if not summary:
            return None

        # Build prompt with entity details (GraphRAG enhancement)
        # Use JSON mode for structured output per Microsoft GraphRAG spec
        prompt = self._build_map_prompt(
            query, summary, key_entities, entity_details,
            use_json_mode=self.use_json_mode
        )

        # Call LLM
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
                    }
                },
                timeout=25
            )

            if response.status_code != 200:
                logger.warning(f"LLM call failed for {community_id}: {response.status_code}")
                return None

            result = response.json()
            generated_text = result.get('generated_text', '').strip()

            # Parse response
            answer, relevance, key_points = self._parse_map_response(generated_text)

            if relevance < 0.1:  # Too low relevance, skip
                return None

            return PartialAnswer(
                community_id=community_id,
                level=level,
                summary=summary[:200],  # Truncate for storage
                answer=answer,
                relevance_score=relevance,
                key_points=key_points,
                entities=key_entities[:5] if key_entities else []
            )

        except Exception as e:
            logger.warning(f"Error generating partial answer for {community_id}: {e}")
            return None

    def _build_map_prompt(
        self,
        query: str,
        summary: str,
        key_entities: List[str],
        entity_details: List[Dict] = None,
        use_json_mode: bool = True
    ) -> str:
        """
        Build prompt for generating partial answer from community summary.

        GraphRAG Enhancement:
        - Includes entity descriptions for richer context
        - Uses JSON mode for structured output (per Microsoft GraphRAG spec)

        Args:
            query: User question
            summary: Community summary
            key_entities: List of key entity names
            entity_details: Detailed entity info with descriptions
            use_json_mode: If True, request JSON output (GraphRAG spec)
        """
        entities_text = ", ".join(key_entities[:10]) if key_entities else "غير محدد"

        # GraphRAG: Add entity descriptions for richer context
        entity_context = ""
        if entity_details:
            entity_lines = []
            for e in entity_details[:5]:  # Limit to 5 for token budget
                name = e.get('name', '')
                desc = e.get('description', '')
                if name and desc:
                    # Truncate long descriptions
                    desc_short = desc[:100] + "..." if len(desc) > 100 else desc
                    entity_lines.append(f"• {name}: {desc_short}")
            if entity_lines:
                entity_context = f"\n\nتفاصيل الكيانات:\n" + "\n".join(entity_lines)

        if use_json_mode:
            # GraphRAG JSON Mode: Structured output for better parsing
            prompt = f"""أنت محلل بيانات. استخدم المعلومات التالية للإجابة على السؤال.

ملخص المجموعة:
{summary}

الكيانات الرئيسية: {entities_text}{entity_context}

السؤال: {query}

أجب بتنسيق JSON فقط:
{{"score": <رقم من 0 إلى 100 يمثل صلة المعلومات بالسؤال>, "answer": "<إجابتك هنا في 2-3 جمل>", "points": ["<نقطة 1>", "<نقطة 2>"], "explanation": "<سبب اختيارك لهذه الدرجة>"}}

إذا لم تكن هناك معلومات ذات صلة:
{{"score": 0, "answer": "لا توجد معلومات ذات صلة", "points": [], "explanation": "المعلومات لا تتعلق بالسؤال"}}

JSON فقط:"""
        else:
            # Fallback: Text format
            prompt = f"""أنت محلل بيانات. استخدم المعلومات التالية للإجابة على السؤال.

ملخص المجموعة:
{summary}

الكيانات الرئيسية: {entities_text}{entity_context}

السؤال: {query}

التعليمات:
1. إذا كانت المعلومات ذات صلة بالسؤال، اكتب إجابة موجزة (2-3 جمل)
2. قيّم مدى صلة المعلومات بالسؤال (0.0 = لا صلة، 1.0 = صلة كاملة)
3. اذكر النقاط الرئيسية المستخلصة (1-3 نقاط)

الإجابة بالتنسيق التالي:
الصلة: [رقم بين 0 و 1]
الإجابة: [إجابتك هنا]
النقاط: [نقطة 1 | نقطة 2 | نقطة 3]

إذا لم تكن هناك معلومات ذات صلة، اكتب:
الصلة: 0.0
الإجابة: لا توجد معلومات ذات صلة
النقاط: لا يوجد"""

        return prompt

    def _parse_map_response(
        self,
        response: str
    ) -> Tuple[str, float, List[str]]:
        """
        Parse LLM response from map phase.

        GraphRAG Enhancement: First tries JSON parsing (per spec), then falls back to text.
        """
        answer = ""
        relevance = 0.0
        key_points = []

        # =====================================================================
        # GraphRAG JSON Mode: Try JSON parsing first
        # =====================================================================
        try:
            # Find JSON in response - handle nested structures like arrays
            # Look for balanced braces to extract complete JSON object
            json_str = self._extract_json_from_text(response)
            if json_str:
                data = json.loads(json_str)

                # Parse score (0-100 in JSON, normalize to 0-1)
                score = data.get('score', 0)
                if isinstance(score, (int, float)):
                    relevance = min(1.0, max(0.0, score / 100.0))
                elif isinstance(score, str):
                    # Handle string scores like "75" or "0.75"
                    try:
                        score_val = float(score)
                        if score_val > 1:
                            relevance = min(1.0, max(0.0, score_val / 100.0))
                        else:
                            relevance = min(1.0, max(0.0, score_val))
                    except ValueError:
                        pass

                # Parse answer
                answer = data.get('answer', '').strip()

                # Parse key points - handle various formats
                points = data.get('points', [])
                if isinstance(points, list):
                    key_points = [str(p).strip() for p in points if p][:3]
                elif isinstance(points, str):
                    # Handle comma-separated string
                    key_points = [p.strip() for p in points.split(',') if p.strip()][:3]

                # Log successful JSON parse
                if answer and relevance > 0:
                    logger.debug(f"GraphRAG JSON parse successful: score={score}, answer_len={len(answer)}")
                    return answer, relevance, key_points

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"JSON parse failed, falling back to text: {e}")

        # =====================================================================
        # Fallback: Text-based parsing
        # =====================================================================
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Parse relevance
            if line.startswith('الصلة:') or line.startswith('Relevance:'):
                try:
                    score_text = line.split(':')[1].strip()
                    # Extract number from text
                    match = re.search(r'(\d+\.?\d*)', score_text)
                    if match:
                        relevance = float(match.group(1))
                        relevance = min(1.0, max(0.0, relevance))
                except:
                    pass

            # Parse answer
            elif line.startswith('الإجابة:') or line.startswith('Answer:'):
                answer = line.split(':', 1)[1].strip() if ':' in line else ""

            # Parse key points
            elif line.startswith('النقاط:') or line.startswith('Points:'):
                points_text = line.split(':', 1)[1].strip() if ':' in line else ""
                if '|' in points_text:
                    key_points = [p.strip() for p in points_text.split('|') if p.strip()]
                elif '،' in points_text:
                    key_points = [p.strip() for p in points_text.split('،') if p.strip()]

        # If no structured response, try to extract from raw text
        if not answer and response:
            # Use first substantial sentence as answer
            sentences = response.split('.')
            for sent in sentences:
                if len(sent.strip()) > 20:
                    answer = sent.strip()
                    break

            # Estimate relevance based on content
            if "لا توجد" in response or "لا يوجد" in response or "no relevant" in response.lower():
                relevance = 0.1
            else:
                relevance = 0.5  # Default moderate relevance

        return answer, relevance, key_points[:3]

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON object from text, handling nested structures.

        Uses balanced brace matching to find complete JSON objects,
        even when they contain nested arrays or objects.

        Args:
            text: Text that may contain JSON

        Returns:
            Extracted JSON string or None if not found
        """
        if not text:
            return None

        # Find the first opening brace
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        # Count braces to find matching closing brace
        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found complete JSON object
                    json_str = text[start_idx:i + 1]
                    # Validate it's valid JSON
                    try:
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError:
                        # Try to continue finding another JSON
                        continue

        return None

    def _filter_phase(
        self,
        partial_answers: List[PartialAnswer]
    ) -> List[PartialAnswer]:
        """FILTER phase: Keep only relevant partial answers."""

        # Filter by minimum relevance
        relevant = [
            pa for pa in partial_answers
            if pa.relevance_score >= self.min_relevance
        ]

        # Sort by relevance (descending)
        relevant.sort(key=lambda x: x.relevance_score, reverse=True)

        # Keep top answers - limit to 6 to keep reduce prompt manageable
        # for smaller LLMs with 2K context limit
        max_answers = 6
        return relevant[:max_answers]

    def _reduce_phase(
        self,
        query: str,
        partial_answers: List[PartialAnswer]
    ) -> Tuple[str, List[str], float]:
        """
        REDUCE phase: Combine partial answers into final coherent answer.

        Returns:
            (final_answer, themes, confidence)
        """

        # Format partial answers for synthesis
        answers_text = self._format_partial_answers(partial_answers)

        prompt = self._build_reduce_prompt(query, answers_text, len(partial_answers))

        try:
            response = requests.post(
                f"{self.llm_endpoint}/generate",
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 600,  # Longer for final answer
                        "temperature": self.temperature,
                        "top_p": 0.95,
                        "do_sample": True,
                    }
                },
                timeout=45
            )

            if response.status_code != 200:
                logger.error(f"Reduce phase LLM call failed: {response.status_code}")
                # Fallback: concatenate top answers
                return self._fallback_reduce(partial_answers), [], 0.5

            result = response.json()
            generated_text = result.get('generated_text', '').strip()

            logger.debug(f"Reduce phase raw response: {generated_text[:300]}...")

            # Parse and clean response
            final_answer = self._clean_reduce_response(generated_text)

            # If cleaning resulted in empty/too short, use fallback
            if not final_answer or len(final_answer) < 30:
                logger.warning("Reduce phase produced insufficient answer, using fallback")
                final_answer = self._fallback_reduce(partial_answers)

            # Extract themes
            themes = self._extract_themes(final_answer, partial_answers)

            # Calculate confidence based on coverage and relevance
            avg_relevance = sum(pa.relevance_score for pa in partial_answers) / len(partial_answers)
            coverage = len(partial_answers) / 10  # Normalize by max expected
            confidence = (avg_relevance * 0.7) + (min(coverage, 1.0) * 0.3)

            return final_answer, themes, confidence

        except Exception as e:
            logger.error(f"Error in reduce phase: {e}")
            return self._fallback_reduce(partial_answers), [], 0.4

    def _format_partial_answers(self, partial_answers: List[PartialAnswer]) -> str:
        """Format partial answers for reduce prompt - keep concise for 2K context LLM."""

        lines = []
        for i, pa in enumerate(partial_answers, 1):
            # Truncate to 150 chars to keep total prompt under 2K tokens
            answer = pa.answer[:150].strip()
            if len(pa.answer) > 150:
                answer += "..."
            lines.append(f"[{i}] {answer}")

        return "\n".join(lines)

    def _build_reduce_prompt(
        self,
        query: str,
        answers_text: str,
        num_answers: int
    ) -> str:
        """
        Build prompt for reduce phase.

        GraphRAG Enhancement: Improved synthesis prompt that:
        - Preserves key insights from each partial answer
        - Identifies common themes across communities
        - Produces coherent, well-structured response
        """
        # Adaptive prompt based on number of answers
        if num_answers <= 2:
            instruction = "اكتب إجابة موجزة في 2-3 جمل"
        elif num_answers <= 4:
            instruction = "اكتب إجابة شاملة في 3-4 جمل تجمع المعلومات"
        else:
            instruction = "اكتب إجابة شاملة في 4-5 جمل تجمع أهم النقاط من جميع المصادر"

        prompt = f"""أنت محلل بيانات متخصص. مهمتك دمج المعلومات التالية في إجابة واحدة متماسكة.

السؤال: {query}

المعلومات من مصادر متعددة ({num_answers} مصادر):
{answers_text}

التعليمات:
1. {instruction}
2. احتفظ بالمعلومات الرئيسية من كل مصدر
3. تجنب التكرار
4. إذا تعارضت المعلومات، اذكر وجهات النظر المختلفة
5. ابدأ الإجابة مباشرة بدون مقدمات

الإجابة:"""

        return prompt

    def _clean_reduce_response(self, response: str) -> str:
        """Clean and format the reduce phase response."""

        if not response:
            return ""

        # Remove common artifacts
        response = response.strip()

        # Remove LLM hallucination patterns
        hallucination_patterns = [
            r'هل ترغب في.*$',             # "Would you like to..." (to end of text)
            r'هل تريد.*$',                 # "Do you want..." (to end of text)
            r'نعم،?\s*أرغب.*$',            # "Yes, I want..." (to end of text)
            r'نعم،?\s*يمكن.*$',            # "Yes, can..." (to end of text)
            r'الإجابة المُحسنة:.*$',       # "Improved answer:" (to end of text)
            r'^التعليمات:.*$',
            r'^الإجابة الشاملة:?\s*$',     # Empty header line
        ]

        for pattern in hallucination_patterns:
            response = re.sub(pattern, '', response, flags=re.MULTILINE | re.UNICODE)

        lines = response.split('\n')
        clean_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip very short lines that are likely artifacts
            if len(line) < 5:
                continue

            # Skip numbered list prefixes but keep content
            line = re.sub(r'^\d+\.\s*', '', line)
            line = re.sub(r'^-\s*', '', line)

            if line:
                clean_lines.append(line)

        result = ' '.join(clean_lines)

        # If result is too short, something went wrong
        if len(result) < 20:
            logger.warning(f"Reduce response too short ({len(result)} chars), raw: {response[:200]}")
            return ""

        return result

    def _extract_themes(
        self,
        answer: str,
        partial_answers: List[PartialAnswer]
    ) -> List[str]:
        """Extract themes from the answer and partial answers."""

        themes = set()

        # Theme keywords
        theme_keywords = [
            'التكنولوجيا', 'الذكاء الاصطناعي', 'التعليم', 'الصحة',
            'الاقتصاد', 'الحكومة', 'الرقمي', 'التحول', 'الابتكار',
            'الخدمات', 'البيانات', 'الأمن', 'التجارة', 'الطاقة',
            'technology', 'digital', 'innovation', 'government',
            'education', 'health', 'economy', 'security'
        ]

        # Search in answer
        answer_lower = answer.lower()
        for keyword in theme_keywords:
            if keyword in answer_lower:
                themes.add(keyword)

        # Search in partial answers
        for pa in partial_answers:
            if pa.key_points:
                for point in pa.key_points:
                    for keyword in theme_keywords:
                        if keyword in point.lower():
                            themes.add(keyword)

        return list(themes)[:5]

    def _fallback_reduce(self, partial_answers: List[PartialAnswer]) -> str:
        """Fallback reduce: structured concatenation of top answers."""

        if not partial_answers:
            return "لم يتم العثور على معلومات كافية للإجابة."

        # Take top 3 answers with valid content
        top_answers = [pa for pa in partial_answers[:4] if pa.answer and len(pa.answer) > 20]

        if not top_answers:
            return "لم يتم العثور على معلومات كافية للإجابة."

        # Build structured response
        parts = []
        for pa in top_answers:
            answer = pa.answer[:200].strip()
            if answer:
                parts.append(answer)

        combined = " كما أن ".join(parts) if len(parts) > 1 else parts[0]

        return combined[:600] if combined else "لم يتم العثور على معلومات كافية."

    def search_auto_level(self, query: str) -> GlobalSearchResult:
        """
        GraphRAG Enhancement: Search with automatic level selection.

        Automatically selects the optimal community level based on query type:
        - Broad/thematic queries → Higher levels (coarser communities)
        - Specific/entity queries → Lower levels (finer communities)

        Args:
            query: User query string

        Returns:
            GlobalSearchResult with auto-selected level
        """
        optimal_level = self._select_optimal_level(query)
        original_level = self.community_level

        logger.info(f"Auto-selected level {optimal_level} for query: '{query[:50]}...'")

        self.community_level = optimal_level
        result = self.search(query)
        self.community_level = original_level  # Restore original

        return result

    def search_multi_level(
        self,
        query: str,
        levels: List[int] = [0, 1]
    ) -> GlobalSearchResult:
        """
        Search across multiple community levels and combine results.

        Args:
            query: User query
            levels: List of levels to search (e.g., [0, 1] for fine and coarse)

        Returns:
            Combined GlobalSearchResult
        """
        all_partial_answers = []
        total_communities = 0
        searched_communities = 0

        for level in levels:
            self.community_level = level
            communities = self._get_communities_at_level(level)

            if communities:
                total_communities += len(communities)
                communities = communities[:self.max_communities // len(levels)]
                searched_communities += len(communities)

                partial = self._map_phase(query, communities)
                all_partial_answers.extend(partial)

        if not all_partial_answers:
            return GlobalSearchResult(
                query=query,
                answer="لم يتم العثور على معلومات ذات صلة.",
                partial_answers=[],
                communities_searched=searched_communities,
                total_communities=total_communities,
                search_level=-1  # Multiple levels
            )

        # Filter and reduce across all levels
        relevant = self._filter_phase(all_partial_answers)
        final_answer, themes, confidence = self._reduce_phase(query, relevant)

        return GlobalSearchResult(
            query=query,
            answer=final_answer,
            partial_answers=relevant,
            communities_searched=searched_communities,
            total_communities=total_communities,
            search_level=-1,  # Indicates multi-level
            confidence=confidence,
            themes=themes
        )

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# Singleton instance
_global_search_engine: Optional[GlobalSearchEngine] = None


def get_global_search_engine(
    neo4j_client=None,
    **kwargs
) -> GlobalSearchEngine:
    """Get or create global search engine singleton."""
    global _global_search_engine

    if _global_search_engine is None:
        if neo4j_client is None:
            # Try to get default client
            try:
                from ..graph_builder import Neo4jClient
                neo4j_client = Neo4jClient()
            except Exception as e:
                logger.error(f"Could not initialize Neo4j client: {e}")
                raise

        _global_search_engine = GlobalSearchEngine(
            neo4j_client=neo4j_client,
            **kwargs
        )

    return _global_search_engine
