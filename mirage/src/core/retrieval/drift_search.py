"""
DRIFT Search Engine for GraphRAG

Microsoft GraphRAG's DRIFT (Dynamic Reasoning and Inference with Flexible Traversal)
search strategy that uses iterative refinement with follow-up questions.

DRIFT Search Algorithm (per Microsoft spec):
1. PRIMER PHASE:
   - Compare query against top-K community reports
   - Generate initial broad answer
   - Generate 0-N follow-up questions for deeper exploration

2. FOLLOW-UP PHASE (Iterative):
   - For each follow-up question:
     - Search community reports for relevant info
     - Use local search for detailed context
     - Generate intermediate answer
     - Potentially generate more follow-ups
   - Continue until confidence is high or max iterations

3. OUTPUT PHASE:
   - Hierarchically organize intermediate answers
   - Synthesize final comprehensive answer

Key Innovation:
- Uses LLM to dynamically generate follow-up questions
- Iteratively refines answers through multi-hop reasoning
- Provides both breadth (global themes) and depth (local facts)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import time
import json
import re

from ..graph_builder import Neo4jClient
from ..config.constants import TGI_ENDPOINT_DEFAULT


class DriftPhase(Enum):
    """Phases of DRIFT search"""
    PRIMER = "primer"       # Initial broad search + follow-up generation
    FOLLOW_UP = "follow_up" # Iterative refinement with generated questions
    LOCAL = "local"         # Targeted local search
    OUTPUT = "output"       # Final synthesis


@dataclass
class FollowUpQuestion:
    """A generated follow-up question"""
    question: str
    priority: int  # 1-5, higher = more important
    context_needed: str  # What type of context this question needs
    answered: bool = False
    answer: str = ""


@dataclass
class DriftIteration:
    """Result from one iteration of DRIFT"""
    question: str
    answer: str
    confidence: float
    sources: List[str]
    new_follow_ups: List[FollowUpQuestion]


@dataclass
class DriftSearchResult:
    """Result from DRIFT search"""
    query: str
    answer: str
    primer_answer: str  # Initial broad answer
    follow_up_questions: List[FollowUpQuestion]
    iterations: List[DriftIteration]
    global_context: List[str]  # Community summaries used
    local_context: List[Dict[str, Any]]  # Chunks retrieved
    entities_traversed: List[str]
    communities_visited: List[str]
    confidence: float
    phases_executed: List[DriftPhase]
    total_time_ms: float
    drift_path: List[str]


@dataclass
class DriftConfig:
    """Configuration for DRIFT search"""
    # Primer phase
    max_communities_primer: int = 10
    primer_min_confidence: float = 0.5
    max_initial_follow_ups: int = 3

    # Follow-up phase
    max_iterations: int = 3
    max_follow_ups_per_iteration: int = 2
    confidence_threshold: float = 0.7  # Stop if confidence exceeds this

    # Local search
    max_chunks_local: int = 10
    max_entities_per_search: int = 5

    # Output
    max_answer_tokens: int = 500


class DriftSearchEngine:
    """
    DRIFT Search implementation following Microsoft GraphRAG specification.

    DRIFT dynamically generates follow-up questions and iteratively refines
    answers through multi-hop reasoning over the knowledge graph.
    """

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        config: Optional[DriftConfig] = None,
        llm_endpoint: str = TGI_ENDPOINT_DEFAULT
    ):
        """Initialize DRIFT search engine"""
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.config = config or DriftConfig()
        self.llm_endpoint = llm_endpoint

        self._ensure_connected()
        logger.info("DriftSearchEngine initialized with iterative refinement")

    def _ensure_connected(self):
        """Ensure Neo4j connection"""
        if not hasattr(self.neo4j_client, '_driver') or self.neo4j_client._driver is None:
            self.neo4j_client.connect()

    def search(self, query: str) -> DriftSearchResult:
        """
        Execute DRIFT search with iterative refinement.

        Args:
            query: User question

        Returns:
            DriftSearchResult with comprehensive answer
        """
        start_time = time.time()
        drift_path = []
        phases_executed = []
        all_global_context = []
        all_local_context = []
        all_entities = set()
        all_communities = set()

        logger.info(f"DRIFT Search starting for: {query[:80]}...")

        # =========================================================================
        # PHASE 1: PRIMER - Broad search + follow-up question generation
        # =========================================================================
        drift_path.append("PRIMER: Starting broad community search")
        phases_executed.append(DriftPhase.PRIMER)

        # Get community summaries
        community_results = self._search_communities(query)
        primer_context = [r['summary'] for r in community_results if r.get('summary')]
        all_global_context.extend(primer_context)
        all_communities.update(r.get('community_id', '') for r in community_results)

        logger.info(f"  Primer: Found {len(community_results)} community summaries")

        # Generate primer answer and follow-up questions
        primer_answer, initial_confidence, follow_ups = self._primer_phase(
            query, primer_context
        )
        drift_path.append(f"PRIMER: Generated {len(follow_ups)} follow-up questions")

        logger.info(f"  Primer answer confidence: {initial_confidence:.2f}")
        logger.info(f"  Generated {len(follow_ups)} follow-up questions")

        # =========================================================================
        # PHASE 2: FOLLOW-UP - Iterative refinement
        # =========================================================================
        iterations = []
        current_confidence = initial_confidence

        if follow_ups and current_confidence < self.config.confidence_threshold:
            phases_executed.append(DriftPhase.FOLLOW_UP)
            drift_path.append("FOLLOW_UP: Beginning iterative refinement")

            for iteration_num in range(self.config.max_iterations):
                if current_confidence >= self.config.confidence_threshold:
                    drift_path.append(f"FOLLOW_UP: Confidence {current_confidence:.2f} sufficient, stopping")
                    break

                # Get unanswered follow-ups, sorted by priority
                unanswered = [f for f in follow_ups if not f.answered]
                if not unanswered:
                    break

                unanswered.sort(key=lambda x: x.priority, reverse=True)
                question_to_answer = unanswered[0]

                drift_path.append(f"ITERATION {iteration_num + 1}: Answering '{question_to_answer.question[:50]}...'")

                # Execute local search for this follow-up
                phases_executed.append(DriftPhase.LOCAL)
                local_results = self._local_search_for_question(question_to_answer.question)
                all_local_context.extend(local_results)
                all_entities.update(r.get('entity_name', '') for r in local_results)

                # Get additional community context if needed
                additional_communities = self._search_communities(question_to_answer.question, limit=3)
                additional_context = [r['summary'] for r in additional_communities if r.get('summary')]
                all_global_context.extend(additional_context)

                # Generate answer for this follow-up
                iteration_result = self._answer_follow_up(
                    question_to_answer,
                    local_results,
                    additional_context,
                    primer_answer
                )

                question_to_answer.answered = True
                question_to_answer.answer = iteration_result.answer
                iterations.append(iteration_result)

                # Add new follow-ups if any
                follow_ups.extend(iteration_result.new_follow_ups)

                # Update confidence
                current_confidence = max(current_confidence, iteration_result.confidence)
                drift_path.append(f"ITERATION {iteration_num + 1}: Confidence now {current_confidence:.2f}")

                logger.info(f"  Iteration {iteration_num + 1}: answered, confidence {iteration_result.confidence:.2f}")

        # =========================================================================
        # PHASE 3: OUTPUT - Synthesize final answer
        # =========================================================================
        phases_executed.append(DriftPhase.OUTPUT)
        drift_path.append("OUTPUT: Synthesizing final answer")

        final_answer = self._synthesize_final_answer(
            query,
            primer_answer,
            iterations,
            all_global_context,
            all_local_context
        )

        total_time = (time.time() - start_time) * 1000

        return DriftSearchResult(
            query=query,
            answer=final_answer,
            primer_answer=primer_answer,
            follow_up_questions=follow_ups,
            iterations=iterations,
            global_context=list(set(all_global_context)),
            local_context=all_local_context,
            entities_traversed=list(all_entities),
            communities_visited=list(all_communities),
            confidence=current_confidence,
            phases_executed=phases_executed,
            total_time_ms=total_time,
            drift_path=drift_path
        )

    def _search_communities(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """Search community summaries"""
        limit = limit or self.config.max_communities_primer

        search_query = """
        MATCH (c:Community)
        WHERE c.summary IS NOT NULL AND c.summary <> ''
        OPTIONAL MATCH (c)<-[:BELONGS_TO]-(e:Entity)
        WITH c, collect(DISTINCT e.name) as entities
        RETURN c.id as community_id,
               c.level as level,
               c.summary as summary,
               c.themes as themes,
               c.title as title,
               entities[0..5] as key_entities,
               size(entities) as entity_count
        ORDER BY c.level DESC, entity_count DESC
        LIMIT $limit
        """

        try:
            return self.neo4j_client.execute_query(search_query, {'limit': limit})
        except Exception as e:
            logger.error(f"Community search error: {e}")
            return []

    def _primer_phase(
        self,
        query: str,
        community_summaries: List[str]
    ) -> Tuple[str, float, List[FollowUpQuestion]]:
        """
        Execute PRIMER phase: generate initial answer and follow-up questions.

        Returns:
            Tuple of (primer_answer, confidence, follow_up_questions)
        """
        import requests

        # Build context from community summaries
        context = "\n".join(f"- {s[:500]}" for s in community_summaries[:5])

        # Detect language
        is_arabic = self._is_arabic_query(query)

        # Prompt to generate answer AND follow-up questions
        if is_arabic:
            prompt = f"""أنت مساعد بحث ذكي. بناءً على السياق التالي، أجب على السؤال ثم اقترح أسئلة متابعة للحصول على معلومات أعمق.

السياق:
{context}

السؤال: {query}

أجب بصيغة JSON التالية:
{{
    "answer": "إجابتك الأولية هنا",
    "confidence": 0.0 إلى 1.0,
    "follow_up_questions": [
        {{"question": "سؤال المتابعة", "priority": 1-5, "context_needed": "نوع السياق المطلوب"}}
    ]
}}

أجب بصيغة JSON فقط:"""
        else:
            prompt = f"""You are an intelligent research assistant. Based on the context below, answer the question and suggest follow-up questions for deeper exploration.

Context:
{context}

Question: {query}

Respond in this JSON format:
{{
    "answer": "Your initial answer here",
    "confidence": 0.0 to 1.0,
    "follow_up_questions": [
        {{"question": "Follow-up question", "priority": 1-5, "context_needed": "Type of context needed"}}
    ]
}}

Respond with JSON only:"""

        try:
            response = requests.post(
                f"{self.llm_endpoint}/v1/chat/completions",
                json={
                    "model": "tgi",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 800
                },
                timeout=30
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()

            # Parse JSON response
            result = self._parse_json_response(content)

            answer = result.get("answer", "Unable to generate initial answer.")
            confidence = float(result.get("confidence", 0.5))

            follow_ups = []
            for fq in result.get("follow_up_questions", [])[:self.config.max_initial_follow_ups]:
                follow_ups.append(FollowUpQuestion(
                    question=fq.get("question", ""),
                    priority=int(fq.get("priority", 3)),
                    context_needed=fq.get("context_needed", "general")
                ))

            return answer, confidence, follow_ups

        except Exception as e:
            logger.error(f"Primer phase error: {e}")
            return "Unable to generate initial answer.", 0.3, []

    def _local_search_for_question(self, question: str) -> List[Dict[str, Any]]:
        """Execute local search to answer a follow-up question"""
        # Extract potential entities from the question
        entities = self._extract_entities_from_text(question)

        results = []

        if entities:
            entity_query = """
            UNWIND $entities as entity_name
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower(entity_name)
               OR toLower(e.description) CONTAINS toLower(entity_name)
            MATCH (c:Chunk)-[:MENTIONS]->(e)
            RETURN DISTINCT c.id as chunk_id,
                   c.text as text,
                   c.document_id as document_id,
                   e.name as entity_name,
                   e.description as entity_description
            LIMIT $limit
            """

            try:
                results = self.neo4j_client.execute_query(
                    entity_query,
                    {'entities': entities[:5], 'limit': self.config.max_chunks_local}
                )
            except Exception as e:
                logger.error(f"Local search error: {e}")

        # Fallback: keyword search in chunks
        if not results:
            keywords = question.split()[:5]
            fallback_query = """
            MATCH (c:Chunk)
            WHERE any(kw IN $keywords WHERE toLower(c.text) CONTAINS toLower(kw))
            RETURN c.id as chunk_id,
                   c.text as text,
                   c.document_id as document_id
            LIMIT $limit
            """

            try:
                results = self.neo4j_client.execute_query(
                    fallback_query,
                    {'keywords': keywords, 'limit': self.config.max_chunks_local}
                )
            except Exception as e:
                logger.error(f"Fallback search error: {e}")

        return results

    def _answer_follow_up(
        self,
        follow_up: FollowUpQuestion,
        local_context: List[Dict[str, Any]],
        additional_community_context: List[str],
        primer_answer: str
    ) -> DriftIteration:
        """Answer a follow-up question and potentially generate more follow-ups"""
        import requests

        # Build context
        local_text = "\n".join(
            f"- {c.get('text', '')[:300]}" for c in local_context[:5]
        )
        community_text = "\n".join(
            f"- {s[:300]}" for s in additional_community_context[:3]
        )

        is_arabic = self._is_arabic_query(follow_up.question)

        if is_arabic:
            prompt = f"""بناءً على السياق التالي والإجابة الأولية، أجب على سؤال المتابعة.

الإجابة الأولية: {primer_answer[:300]}

سياق محلي جديد:
{local_text}

سياق مجتمعي إضافي:
{community_text}

سؤال المتابعة: {follow_up.question}

أجب بصيغة JSON:
{{
    "answer": "إجابتك هنا",
    "confidence": 0.0 إلى 1.0,
    "new_follow_ups": []
}}

JSON فقط:"""
        else:
            prompt = f"""Based on the context below and the initial answer, answer the follow-up question.

Initial Answer: {primer_answer[:300]}

New Local Context:
{local_text}

Additional Community Context:
{community_text}

Follow-up Question: {follow_up.question}

Respond in JSON format:
{{
    "answer": "Your answer here",
    "confidence": 0.0 to 1.0,
    "new_follow_ups": []
}}

JSON only:"""

        try:
            response = requests.post(
                f"{self.llm_endpoint}/v1/chat/completions",
                json={
                    "model": "tgi",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=30
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()

            result = self._parse_json_response(content)

            answer = result.get("answer", "Unable to answer follow-up.")
            confidence = float(result.get("confidence", 0.5))

            new_follow_ups = []
            for fq in result.get("new_follow_ups", [])[:self.config.max_follow_ups_per_iteration]:
                new_follow_ups.append(FollowUpQuestion(
                    question=fq.get("question", ""),
                    priority=int(fq.get("priority", 2)),
                    context_needed=fq.get("context_needed", "general")
                ))

            sources = [c.get('chunk_id', '') for c in local_context]

            return DriftIteration(
                question=follow_up.question,
                answer=answer,
                confidence=confidence,
                sources=sources,
                new_follow_ups=new_follow_ups
            )

        except Exception as e:
            logger.error(f"Follow-up answer error: {e}")
            return DriftIteration(
                question=follow_up.question,
                answer="Unable to answer.",
                confidence=0.3,
                sources=[],
                new_follow_ups=[]
            )

    def _synthesize_final_answer(
        self,
        query: str,
        primer_answer: str,
        iterations: List[DriftIteration],
        global_context: List[str],
        local_context: List[Dict[str, Any]]
    ) -> str:
        """Synthesize final comprehensive answer from all gathered information"""
        import requests

        # Build comprehensive context from all phases
        iteration_insights = "\n".join(
            f"- Q: {it.question[:100]}\n  A: {it.answer[:200]}"
            for it in iterations
        )

        is_arabic = self._is_arabic_query(query)

        if is_arabic:
            prompt = f"""بناءً على جميع المعلومات المجمعة، قدم إجابة شاملة ونهائية.

السؤال الأصلي: {query}

الإجابة الأولية: {primer_answer[:400]}

رؤى من الأسئلة المتابعة:
{iteration_insights}

قدم إجابة نهائية شاملة ومنظمة (باللغة العربية):"""
        else:
            prompt = f"""Based on all gathered information, provide a comprehensive final answer.

Original Question: {query}

Initial Answer: {primer_answer[:400]}

Insights from Follow-up Questions:
{iteration_insights}

Provide a comprehensive, well-organized final answer (in English):"""

        try:
            response = requests.post(
                f"{self.llm_endpoint}/v1/chat/completions",
                json={
                    "model": "tgi",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": self.config.max_answer_tokens
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Final synthesis error: {e}")
            # Fallback: return primer answer with iteration summaries
            if iterations:
                additional = " ".join(it.answer[:100] for it in iterations[:2])
                return f"{primer_answer} {additional}"
            return primer_answer

    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract potential entity names from text"""
        entities = []

        # Arabic entity patterns
        arabic_prefixes = ["شركة", "هيئة", "وزارة", "جامعة", "مركز", "جائزة"]
        for prefix in arabic_prefixes:
            pattern = rf"{prefix}\s+([^\s،,]+(?:\s+[^\s،,]+)?)"
            matches = re.findall(pattern, text)
            entities.extend(matches)

        # English capitalized words (potential entities)
        english_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        entities.extend(re.findall(english_pattern, text))

        # Acronyms
        acronym_pattern = r'\b([A-Z]{2,6})\b'
        entities.extend(re.findall(acronym_pattern, text))

        return list(set(entities))[:10]

    def _is_arabic_query(self, query: str) -> bool:
        """Detect if query is primarily Arabic"""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', query))
        english_chars = len(re.findall(r'[a-zA-Z]', query))
        return arabic_chars > english_chars

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common issues"""
        # Try to extract JSON from response
        try:
            # First, try direct parse
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: return empty dict
        logger.warning(f"Could not parse JSON from: {content[:100]}...")
        return {}


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_drift_engine: Optional[DriftSearchEngine] = None


def get_drift_search_engine(
    neo4j_client: Optional[Neo4jClient] = None,
    config: Optional[DriftConfig] = None
) -> DriftSearchEngine:
    """Get or create global DRIFT search engine"""
    global _drift_engine

    if _drift_engine is None:
        _drift_engine = DriftSearchEngine(neo4j_client=neo4j_client, config=config)

    return _drift_engine
