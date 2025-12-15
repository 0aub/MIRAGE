"""
Drift Search Engine for GraphRAG
Microsoft GraphRAG's dynamic search strategy that "drifts" between
global and local search based on query context and intermediate results.

Drift Search Flow:
1. Start with global search over community summaries (broad context)
2. Identify relevant communities/entities from global results
3. Drift to local search in relevant communities (detailed context)
4. Combine results with progressive refinement
5. Optionally iterate for complex queries

Key Innovation:
- Unlike static hybrid search, drift search adapts dynamically
- Uses global results to guide local search scope
- Provides both breadth (global themes) and depth (local facts)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import time

from ..graph_builder import Neo4jClient


class DriftPhase(Enum):
    """Phases of drift search"""
    GLOBAL = "global"  # Broad search over community summaries
    LOCAL = "local"    # Targeted search in relevant communities
    CLAIM = "claim"    # Factual claim search
    MERGED = "merged"  # Final merged results


@dataclass
class DriftSearchResult:
    """Result from drift search"""
    query: str
    answer: str
    global_context: List[str]  # Community summaries used
    local_context: List[Dict[str, Any]]  # Chunks retrieved
    claims_used: List[Dict[str, Any]]  # Claims matched
    entities_traversed: List[str]  # Entities visited during drift
    communities_visited: List[str]  # Communities visited
    confidence: float  # Overall confidence
    phases_executed: List[DriftPhase]  # Which phases ran
    total_time_ms: float
    drift_path: List[str]  # Trace of drift decisions


@dataclass
class DriftConfig:
    """Configuration for drift search"""
    # Global search settings
    max_communities_global: int = 10  # Max communities to search globally
    global_confidence_threshold: float = 0.5  # Min confidence to proceed

    # Local search settings
    max_entities_per_community: int = 5  # Max entities to explore per community
    max_chunks_local: int = 10  # Max chunks from local search
    local_hop_depth: int = 2  # How many hops in graph traversal

    # Drift settings
    drift_threshold: float = 0.6  # Confidence below which we drift more
    max_drift_iterations: int = 3  # Max number of drift iterations

    # Claim settings
    include_claims: bool = True  # Whether to search claims
    max_claims: int = 5  # Max claims to include

    # Output
    merge_strategy: str = "weighted"  # "weighted", "sequential", "interleave"


class DriftSearchEngine:
    """
    Drift Search implementation for GraphRAG.

    Drift search dynamically moves between global (broad themes) and
    local (specific facts) search based on query needs and intermediate
    results.

    Strategy:
    1. GLOBAL: Query community summaries for thematic understanding
    2. IDENTIFY: Extract relevant communities and entities
    3. LOCAL: Dive into identified areas for detailed context
    4. CLAIM: (optional) Add factual claims for precision
    5. MERGE: Combine all context for final answer
    """

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        config: Optional[DriftConfig] = None,
        llm_endpoint: str = "http://tgi:80"
    ):
        """Initialize drift search engine"""
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.config = config or DriftConfig()
        self.llm_endpoint = llm_endpoint

        self._ensure_connected()
        logger.info("DriftSearchEngine initialized")

    def _ensure_connected(self):
        """Ensure Neo4j connection"""
        if not hasattr(self.neo4j_client, '_driver') or self.neo4j_client._driver is None:
            self.neo4j_client.connect()

    def search(self, query: str) -> DriftSearchResult:
        """
        Execute drift search for query.

        Args:
            query: User question

        Returns:
            DriftSearchResult with combined context
        """
        start_time = time.time()
        drift_path = []
        phases_executed = []

        logger.info(f"DriftSearch starting for: {query[:80]}...")

        # Phase 1: Global Search (Community Summaries)
        drift_path.append("GLOBAL: Starting broad search")
        phases_executed.append(DriftPhase.GLOBAL)

        global_results = self._global_search(query)
        global_context = [r['summary'] for r in global_results if r.get('summary')]
        communities_visited = [r['community_id'] for r in global_results]

        logger.info(f"  Global: {len(global_results)} communities found")

        # Extract entities mentioned in global summaries
        global_entities = self._extract_entities_from_summaries(global_results)
        drift_path.append(f"IDENTIFY: Found {len(global_entities)} key entities")

        # Phase 2: Determine if we need to drift to local
        global_confidence = self._calculate_global_confidence(global_results, query)

        if global_confidence < self.config.drift_threshold or len(global_entities) > 0:
            # Drift to local search
            drift_path.append(f"DRIFT: Confidence {global_confidence:.2f} < {self.config.drift_threshold}, going local")
            phases_executed.append(DriftPhase.LOCAL)

            local_results = self._local_search(query, global_entities, communities_visited)
            local_context = local_results
            entities_traversed = [e['entity_name'] for e in local_results if e.get('entity_name')]

            logger.info(f"  Local: {len(local_results)} chunks retrieved")
        else:
            drift_path.append(f"STAY: Confidence {global_confidence:.2f} sufficient")
            local_context = []
            entities_traversed = global_entities

        # Phase 3: Claim Search (if enabled)
        claims_used = []
        if self.config.include_claims:
            phases_executed.append(DriftPhase.CLAIM)
            claims_used = self._claim_search(query, entities_traversed)
            drift_path.append(f"CLAIMS: Found {len(claims_used)} relevant claims")
            logger.info(f"  Claims: {len(claims_used)} claims matched")

        # Phase 4: Merge and Generate Answer
        phases_executed.append(DriftPhase.MERGED)

        answer, confidence = self._merge_and_answer(
            query,
            global_context,
            local_context,
            claims_used
        )

        drift_path.append(f"MERGED: Final confidence {confidence:.2f}")

        total_time = (time.time() - start_time) * 1000

        return DriftSearchResult(
            query=query,
            answer=answer,
            global_context=global_context,
            local_context=local_context,
            claims_used=claims_used,
            entities_traversed=entities_traversed,
            communities_visited=communities_visited,
            confidence=confidence,
            phases_executed=phases_executed,
            total_time_ms=total_time,
            drift_path=drift_path
        )

    def _global_search(self, query: str) -> List[Dict[str, Any]]:
        """Search community summaries globally"""
        # Find communities with summaries that might relate to query
        search_query = """
        MATCH (c:Community)
        WHERE c.summary IS NOT NULL AND c.summary <> ''
        OPTIONAL MATCH (c)<-[:BELONGS_TO]-(e:Entity)
        WITH c, collect(DISTINCT e.name) as entities
        RETURN c.id as community_id,
               c.level as level,
               c.summary as summary,
               c.themes as themes,
               c.key_entities as key_entities,
               entities[0..5] as sample_entities,
               size(entities) as entity_count
        ORDER BY c.level DESC, entity_count DESC
        LIMIT $limit
        """

        try:
            results = self.neo4j_client.execute_query(
                search_query,
                {'limit': self.config.max_communities_global}
            )
            return results
        except Exception as e:
            logger.error(f"Global search error: {e}")
            return []

    def _extract_entities_from_summaries(
        self,
        global_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract key entities from community summaries"""
        entities = set()

        for result in global_results:
            # From key_entities field
            key_entities = result.get('key_entities') or []
            for e in key_entities:
                if isinstance(e, str):
                    entities.add(e)

            # From sample_entities field
            sample = result.get('sample_entities') or []
            for e in sample:
                if isinstance(e, str):
                    entities.add(e)

        return list(entities)[:20]  # Limit to top 20

    def _calculate_global_confidence(
        self,
        global_results: List[Dict[str, Any]],
        query: str
    ) -> float:
        """Calculate confidence in global results"""
        if not global_results:
            return 0.0

        # Simple heuristic based on:
        # - Number of communities found
        # - Whether summaries exist
        # - Entity coverage

        summaries_found = sum(1 for r in global_results if r.get('summary'))
        total_entities = sum(r.get('entity_count', 0) for r in global_results)

        # Score components
        community_score = min(1.0, summaries_found / 5)  # 5+ communities = max
        entity_score = min(1.0, total_entities / 50)  # 50+ entities = max

        return 0.5 * community_score + 0.5 * entity_score

    def _local_search(
        self,
        query: str,
        entities: List[str],
        communities: List[str]
    ) -> List[Dict[str, Any]]:
        """Execute local search in identified communities/entities"""
        results = []

        # Search chunks related to identified entities
        if entities:
            entity_query = """
            UNWIND $entities as entity_name
            MATCH (e:Entity)
            WHERE toLower(e.name) = toLower(entity_name)
               OR toLower(e.name) CONTAINS toLower(entity_name)
            MATCH (c:Chunk)-[:MENTIONS]->(e)
            RETURN DISTINCT c.id as chunk_id,
                   c.text as text,
                   c.document_id as document_id,
                   e.name as entity_name,
                   e.description as entity_description
            LIMIT $limit
            """

            try:
                entity_results = self.neo4j_client.execute_query(
                    entity_query,
                    {'entities': entities[:10], 'limit': self.config.max_chunks_local}
                )
                results.extend(entity_results)
            except Exception as e:
                logger.error(f"Local entity search error: {e}")

        # Search chunks in identified communities
        if communities and len(results) < self.config.max_chunks_local:
            community_query = """
            UNWIND $communities as comm_id
            MATCH (comm:Community {id: comm_id})<-[:BELONGS_TO]-(e:Entity)
            MATCH (c:Chunk)-[:MENTIONS]->(e)
            RETURN DISTINCT c.id as chunk_id,
                   c.text as text,
                   c.document_id as document_id,
                   e.name as entity_name,
                   comm.id as community_id
            LIMIT $limit
            """

            try:
                comm_results = self.neo4j_client.execute_query(
                    community_query,
                    {
                        'communities': communities[:5],
                        'limit': self.config.max_chunks_local - len(results)
                    }
                )
                results.extend(comm_results)
            except Exception as e:
                logger.error(f"Local community search error: {e}")

        return results[:self.config.max_chunks_local]

    def _claim_search(
        self,
        query: str,
        entities: List[str]
    ) -> List[Dict[str, Any]]:
        """Search for relevant claims"""
        claims = []

        if not entities:
            return claims

        # Find claims involving identified entities
        claim_query = """
        UNWIND $entities as entity_name
        MATCH (c:Claim)
        WHERE toLower(c.subject) CONTAINS toLower(entity_name)
           OR toLower(c.object) CONTAINS toLower(entity_name)
        RETURN c.claim_id as claim_id,
               c.subject as subject,
               c.predicate as predicate,
               c.object as object,
               c.description as description,
               c.confidence as confidence
        ORDER BY c.confidence DESC
        LIMIT $limit
        """

        try:
            claims = self.neo4j_client.execute_query(
                claim_query,
                {'entities': entities[:10], 'limit': self.config.max_claims}
            )
        except Exception as e:
            logger.error(f"Claim search error: {e}")

        return claims

    def _merge_and_answer(
        self,
        query: str,
        global_context: List[str],
        local_context: List[Dict[str, Any]],
        claims: List[Dict[str, Any]]
    ) -> Tuple[str, float]:
        """Merge all context and generate answer"""
        import requests

        # Build merged context
        context_parts = []

        # Add global context (community summaries)
        if global_context:
            context_parts.append("## Thematic Overview:")
            for i, summary in enumerate(global_context[:5], 1):
                context_parts.append(f"{i}. {summary[:500]}")

        # Add local context (specific chunks)
        if local_context:
            context_parts.append("\n## Detailed Context:")
            for chunk in local_context[:7]:
                text = chunk.get('text', '')[:400]
                entity = chunk.get('entity_name', 'Unknown')
                context_parts.append(f"- [{entity}]: {text}")

        # Add claims
        if claims:
            context_parts.append("\n## Key Facts:")
            for claim in claims[:5]:
                fact = f"{claim.get('subject', '')} {claim.get('predicate', '')} {claim.get('object', '')}"
                context_parts.append(f"- {fact}")

        merged_context = '\n'.join(context_parts)

        # Detect query language and generate answer using LLM
        import re
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', query))
        english_chars = len(re.findall(r'[a-zA-Z]', query))
        is_arabic = arabic_chars > english_chars

        if is_arabic:
            prompt = f"""أنت مساعد ذكي متخصص في الإجابة على الأسئلة بناءً على السياق المقدم.
يجب أن تجيب باللغة العربية فقط. لا تستخدم أي لغة أخرى.

السياق:
{merged_context}

السؤال: {query}

الإجابة (باللغة العربية فقط):"""
        else:
            prompt = f"""You are an intelligent assistant specialized in answering questions based on provided context.
Answer in English only. Be concise and accurate.

Context:
{merged_context}

Question: {query}

Answer (in English only):"""

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
            answer = response.json()["choices"][0]["message"]["content"].strip()

            # Calculate confidence based on context coverage
            confidence = self._calculate_answer_confidence(
                global_context, local_context, claims, answer
            )

            return answer, confidence

        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return "Unable to generate answer due to an error.", 0.0

    def _calculate_answer_confidence(
        self,
        global_context: List[str],
        local_context: List[Dict[str, Any]],
        claims: List[Dict[str, Any]],
        answer: str
    ) -> float:
        """Calculate confidence in the generated answer"""
        # Heuristic based on context coverage
        global_score = min(1.0, len(global_context) / 3)  # 3+ summaries = max
        local_score = min(1.0, len(local_context) / 5)    # 5+ chunks = max
        claim_score = min(1.0, len(claims) / 3)           # 3+ claims = max

        # Answer quality heuristics
        answer_length_score = min(1.0, len(answer) / 200)  # ~200 chars = good answer

        # Check for uncertainty phrases
        uncertainty_phrases = ["don't know", "not sure", "unable to", "no information", "لا أعرف", "غير متأكد"]
        has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
        uncertainty_penalty = 0.3 if has_uncertainty else 0.0

        # Weighted combination
        confidence = (
            0.25 * global_score +
            0.35 * local_score +
            0.20 * claim_score +
            0.20 * answer_length_score -
            uncertainty_penalty
        )

        return max(0.0, min(1.0, confidence))


def get_drift_search_engine(
    neo4j_client: Optional[Neo4jClient] = None,
    config: Optional[DriftConfig] = None
) -> DriftSearchEngine:
    """Factory function to get drift search engine"""
    return DriftSearchEngine(neo4j_client=neo4j_client, config=config)
