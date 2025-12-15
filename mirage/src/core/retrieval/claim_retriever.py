"""
Claim-Based Retriever for GraphRAG
Retrieves factual claims to answer precise factual questions.

Claims are structured facts (subject-predicate-object) with evidence.
This retriever is optimal for questions like:
- "When did X launch Y?"
- "Who leads organization Z?"
- "What did X announce?"
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from ..graph_builder import Neo4jClient


@dataclass
class ClaimMatch:
    """A matched claim from the database"""
    claim_id: str
    subject: str
    predicate: str
    object: str
    description: str
    evidence_text: str
    confidence: float
    match_score: float  # How well it matches the query
    match_type: str  # "subject", "predicate", "object", "description"


@dataclass
class ClaimRetrievalResult:
    """Result from claim-based retrieval"""
    claims: List[ClaimMatch]
    total_claims_searched: int
    query_entities: List[str]  # Entities found in the query


class ClaimRetriever:
    """
    Retrieve claims to answer factual questions.

    Search strategies:
    1. Entity match: Find claims where subject/object matches query entities
    2. Predicate match: Find claims where predicate matches query verbs
    3. Full-text: Search claim descriptions for query terms
    """

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """Initialize claim retriever"""
        self.neo4j_client = neo4j_client or Neo4jClient()
        self._ensure_connected()
        logger.info("ClaimRetriever initialized")

    def _ensure_connected(self):
        """Ensure Neo4j connection"""
        if not hasattr(self.neo4j_client, '_driver') or self.neo4j_client._driver is None:
            self.neo4j_client.connect()

    def retrieve_claims(
        self,
        query: str,
        top_k: int = 10,
        min_confidence: float = 0.5
    ) -> ClaimRetrievalResult:
        """
        Retrieve claims relevant to the query.

        Args:
            query: User question
            top_k: Maximum claims to return
            min_confidence: Minimum claim confidence

        Returns:
            ClaimRetrievalResult with matched claims
        """
        logger.info(f"ClaimRetriever searching for: {query[:80]}...")

        # Step 1: Extract potential entities from query
        query_entities = self._extract_query_entities(query)

        # Step 2: Search claims by multiple strategies
        all_claims = []

        # Strategy 1: Entity-based search (subject/object match)
        if query_entities:
            entity_claims = self._search_by_entities(query_entities, min_confidence)
            all_claims.extend(entity_claims)
            logger.info(f"  Entity search: {len(entity_claims)} claims")

        # Strategy 2: Keyword/predicate search
        keyword_claims = self._search_by_keywords(query, min_confidence)
        all_claims.extend(keyword_claims)
        logger.info(f"  Keyword search: {len(keyword_claims)} claims")

        # Strategy 3: Full-text description search
        description_claims = self._search_by_description(query, min_confidence)
        all_claims.extend(description_claims)
        logger.info(f"  Description search: {len(description_claims)} claims")

        # Deduplicate by claim_id
        seen_ids = set()
        unique_claims = []
        for claim in all_claims:
            if claim.claim_id not in seen_ids:
                seen_ids.add(claim.claim_id)
                unique_claims.append(claim)

        # Sort by match_score, then confidence
        unique_claims.sort(key=lambda c: (c.match_score, c.confidence), reverse=True)

        # Take top_k
        top_claims = unique_claims[:top_k]

        # Get total claims count
        total_claims = self._get_total_claims()

        logger.info(f"ClaimRetriever returning {len(top_claims)}/{total_claims} claims")

        return ClaimRetrievalResult(
            claims=top_claims,
            total_claims_searched=total_claims,
            query_entities=query_entities
        )

    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Extract potential entity names from query.

        Simple heuristic approach - looks for:
        - Capitalized words/phrases
        - Known entity patterns
        - Arabic proper nouns
        """
        entities = []

        # Check for entities in Neo4j that appear in query
        query_lower = query.lower()

        # Search for entity names that appear in the query
        search_query = """
        MATCH (e:Entity)
        WHERE toLower($query) CONTAINS toLower(e.name)
           OR toLower(e.name) CONTAINS toLower($query_part)
        RETURN DISTINCT e.name as name
        LIMIT 10
        """

        # Try with different query parts
        query_words = query.split()
        for i in range(len(query_words)):
            for j in range(i + 1, min(i + 5, len(query_words) + 1)):
                query_part = ' '.join(query_words[i:j])
                if len(query_part) >= 3:  # Skip very short parts
                    try:
                        results = self.neo4j_client.execute_query(
                            search_query,
                            {'query': query, 'query_part': query_part}
                        )
                        for r in results:
                            if r['name'] and r['name'] not in entities:
                                entities.append(r['name'])
                    except Exception:
                        pass

        return entities[:5]  # Limit to top 5 entities

    def _search_by_entities(
        self,
        entities: List[str],
        min_confidence: float
    ) -> List[ClaimMatch]:
        """Search claims by entity names (subject/object)"""
        claims = []

        query = """
        MATCH (c:Claim)
        WHERE (toLower(c.subject) CONTAINS toLower($entity)
               OR toLower(c.object) CONTAINS toLower($entity))
          AND c.confidence >= $min_confidence
        RETURN c.claim_id as claim_id,
               c.subject as subject,
               c.predicate as predicate,
               c.object as object,
               c.description as description,
               c.evidence_text as evidence_text,
               c.confidence as confidence,
               CASE
                   WHEN toLower(c.subject) = toLower($entity) THEN 1.0
                   WHEN toLower(c.object) = toLower($entity) THEN 0.95
                   WHEN toLower(c.subject) CONTAINS toLower($entity) THEN 0.8
                   ELSE 0.7
               END as match_score
        ORDER BY match_score DESC, c.confidence DESC
        LIMIT 20
        """

        for entity in entities:
            try:
                results = self.neo4j_client.execute_query(
                    query,
                    {'entity': entity, 'min_confidence': min_confidence}
                )
                for r in results:
                    claims.append(ClaimMatch(
                        claim_id=r['claim_id'],
                        subject=r['subject'] or '',
                        predicate=r['predicate'] or '',
                        object=r['object'] or '',
                        description=r['description'] or '',
                        evidence_text=r['evidence_text'] or '',
                        confidence=r['confidence'] or 0.0,
                        match_score=r['match_score'] or 0.0,
                        match_type='entity'
                    ))
            except Exception as e:
                logger.error(f"Entity search error: {e}")

        return claims

    def _search_by_keywords(
        self,
        query: str,
        min_confidence: float
    ) -> List[ClaimMatch]:
        """Search claims by predicate keywords"""
        claims = []

        # Common predicate mappings (query words -> predicates)
        predicate_mappings = {
            # English
            'launch': ['launched', 'أطلق', 'أطلقت'],
            'announce': ['announced', 'أعلن', 'أعلنت'],
            'lead': ['leads', 'يرأس', 'يقود'],
            'manage': ['manages', 'يدير'],
            'organize': ['organized', 'نظم', 'نظمت'],
            'establish': ['established', 'أسس', 'أسست'],
            'create': ['created', 'أنشأ', 'أنشأت'],
            'develop': ['developed', 'طور', 'طورت'],
            'support': ['supports', 'يدعم', 'تدعم'],
            'achieve': ['achieved', 'حقق', 'حققت'],
            # Arabic
            'أطلق': ['launched', 'أطلق', 'أطلقت'],
            'أعلن': ['announced', 'أعلن', 'أعلنت'],
            'يرأس': ['leads', 'يرأس', 'يقود'],
            'نظم': ['organized', 'نظم', 'نظمت'],
        }

        # Find matching predicates
        query_lower = query.lower()
        matching_predicates = []

        for keyword, predicates in predicate_mappings.items():
            if keyword.lower() in query_lower:
                matching_predicates.extend(predicates)

        if not matching_predicates:
            return []

        # Search for claims with matching predicates
        search_query = """
        MATCH (c:Claim)
        WHERE ANY(p IN $predicates WHERE toLower(c.predicate) CONTAINS toLower(p))
          AND c.confidence >= $min_confidence
        RETURN c.claim_id as claim_id,
               c.subject as subject,
               c.predicate as predicate,
               c.object as object,
               c.description as description,
               c.evidence_text as evidence_text,
               c.confidence as confidence,
               0.75 as match_score
        ORDER BY c.confidence DESC
        LIMIT 15
        """

        try:
            results = self.neo4j_client.execute_query(
                search_query,
                {'predicates': matching_predicates, 'min_confidence': min_confidence}
            )
            for r in results:
                claims.append(ClaimMatch(
                    claim_id=r['claim_id'],
                    subject=r['subject'] or '',
                    predicate=r['predicate'] or '',
                    object=r['object'] or '',
                    description=r['description'] or '',
                    evidence_text=r['evidence_text'] or '',
                    confidence=r['confidence'] or 0.0,
                    match_score=r['match_score'] or 0.0,
                    match_type='predicate'
                ))
        except Exception as e:
            logger.error(f"Keyword search error: {e}")

        return claims

    def _search_by_description(
        self,
        query: str,
        min_confidence: float
    ) -> List[ClaimMatch]:
        """Search claims by description full-text"""
        claims = []

        # Extract significant words from query (skip stop words)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'when', 'where',
                      'who', 'which', 'how', 'does', 'did', 'do', 'in', 'on', 'at', 'to',
                      'ما', 'من', 'في', 'على', 'هل', 'متى', 'أين', 'كيف', 'هي', 'هو'}

        query_words = [w for w in query.split() if w.lower() not in stop_words and len(w) >= 3]

        if not query_words:
            return []

        # Build search query
        search_query = """
        MATCH (c:Claim)
        WHERE c.confidence >= $min_confidence
          AND ANY(word IN $words WHERE
              toLower(c.description) CONTAINS toLower(word)
              OR toLower(c.evidence_text) CONTAINS toLower(word)
          )
        RETURN c.claim_id as claim_id,
               c.subject as subject,
               c.predicate as predicate,
               c.object as object,
               c.description as description,
               c.evidence_text as evidence_text,
               c.confidence as confidence,
               0.6 as match_score
        ORDER BY c.confidence DESC
        LIMIT 15
        """

        try:
            results = self.neo4j_client.execute_query(
                search_query,
                {'words': query_words, 'min_confidence': min_confidence}
            )
            for r in results:
                claims.append(ClaimMatch(
                    claim_id=r['claim_id'],
                    subject=r['subject'] or '',
                    predicate=r['predicate'] or '',
                    object=r['object'] or '',
                    description=r['description'] or '',
                    evidence_text=r['evidence_text'] or '',
                    confidence=r['confidence'] or 0.0,
                    match_score=r['match_score'] or 0.0,
                    match_type='description'
                ))
        except Exception as e:
            logger.error(f"Description search error: {e}")

        return claims

    def _get_total_claims(self) -> int:
        """Get total number of claims in database"""
        try:
            result = self.neo4j_client.execute_query(
                "MATCH (c:Claim) RETURN count(c) as total",
                {}
            )
            return result[0]['total'] if result else 0
        except Exception:
            return 0

    def format_claims_for_context(self, claims: List[ClaimMatch]) -> str:
        """
        Format claims as context for LLM answer generation.

        Returns a structured text representation of claims.
        """
        if not claims:
            return ""

        context_parts = ["## Relevant Facts (Claims):\n"]

        for i, claim in enumerate(claims, 1):
            # Format: "1. [Subject] [predicate] [Object]"
            fact_line = f"{i}. **{claim.subject}** {claim.predicate} **{claim.object}**"

            # Add description if available
            if claim.description:
                fact_line += f"\n   - {claim.description}"

            # Add evidence if available
            if claim.evidence_text and len(claim.evidence_text) < 200:
                fact_line += f"\n   - Evidence: \"{claim.evidence_text}\""

            # Add confidence
            fact_line += f"\n   - (Confidence: {claim.confidence:.2f})"

            context_parts.append(fact_line)

        return "\n".join(context_parts)


def get_claim_retriever(neo4j_client: Optional[Neo4jClient] = None) -> ClaimRetriever:
    """Factory function to get claim retriever"""
    return ClaimRetriever(neo4j_client=neo4j_client)
