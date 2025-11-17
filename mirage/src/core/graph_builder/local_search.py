"""
Local Search Module for GraphRAG

Implements entity-centric search using multi-hop graph traversal.
Answers specific questions about entities and their relationships.

Local Search vs Global Search:
- Local: "Who works at IBM?" → traverses from IBM entity
- Global: "What are the main themes?" → queries community summaries

Uses graph traversal (Phase 1) + Allam LLM for answer generation.
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import requests

logger = logging.getLogger(__name__)


@dataclass
class LocalSearchResult:
    """Result from local search."""
    query: str
    answer: str
    seed_entities: List[str]
    discovered_entities: int
    relationships: int
    confidence: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "seed_entities": self.seed_entities,
            "discovered_entities": self.discovered_entities,
            "relationships": self.relationships,
            "confidence": self.confidence
        }


class LocalSearchEngine:
    """
    Local search engine using entity-centric graph traversal.

    Strategy:
    1. Extract entities from query
    2. Traverse graph from seed entities (2-3 hops)
    3. Format context (entities + relationships)
    4. Query LLM with context

    Best for:
    - "Who works at IBM?"
    - "What is Ahmed Hassan's role?"
    - "How is X related to Y?"
    - Specific fact retrieval

    Not good for:
    - "What are the main themes?" (use Global Search)
    - Holistic questions about entire knowledge base

    Usage:
        local_search = LocalSearchEngine(neo4j_client, graph_traversal, llm_endpoint)
        result = local_search.search("Who works at IBM?")
        print(result.answer)
    """

    def __init__(
        self,
        neo4j_client,
        graph_traversal,
        llm_endpoint: str = "http://tgi:8765",
        max_hops: int = 2,
        max_tokens: int = 600,
        temperature: float = 0.3
    ):
        """
        Initialize local search engine.

        Args:
            neo4j_client: Neo4j client instance
            graph_traversal: GraphTraversal instance (from Phase 1)
            llm_endpoint: TGI endpoint for Allam
            max_hops: Maximum hops for graph traversal (1-3 recommended)
            max_tokens: Maximum tokens for answer
            temperature: LLM temperature
        """
        self.neo4j_client = neo4j_client
        self.graph_traversal = graph_traversal
        self.llm_endpoint = llm_endpoint
        self.max_hops = max_hops
        self.max_tokens = max_tokens
        self.temperature = temperature

    def search(
        self,
        query: str,
        seed_entities: Optional[List[str]] = None
    ) -> LocalSearchResult:
        """
        Execute local search from seed entities.

        Args:
            query: User question (specific, entity-focused)
            seed_entities: Optional seed entities (if None, extracted from query)

        Returns:
            LocalSearchResult with answer and metadata

        Example:
            >>> engine = LocalSearchEngine(neo4j_client, graph_traversal)
            >>> result = engine.search("Who works at IBM?")
            >>> print(result.answer)
        """
        logger.info(f"Local search: '{query}'")

        # Step 1: Extract or use provided seed entities
        if not seed_entities:
            seed_entities = self._extract_entities_from_query(query)

        if not seed_entities:
            logger.warning("No seed entities found for local search")
            return LocalSearchResult(
                query=query,
                answer="Could not identify entities in the query. Please be more specific.",
                seed_entities=[],
                discovered_entities=0,
                relationships=0,
                confidence=0.0
            )

        logger.info(f"Seed entities: {seed_entities}")

        # Step 2: Traverse graph from seed entities
        traversal_result = self.graph_traversal.traverse_from_entities(
            entity_names=seed_entities,
            max_hops=self.max_hops
        )

        logger.info(
            f"Traversal complete: {traversal_result.total_nodes} nodes, "
            f"{traversal_result.total_edges} edges"
        )

        # Step 3: Format context for LLM
        context = self._format_context(
            seed_entities=seed_entities,
            discovered_entities=traversal_result.discovered_entities[:15],  # Limit for Allam 2K
            relationships=traversal_result.relationships[:15]
        )

        # Step 4: Generate answer using LLM
        answer = self._generate_answer(query, context)

        # Calculate confidence
        confidence = min(1.0, traversal_result.total_nodes / 10.0)  # Max at 10+ nodes

        return LocalSearchResult(
            query=query,
            answer=answer,
            seed_entities=seed_entities,
            discovered_entities=traversal_result.total_nodes,
            relationships=traversal_result.total_edges,
            confidence=confidence
        )

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract potential entity names from query.

        Uses keyword matching against known entities in Neo4j.

        Args:
            query: User question

        Returns:
            List of entity names found in query
        """
        # Get all entity names from Neo4j
        entity_query = """
        MATCH (e:Entity)
        RETURN e.name as name
        LIMIT 1000
        """

        try:
            results = self.neo4j_client.execute_query(entity_query)
            all_entities = [r.get('name') for r in results if r.get('name')]

            # Find entities mentioned in query (case-insensitive)
            query_lower = query.lower()
            found_entities = []

            for entity in all_entities:
                if entity.lower() in query_lower:
                    found_entities.append(entity)

            # Sort by length (prefer longer, more specific matches)
            found_entities.sort(key=len, reverse=True)

            # Return top 3 entities
            return found_entities[:3]

        except Exception as e:
            logger.error(f"Error extracting entities from query: {e}")
            return []

    def _format_context(
        self,
        seed_entities: List[str],
        discovered_entities: List[Dict],
        relationships: List[Dict]
    ) -> str:
        """
        Format graph context for LLM.

        Args:
            seed_entities: Starting entities
            discovered_entities: Discovered entities from traversal
            relationships: Relationships found

        Returns:
            Formatted context string
        """
        context_parts = []

        # Seed entities
        context_parts.append("الكيانات الأساسية:")
        for entity in seed_entities:
            context_parts.append(f"- {entity}")

        # Discovered entities (limit to 15 for Allam 2K)
        if discovered_entities:
            context_parts.append("\nالكيانات المرتبطة:")
            for entity in discovered_entities:
                name = entity.get('name', 'Unknown')
                entity_type = entity.get('type', 'Unknown')
                context_parts.append(f"- {name} ({entity_type})")

        # Relationships (limit to 15)
        if relationships:
            context_parts.append("\nالعلاقات:")
            for rel in relationships:
                source = rel.get('source', 'Unknown')
                target = rel.get('target', 'Unknown')
                rel_type = rel.get('type', 'RELATED_TO')
                context_parts.append(f"- {source} → {rel_type} → {target}")

        return "\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using LLM with graph context.

        Args:
            query: User question
            context: Formatted graph context

        Returns:
            Generated answer
        """
        # Build prompt (Arabic-first, bilingual)
        prompt = f"""استنادًا إلى السياق التالي من قاعدة المعرفة، أجب على السؤال.

السياق:
{context}

السؤال: {query}

التعليمات:
1. أجب بناءً على المعلومات المتاحة في السياق فقط
2. كن دقيقًا ومحددًا
3. إذا لم يحتوي السياق على معلومات كافية، اذكر ذلك
4. اكتب إجابة موجزة (2-4 فقرات)

الإجابة:
"""

        # Call Allam
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
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get('generated_text', '').strip()

                if not answer:
                    return "No answer could be generated from the available context."

                return answer

            else:
                logger.error(f"LLM request failed: {response.status_code}")
                return "Error generating answer. Please try again."

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"

    def search_entity_neighborhood(
        self,
        entity_name: str,
        hops: int = 1
    ) -> LocalSearchResult:
        """
        Search the neighborhood of a specific entity.

        Args:
            entity_name: Entity to explore
            hops: Number of hops (default 1 for direct neighbors)

        Returns:
            LocalSearchResult with neighborhood information

        Example:
            >>> result = engine.search_entity_neighborhood("IBM", hops=2)
        """
        query = f"What information is available about {entity_name}?"

        # Use this entity as seed
        return self.search(query=query, seed_entities=[entity_name])

    def find_connection(
        self,
        entity1: str,
        entity2: str
    ) -> LocalSearchResult:
        """
        Find connection/relationship between two entities.

        Args:
            entity1: First entity
            entity2: Second entity

        Returns:
            LocalSearchResult describing the connection

        Example:
            >>> result = engine.find_connection("Ahmed Hassan", "IBM")
        """
        query = f"How are {entity1} and {entity2} related?"

        # Use both as seeds
        return self.search(query=query, seed_entities=[entity1, entity2])
