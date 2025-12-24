"""
Global Retrieval Modes

Relationship-focused retrieval and TRUE GraphRAG global search.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from ..base_retriever import RetrievalMode, RetrievalResult, RetrievalResponse


class GlobalModeMixin:
    """Mixin providing global retrieval modes"""

    def _global_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """Relationship-focused retrieval: query → relationships → entities → chunks (via Neo4j)"""
        if query_embedding is None or self.index_manager is None:
            return self._naive_retrieve(query, query_embedding, top_k, **kwargs)

        # 1. Get naive results as base
        naive_response = self._naive_retrieve(query, query_embedding, top_k * 2, **kwargs)

        # 2. Try to find relationships in Neo4j related to the query
        relationship_chunks = []
        if self.graph_client:
            try:
                query_terms = [t for t in query.split() if len(t) > 2]

                for term in query_terms[:3]:
                    entities = self.graph_client.search_entities_by_name(term, limit=3)
                    for entity in entities:
                        entity_name = entity.get("name", "")
                        relationships = self.graph_client.get_entity_relationships(
                            entity_name, limit=5
                        )
                        for rel in relationships:
                            source_chunks = self.graph_client.get_entity_chunks(
                                rel.get("source", ""), limit=2
                            )
                            target_chunks = self.graph_client.get_entity_chunks(
                                rel.get("target", ""), limit=2
                            )
                            source_ids = {c.get("chunk_id") for c in source_chunks}
                            for chunk in target_chunks:
                                if chunk.get("chunk_id") in source_ids:
                                    relationship_chunks.append({
                                        "chunk_id": chunk.get("chunk_id", ""),
                                        "document_id": chunk.get("document_id", ""),
                                        "text": chunk.get("text", ""),
                                        "relationship": rel.get("type", "RELATED_TO"),
                                        "source": rel.get("source", ""),
                                        "target": rel.get("target", "")
                                    })
                            for chunk in source_chunks + target_chunks:
                                relationship_chunks.append({
                                    "chunk_id": chunk.get("chunk_id", ""),
                                    "document_id": chunk.get("document_id", ""),
                                    "text": chunk.get("text", ""),
                                    "relationship": rel.get("type", "RELATED_TO"),
                                    "source": rel.get("source", ""),
                                    "target": rel.get("target", "")
                                })
            except Exception as e:
                logger.warning(f"Neo4j relationship search failed: {e}")

        # 3. Combine results
        results = []
        seen_chunks = set()

        for rc in relationship_chunks[:top_k // 2]:
            if rc["chunk_id"] and rc["chunk_id"] not in seen_chunks:
                seen_chunks.add(rc["chunk_id"])
                results.append(RetrievalResult(
                    chunk_id=rc["chunk_id"],
                    document_id=rc["document_id"],
                    text=rc["text"],
                    score=0.9,
                    retrieval_mode="global",
                    via_relationship=rc["relationship"],
                    hop_distance=2
                ))

        for r in naive_response.results:
            if r.chunk_id not in seen_chunks and len(results) < top_k:
                seen_chunks.add(r.chunk_id)
                r.retrieval_mode = "global"
                results.append(r)

        return RetrievalResponse(
            results=results[:top_k],
            query=query,
            mode=RetrievalMode.GLOBAL,
            total_candidates=len(relationship_chunks) + len(naive_response.results),
            metadata={"relationships_found": len(relationship_chunks)}
        )

    def _global_search_retrieve(
        self,
        query: str,
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """
        TRUE GraphRAG Global Search: Map-reduce over community summaries.

        Enables answering holistic queries like:
        - "What are the main themes across all documents?"
        - "Summarize the key topics in this knowledge base"
        """
        try:
            from ..global_search import GlobalSearchEngine

            global_engine = GlobalSearchEngine(
                neo4j_client=self.graph_client,
                llm_endpoint="http://tgi:80",
                max_communities=kwargs.get('max_communities', 30),
                min_relevance=kwargs.get('min_relevance', 0.3),
                community_level=kwargs.get('community_level', 0)
            )

            result = global_engine.search(query)

            logger.info(
                f"Global search: {result.communities_searched} communities queried, "
                f"{len(result.partial_answers)} relevant answers"
            )

            supporting_chunks = []
            if result.partial_answers and self.graph_client:
                community_ids = [pa.community_id for pa in result.partial_answers[:5]]
                supporting_chunks = self._get_chunks_from_communities(community_ids, top_k)

            results = [
                RetrievalResult(
                    chunk_id=chunk.get("chunk_id", ""),
                    document_id=chunk.get("document_id", ""),
                    text=chunk.get("text", ""),
                    score=0.85,
                    retrieval_mode="global_search",
                    metadata={
                        "community_id": chunk.get("community_id", ""),
                        "from_global_search": True
                    }
                )
                for chunk in supporting_chunks
            ]

            return RetrievalResponse(
                results=results[:top_k],
                query=query,
                mode=RetrievalMode.GLOBAL_SEARCH,
                total_candidates=result.communities_searched,
                metadata={
                    "global_answer": result.answer,
                    "communities_searched": result.communities_searched,
                    "total_communities": result.total_communities,
                    "partial_answers_count": len(result.partial_answers),
                    "themes": result.themes,
                    "confidence": result.confidence,
                    "is_global_search": True
                }
            )

        except Exception as e:
            logger.error(f"Global search failed: {e}")
            query_embedding = self.embedder.embed(query) if self.embedder else None
            return self._global_retrieve(query, query_embedding, top_k, **kwargs)

    def _get_chunks_from_communities(
        self,
        community_ids: List[str],
        limit: int = 10
    ) -> List[Dict]:
        """Get representative chunks from specified communities."""
        if not self.graph_client or not community_ids:
            return []

        try:
            query = """
            MATCH (c:Community)
            WHERE c.id IN $community_ids
            MATCH (e:Entity)-[:BELONGS_TO]->(c)
            MATCH (chunk:Chunk)-[:MENTIONS]->(e)
            RETURN DISTINCT
                chunk.id as chunk_id,
                chunk.text as text,
                chunk.document_id as document_id,
                c.id as community_id
            LIMIT $limit
            """

            results = self.graph_client.execute_query(query, {
                'community_ids': community_ids,
                'limit': limit
            })

            return results

        except Exception as e:
            logger.warning(f"Error getting chunks from communities: {e}")
            return []
