"""
Hybrid Vector-Graph Retriever
Implements the "Anchor & Traverse" strategy for hybrid retrieval:
1. Vector search in Qdrant (find semantically similar anchor chunks)
2. Graph expansion in Neo4j (traverse to related entities and chunks)
3. Relevance scoring and ranking
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from ..vector_store import QdrantVectorStore
from ..graph_builder import Neo4jClient
from ..embeddings import JinaEmbedder


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval"""
    # Vector search
    top_k_vector: int = 5  # Number of anchor chunks from vector search

    # Graph expansion limits
    max_chunks_hop1: int = 10  # Max chunks from 1-hop (same entities)
    max_chunks_hop2: int = 5   # Max chunks from 2-hop (related entities)

    # Relevance scoring
    relevance_decay: float = 0.7  # Decay factor per hop (0.7 = 30% reduction per hop)
    min_graph_score: float = 0.4  # Minimum graph relevance to include

    # Filters
    document_ids: Optional[List[str]] = None  # Limit to specific documents


class HybridRetriever:
    """
    Hybrid retrieval combining vector similarity and graph traversal

    Implements the "Anchor & Traverse" pattern:
    - Anchor: Vector search finds semantically similar chunks
    - Traverse: Graph expansion finds structurally related chunks
    """

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        graph_client: Optional[Neo4jClient] = None,
        embedder: Optional[JinaEmbedder] = None,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Initialize hybrid retriever

        Args:
            vector_store: Qdrant client for vector search
            graph_client: Neo4j client for graph traversal
            embedder: Jina embedder for query encoding
            config: Retrieval configuration
        """
        self.vector_store = vector_store or QdrantVectorStore()
        self.graph_client = graph_client or Neo4jClient()
        self.embedder = embedder or JinaEmbedder()
        self.config = config or RetrievalConfig()

        logger.info(
            f"Initialized HybridRetriever: "
            f"top_k={self.config.top_k_vector}, "
            f"1-hop={self.config.max_chunks_hop1}, "
            f"2-hop={self.config.max_chunks_hop2}"
        )

    def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid vector-graph approach

        Args:
            query: User query string
            config: Optional retrieval config (overrides default)

        Returns:
            List of chunks with relevance scores, sorted by combined score
        """
        cfg = config or self.config

        logger.info(f"Hybrid retrieval for query: {query[:100]}...")

        # Step 1: Vector search (anchor chunks)
        anchor_chunks = self._vector_search(query, cfg)

        if not anchor_chunks:
            logger.warning("No anchor chunks found from vector search")
            return []

        logger.info(f"Found {len(anchor_chunks)} anchor chunks from vector search")

        # Step 2: Graph expansion (traverse to related chunks)
        expanded_chunks = self._graph_expansion(anchor_chunks, cfg)

        logger.info(f"Graph expansion found {len(expanded_chunks)} additional chunks")

        # Step 3: Combine and rank results
        all_chunks = self._combine_and_rank(anchor_chunks, expanded_chunks, cfg)

        logger.info(f"Returning {len(all_chunks)} total chunks (vector + graph)")

        return all_chunks

    def _vector_search(
        self,
        query: str,
        config: RetrievalConfig
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search in Qdrant

        Returns:
            List of anchor chunks with vector scores
        """
        # Encode query
        query_embedding = self.embedder.embed([query], task="retrieval.query")[0]

        # Search Qdrant
        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=config.top_k_vector,
            score_threshold=0.3  # Minimum semantic similarity
        )

        # Format results
        anchor_chunks = []
        for result in results:
            anchor_chunks.append({
                "chunk_id": result.get("id"),
                "text": result.get("text"),
                "document_id": result.get("document_id"),
                "metadata": result.get("metadata", {}),
                "vector_score": result.get("score", 0.0),
                "graph_score": 1.0,  # Anchor chunks have max graph score
                "hop_distance": 0,
                "via_entity": None,
                "retrieval_method": "vector_search"
            })

        return anchor_chunks

    def _graph_expansion(
        self,
        anchor_chunks: List[Dict[str, Any]],
        config: RetrievalConfig
    ) -> List[Dict[str, Any]]:
        """
        Expand from anchor chunks using graph traversal

        Performs:
        - 1-hop: Chunks mentioning the same entities
        - 2-hop: Chunks mentioning entities related to anchor entities

        Returns:
            List of expanded chunks with graph scores
        """
        # Get anchor chunk IDs
        anchor_ids = [chunk["chunk_id"] for chunk in anchor_chunks]

        # Build Cypher query for 2-hop expansion with relevance scoring
        query = """
        // Input: List of anchor chunk IDs
        UNWIND $anchor_ids AS anchor_id
        MATCH (anchor:Chunk {id: anchor_id})

        // 1-HOP: Find chunks mentioning SAME entities as anchor
        OPTIONAL MATCH (anchor)-[:MENTIONS]->(entity:Entity)
        OPTIONAL MATCH (same_entity_chunk:Chunk)-[:MENTIONS]->(entity)
        WHERE same_entity_chunk IS NOT NULL
          AND same_entity_chunk.id <> anchor_id
          AND NOT same_entity_chunk.id IN $anchor_ids

        // Collect 1-hop results with chunk IDs for filtering
        WITH [id IN collect(DISTINCT same_entity_chunk.id) WHERE id IS NOT NULL] AS hop1_chunk_ids,
             [result IN collect(DISTINCT {
                 chunk_id: same_entity_chunk.id,
                 text: same_entity_chunk.text,
                 document_id: same_entity_chunk.document_id,
                 entity_name: entity.name,
                 entity_type: entity.type,
                 distance: 1
             }) WHERE result.chunk_id IS NOT NULL] AS hop1_results,
             $anchor_ids AS anchor_ids

        // 2-HOP: Find chunks mentioning RELATED entities
        // Use ALL relationship types: RELATED_TO (LLM), COOCCURS_WITH, SIMILAR_TO
        // First, get entities from 1-hop chunks
        UNWIND hop1_chunk_ids AS hop1_id
        MATCH (hop1_chunk:Chunk {id: hop1_id})
        OPTIONAL MATCH (hop1_chunk)-[:MENTIONS]->(entity1:Entity)
        // Match any relationship type (RELATED_TO, COOCCURS_WITH, SIMILAR_TO)
        OPTIONAL MATCH (entity1)-[rel]-(entity2:Entity)
        WHERE type(rel) IN ['RELATED_TO', 'COOCCURS_WITH', 'SIMILAR_TO']
          AND (
            (rel.source = 'llm' AND rel.confidence >= 0.7) OR
            (rel.source = 'cooccurrence' AND rel.frequency >= 2) OR
            (rel.source = 'semantic' AND rel.score >= 0.75)
          )
        OPTIONAL MATCH (related_chunk:Chunk)-[:MENTIONS]->(entity2)
        WHERE related_chunk.id IS NOT NULL
          AND NOT related_chunk.id IN anchor_ids
          AND NOT related_chunk.id IN hop1_chunk_ids

        WITH hop1_results,
             [result IN collect(DISTINCT {
                 chunk_id: related_chunk.id,
                 text: related_chunk.text,
                 document_id: related_chunk.document_id,
                 entity_name: entity2.name,
                 entity_type: entity2.type,
                 distance: 2
             }) WHERE result.chunk_id IS NOT NULL] AS hop2_results

        // Combine and return
        UNWIND (hop1_results + hop2_results) AS result

        RETURN
            result.chunk_id AS chunk_id,
            result.text AS text,
            result.document_id AS document_id,
            result.entity_name AS via_entity,
            result.entity_type AS entity_type,
            result.distance AS hop_distance
        LIMIT $limit
        """

        # Calculate limit (1-hop + 2-hop)
        total_limit = config.max_chunks_hop1 + config.max_chunks_hop2

        # Execute query
        try:
            if not self.graph_client._connected:
                self.graph_client.connect()

            with self.graph_client.driver.session() as session:
                result = session.run(
                    query,
                    anchor_ids=anchor_ids,
                    limit=total_limit
                )

                expanded_chunks = []
                for record in result:
                    # Calculate graph relevance score with decay
                    hop_distance = record["hop_distance"]
                    graph_score = config.relevance_decay ** hop_distance

                    # Skip if below threshold
                    if graph_score < config.min_graph_score:
                        continue

                    expanded_chunks.append({
                        "chunk_id": record["chunk_id"],
                        "text": record["text"],
                        "document_id": record["document_id"],
                        "metadata": {},
                        "vector_score": 0.0,  # Not from vector search
                        "graph_score": graph_score,
                        "hop_distance": hop_distance,
                        "via_entity": record["via_entity"],
                        "retrieval_method": f"{hop_distance}-hop_graph"
                    })

                return expanded_chunks

        except Exception as e:
            logger.error(f"Error in graph expansion: {e}")
            return []

    def _combine_and_rank(
        self,
        anchor_chunks: List[Dict[str, Any]],
        expanded_chunks: List[Dict[str, Any]],
        config: RetrievalConfig
    ) -> List[Dict[str, Any]]:
        """
        Combine anchor and expanded chunks, compute final scores, and rank

        Final score = (vector_score * 0.6) + (graph_score * 0.4)

        Returns:
            Ranked list of chunks
        """
        all_chunks = anchor_chunks + expanded_chunks

        # Deduplicate by chunk_id
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk["chunk_id"] not in seen_ids:
                seen_ids.add(chunk["chunk_id"])
                unique_chunks.append(chunk)

        # Compute combined scores
        for chunk in unique_chunks:
            # Weighted combination: 60% vector, 40% graph
            chunk["combined_score"] = (
                chunk["vector_score"] * 0.6 +
                chunk["graph_score"] * 0.4
            )

        # Sort by combined score (descending)
        ranked_chunks = sorted(
            unique_chunks,
            key=lambda x: (x["combined_score"], x["graph_score"], x["vector_score"]),
            reverse=True
        )

        return ranked_chunks

    def retrieve_with_explanation(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> Dict[str, Any]:
        """
        Retrieve chunks with detailed explanation of retrieval process

        Useful for debugging and understanding retrieval decisions

        Returns:
            Dict with:
            - chunks: Retrieved chunks
            - explanation: Step-by-step breakdown
            - stats: Retrieval statistics
        """
        cfg = config or self.config

        # Retrieve
        chunks = self.retrieve(query, cfg)

        # Build explanation
        vector_chunks = [c for c in chunks if c["hop_distance"] == 0]
        hop1_chunks = [c for c in chunks if c["hop_distance"] == 1]
        hop2_chunks = [c for c in chunks if c["hop_distance"] == 2]

        explanation = {
            "query": query,
            "steps": [
                {
                    "step": 1,
                    "name": "Vector Search (Anchor)",
                    "description": f"Found {len(vector_chunks)} semantically similar chunks",
                    "chunks": [c["chunk_id"] for c in vector_chunks]
                },
                {
                    "step": 2,
                    "name": "1-Hop Expansion (Same Entities)",
                    "description": f"Found {len(hop1_chunks)} chunks mentioning same entities",
                    "chunks": [
                        {
                            "id": c["chunk_id"],
                            "via": c["via_entity"],
                            "score": c["graph_score"]
                        }
                        for c in hop1_chunks
                    ]
                },
                {
                    "step": 3,
                    "name": "2-Hop Expansion (Related Entities)",
                    "description": f"Found {len(hop2_chunks)} chunks mentioning related entities",
                    "chunks": [
                        {
                            "id": c["chunk_id"],
                            "via": c["via_entity"],
                            "score": c["graph_score"]
                        }
                        for c in hop2_chunks
                    ]
                }
            ]
        }

        stats = {
            "total_chunks": len(chunks),
            "vector_chunks": len(vector_chunks),
            "hop1_chunks": len(hop1_chunks),
            "hop2_chunks": len(hop2_chunks),
            "avg_vector_score": sum(c["vector_score"] for c in chunks) / len(chunks) if chunks else 0,
            "avg_graph_score": sum(c["graph_score"] for c in chunks) / len(chunks) if chunks else 0,
            "avg_combined_score": sum(c["combined_score"] for c in chunks) / len(chunks) if chunks else 0
        }

        return {
            "chunks": chunks,
            "explanation": explanation,
            "stats": stats
        }
