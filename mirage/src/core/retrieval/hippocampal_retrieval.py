"""
Hippocampal Retrieval - Personalized PageRank for Graph RAG

Key Innovation from HippoRAG (NeurIPS 2024):
- Mimics human hippocampal memory indexing
- Uses Personalized PageRank (PPR) for associative retrieval
- Key insight: Human memory retrieves by association, not just similarity

How it works:
1. Query entities become "seed nodes" (like pattern separation in hippocampus)
2. PPR spreads activation through graph (like pattern completion)
3. Highly activated nodes are retrieved (like memory recall)

Benefits:
- 7% improvement on multi-hop QA (HippoRAG paper)
- Better handling of indirect relationships
- More human-like retrieval patterns

Paper: "HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs" (NeurIPS 2024)
"""

import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger
import time


@dataclass
class PPRResult:
    """Result from Personalized PageRank computation."""
    node_scores: Dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    converged: bool = False
    total_nodes: int = 0
    seed_nodes: List[str] = field(default_factory=list)


@dataclass
class HippocampalRetrievalResult:
    """Result from hippocampal retrieval."""
    query: str
    chunks: List[Dict] = field(default_factory=list)
    seed_entities: List[str] = field(default_factory=list)
    activated_entities: List[Tuple[str, float]] = field(default_factory=list)
    ppr_iterations: int = 0
    retrieval_time_ms: float = 0.0


class PersonalizedPageRank:
    """
    Personalized PageRank implementation for graph traversal.

    PPR Formula:
    r = (1-d) * p + d * M * r

    Where:
    - r = PPR scores vector
    - d = damping factor (0.85 typical)
    - p = personalization vector (seed nodes)
    - M = transition matrix (normalized adjacency)

    The result gives each node a score based on:
    - Proximity to seed nodes
    - Connectivity patterns
    - Graph structure
    """

    def __init__(
        self,
        damping_factor: float = 0.85,
        max_iterations: int = 50,
        tolerance: float = 1e-6
    ):
        """
        Initialize PPR.

        Args:
            damping_factor: Probability of random walk (0.85 = 85% walk, 15% teleport)
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
        """
        self.damping = damping_factor
        self.max_iter = max_iterations
        self.tolerance = tolerance

    def compute(
        self,
        adjacency: Dict[str, List[str]],
        seed_nodes: List[str],
        node_weights: Optional[Dict[str, float]] = None
    ) -> PPRResult:
        """
        Compute Personalized PageRank scores.

        Args:
            adjacency: Graph as adjacency list {node: [neighbors]}
            seed_nodes: Nodes to teleport to (personalization)
            node_weights: Optional weights for seed nodes

        Returns:
            PPRResult with node scores
        """
        if not adjacency or not seed_nodes:
            return PPRResult()

        all_nodes = set(adjacency.keys())
        for neighbors in adjacency.values():
            all_nodes.update(neighbors)

        n = len(all_nodes)
        node_list = list(all_nodes)
        node_to_idx = {node: i for i, node in enumerate(node_list)}

        # Build personalization vector
        personalization = np.zeros(n)
        if node_weights:
            total_weight = sum(node_weights.get(s, 1.0) for s in seed_nodes)
            for seed in seed_nodes:
                if seed in node_to_idx:
                    personalization[node_to_idx[seed]] = node_weights.get(seed, 1.0) / total_weight
        else:
            seed_weight = 1.0 / len(seed_nodes)
            for seed in seed_nodes:
                if seed in node_to_idx:
                    personalization[node_to_idx[seed]] = seed_weight

        # Initialize scores uniformly
        scores = np.ones(n) / n

        # Iterate
        converged = False
        iteration = 0

        for iteration in range(self.max_iter):
            new_scores = np.zeros(n)

            # Teleport component
            new_scores += (1 - self.damping) * personalization

            # Random walk component
            for node, neighbors in adjacency.items():
                if node not in node_to_idx:
                    continue
                node_idx = node_to_idx[node]

                if len(neighbors) == 0:
                    continue

                contribution = scores[node_idx] * self.damping / len(neighbors)
                for neighbor in neighbors:
                    if neighbor in node_to_idx:
                        new_scores[node_to_idx[neighbor]] += contribution

            # Check convergence
            diff = np.sum(np.abs(new_scores - scores))
            scores = new_scores

            if diff < self.tolerance:
                converged = True
                break

        # Build result
        node_scores = {
            node_list[i]: float(scores[i])
            for i in range(n)
            if scores[i] > 0.001  # Filter low scores
        }

        return PPRResult(
            node_scores=node_scores,
            iterations=iteration + 1,
            converged=converged,
            total_nodes=n,
            seed_nodes=seed_nodes
        )


class HippocampalRetriever:
    """
    HippoRAG-inspired retrieval using Personalized PageRank.

    Mimics human hippocampal memory:
    1. Parahippocampal region: Pattern separation (identify seed entities)
    2. Hippocampus: Pattern completion (PPR spreads activation)
    3. Neocortex: Memory retrieval (get chunks for activated entities)

    Usage:
        retriever = HippocampalRetriever(neo4j_client, embedder)
        result = retriever.retrieve(query, top_k=10)
        # Use result.chunks for generation
    """

    def __init__(
        self,
        neo4j_client,
        embedder,
        entity_disambiguator=None,
        damping_factor: float = 0.85,
        max_iterations: int = 50,
        activation_threshold: float = 0.01,
        max_activated_entities: int = 20
    ):
        """
        Initialize hippocampal retriever.

        Args:
            neo4j_client: Neo4j client for graph operations
            embedder: Embedding manager for entity linking
            entity_disambiguator: Optional entity disambiguator
            damping_factor: PPR damping factor
            max_iterations: PPR max iterations
            activation_threshold: Min activation score to include
            max_activated_entities: Max entities to retrieve chunks from
        """
        self.neo4j_client = neo4j_client
        self.embedder = embedder
        self.entity_disambiguator = entity_disambiguator
        self.activation_threshold = activation_threshold
        self.max_activated_entities = max_activated_entities

        self.ppr = PersonalizedPageRank(
            damping_factor=damping_factor,
            max_iterations=max_iterations
        )

        # Cache for graph structure
        self._adjacency_cache: Optional[Dict[str, List[str]]] = None
        self._cache_timestamp: float = 0

        logger.info(
            f"HippocampalRetriever initialized: "
            f"damping={damping_factor}, threshold={activation_threshold}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        force_refresh: bool = False
    ) -> HippocampalRetrievalResult:
        """
        Retrieve using hippocampal-inspired PPR algorithm.

        Algorithm:
        1. Identify seed entities from query (pattern separation)
        2. Load graph structure
        3. Run PPR with seeds as personalization (pattern completion)
        4. Get chunks for top activated entities (retrieval)

        Args:
            query: User query string
            top_k: Number of chunks to return
            force_refresh: Force reload of graph cache

        Returns:
            HippocampalRetrievalResult with chunks and metadata
        """
        start_time = time.time()

        # Step 1: Identify seed entities
        seed_entities = self._identify_seeds(query)

        if not seed_entities:
            logger.info("No seed entities found, falling back to vector search")
            return self._vector_fallback(query, top_k, start_time)

        logger.debug(f"Seed entities: {seed_entities}")

        # Step 2: Load graph adjacency
        adjacency = self._load_adjacency(force_refresh)

        if not adjacency:
            logger.warning("Empty graph, falling back to vector search")
            return self._vector_fallback(query, top_k, start_time)

        # Filter seeds to those in graph
        valid_seeds = [s for s in seed_entities if s in adjacency]
        if not valid_seeds:
            logger.info("No seed entities in graph, using fuzzy matching")
            valid_seeds = self._fuzzy_match_seeds(seed_entities, adjacency)

        if not valid_seeds:
            return self._vector_fallback(query, top_k, start_time)

        # Step 3: Run PPR
        ppr_result = self.ppr.compute(adjacency, valid_seeds)

        logger.debug(
            f"PPR completed: {ppr_result.iterations} iterations, "
            f"converged={ppr_result.converged}, "
            f"{len(ppr_result.node_scores)} nodes scored"
        )

        # Step 4: Get top activated entities
        activated = [
            (node, score)
            for node, score in ppr_result.node_scores.items()
            if score >= self.activation_threshold
        ]
        activated.sort(key=lambda x: x[1], reverse=True)
        top_entities = activated[:self.max_activated_entities]

        # Step 5: Get chunks for activated entities
        chunks = self._get_entity_chunks(
            [e[0] for e in top_entities],
            top_k
        )

        retrieval_time = (time.time() - start_time) * 1000

        logger.info(
            f"Hippocampal retrieval: {len(valid_seeds)} seeds, "
            f"{len(top_entities)} activated, "
            f"{len(chunks)} chunks, {retrieval_time:.1f}ms"
        )

        return HippocampalRetrievalResult(
            query=query,
            chunks=chunks,
            seed_entities=valid_seeds,
            activated_entities=top_entities,
            ppr_iterations=ppr_result.iterations,
            retrieval_time_ms=retrieval_time
        )

    def _identify_seeds(self, query: str) -> List[str]:
        """
        Identify seed entities from query (parahippocampal pattern separation).

        Uses entity extraction and disambiguation.
        """
        seeds = []

        # Method 1: Entity disambiguation (if available)
        if self.entity_disambiguator:
            try:
                # Extract terms from query
                terms = self._extract_terms(query)

                for term in terms[:10]:
                    result = self.entity_disambiguator.disambiguate(
                        query_entity=term,
                        context=query
                    )
                    if result.matched_entity and result.similarity_score > 0.5:
                        seeds.append(result.matched_entity)

            except Exception as e:
                logger.debug(f"Disambiguation failed: {e}")

        # Method 2: Direct entity search
        if not seeds:
            terms = self._extract_terms(query)
            for term in terms[:5]:
                try:
                    cypher_query = """
                    MATCH (e:Entity)
                    WHERE e.name CONTAINS $term
                    RETURN e.name as name
                    LIMIT 3
                    """
                    results = self.neo4j_client.execute_query(cypher_query, {'term': term})
                    for r in results:
                        name = r.get('name')
                        if name and name not in seeds:
                            seeds.append(name)
                except Exception:
                    pass

        return list(set(seeds))

    def _extract_terms(self, query: str) -> List[str]:
        """Extract potential entity terms from query."""
        # Simple approach: take words > 2 chars, not stopwords
        stopwords = {
            "من", "في", "على", "إلى", "هل", "ما", "هو", "هي", "أن", "كان",
            "the", "is", "are", "what", "who", "where", "when", "how"
        }

        words = query.split()
        terms = [w for w in words if len(w) > 2 and w.lower() not in stopwords]

        return terms

    def _load_adjacency(self, force_refresh: bool = False) -> Dict[str, List[str]]:
        """Load graph adjacency list from Neo4j."""
        # Check cache
        if not force_refresh and self._adjacency_cache:
            cache_age = time.time() - self._cache_timestamp
            if cache_age < 300:  # 5 minute cache
                return self._adjacency_cache

        logger.debug("Loading graph adjacency from Neo4j...")

        try:
            query = """
            MATCH (e1:Entity)-[r]->(e2:Entity)
            RETURN e1.name as source, e2.name as target
            """

            results = self.neo4j_client.execute_query(query)

            adjacency = defaultdict(list)
            for r in results:
                source = r.get('source')
                target = r.get('target')
                if source and target:
                    adjacency[source].append(target)
                    # Also add reverse for undirected traversal
                    adjacency[target].append(source)

            self._adjacency_cache = dict(adjacency)
            self._cache_timestamp = time.time()

            logger.debug(f"Loaded adjacency: {len(self._adjacency_cache)} nodes")
            return self._adjacency_cache

        except Exception as e:
            logger.error(f"Failed to load adjacency: {e}")
            return {}

    def _fuzzy_match_seeds(
        self,
        seeds: List[str],
        adjacency: Dict[str, List[str]]
    ) -> List[str]:
        """Fuzzy match seeds to graph nodes."""
        graph_nodes = set(adjacency.keys())
        matched = []

        for seed in seeds:
            seed_lower = seed.lower()

            # Exact match first
            for node in graph_nodes:
                if seed_lower in node.lower() or node.lower() in seed_lower:
                    matched.append(node)
                    break

        return matched

    def _get_entity_chunks(
        self,
        entities: List[str],
        top_k: int
    ) -> List[Dict]:
        """Get chunks for activated entities."""
        if not entities:
            return []

        try:
            query = """
            MATCH (chunk:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE e.name IN $entities
            RETURN DISTINCT
                chunk.id as chunk_id,
                chunk.text as text,
                chunk.document_id as document_id,
                e.name as via_entity
            LIMIT $limit
            """

            results = self.neo4j_client.execute_query(query, {
                'entities': entities,
                'limit': top_k * 2  # Get more, then dedupe
            })

            # Deduplicate by chunk_id
            seen = set()
            chunks = []
            for r in results:
                chunk_id = r.get('chunk_id')
                if chunk_id and chunk_id not in seen:
                    seen.add(chunk_id)
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": r.get("text", ""),
                        "document_id": r.get("document_id", ""),
                        "via_entity": r.get("via_entity", ""),
                        "retrieval_mode": "hippocampal"
                    })

            return chunks[:top_k]

        except Exception as e:
            logger.error(f"Failed to get entity chunks: {e}")
            return []

    def _vector_fallback(
        self,
        query: str,
        top_k: int,
        start_time: float
    ) -> HippocampalRetrievalResult:
        """Fallback to vector search when PPR can't be used."""
        chunks = []

        try:
            # Use vector search via chunk embeddings
            query_embedding = self.embedder.embed(query)

            # Search in Neo4j using vector index (if available)
            # or fallback to Qdrant
            cypher_query = """
            MATCH (chunk:Chunk)
            WHERE chunk.text IS NOT NULL
            RETURN chunk.id as chunk_id, chunk.text as text, chunk.document_id as document_id
            LIMIT $limit
            """

            results = self.neo4j_client.execute_query(cypher_query, {'limit': top_k * 3})

            # Score by embedding similarity
            scored = []
            for r in results:
                text = r.get('text', '')
                if text:
                    try:
                        text_emb = self.embedder.embed(text[:512])  # Truncate
                        similarity = np.dot(query_embedding, text_emb) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(text_emb)
                        )
                        scored.append((r, similarity))
                    except Exception:
                        pass

            # Sort by similarity
            scored.sort(key=lambda x: x[1], reverse=True)

            for r, score in scored[:top_k]:
                chunks.append({
                    "chunk_id": r.get("chunk_id"),
                    "text": r.get("text"),
                    "document_id": r.get("document_id"),
                    "score": float(score),
                    "retrieval_mode": "vector_fallback"
                })

        except Exception as e:
            logger.error(f"Vector fallback failed: {e}")

        retrieval_time = (time.time() - start_time) * 1000

        return HippocampalRetrievalResult(
            query=query,
            chunks=chunks,
            seed_entities=[],
            activated_entities=[],
            ppr_iterations=0,
            retrieval_time_ms=retrieval_time
        )

    def invalidate_cache(self):
        """Invalidate graph cache."""
        self._adjacency_cache = None
        self._cache_timestamp = 0
        logger.info("Graph cache invalidated")


# Singleton
_hippocampal_retriever: Optional[HippocampalRetriever] = None


def get_hippocampal_retriever(
    neo4j_client=None,
    embedder=None,
    **kwargs
) -> HippocampalRetriever:
    """Get or create hippocampal retriever singleton."""
    global _hippocampal_retriever

    if _hippocampal_retriever is None:
        if neo4j_client is None:
            from ..graph_builder import Neo4jClient
            neo4j_client = Neo4jClient()

        if embedder is None:
            from ..models.embedding_manager import get_embedding_manager
            embedder = get_embedding_manager()

        _hippocampal_retriever = HippocampalRetriever(
            neo4j_client=neo4j_client,
            embedder=embedder,
            **kwargs
        )

    return _hippocampal_retriever
