"""
Enhanced Neo4j Client with Vector Search and Bilingual Support
Implements research-based GraphRAG improvements:
- Vector embeddings for semantic search
- Bilingual entity properties (Arabic + English)
- Multiple search strategies (keyword, semantic, hybrid)
- Knowledge enrichment using LLM
"""

import re
import requests
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase, Driver
from loguru import logger
import numpy as np

from ...config import settings


class EnhancedNeo4jClient:
    """Enhanced Neo4j client with vector search and bilingual support"""

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        embedder=None,  # Jina embedder
        llm_client=None  # ClaudeClient for enrichment
    ):
        """
        Initialize enhanced Neo4j client

        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
            embedder: Jina embedder for vector search
            llm_client: LLM client for knowledge enrichment
        """
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password

        self.driver: Optional[Driver] = None
        self._connected = False

        # Vector search support
        self.embedder = embedder
        self.llm_client = llm_client

        # Cache for embeddings
        self._embedding_cache = {}

    def connect(self):
        """Establish connection to Neo4j"""
        if self._connected:
            return

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Test connection
            self.driver.verify_connectivity()
            self._connected = True
            logger.info(f"Connected to Enhanced Neo4j at {self.uri}")

            # Create vector index if it doesn't exist
            self._create_vector_index()

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def _create_vector_index(self):
        """Create vector similarity index for semantic search"""
        try:
            with self.driver.session() as session:
                # Check if index already exists
                result = session.run("SHOW INDEXES")
                indexes = [record["name"] for record in result]

                if "entity_embedding_index" not in indexes:
                    # Create index on embedding property (for approximate nearest neighbor search)
                    # Note: Full vector index requires Neo4j Enterprise or specific plugins
                    # For now, we'll use property-based indexing and do cosine similarity in application
                    session.run("""
                        CREATE INDEX entity_name_index IF NOT EXISTS
                        FOR (n:Entity) ON (n.name)
                    """)

                    session.run("""
                        CREATE INDEX entity_name_en_index IF NOT EXISTS
                        FOR (n:Entity) ON (n.name_en)
                    """)

                    session.run("""
                        CREATE INDEX entity_name_ar_index IF NOT EXISTS
                        FOR (n:Entity) ON (n.name_ar)
                    """)

                    logger.info("Created entity indexes for enhanced search")

        except Exception as e:
            logger.warning(f"Could not create vector indexes: {e}")

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self._connected = False
            logger.info("Closed Enhanced Neo4j connection")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def enrich_entity_with_llm(self, entity_name: str, entity_type: str, language: str = "ar") -> Dict[str, str]:
        """
        Use LLM to enrich entity with translations and descriptions

        Args:
            entity_name: Original entity name
            entity_type: Entity type
            language: Source language

        Returns:
            Dict with name_en, name_ar, description
        """
        if not self.llm_client:
            return {"name_en": entity_name, "name_ar": entity_name, "description": ""}

        try:
            if language == "ar":
                prompt = f"""For the Arabic entity "{entity_name}" (type: {entity_type}):
1. Translate to English
2. Provide a brief 1-sentence description in English

Respond ONLY in this format:
English: [translation]
Description: [description]

Do not include any other text."""
            else:
                prompt = f"""For the English entity "{entity_name}" (type: {entity_type}):
1. Translate to Arabic
2. Provide a brief 1-sentence description in English

Respond ONLY in this format:
Arabic: [translation]
Description: [description]

Do not include any other text."""

            result = self.llm_client.generate_response(
                query=prompt,
                context="",
                system_prompt="You are a bilingual translator. Respond ONLY in the exact format requested.",
                max_tokens=150
            )

            response_text = result.get("response", "").strip()

            # Parse response
            name_en = entity_name
            name_ar = entity_name
            description = ""

            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith("English:"):
                    name_en = line.replace("English:", "").strip()
                elif line.startswith("Arabic:"):
                    name_ar = line.replace("Arabic:", "").strip()
                elif line.startswith("Description:"):
                    description = line.replace("Description:", "").strip()

            if language == "ar":
                name_ar = entity_name  # Keep original
            else:
                name_en = entity_name  # Keep original

            return {
                "name_en": name_en,
                "name_ar": name_ar,
                "description": description
            }

        except Exception as e:
            logger.warning(f"LLM enrichment failed for {entity_name}: {e}")
            return {"name_en": entity_name, "name_ar": entity_name, "description": ""}

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None
        """
        if not self.embedder:
            return None

        # Check cache
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        try:
            embedding = self.embedder.embed_text(text)
            self._embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding for '{text}': {e}")
            return None

    def create_enriched_entity_node(
        self,
        entity: Dict[str, Any],
        document_id: str,
        language: str = "ar",
        enrich: bool = True
    ) -> Dict[str, Any]:
        """
        Create or update an enriched entity node with bilingual support

        Args:
            entity: Entity dict with text, type, etc.
            document_id: Source document ID
            language: Source language
            enrich: Whether to enrich with LLM

        Returns:
            Created/updated node
        """
        if not self._connected:
            self.connect()

        entity_name = entity["text"]
        entity_type = entity["type"].replace(" ", "_").replace("-", "_")

        # Get enrichment data
        if enrich and self.llm_client:
            enrichment = self.enrich_entity_with_llm(entity_name, entity_type, language)
        else:
            enrichment = {
                "name_en": entity_name if language == "en" else "",
                "name_ar": entity_name if language == "ar" else "",
                "description": ""
            }

        # Get embedding
        embedding = self.get_embedding(entity_name)
        embedding_list = embedding if embedding else []

        query = f"""
        MERGE (n:Entity {{name: $name}})
        ON CREATE SET
            n.type = $type,
            n.name_en = $name_en,
            n.name_ar = $name_ar,
            n.description = $description,
            n.embedding = $embedding,
            n.created_at = timestamp(),
            n.confidence = $confidence,
            n.source_documents = [$document_id],
            n.language = $language,
            n:{entity_type}
        ON MATCH SET
            n.confidence = CASE
                WHEN $confidence > n.confidence THEN $confidence
                ELSE n.confidence
            END,
            n.name_en = CASE
                WHEN $name_en <> '' AND (n.name_en IS NULL OR n.name_en = '') THEN $name_en
                ELSE n.name_en
            END,
            n.name_ar = CASE
                WHEN $name_ar <> '' AND (n.name_ar IS NULL OR n.name_ar = '') THEN $name_ar
                ELSE n.name_ar
            END,
            n.description = CASE
                WHEN $description <> '' AND (n.description IS NULL OR n.description = '') THEN $description
                ELSE n.description
            END,
            n.embedding = CASE
                WHEN n.embedding IS NULL AND size($embedding) > 0 THEN $embedding
                ELSE n.embedding
            END,
            n.source_documents = CASE
                WHEN NOT $document_id IN n.source_documents
                THEN n.source_documents + [$document_id]
                ELSE n.source_documents
            END,
            n.updated_at = timestamp(),
            n:{entity_type}
        RETURN n
        """

        parameters = {
            "name": entity_name,
            "type": entity["type"],
            "name_en": enrichment["name_en"],
            "name_ar": enrichment["name_ar"],
            "description": enrichment["description"],
            "embedding": embedding_list,
            "confidence": entity.get("confidence", 0.5),
            "document_id": document_id,
            "language": language,
        }

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                record = result.single()

                if record:
                    return dict(record["n"])

        except Exception as e:
            logger.error(f"Error creating enriched entity node: {e}")
            raise

        return {}

    def semantic_search_entities(
        self,
        query: str,
        limit: int = 20,
        threshold: float = 0.5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search entities using semantic similarity (vector search)

        Args:
            query: Search query
            limit: Maximum results
            threshold: Minimum similarity threshold

        Returns:
            List of (entity, similarity_score) tuples
        """
        if not self._connected:
            self.connect()

        if not self.embedder:
            logger.warning("No embedder available for semantic search")
            return []

        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        try:
            with self.driver.session() as session:
                # Get all entities with embeddings
                result = session.run("""
                    MATCH (n:Entity)
                    WHERE n.embedding IS NOT NULL AND size(n.embedding) > 0
                    RETURN n, n.embedding as embedding
                """)

                # Calculate cosine similarity in application
                # (Neo4j Community Edition doesn't have built-in vector similarity)
                entities_with_scores = []

                query_vec = np.array(query_embedding)
                query_norm = np.linalg.norm(query_vec)

                for record in result:
                    entity = dict(record["n"])
                    entity_embedding = record["embedding"]

                    if not entity_embedding:
                        continue

                    entity_vec = np.array(entity_embedding)
                    entity_norm = np.linalg.norm(entity_vec)

                    if entity_norm == 0 or query_norm == 0:
                        continue

                    # Cosine similarity
                    similarity = np.dot(query_vec, entity_vec) / (query_norm * entity_norm)

                    if similarity >= threshold:
                        entities_with_scores.append((entity, float(similarity)))

                # Sort by similarity (descending) and limit
                entities_with_scores.sort(key=lambda x: x[1], reverse=True)
                return entities_with_scores[:limit]

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def hybrid_search_entities(
        self,
        query: str,
        limit: int = 20,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining keyword and semantic search

        Args:
            query: Search query
            limit: Maximum results
            keyword_weight: Weight for keyword match score
            semantic_weight: Weight for semantic similarity score

        Returns:
            List of entities sorted by combined score
        """
        if not self._connected:
            self.connect()

        # Get keyword search results
        keyword_results = self.bilingual_keyword_search(query, limit=limit * 2)

        # Get semantic search results
        semantic_results = self.semantic_search_entities(query, limit=limit * 2, threshold=0.3)

        # Combine scores
        entity_scores = {}

        # Add keyword scores (normalized)
        max_keyword_score = len(keyword_results)
        for i, entity in enumerate(keyword_results):
            entity_name = entity["name"]
            # Reverse ranking score (first result gets highest score)
            keyword_score = (max_keyword_score - i) / max_keyword_score
            entity_scores[entity_name] = {
                "entity": entity,
                "keyword_score": keyword_score,
                "semantic_score": 0.0
            }

        # Add semantic scores
        for entity, similarity in semantic_results:
            entity_name = entity["name"]
            if entity_name in entity_scores:
                entity_scores[entity_name]["semantic_score"] = similarity
            else:
                entity_scores[entity_name] = {
                    "entity": entity,
                    "keyword_score": 0.0,
                    "semantic_score": similarity
                }

        # Calculate combined scores
        ranked_entities = []
        for entity_name, scores in entity_scores.items():
            combined_score = (
                keyword_weight * scores["keyword_score"] +
                semantic_weight * scores["semantic_score"]
            )
            ranked_entities.append((scores["entity"], combined_score))

        # Sort by combined score
        ranked_entities.sort(key=lambda x: x[1], reverse=True)

        # Return top N entities
        return [entity for entity, score in ranked_entities[:limit]]

    def bilingual_keyword_search(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search entities by keyword in both English and Arabic names

        Args:
            query: Search query
            entity_type: Optional entity type filter
            limit: Maximum results

        Returns:
            List of matching entities
        """
        if not self._connected:
            self.connect()

        # Build query to search in name, name_en, and name_ar
        cypher_query = """
        MATCH (n:Entity)
        WHERE
            toLower(n.name) CONTAINS toLower($query)
            OR toLower(n.name_en) CONTAINS toLower($query)
            OR toLower(n.name_ar) CONTAINS toLower($query)
            OR toLower(n.description) CONTAINS toLower($query)
        """

        if entity_type:
            cypher_query += " AND n.type = $entity_type"

        cypher_query += """
        RETURN n
        ORDER BY n.confidence DESC
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher_query,
                    {
                        "query": query,
                        "entity_type": entity_type,
                        "limit": limit,
                    }
                )

                return [dict(record["n"]) for record in result]

        except Exception as e:
            logger.error(f"Error in bilingual keyword search: {e}")
            return []

    def query_subgraph(
        self,
        entity_name: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Query a subgraph around an entity

        Args:
            entity_name: Entity to center on
            depth: Depth of traversal

        Returns:
            Subgraph dict with nodes and edges
        """
        if not self._connected:
            self.connect()

        query = f"""
        MATCH path = (start:Entity {{name: $entity_name}})-[*1..{depth}]-(connected)
        WITH nodes(path) as nodes, relationships(path) as rels
        UNWIND nodes as node
        WITH collect(DISTINCT node) as unique_nodes, rels
        UNWIND rels as rel
        WITH unique_nodes, collect(DISTINCT rel) as unique_rels
        RETURN unique_nodes, unique_rels
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, {"entity_name": entity_name})
                record = result.single()

                if not record:
                    return {"nodes": [], "edges": []}

                nodes = [dict(node) for node in record["unique_nodes"]]
                edges = []

                for rel in record["unique_rels"]:
                    edge_dict = dict(rel)
                    # Add start and end node info
                    edge_dict["start"] = {"name": rel.start_node["name"]}
                    edge_dict["end"] = {"name": rel.end_node["name"]}
                    edges.append(edge_dict)

                return {
                    "nodes": nodes,
                    "edges": edges,
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                }

        except Exception as e:
            logger.error(f"Error querying subgraph: {e}")
            return {"nodes": [], "edges": []}
