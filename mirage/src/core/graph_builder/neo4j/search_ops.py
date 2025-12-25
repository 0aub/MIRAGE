"""
Neo4j Client - Search and Query Operations

Entity search, graph traversal, and advanced query methods.
"""

from typing import List, Dict, Any, Optional
from loguru import logger


class SearchOperationsMixin:
    """Mixin for search and query operations"""

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
                edges = [dict(rel) for rel in record["unique_rels"]]

                return {
                    "nodes": nodes,
                    "edges": edges,
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                }

        except Exception as e:
            logger.error(f"Error querying subgraph: {e}")
            return {"nodes": [], "edges": []}

    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for entities by name

        Args:
            query: Search query
            entity_type: Optional entity type filter
            limit: Maximum results

        Returns:
            List of matching entities
        """
        if not self._connected:
            self.connect()

        cypher_query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($query)
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
            logger.error(f"Error searching entities: {e}")
            return []

    def search_entities_by_name(
        self,
        name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for entities by name (partial match)

        Args:
            name: Entity name to search for
            limit: Maximum results

        Returns:
            List of matching entities with name, type, confidence
        """
        return self.search_entities(query=name, limit=limit)

    def get_entity_chunks(
        self,
        entity_name: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get chunks that mention a specific entity

        Args:
            entity_name: Name of the entity
            limit: Maximum chunks to return

        Returns:
            List of chunks with chunk_id, document_id, text
        """
        if not self._connected:
            self.connect()

        try:
            with self.driver.session() as session:
                # Find entity and get chunks that MENTION it
                # Schema: Chunk -[:MENTIONS]-> Entity (direction matters!)
                # Chunk has: id, document_id, text
                cypher_query = """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($entity_name)
                RETURN c.id as chunk_id,
                       c.document_id as document_id,
                       c.text as text
                LIMIT $limit
                """
                result = session.run(
                    cypher_query,
                    {"entity_name": entity_name, "limit": limit}
                )

                chunks = []
                for record in result:
                    chunks.append({
                        "chunk_id": record["chunk_id"],
                        "document_id": record["document_id"],
                        "text": record["text"] or ""
                    })

                return chunks

        except Exception as e:
            logger.error(f"Error getting entity chunks: {e}")
            return []

    def get_entity_relationships(
        self,
        entity_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get relationships involving a specific entity

        Args:
            entity_name: Name of the entity
            limit: Maximum relationships to return

        Returns:
            List of relationships with source, target, type
        """
        if not self._connected:
            self.connect()

        try:
            with self.driver.session() as session:
                # Get outgoing relationships
                cypher_query = """
                MATCH (e:Entity)-[r]->(t:Entity)
                WHERE toLower(e.name) CONTAINS toLower($entity_name)
                RETURN e.name as source, t.name as target, type(r) as type
                UNION
                MATCH (s:Entity)-[r]->(e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($entity_name)
                RETURN s.name as source, e.name as target, type(r) as type
                LIMIT $limit
                """
                result = session.run(
                    cypher_query,
                    {"entity_name": entity_name, "limit": limit}
                )

                relationships = []
                for record in result:
                    relationships.append({
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["type"]
                    })

                return relationships

        except Exception as e:
            logger.error(f"Error getting entity relationships: {e}")
            return []

    def get_entities_by_type(
        self,
        entity_types: List[str],
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get entities filtered by type(s)

        Args:
            entity_types: List of entity types (e.g., ['Organization', 'Person'])
            limit: Maximum results

        Returns:
            List of entities with name, type, confidence
        """
        if not self._connected:
            self.connect()

        try:
            with self.driver.session() as session:
                cypher_query = """
                MATCH (e:Entity)
                WHERE e.type IN $entity_types
                RETURN e.name as name, e.type as type, e.confidence as confidence
                ORDER BY e.confidence DESC
                LIMIT $limit
                """
                result = session.run(
                    cypher_query,
                    {"entity_types": entity_types, "limit": limit}
                )

                entities = []
                for record in result:
                    entities.append({
                        "name": record["name"],
                        "type": record["type"],
                        "confidence": record["confidence"] or 0.5
                    })

                return entities

        except Exception as e:
            logger.error(f"Error getting entities by type: {e}")
            return []

    def get_entities_from_chunks(
        self,
        chunk_ids: List[str],
        entity_types: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get entities mentioned in specific chunks

        Args:
            chunk_ids: List of chunk IDs
            entity_types: Optional filter for entity types
            limit: Maximum results

        Returns:
            List of entities with name, type, chunks
        """
        if not self._connected:
            self.connect()

        if not chunk_ids:
            return []

        try:
            with self.driver.session() as session:
                if entity_types:
                    cypher_query = """
                    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                    WHERE c.id IN $chunk_ids AND e.type IN $entity_types
                    WITH e, collect(DISTINCT c.id) as chunks
                    RETURN e.name as name, e.type as type, e.confidence as confidence, chunks
                    ORDER BY size(chunks) DESC, e.confidence DESC
                    LIMIT $limit
                    """
                    params = {"chunk_ids": chunk_ids, "entity_types": entity_types, "limit": limit}
                else:
                    cypher_query = """
                    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                    WHERE c.id IN $chunk_ids
                    WITH e, collect(DISTINCT c.id) as chunks
                    RETURN e.name as name, e.type as type, e.confidence as confidence, chunks
                    ORDER BY size(chunks) DESC, e.confidence DESC
                    LIMIT $limit
                    """
                    params = {"chunk_ids": chunk_ids, "limit": limit}

                result = session.run(cypher_query, params)

                entities = []
                for record in result:
                    entities.append({
                        "name": record["name"],
                        "type": record["type"],
                        "confidence": record["confidence"] or 0.5,
                        "chunks": list(record["chunks"])
                    })

                return entities

        except Exception as e:
            logger.error(f"Error getting entities from chunks: {e}")
            return []

    def find_related_entities(
        self,
        entity_name: str,
        depth: int = 1,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to a given entity through relationships

        Args:
            entity_name: Starting entity name
            depth: How many hops to traverse
            limit: Maximum results

        Returns:
            List of related entities with relationship info
        """
        if not self._connected:
            self.connect()

        try:
            with self.driver.session() as session:
                cypher_query = f"""
                MATCH (e:Entity)-[r*1..{depth}]-(related:Entity)
                WHERE toLower(e.name) CONTAINS toLower($entity_name)
                WITH related, min(length(r)) as distance
                RETURN DISTINCT related.name as name, related.type as type,
                       related.confidence as confidence, distance
                ORDER BY distance, related.confidence DESC
                LIMIT $limit
                """
                result = session.run(
                    cypher_query,
                    {"entity_name": entity_name, "limit": limit}
                )

                entities = []
                for record in result:
                    entities.append({
                        "name": record["name"],
                        "type": record["type"],
                        "confidence": record["confidence"] or 0.5,
                        "distance": record["distance"]
                    })

                return entities

        except Exception as e:
            logger.error(f"Error finding related entities: {e}")
            return []

    def search_entities_semantic(
        self,
        query_terms: List[str],
        entity_types: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search entities using multiple query terms with OR logic

        Args:
            query_terms: List of terms to search for
            entity_types: Optional filter for entity types
            limit: Maximum results

        Returns:
            List of matching entities with match count
        """
        if not self._connected:
            self.connect()

        if not query_terms:
            return []

        try:
            with self.driver.session() as session:
                # Build regex pattern for any term match
                patterns = "|".join([f"(?i){term}" for term in query_terms if len(term) > 2])
                if not patterns:
                    return []

                if entity_types:
                    cypher_query = """
                    MATCH (e:Entity)
                    WHERE e.type IN $entity_types
                    WITH e, [term IN $query_terms WHERE toLower(e.name) CONTAINS toLower(term)] as matches
                    WHERE size(matches) > 0
                    RETURN e.name as name, e.type as type, e.confidence as confidence,
                           size(matches) as match_count
                    ORDER BY match_count DESC, e.confidence DESC
                    LIMIT $limit
                    """
                    params = {"query_terms": query_terms, "entity_types": entity_types, "limit": limit}
                else:
                    cypher_query = """
                    MATCH (e:Entity)
                    WITH e, [term IN $query_terms WHERE toLower(e.name) CONTAINS toLower(term)] as matches
                    WHERE size(matches) > 0
                    RETURN e.name as name, e.type as type, e.confidence as confidence,
                           size(matches) as match_count
                    ORDER BY match_count DESC, e.confidence DESC
                    LIMIT $limit
                    """
                    params = {"query_terms": query_terms, "limit": limit}

                result = session.run(cypher_query, params)

                entities = []
                for record in result:
                    entities.append({
                        "name": record["name"],
                        "type": record["type"],
                        "confidence": record["confidence"] or 0.5,
                        "match_count": record["match_count"]
                    })

                return entities

        except Exception as e:
            logger.error(f"Error searching entities semantically: {e}")
            return []

    def search_entities_by_embedding(
        self,
        query_embedding: "np.ndarray",
        entity_types: Optional[List[str]] = None,
        limit: int = 20,
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        GraphRAG Local Search: Search entities using embedding similarity.

        Per Microsoft GraphRAG spec, this finds semantically similar entities
        by computing cosine similarity between query embedding and entity
        description embeddings stored in Neo4j.

        Args:
            query_embedding: Query vector (numpy array)
            entity_types: Optional filter for entity types
            limit: Maximum results
            threshold: Minimum similarity threshold

        Returns:
            List of entities with similarity scores, sorted by relevance
        """
        import numpy as np

        if not self._connected:
            self.connect()

        if query_embedding is None or len(query_embedding) == 0:
            return []

        try:
            with self.driver.session() as session:
                # Fetch entities with embeddings
                if entity_types:
                    cypher_query = """
                    MATCH (e:Entity)
                    WHERE e.embedding IS NOT NULL AND size(e.embedding) > 0
                    AND e.type IN $entity_types
                    RETURN e.name as name, e.type as type, e.description as description,
                           e.embedding as embedding, e.confidence as confidence
                    """
                    params = {"entity_types": entity_types}
                else:
                    cypher_query = """
                    MATCH (e:Entity)
                    WHERE e.embedding IS NOT NULL AND size(e.embedding) > 0
                    RETURN e.name as name, e.type as type, e.description as description,
                           e.embedding as embedding, e.confidence as confidence
                    """
                    params = {}

                result = session.run(cypher_query, params)

                # Compute cosine similarity in application
                query_vec = np.array(query_embedding)
                query_norm = np.linalg.norm(query_vec)

                if query_norm == 0:
                    return []

                entities_with_scores = []
                for record in result:
                    entity_embedding = record["embedding"]
                    if not entity_embedding:
                        continue

                    entity_vec = np.array(entity_embedding)
                    entity_norm = np.linalg.norm(entity_vec)

                    if entity_norm == 0:
                        continue

                    # Cosine similarity
                    similarity = float(np.dot(query_vec, entity_vec) / (query_norm * entity_norm))

                    if similarity >= threshold:
                        entities_with_scores.append({
                            "name": record["name"],
                            "type": record["type"],
                            "description": record["description"] or "",
                            "confidence": record["confidence"] or 0.5,
                            "similarity": similarity
                        })

                # Sort by similarity descending
                entities_with_scores.sort(key=lambda x: x["similarity"], reverse=True)
                return entities_with_scores[:limit]

        except Exception as e:
            logger.error(f"Error searching entities by embedding: {e}")
            return []

    def get_entity_with_context(
        self,
        entity_name: str
    ) -> Dict[str, Any]:
        """
        GraphRAG Local Search: Get entity with full context including
        description, relationships, and community membership.

        Per Microsoft GraphRAG spec, this builds comprehensive entity context.

        Args:
            entity_name: Entity name to look up

        Returns:
            Dict with entity info, relationships, and community context
        """
        if not self._connected:
            self.connect()

        try:
            with self.driver.session() as session:
                # Get entity with relationships and community
                query = """
                MATCH (e:Entity {name: $name})
                OPTIONAL MATCH (e)-[r]->(target:Entity)
                OPTIONAL MATCH (e)<-[r2]-(source:Entity)
                OPTIONAL MATCH (e)-[:IN_COMMUNITY]->(c:Community)
                WITH e,
                     collect(DISTINCT {
                         type: type(r),
                         target: target.name,
                         target_type: target.type,
                         description: r.description
                     }) as outgoing,
                     collect(DISTINCT {
                         type: type(r2),
                         source: source.name,
                         source_type: source.type,
                         description: r2.description
                     }) as incoming,
                     collect(DISTINCT {
                         id: c.id,
                         title: c.title,
                         summary: c.summary,
                         level: c.level
                     }) as communities
                RETURN e.name as name, e.type as type, e.description as description,
                       e.confidence as confidence, outgoing, incoming, communities
                """

                result = session.run(query, {"name": entity_name})
                record = result.single()

                if not record:
                    return {}

                # Filter out null relationships
                outgoing = [r for r in record["outgoing"] if r.get("target")]
                incoming = [r for r in record["incoming"] if r.get("source")]
                communities = [c for c in record["communities"] if c.get("id")]

                return {
                    "name": record["name"],
                    "type": record["type"],
                    "description": record["description"] or "",
                    "confidence": record["confidence"] or 0.5,
                    "outgoing_relationships": outgoing,
                    "incoming_relationships": incoming,
                    "communities": communities
                }

        except Exception as e:
            logger.error(f"Error getting entity context: {e}")
            return {}

    def get_entities_with_context_batch(
        self,
        entity_names: List[str],
        max_relationships_per_entity: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        GraphRAG Local Search: Batch fetch entity contexts.

        Fixes N+1 query issue by fetching all entity contexts in a single query.

        Args:
            entity_names: List of entity names to look up
            max_relationships_per_entity: Max relationships per entity

        Returns:
            Dict mapping entity name to context dict
        """
        if not self._connected:
            self.connect()

        if not entity_names:
            return {}

        try:
            with self.driver.session() as session:
                # Batch query for all entities at once
                query = """
                UNWIND $names as entity_name
                MATCH (e:Entity {name: entity_name})
                OPTIONAL MATCH (e)-[r]->(target:Entity)
                OPTIONAL MATCH (e)<-[r2]-(source:Entity)
                OPTIONAL MATCH (e)-[:IN_COMMUNITY]->(c:Community)
                WITH e,
                     collect(DISTINCT {
                         type: type(r),
                         target: target.name,
                         target_type: target.type,
                         description: r.description
                     })[0..$max_rels] as outgoing,
                     collect(DISTINCT {
                         type: type(r2),
                         source: source.name,
                         source_type: source.type,
                         description: r2.description
                     })[0..$max_rels] as incoming,
                     collect(DISTINCT {
                         id: c.id,
                         title: c.title,
                         summary: c.summary,
                         level: c.level
                     })[0..3] as communities
                RETURN e.name as name, e.type as type, e.description as description,
                       e.confidence as confidence, outgoing, incoming, communities
                """

                result = session.run(query, {
                    "names": entity_names,
                    "max_rels": max_relationships_per_entity
                })

                contexts = {}
                for record in result:
                    name = record["name"]
                    if name:
                        # Filter out null relationships
                        outgoing = [r for r in record["outgoing"] if r.get("target")]
                        incoming = [r for r in record["incoming"] if r.get("source")]
                        communities = [c for c in record["communities"] if c.get("id")]

                        contexts[name] = {
                            "name": name,
                            "type": record["type"],
                            "description": record["description"] or "",
                            "confidence": record["confidence"] or 0.5,
                            "outgoing_relationships": outgoing,
                            "incoming_relationships": incoming,
                            "communities": communities
                        }

                return contexts

        except Exception as e:
            logger.error(f"Error batch fetching entity contexts: {e}")
            return {}

    def get_relationships_between(
        self,
        entity_names: List[str],
        limit: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Get relationships strictly between a set of entities using simple type matching.
        Optimized for SLM context window (limited to most relevant).

        Args:
            entity_names: List of entity names to check
            limit: Maximum relationships to return

        Returns:
            List of relationship dicts (formatted for context)
        """
        if not self._connected:
            self.connect()

        if not entity_names:
            return []

        # Simple query matching relationships between any two entities in the list
        query = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.name IN $names AND b.name IN $names
        RETURN
            a.name as source,
            type(r) as type,
            b.name as target,
            r.description as description,
            r.confidence as confidence
        ORDER BY r.confidence DESC
        LIMIT $limit
        """

        try:
            results = self.execute_query(query, {"names": entity_names, "limit": limit})
            return results
        except Exception as e:
            logger.error(f"Error getting relationships between entities: {e}")
            return []

    def get_entity_communities(
        self,
        entity_names: List[str],
        level: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get community summaries for a list of entities.
        Used to provide broader context ("This entity belongs to Community X which is about...").

        Args:
            entity_names: Entities to look up
            level: Community level (0 = lowest/most specific)

        Returns:
            List of community summaries
        """
        if not self._connected:
            self.connect()

        # Assuming community structure: (Entity)-[:IN_COMMUNITY]->(Community)
        # Note: Adjust logic if your community structure differs (e.g. Leiden algo properties)
        query = """
        MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community)
        WHERE e.name IN $names AND c.level = $level
        RETURN DISTINCT
            c.id as community_id,
            c.summary as summary,
            c.title as title,
            count(e) as entity_count_in_context
        ORDER BY entity_count_in_context DESC
        LIMIT 3
        """

        try:
            results = self.execute_query(query, {"names": entity_names, "level": level})
            return results
        except Exception as e:
            logger.error(f"Error getting entity communities: {e}")
            return []
