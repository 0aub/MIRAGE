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
