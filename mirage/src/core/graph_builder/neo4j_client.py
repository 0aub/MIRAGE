"""
Neo4j Client
Dynamic graph storage without predefined schemas
Nodes and relationships emerge from the data
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from loguru import logger

from ...config import settings


class Neo4jClient:
    """Client for interacting with Neo4j graph database"""

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None
    ):
        """
        Initialize Neo4j client

        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
        """
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password

        self.driver: Optional[Driver] = None
        self._connected = False

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
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self._connected = False
            logger.info("Closed Neo4j connection")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def create_entity_node(
        self,
        entity: Dict[str, Any],
        document_id: str
    ) -> Dict[str, Any]:
        """
        Create or update an entity node

        Args:
            entity: Entity dict with text, type, etc.
            document_id: Source document ID

        Returns:
            Created/updated node
        """
        if not self._connected:
            self.connect()

        # Sanitize label for Cypher (remove spaces, special chars)
        entity_type = entity["type"].replace(" ", "_").replace("-", "_")

        # Dynamic query with type as additional label
        query = f"""
        MERGE (n:Entity {{name: $name, type: $type}})
        ON CREATE SET
            n.created_at = timestamp(),
            n.confidence = $confidence,
            n.source_documents = [$document_id],
            n.description = $description,
            n.attributes = $attributes,
            n:{entity_type}
        ON MATCH SET
            n.confidence = CASE
                WHEN $confidence > n.confidence THEN $confidence
                ELSE n.confidence
            END,
            n.description = CASE
                WHEN $description <> '' AND (n.description IS NULL OR n.description = '') THEN $description
                ELSE n.description
            END,
            n.attributes = CASE
                WHEN n.attributes IS NULL AND $attributes <> '' THEN $attributes
                ELSE n.attributes
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

        # Serialize attributes to JSON string for Neo4j storage
        import json
        attributes_json = json.dumps(entity.get("attributes", {})) if entity.get("attributes") else ""

        parameters = {
            "name": entity["text"],
            "type": entity["type"],
            "confidence": entity.get("confidence", 0.5),
            "description": entity.get("description", ""),
            "attributes": attributes_json,
            "document_id": document_id,
        }

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                record = result.single()

                if record:
                    return dict(record["n"])

        except Exception as e:
            logger.error(f"Error creating entity node: {e}")
            raise

        return {}

    def create_relationship(
        self,
        relationship: Dict[str, Any],
        document_id: str
    ) -> Dict[str, Any]:
        """
        Create a relationship between two entities

        For LLM-extracted relationships, stores as RELATED_TO with the semantic type as a property.
        For co-occurrence and semantic relationships, uses specific relationship types.

        Args:
            relationship: Relationship dict
            document_id: Source document ID

        Returns:
            Created relationship
        """
        if not self._connected:
            self.connect()

        # Determine relationship storage strategy based on source
        rel_source = relationship.get("source_type", "llm")  # Default to "llm" for backward compatibility

        # For LLM relationships: Store as RELATED_TO with semantic type as property
        # For co-occurrence/semantic: Use specific relationship types (COOCCURS_WITH, SIMILAR_TO)
        if rel_source == "llm":
            cypher_rel_type = "RELATED_TO"
            semantic_type = relationship.get("type", "RELATED_TO").lower().replace(" ", "_").replace("-", "_")
        elif rel_source == "cooccurrence":
            cypher_rel_type = "COOCCURS_WITH"
            semantic_type = None
        elif rel_source == "semantic":
            cypher_rel_type = "SIMILAR_TO"
            semantic_type = None
        else:
            # Fallback for unknown sources
            cypher_rel_type = relationship.get("type", "RELATED_TO").upper().replace(" ", "_").replace("-", "_")
            semantic_type = None

        query = f"""
        MATCH (source:Entity {{name: $source_name}})
        MATCH (target:Entity {{name: $target_name}})
        MERGE (source)-[r:{cypher_rel_type}]->(target)
        ON CREATE SET
            r.created_at = timestamp(),
            r.confidence = $confidence,
            r.method = $method,
            r.source = $source_type,
            r.source_documents = [$document_id],
            r.description = $description,
            r.attributes = $attributes,
            r.relationship_type = $semantic_type,
            r.frequency = $frequency,
            r.score = $score,
            r.chunks = $chunks
        ON MATCH SET
            r.confidence = CASE
                WHEN $confidence > r.confidence THEN $confidence
                ELSE r.confidence
            END,
            r.description = CASE
                WHEN $description <> '' AND (r.description IS NULL OR r.description = '') THEN $description
                ELSE r.description
            END,
            r.attributes = CASE
                WHEN r.attributes IS NULL AND $attributes <> '' THEN $attributes
                ELSE r.attributes
            END,
            r.frequency = CASE
                WHEN $frequency IS NOT NULL THEN r.frequency + $frequency
                ELSE r.frequency
            END,
            r.chunks = CASE
                WHEN $chunks IS NOT NULL THEN r.chunks + $chunks
                ELSE r.chunks
            END,
            r.source_documents = CASE
                WHEN NOT $document_id IN r.source_documents
                THEN r.source_documents + [$document_id]
                ELSE r.source_documents
            END,
            r.updated_at = timestamp()
        RETURN r
        """

        # Serialize attributes to JSON string for Neo4j storage
        import json
        attributes_json = json.dumps(relationship.get("attributes", {})) if relationship.get("attributes") else ""

        parameters = {
            "source_name": relationship["source"],
            "target_name": relationship["target"],
            "confidence": relationship.get("confidence", 0.5),
            "method": relationship.get("method", "llm"),
            "source_type": rel_source,
            "semantic_type": semantic_type,
            "description": relationship.get("description", ""),
            "attributes": attributes_json,
            "frequency": relationship.get("frequency"),  # For co-occurrence
            "score": relationship.get("score"),  # For semantic similarity
            "chunks": relationship.get("chunks"),  # For co-occurrence (list of chunk IDs)
            "document_id": document_id,
        }

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                record = result.single()

                if record:
                    return dict(record["r"])

        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            # Non-fatal, continue processing
            logger.warning(f"Skipping relationship: {relationship['source']} -> {relationship['target']}")

        return {}

    def store_document_metadata(
        self,
        document_id: str,
        title: str,
        content_type: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Store document metadata in Neo4j

        Args:
            document_id: Unique document ID
            title: Document title
            content_type: Type (file, webpage, youtube)
            metadata: Additional metadata

        Returns:
            Created document node
        """
        if not self._connected:
            self.connect()

        metadata = metadata or {}

        query = """
        MERGE (d:Document {document_id: $document_id})
        ON CREATE SET
            d.title = $title,
            d.content_type = $content_type,
            d.created_at = timestamp(),
            d.url = $url,
            d.author = $author,
            d.language = $language,
            d.total_chars = $total_chars,
            d.total_words = $total_words,
            d.video_id = $video_id,
            d.transcript_length = $transcript_length,
            d.full_text = $full_text,
            d.processed_text = $processed_text,
            d.processing_time_seconds = $processing_time_seconds
        ON MATCH SET
            d.title = $title,
            d.url = $url,
            d.author = $author,
            d.language = $language,
            d.total_chars = $total_chars,
            d.total_words = $total_words,
            d.video_id = $video_id,
            d.transcript_length = $transcript_length,
            d.full_text = $full_text,
            d.processed_text = $processed_text,
            d.processing_time_seconds = $processing_time_seconds,
            d.updated_at = timestamp()
        RETURN d
        """

        parameters = {
            "document_id": document_id,
            "title": title,
            "content_type": content_type,
            "url": metadata.get("url"),
            "author": metadata.get("author"),
            "language": metadata.get("language"),
            "total_chars": metadata.get("total_chars", 0),
            "total_words": metadata.get("total_words", 0),
            "video_id": metadata.get("video_id"),
            "transcript_length": metadata.get("transcript_length", 0),
            "full_text": metadata.get("full_text"),
            "processed_text": metadata.get("processed_text"),
            "processing_time_seconds": metadata.get("processing_time_seconds"),
        }

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                record = result.single()

                if record:
                    logger.info(f"Stored document metadata for {document_id}")
                    return dict(record["d"])

        except Exception as e:
            logger.error(f"Error storing document metadata: {e}")
            raise

        return {}

    def store_graph(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        document_id: str,
        document_title: str = None,
        document_type: str = "file",
        document_metadata: Dict[str, Any] = None,
        enhanced_neo4j_client=None
    ) -> Dict[str, Any]:
        """
        Store entities and relationships in Neo4j

        Args:
            entities: List of entities
            relationships: List of relationships
            document_id: Document ID
            document_title: Document title (optional)
            document_type: Document type (file, webpage, youtube)
            document_metadata: Additional document metadata
            enhanced_neo4j_client: Optional EnhancedNeo4jClient for enriched entity creation

        Returns:
            Storage statistics
        """
        if not self._connected:
            self.connect()

        logger.info(f"Storing graph for document {document_id}: {len(entities)} entities, {len(relationships)} relationships")

        # Store document metadata if provided
        if document_title:
            try:
                self.store_document_metadata(
                    document_id=document_id,
                    title=document_title,
                    content_type=document_type,
                    metadata=document_metadata or {}
                )
            except Exception as e:
                logger.warning(f"Failed to store document metadata: {e}")

        nodes_created = 0
        edges_created = 0

        # Detect language from metadata if available
        language = "ar"  # Default to Arabic
        if document_metadata and "full_text" in document_metadata:
            # Simple language detection based on character count
            import re
            full_text = document_metadata["full_text"]
            arabic_chars = len(re.findall(r'[\u0600-\u06FF]', full_text))
            latin_chars = len(re.findall(r'[a-zA-Z]', full_text))
            language = "ar" if arabic_chars > latin_chars else "en"

        # Create entity nodes (with enrichment if enhanced client is available)
        for entity in entities:
            try:
                if enhanced_neo4j_client:
                    # Use enriched entity creation (with embeddings, translations, descriptions)
                    enhanced_neo4j_client.create_enriched_entity_node(
                        entity=entity,
                        document_id=document_id,
                        language=language,
                        enrich=True  # Enable LLM enrichment
                    )
                    nodes_created += 1
                else:
                    # Fall back to standard entity creation
                    self.create_entity_node(entity, document_id)
                    nodes_created += 1
            except Exception as e:
                logger.error(f"Failed to create node for {entity['text']}: {e}")

        # Create relationships
        for relationship in relationships:
            try:
                self.create_relationship(relationship, document_id)
                edges_created += 1
            except Exception as e:
                logger.error(f"Failed to create relationship: {e}")

        logger.info(f"Stored {nodes_created} nodes and {edges_created} edges")

        return {
            "nodes_created": nodes_created,
            "edges_created": edges_created,
            "document_id": document_id,
        }

    def store_chunks_with_entities(
        self,
        chunks: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        document_id: str,
        document_title: str = None,
        document_type: str = None,
        document_metadata: Dict[str, Any] = None,
        enhanced_neo4j_client: Any = None
    ) -> Dict[str, Any]:
        """
        Store chunks as nodes and link them to entities (Hybrid Vector-Graph Architecture)

        This creates the topology:
        (:Chunk)-[:MENTIONS]->(:Entity)-[:RELATED_TO]->(:Entity)

        Chunks are also stored in Qdrant with the same IDs for vector search.

        Args:
            chunks: List of chunk dicts with {id, text, index, char_count, word_count}
            entities: List of entity dicts with {name, type, chunks: [chunk_ids], confidence}
            relationships: List of relationship dicts
            document_id: Source document ID
            document_title: Document title
            document_type: Document type (webpage, youtube, file)
            document_metadata: Additional metadata
            enhanced_neo4j_client: Optional enhanced client for enrichment

        Returns:
            Statistics about created nodes and relationships
        """
        if not self._connected:
            self.connect()

        logger.info(f"Storing {len(chunks)} chunks and {len(entities)} entities for document {document_id}")

        # Single transaction for atomicity
        query = """
        // 1. Create chunk nodes
        UNWIND $chunks AS chunk
        MERGE (c:Chunk {id: chunk.id})
        SET c.text = chunk.text,
            c.document_id = $document_id,
            c.chunk_index = chunk.index,
            c.char_count = chunk.char_count,
            c.word_count = chunk.word_count,
            c.created_at = timestamp()

        WITH count(c) as chunks_created

        // 2. Create entity nodes
        UNWIND $entities AS ent
        MERGE (e:Entity {name: ent.name})
        ON CREATE SET
            e.type = ent.type,
            e.created_at = timestamp(),
            e.source_documents = [$document_id],
            e.confidence = ent.confidence,
            e.embedding = CASE
                WHEN ent.embedding IS NOT NULL THEN ent.embedding
                ELSE NULL
            END
        ON MATCH SET
            e.source_documents = CASE
                WHEN NOT $document_id IN e.source_documents
                THEN e.source_documents + [$document_id]
                ELSE e.source_documents
            END,
            e.embedding = CASE
                WHEN ent.embedding IS NOT NULL THEN ent.embedding
                ELSE e.embedding
            END,
            e.updated_at = timestamp()

        WITH chunks_created, count(DISTINCT e) as entities_created

        // 3. Create MENTIONS relationships (Chunk -> Entity)
        UNWIND $entities AS ent
        MATCH (e:Entity {name: ent.name})
        UNWIND ent.chunks AS chunk_id
        MATCH (c:Chunk {id: chunk_id})
        MERGE (c)-[m:MENTIONS]->(e)
        ON CREATE SET
            m.confidence = ent.confidence,
            m.created_at = timestamp()

        WITH chunks_created, entities_created, count(m) as mentions_created

        RETURN chunks_created, entities_created, mentions_created
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, {
                    "chunks": chunks,
                    "entities": entities,
                    "document_id": document_id
                })
                record = result.single()

                stats = {
                    "chunks_created": record["chunks_created"] if record else 0,
                    "entities_created": record["entities_created"] if record else 0,
                    "mentions_created": record["mentions_created"] if record else 0,
                    "document_id": document_id
                }

                # Store entity-to-entity relationships (RELATED_TO, COOCCURS_WITH, SIMILAR_TO)
                edges_created = 0
                if relationships:
                    logger.info(f"Storing {len(relationships)} entity-to-entity relationships...")
                    for relationship in relationships:
                        try:
                            self.create_relationship(relationship, document_id)
                            edges_created += 1
                        except Exception as e:
                            logger.error(f"Failed to create relationship {relationship.get('source')} -> {relationship.get('target')}: {e}")

                    logger.info(f"Successfully stored {edges_created}/{len(relationships)} relationships")

                # Add relationships count to stats
                stats["relationships_created"] = edges_created

                logger.info(f"Stored chunks: {stats}")
                return stats

        except Exception as e:
            logger.error(f"Error storing chunks and entities: {e}")
            raise

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get overall graph statistics

        Returns:
            Statistics dict
        """
        if not self._connected:
            self.connect()

        stats = {}

        try:
            with self.driver.session() as session:
                # Total nodes
                result = session.run("MATCH (n:Entity) RETURN count(n) as count")
                stats["total_nodes"] = result.single()["count"]

                # Total relationships
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                stats["total_edges"] = result.single()["count"]

                # Nodes by type
                result = session.run("""
                    MATCH (n:Entity)
                    RETURN n.type as type, count(n) as count
                    ORDER BY count DESC
                """)
                stats["node_types"] = {record["type"]: record["count"] for record in result}

                # Relationship types
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as type, count(r) as count
                    ORDER BY count DESC
                """)
                stats["edge_types"] = {record["type"]: record["count"] for record in result}

        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")

        return stats

    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a raw Cypher query and return results

        Args:
            query: Cypher query string
            parameters: Optional query parameters

        Returns:
            List of result records as dictionaries
        """
        if not self._connected:
            self.connect()

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
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

    def clear_graph(self) -> Dict[str, Any]:
        """
        Clear all nodes and relationships from the graph
        WARNING: This is a destructive operation

        Returns:
            Statistics about cleared data
        """
        if not self._connected:
            self.connect()

        try:
            with self.driver.session() as session:
                # Get counts before deletion
                node_count_result = session.run("MATCH (n) RETURN count(n) as count")
                nodes_before = node_count_result.single()["count"]

                rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rels_before = rel_count_result.single()["count"]

                # Delete all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")

                logger.info(f"Cleared graph: {nodes_before} nodes, {rels_before} relationships")

                return {
                    "nodes_deleted": nodes_before,
                    "relationships_deleted": rels_before,
                    "status": "success",
                }

        except Exception as e:
            logger.error(f"Error clearing graph: {e}")
            raise

    def delete_by_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete all entities and relationships for a specific document

        Args:
            document_id: Document ID to remove

        Returns:
            Deletion statistics
        """
        if not self._connected:
            self.connect()

        try:
            with self.driver.session() as session:
                # Delete relationships first
                rel_query = """
                MATCH ()-[r]->()
                WHERE $document_id IN r.source_documents
                DELETE r
                RETURN count(r) as count
                """
                rel_result = session.run(rel_query, {"document_id": document_id})
                rels_deleted = rel_result.single()["count"] if rel_result.peek() else 0

                # Delete nodes that only belong to this document
                node_query = """
                MATCH (n:Entity)
                WHERE $document_id IN n.source_documents
                AND size(n.source_documents) = 1
                DELETE n
                RETURN count(n) as count
                """
                node_result = session.run(node_query, {"document_id": document_id})
                nodes_deleted = node_result.single()["count"] if node_result.peek() else 0

                # Update nodes that belong to multiple documents
                update_query = """
                MATCH (n:Entity)
                WHERE $document_id IN n.source_documents
                AND size(n.source_documents) > 1
                SET n.source_documents = [doc IN n.source_documents WHERE doc <> $document_id]
                RETURN count(n) as count
                """
                update_result = session.run(update_query, {"document_id": document_id})
                nodes_updated = update_result.single()["count"] if update_result.peek() else 0

                # Delete the Document node itself (DETACH DELETE removes all relationships first)
                doc_query = """
                MATCH (d:Document {document_id: $document_id})
                DETACH DELETE d
                RETURN count(d) as count
                """
                doc_result = session.run(doc_query, {"document_id": document_id})
                docs_deleted = doc_result.single()["count"] if doc_result.peek() else 0

                logger.info(
                    f"Deleted document {document_id} from graph: "
                    f"{nodes_deleted} nodes deleted, {nodes_updated} nodes updated, "
                    f"{rels_deleted} relationships deleted, {docs_deleted} document nodes deleted"
                )

                return {
                    "nodes_deleted": nodes_deleted,
                    "nodes_updated": nodes_updated,
                    "relationships_deleted": rels_deleted,
                    "documents_deleted": docs_deleted,
                    "document_id": document_id,
                    "status": "success",
                }

        except Exception as e:
            logger.error(f"Error deleting document from graph: {e}")
            raise

    def get_document_ids(self) -> List[str]:
        """
        Get all unique document IDs in the graph

        Returns:
            List of document IDs
        """
        if not self._connected:
            self.connect()

        try:
            with self.driver.session() as session:
                query = """
                MATCH (n:Entity)
                UNWIND n.source_documents as doc_id
                RETURN DISTINCT doc_id
                ORDER BY doc_id
                """
                result = session.run(query)
                return [record["doc_id"] for record in result]

        except Exception as e:
            logger.error(f"Error getting document IDs: {e}")
            return []

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents with their metadata and stats

        Returns:
            List of document dictionaries
        """
        if not self._connected:
            self.connect()

        try:
            with self.driver.session() as session:
                # OPTIMIZED: Only match relationships FROM entities belonging to this document
                # OLD SLOW QUERY: MATCH ()-[r]->() scanned ALL relationships in graph!
                query = """
                MATCH (d:Document)
                OPTIONAL MATCH (e:Entity)
                WHERE d.document_id IN e.source_documents
                WITH d, count(DISTINCT e) as entity_count
                OPTIONAL MATCH (e2:Entity)-[r]->(e3:Entity)
                WHERE d.document_id IN e2.source_documents
                  AND d.document_id IN r.source_documents
                WITH d, entity_count, count(DISTINCT r) as rel_count
                RETURN d, entity_count, rel_count
                ORDER BY d.created_at DESC
                """
                result = session.run(query)

                documents = []
                for record in result:
                    doc = dict(record["d"])
                    doc["entity_count"] = record["entity_count"]
                    doc["relationship_count"] = record["rel_count"]
                    documents.append(doc)

                return documents

        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []

    def get_document_stats(self, document_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific document

        Args:
            document_id: Document ID

        Returns:
            Document statistics (entities, relationships, etc.)
        """
        if not self._connected:
            self.connect()

        try:
            with self.driver.session() as session:
                # Count entities
                entity_query = """
                MATCH (n:Entity)
                WHERE $document_id IN n.source_documents
                RETURN count(n) as entity_count
                """
                entity_result = session.run(entity_query, {"document_id": document_id})
                entity_count = entity_result.single()["entity_count"]

                # Count relationships
                rel_query = """
                MATCH ()-[r]->()
                WHERE $document_id IN r.source_documents
                RETURN count(r) as rel_count
                """
                rel_result = session.run(rel_query, {"document_id": document_id})
                rel_count = rel_result.single()["rel_count"]

                # Get creation time (oldest entity timestamp)
                time_query = """
                MATCH (n:Entity)
                WHERE $document_id IN n.source_documents
                RETURN min(n.created_at) as created_at
                ORDER BY created_at
                LIMIT 1
                """
                time_result = session.run(time_query, {"document_id": document_id})
                time_record = time_result.single()
                created_at = time_record["created_at"] if time_record else None

                return {
                    "document_id": document_id,
                    "entity_count": entity_count,
                    "relationship_count": rel_count,
                    "created_at": created_at,
                }

        except Exception as e:
            logger.error(f"Error getting document stats for {document_id}: {e}")
            return {
                "document_id": document_id,
                "entity_count": 0,
                "relationship_count": 0,
                "created_at": None,
            }

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
