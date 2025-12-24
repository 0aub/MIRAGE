"""
Neo4j Client - Document Storage Operations

Document metadata, graph storage, and chunk management.
"""

import re
from typing import List, Dict, Any
from loguru import logger


class DocumentOperationsMixin:
    """Mixin for document storage operations"""

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
        # GraphRAG Enhancement: Store description for each entity
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

        // 2. Create entity nodes (GraphRAG: with description)
        UNWIND $entities AS ent
        MERGE (e:Entity {name: ent.name})
        ON CREATE SET
            e.type = ent.type,
            e.created_at = timestamp(),
            e.source_documents = [$document_id],
            e.confidence = ent.confidence,
            e.description = COALESCE(ent.description, ''),
            e.importance = COALESCE(ent.importance, 'medium'),
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
            e.description = CASE
                WHEN COALESCE(ent.description, '') <> '' AND COALESCE(e.description, '') = '' THEN ent.description
                WHEN COALESCE(ent.description, '') <> '' AND size(ent.description) > size(COALESCE(e.description, '')) THEN ent.description
                ELSE e.description
            END,
            e.importance = CASE
                WHEN ent.importance = 'high' THEN 'high'
                WHEN e.importance IS NULL THEN ent.importance
                ELSE e.importance
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
                # Step 1: Delete all chunks (DETACH DELETE removes MENTIONS relationships)
                chunk_delete_query = """
                MATCH (c:Chunk {document_id: $document_id})
                DETACH DELETE c
                """
                session.run(chunk_delete_query, {"document_id": document_id})

                # Step 2: Count and delete relationships between entities
                rel_count_query = """
                MATCH ()-[r]->()
                WHERE $document_id IN r.source_documents
                RETURN count(r) as count
                """
                rel_count_result = session.run(rel_count_query, {"document_id": document_id})
                rels_to_delete = rel_count_result.single()["count"] if rel_count_result.peek() else 0

                rel_delete_query = """
                MATCH ()-[r]->()
                WHERE $document_id IN r.source_documents
                DELETE r
                """
                session.run(rel_delete_query, {"document_id": document_id})

                # Step 3: Count and delete Entity nodes that only belong to this document (use DETACH DELETE)
                node_count_query = """
                MATCH (n:Entity)
                WHERE $document_id IN n.source_documents
                AND size(n.source_documents) = 1
                RETURN count(n) as count
                """
                node_count_result = session.run(node_count_query, {"document_id": document_id})
                nodes_to_delete = node_count_result.single()["count"] if node_count_result.peek() else 0

                node_delete_query = """
                MATCH (n:Entity)
                WHERE $document_id IN n.source_documents
                AND size(n.source_documents) = 1
                DETACH DELETE n
                """
                session.run(node_delete_query, {"document_id": document_id})

                # Step 4: Update Entity nodes that belong to multiple documents
                update_query = """
                MATCH (n:Entity)
                WHERE $document_id IN n.source_documents
                AND size(n.source_documents) > 1
                SET n.source_documents = [doc IN n.source_documents WHERE doc <> $document_id]
                RETURN count(n) as count
                """
                update_result = session.run(update_query, {"document_id": document_id})
                nodes_updated = update_result.single()["count"] if update_result.peek() else 0

                # Step 5: Delete the Document node itself (DETACH DELETE removes all remaining relationships)
                doc_query = """
                MATCH (d:Document {document_id: $document_id})
                DETACH DELETE d
                """
                session.run(doc_query, {"document_id": document_id})

                rels_deleted = rels_to_delete
                nodes_deleted = nodes_to_delete
                docs_deleted = 1  # We know we're deleting one document

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
                # Added chunk_count from HAS_CHUNK relationship (stored in Neo4j, not Qdrant)
                query = """
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                WITH d, count(c) as chunk_count
                OPTIONAL MATCH (e:Entity)
                WHERE d.document_id IN e.source_documents
                WITH d, chunk_count, count(DISTINCT e) as entity_count
                OPTIONAL MATCH (e2:Entity)-[r]->(e3:Entity)
                WHERE d.document_id IN e2.source_documents
                  AND d.document_id IN r.source_documents
                WITH d, chunk_count, entity_count, count(DISTINCT r) as rel_count
                RETURN d, chunk_count, entity_count, rel_count
                ORDER BY d.created_at DESC
                """
                result = session.run(query)

                documents = []
                for record in result:
                    doc = dict(record["d"])
                    doc["chunk_count"] = record["chunk_count"]
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
