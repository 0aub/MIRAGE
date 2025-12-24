"""
Neo4j Client - Entity and Relationship Operations

Entity node creation and relationship management.
"""

import json
from typing import Dict, Any
from loguru import logger

from ..relationship_normalizer import get_relationship_normalizer


class EntityOperationsMixin:
    """Mixin for entity and relationship operations"""

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

        MIRAGE V4: Uses dynamic relationship types from LLM extraction.
        - LLM relationships: Normalized to standard Cypher types (FOUNDED, LEADS, WORKS_AT, etc.)
        - Co-occurrence: COOCCURS_WITH
        - Semantic similarity: SIMILAR_TO

        Args:
            relationship: Relationship dict
            document_id: Source document ID

        Returns:
            Created relationship
        """
        if not self._connected:
            self.connect()

        # Get relationship normalizer
        normalizer = get_relationship_normalizer()

        # Determine relationship storage strategy based on source
        rel_source = relationship.get("source_type", "llm")  # Default to "llm" for backward compatibility

        # MIRAGE V4: Use dynamic relationship types for LLM-extracted relationships
        # This enables powerful graph queries like MATCH (p:Person)-[:FOUNDED]->(o:Organization)
        if rel_source == "llm":
            # Normalize and convert to Cypher type
            original_type = relationship.get("type", "RELATED_TO")
            normalized = normalizer.normalize(original_type)
            cypher_rel_type = normalizer.to_cypher_type(original_type)
            semantic_type = normalized.normalized_type
            # Apply confidence modifier from normalizer
            type_confidence = normalized.confidence
        elif rel_source == "cooccurrence":
            cypher_rel_type = "COOCCURS_WITH"
            semantic_type = None
            type_confidence = 0.7
        elif rel_source == "semantic":
            cypher_rel_type = "SIMILAR_TO"
            semantic_type = None
            type_confidence = 0.8
        else:
            # For unknown sources, normalize the type
            original_type = relationship.get("type", "RELATED_TO")
            cypher_rel_type = normalizer.to_cypher_type(original_type)
            semantic_type = normalizer.normalize(original_type).normalized_type
            type_confidence = 0.6

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
        attributes_json = json.dumps(relationship.get("attributes", {})) if relationship.get("attributes") else ""

        # Calculate final confidence: combine LLM confidence with type specificity
        base_confidence = relationship.get("confidence", 0.5)
        final_confidence = base_confidence * type_confidence

        parameters = {
            "source_name": relationship["source"],
            "target_name": relationship["target"],
            "confidence": final_confidence,
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
