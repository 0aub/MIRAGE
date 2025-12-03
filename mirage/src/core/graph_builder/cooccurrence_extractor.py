"""
Co-occurrence Relationship Extractor
Extracts statistical co-occurrence relationships between entities
based on their appearance in the same chunks
"""

from typing import List, Dict, Any
from collections import defaultdict
from loguru import logger


class CooccurrenceExtractor:
    """
    Extract co-occurrence relationships between entities

    Creates COOCCURS_WITH relationships for entities that appear
    together in multiple chunks (frequency ≥ 2)
    """

    def __init__(self, min_frequency: int = 2):
        """
        Initialize co-occurrence extractor

        Args:
            min_frequency: Minimum co-occurrence frequency to create relationship (default: 2)
        """
        self.min_frequency = min_frequency
        logger.info(f"Initialized CooccurrenceExtractor with min_frequency={min_frequency}")

    def extract_cooccurrences(
        self,
        entities_with_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract co-occurrence relationships from entities with chunk references

        Args:
            entities_with_chunks: List of entities with "name" and "chunks" (list of chunk IDs)

        Returns:
            List of co-occurrence relationship dicts with:
            - source: Entity name
            - target: Entity name
            - type: "COOCCURS_WITH"
            - source_type: "cooccurrence"
            - frequency: Number of chunks where both appear
            - chunks: List of chunk IDs where they cooccur
        """
        if not entities_with_chunks:
            logger.warning("No entities provided for co-occurrence extraction")
            return []

        logger.info(f"Extracting co-occurrences from {len(entities_with_chunks)} entities")

        # Build chunk-to-entities mapping
        chunk_to_entities = defaultdict(list)
        for entity in entities_with_chunks:
            entity_name = entity.get("name")
            entity_chunks = entity.get("chunks", [])

            if not entity_name or not entity_chunks:
                continue

            for chunk_id in entity_chunks:
                chunk_to_entities[chunk_id].append(entity_name)

        logger.info(f"Built entity mapping for {len(chunk_to_entities)} chunks")

        # Track co-occurrences: (entity1, entity2) -> [chunk_ids]
        cooccurrences = defaultdict(list)

        # For each chunk, create pairs of entities
        for chunk_id, entities_in_chunk in chunk_to_entities.items():
            # Create all pairs in this chunk
            for i, entity1 in enumerate(entities_in_chunk):
                for entity2 in entities_in_chunk[i+1:]:
                    # Always use sorted order to avoid duplicates (A,B) vs (B,A)
                    pair = tuple(sorted([entity1, entity2]))
                    cooccurrences[pair].append(chunk_id)

        logger.info(f"Found {len(cooccurrences)} unique entity pairs")

        # Create relationships for pairs with frequency ≥ min_frequency
        relationships = []
        for (entity1, entity2), chunk_ids in cooccurrences.items():
            frequency = len(chunk_ids)

            # Filter by minimum frequency
            if frequency < self.min_frequency:
                continue

            # Create bidirectional relationship (undirected)
            # Neo4j will store as directed, but we treat as undirected in queries
            relationships.append({
                "source": entity1,
                "target": entity2,
                "type": "COOCCURS_WITH",
                "source_type": "cooccurrence",
                "frequency": frequency,
                "chunks": chunk_ids,
                "confidence": min(1.0, frequency / 10.0)  # Normalized confidence (max at 10 co-occurrences)
            })

        logger.info(
            f"Extracted {len(relationships)} co-occurrence relationships "
            f"(frequency ≥ {self.min_frequency})"
        )

        # Log statistics
        if relationships:
            frequencies = [r["frequency"] for r in relationships]
            avg_freq = sum(frequencies) / len(frequencies)
            max_freq = max(frequencies)
            logger.info(
                f"Co-occurrence stats: "
                f"avg_frequency={avg_freq:.1f}, "
                f"max_frequency={max_freq}"
            )

        return relationships

    def extract_with_threshold(
        self,
        entities_with_chunks: List[Dict[str, Any]],
        min_frequency: int = None
    ) -> List[Dict[str, Any]]:
        """
        Extract co-occurrences with custom threshold

        Args:
            entities_with_chunks: List of entities with chunk references
            min_frequency: Override default minimum frequency

        Returns:
            List of co-occurrence relationships
        """
        original_min = self.min_frequency
        if min_frequency is not None:
            self.min_frequency = min_frequency

        try:
            return self.extract_cooccurrences(entities_with_chunks)
        finally:
            self.min_frequency = original_min
