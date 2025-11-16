"""
Relationship Extractor
Discovers relationships between entities using:
- Co-occurrence analysis
- Distance-based relationships
- Statistical correlations
No predefined schemas - relationships emerge from data
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
from loguru import logger
import math


class RelationshipExtractor:
    """Extract relationships between entities"""

    def __init__(
        self,
        co_occurrence_window: int = 100,  # Characters
        min_confidence: float = 0.3,
    ):
        """
        Initialize relationship extractor

        Args:
            co_occurrence_window: Character distance for co-occurrence
            min_confidence: Minimum confidence threshold for relationships
        """
        self.co_occurrence_window = co_occurrence_window
        self.min_confidence = min_confidence

    def extract_relationships(
        self,
        entities: List[Dict[str, Any]],
        text: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities

        Args:
            entities: List of extracted entities
            text: Optional source text for context

        Returns:
            List of relationship dicts
        """
        if not entities or len(entities) < 2:
            return []

        logger.debug(f"Extracting relationships from {len(entities)} entities")

        relationships = []

        # Extract co-occurrence relationships
        co_occur_rels = self._extract_co_occurrence_relationships(entities)
        relationships.extend(co_occur_rels)

        # Extract type-based relationships
        type_rels = self._extract_type_based_relationships(entities)
        relationships.extend(type_rels)

        # Deduplicate and filter
        relationships = self._filter_relationships(relationships)

        logger.info(f"Extracted {len(relationships)} relationships")

        return relationships

    def _extract_co_occurrence_relationships(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships based on co-occurrence in text

        Entities that appear close together likely have a relationship
        """
        relationships = []

        # Sort entities by position if available
        entities_with_pos = [e for e in entities if "start" in e and "end" in e]
        entities_with_pos.sort(key=lambda e: e["start"])

        # Find co-occurring entities
        for i, entity1 in enumerate(entities_with_pos):
            for entity2 in entities_with_pos[i + 1:]:
                # Calculate distance
                distance = entity2["start"] - entity1["end"]

                # If within window, create relationship
                if distance >= 0 and distance <= self.co_occurrence_window:
                    # Calculate confidence based on distance (closer = higher confidence)
                    confidence = 1.0 - (distance / self.co_occurrence_window)
                    confidence = max(self.min_confidence, confidence)

                    # Determine relationship type based on entity types
                    rel_type = self._infer_relationship_type(
                        entity1["type"],
                        entity2["type"]
                    )

                    relationships.append({
                        "source": entity1["text"],
                        "source_type": entity1["type"],
                        "target": entity2["text"],
                        "target_type": entity2["type"],
                        "relationship": rel_type,
                        "confidence": confidence,
                        "method": "co-occurrence",
                        "distance": distance,
                    })

                # Break if too far
                if distance > self.co_occurrence_window:
                    break

        return relationships

    def _extract_type_based_relationships(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships based on entity type combinations

        Certain type combinations have likely relationships:
        - Person + Organization = WORKS_AT / MEMBER_OF
        - Person + Location = LOCATED_IN / FROM
        - Organization + Location = BASED_IN
        """
        relationships = []

        # Group entities by type
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity["type"]].append(entity)

        # Define type-based relationship patterns
        type_patterns = [
            ("Person", "Organization", "AFFILIATED_WITH"),
            ("Person", "Location", "ASSOCIATED_WITH"),
            ("Organization", "Location", "LOCATED_IN"),
            ("Person", "Person", "RELATED_TO"),
            ("Organization", "Organization", "RELATED_TO"),
            ("Person", "Date", "ASSOCIATED_WITH"),
            ("Event", "Location", "OCCURRED_IN"),
            ("Event", "Person", "INVOLVES"),
        ]

        # Create relationships based on type patterns
        for type1, type2, rel_type in type_patterns:
            entities_type1 = entities_by_type.get(type1, [])
            entities_type2 = entities_by_type.get(type2, [])

            # Connect entities of these types
            for entity1 in entities_type1:
                for entity2 in entities_type2:
                    # Only create relationship if they co-occur
                    if self._entities_co_occur(entity1, entity2):
                        relationships.append({
                            "source": entity1["text"],
                            "source_type": entity1["type"],
                            "target": entity2["text"],
                            "target_type": entity2["type"],
                            "relationship": rel_type,
                            "confidence": 0.5,  # Type-based confidence
                            "method": "type-based",
                        })

        return relationships

    def _entities_co_occur(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any]
    ) -> bool:
        """Check if two entities co-occur in the same context"""
        # Check if they're in the same chunk
        if "chunk_index" in entity1 and "chunk_index" in entity2:
            return entity1["chunk_index"] == entity2["chunk_index"]

        # Check if they're close in position
        if "start" in entity1 and "start" in entity2:
            distance = abs(entity1["start"] - entity2["start"])
            return distance <= self.co_occurrence_window * 2

        # Default: assume they co-occur if in same document
        return True

    def _infer_relationship_type(
        self,
        type1: str,
        type2: str
    ) -> str:
        """
        Infer relationship type from entity types

        Args:
            type1: First entity type
            type2: Second entity type

        Returns:
            Inferred relationship type
        """
        # Create a normalized key for lookup
        types = tuple(sorted([type1, type2]))

        # Relationship inference rules
        inference_rules = {
            ("Location", "Organization"): "LOCATED_IN",
            ("Location", "Person"): "LOCATED_IN",
            ("Organization", "Person"): "AFFILIATED_WITH",
            ("Date", "Event"): "OCCURRED_ON",
            ("Date", "Person"): "ASSOCIATED_WITH",
            ("Event", "Location"): "OCCURRED_IN",
            ("Event", "Person"): "INVOLVES",
            ("Organization", "Organization"): "RELATED_TO",
            ("Person", "Person"): "RELATED_TO",
            ("Product", "Organization"): "PRODUCED_BY",
        }

        # Look up relationship type
        rel_type = inference_rules.get(types, "RELATED_TO")

        return rel_type

    def _filter_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter and deduplicate relationships

        Args:
            relationships: List of relationships

        Returns:
            Filtered relationships
        """
        # Remove low-confidence relationships
        filtered = [
            rel for rel in relationships
            if rel.get("confidence", 0) >= self.min_confidence
        ]

        # Deduplicate by source-target-relationship tuple
        seen = set()
        unique_rels = []

        for rel in filtered:
            # Create key (bidirectional - order doesn't matter for undirected)
            entities = tuple(sorted([rel["source"], rel["target"]]))
            key = (*entities, rel["relationship"])

            if key not in seen:
                seen.add(key)
                unique_rels.append(rel)

        return unique_rels

    def build_relationship_graph(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build a graph representation from entities and relationships

        Args:
            entities: List of entities
            relationships: List of relationships

        Returns:
            Graph dict with nodes and edges
        """
        # Build nodes
        nodes = {}
        for entity in entities:
            node_id = f"{entity['type']}:{entity['text']}"
            nodes[node_id] = {
                "id": node_id,
                "label": entity["text"],
                "type": entity["type"],
                "properties": {
                    "confidence": entity.get("confidence", 0.5),
                    "occurrences": entity.get("occurrence_count", 1),
                },
            }

        # Build edges
        edges = []
        for i, rel in enumerate(relationships):
            source_id = f"{rel['source_type']}:{rel['source']}"
            target_id = f"{rel['target_type']}:{rel['target']}"

            # Only add edge if both nodes exist
            if source_id in nodes and target_id in nodes:
                edges.append({
                    "id": f"rel_{i}",
                    "source": source_id,
                    "target": target_id,
                    "relationship": rel["relationship"],
                    "properties": {
                        "confidence": rel.get("confidence", 0.5),
                        "method": rel.get("method", "unknown"),
                    },
                })

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "relationship_types": list(set(e["relationship"] for e in edges)),
            },
        }
