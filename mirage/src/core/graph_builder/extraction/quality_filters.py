"""
LLM Entity Extraction - Quality Filters
Deduplication and quality filtering for entities and relationships
"""

from typing import List, Dict, Any, Tuple
from loguru import logger


def is_generic_entity(entity: Dict[str, Any]) -> bool:
    """
    Check if entity is too generic/vague to be useful.

    Args:
        entity: Entity dict with 'text' and 'type'

    Returns:
        True if entity should be filtered out
    """
    text = entity.get("text", "").strip()
    entity_type = entity.get("type", "")

    # Filter very short entities
    if len(text) < 2:
        return True

    # Filter overly long entity names (likely contaminated with description)
    if len(text) > 150:
        logger.warning(f"Filtering overly long entity name ({len(text)} chars): {text[:50]}...")
        return True

    # Filter generic date/time without context
    if entity_type in ["Date", "Time", "Money", "Percentage"] and len(text.split()) <= 2:
        return True

    # Filter single-word Miscellaneous entities
    if entity_type == "Miscellaneous" and len(text.split()) <= 1:
        return True

    return False


def is_weak_relationship(relationship: Dict[str, Any]) -> bool:
    """
    Check if relationship is too weak/generic to be useful.

    Args:
        relationship: Relationship dict

    Returns:
        True if relationship should be filtered out
    """
    # Filter self-referential relationships
    source = relationship.get("source", "").lower().strip()
    target = relationship.get("target", "").lower().strip()
    if source == target:
        logger.debug(f"Filtering self-referential relationship: {source} -> {target}")
        return True

    return False


def deduplicate_entities(entities: List[Dict[str, Any]], normalizer) -> List[Dict[str, Any]]:
    """
    Remove duplicate entities using entity normalizer.

    Merges descriptions from duplicates to create richer entity profiles.

    Args:
        entities: List of entities
        normalizer: EntityNormalizer instance

    Returns:
        Deduplicated list of entities
    """
    seen = {}
    original_count = len(entities)

    for entity in entities:
        entity_text = entity.get("text", "")
        entity_type = entity.get("type", "Unknown")

        normalized_name = normalizer.normalize_entity_name(entity_text, entity_type)
        key = (normalized_name.lower().strip(), entity_type)

        if key not in seen:
            entity["text"] = normalized_name
            entity["original_text"] = entity_text
            seen[key] = entity
        else:
            existing = seen[key]

            # Keep higher confidence
            if entity.get("confidence", 0) > existing.get("confidence", 0):
                existing["confidence"] = entity.get("confidence", 0)

            # Merge descriptions
            new_desc = entity.get("description", "")
            existing_desc = existing.get("description", "")
            if new_desc and new_desc not in existing_desc:
                if existing_desc and len(existing_desc) < 500:
                    existing["description"] = f"{existing_desc} {new_desc}"
                elif not existing_desc:
                    existing["description"] = new_desc

            # Merge attributes
            existing["attributes"] = {
                **entity.get("attributes", {}),
                **existing.get("attributes", {})
            }

            # Update importance if higher
            importance_order = {"high": 3, "medium": 2, "low": 1}
            new_importance = importance_order.get(entity.get("importance", "low"), 1)
            existing_importance = importance_order.get(existing.get("importance", "low"), 1)
            if new_importance > existing_importance:
                existing["importance"] = entity.get("importance")

    deduplicated = list(seen.values())
    removed_count = original_count - len(deduplicated)

    if removed_count > 0:
        logger.info(f"Entity normalizer removed {removed_count} duplicate entities")

    return deduplicated


def deduplicate_relationships(relationships: List[Dict[str, Any]], normalizer) -> List[Dict[str, Any]]:
    """
    Remove duplicate relationships and normalize entity names.

    Args:
        relationships: List of relationships
        normalizer: EntityNormalizer instance

    Returns:
        Deduplicated list of relationships
    """
    seen = {}
    unique = []

    for rel in relationships:
        if not all(key in rel for key in ["source", "target", "type"]):
            logger.warning(f"Skipping invalid relationship (missing fields): {rel}")
            continue

        source_normalized = normalizer.normalize_entity_name(rel["source"], "Person")
        target_normalized = normalizer.normalize_entity_name(rel["target"], "Person")

        rel["source"] = source_normalized
        rel["target"] = target_normalized

        key = (source_normalized.lower().strip(), target_normalized.lower().strip(), rel["type"].lower())

        if key not in seen:
            seen[key] = rel
            unique.append(rel)
        else:
            existing = seen[key]

            # Merge descriptions
            new_desc = rel.get("description", "")
            existing_desc = existing.get("description", "")
            if new_desc and new_desc not in existing_desc:
                if existing_desc and len(existing_desc) < 300:
                    existing["description"] = f"{existing_desc} {new_desc}"
                elif not existing_desc:
                    existing["description"] = new_desc

            # Keep higher weight
            if rel.get("weight", 0.5) > existing.get("weight", 0.5):
                existing["weight"] = rel.get("weight", 0.5)

            # Merge attributes
            existing["attributes"] = {
                **rel.get("attributes", {}),
                **existing.get("attributes", {})
            }

    return unique


def filter_quality(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Apply quality filters to entities and relationships.

    Args:
        entities: List of entities
        relationships: List of relationships

    Returns:
        Tuple of (filtered_entities, filtered_relationships)
    """
    # Filter entities
    quality_entities = [e for e in entities if not is_generic_entity(e)]

    # Create set of valid entity names
    valid_entity_names = {e["text"].lower().strip() for e in quality_entities}

    # Filter relationships
    quality_relationships = []
    for rel in relationships:
        if is_weak_relationship(rel):
            continue

        source = rel.get("source", "").lower().strip()
        target = rel.get("target", "").lower().strip()

        if source not in valid_entity_names or target not in valid_entity_names:
            logger.debug(f"Filtering relationship with missing entity: {source} -> {target}")
            continue

        quality_relationships.append(rel)

    filtered_entity_count = len(entities) - len(quality_entities)
    filtered_rel_count = len(relationships) - len(quality_relationships)

    if filtered_entity_count > 0 or filtered_rel_count > 0:
        logger.info(
            f"Quality filter: removed {filtered_entity_count} generic entities "
            f"and {filtered_rel_count} weak relationships"
        )

    return quality_entities, quality_relationships
