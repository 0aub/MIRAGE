"""
Entity Resolver - Deduplicates entities using embedding-based clustering
Handles cross-lingual entity variants (Arabic + English)
"""

import numpy as np
from typing import List, Dict, Any, Set, Tuple
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity


class EntityResolver:
    """Resolve duplicate entities using semantic embeddings"""

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize entity resolver

        Args:
            similarity_threshold: Cosine similarity threshold for merging entities (0.0-1.0)
                                 0.85 = very similar (recommended for entity deduplication)
                                 0.90 = extremely similar (may miss some variants)
                                 0.80 = somewhat similar (may create false positives)
        """
        self.similarity_threshold = similarity_threshold
        logger.info(f"Entity resolver initialized with similarity threshold: {similarity_threshold}")

    def resolve_entities(
        self,
        entities: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[Dict[str, Any]]:
        """
        Resolve duplicate entities by clustering similar embeddings

        Args:
            entities: List of entity dicts with: text, type, label, confidence
            embeddings: List of embedding vectors (same order as entities)

        Returns:
            List of resolved entities with merged variants

        Example:
            Input entities: [
                {"text": "السعودية", "type": "Location"},
                {"text": "Saudi Arabia", "type": "Location"},
                {"text": "KSA", "type": "Location"},
            ]
            Output: [
                {
                    "text": "Saudi Arabia",  # Canonical name (longest variant)
                    "variants": ["السعودية", "KSA"],
                    "type": "Location",
                    "confidence": 0.8,
                    "cluster_size": 3
                }
            ]
        """
        if not entities or len(embeddings) == 0:
            return entities

        if len(entities) != len(embeddings):
            logger.error(f"Mismatch: {len(entities)} entities but {len(embeddings)} embeddings")
            return entities

        logger.info(f"Resolving {len(entities)} entities using embedding similarity")

        # Convert embeddings to numpy array for fast cosine similarity
        embedding_matrix = np.array(embeddings)

        # Compute pairwise cosine similarities
        similarities = cosine_similarity(embedding_matrix)

        # Find clusters of similar entities
        clusters = self._cluster_entities(entities, similarities)

        # Merge entities within each cluster
        resolved_entities = []
        for cluster_indices in clusters:
            merged_entity = self._merge_cluster(entities, cluster_indices)
            resolved_entities.append(merged_entity)

        reduction_pct = (1 - len(resolved_entities) / len(entities)) * 100 if entities else 0
        logger.info(
            f"Entity resolution complete: {len(entities)} → {len(resolved_entities)} entities "
            f"({reduction_pct:.1f}% reduction)"
        )

        return resolved_entities

    def _cluster_entities(
        self,
        entities: List[Dict[str, Any]],
        similarities: np.ndarray
    ) -> List[List[int]]:
        """
        Cluster entities based on similarity matrix using greedy approach

        Args:
            entities: List of entities
            similarities: Pairwise similarity matrix (n x n)

        Returns:
            List of clusters, where each cluster is a list of entity indices
        """
        n = len(entities)
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue

            # Start new cluster with current entity
            cluster = [i]
            visited.add(i)

            # Find all entities similar to this one
            for j in range(i + 1, n):
                if j in visited:
                    continue

                # Check similarity with ANY entity in the current cluster (transitive)
                max_similarity = max(similarities[cluster_idx][j] for cluster_idx in cluster)

                if max_similarity >= self.similarity_threshold:
                    cluster.append(j)
                    visited.add(j)

            clusters.append(cluster)

        return clusters

    def _merge_cluster(
        self,
        entities: List[Dict[str, Any]],
        cluster_indices: List[int]
    ) -> Dict[str, Any]:
        """
        Merge multiple entities into a single resolved entity

        Args:
            entities: All entities
            cluster_indices: Indices of entities to merge

        Returns:
            Merged entity with canonical name and variants
        """
        cluster_entities = [entities[i] for i in cluster_indices]

        if len(cluster_entities) == 1:
            # No merging needed
            return cluster_entities[0]

        # Choose canonical name (longest text as it's usually most complete)
        canonical = max(cluster_entities, key=lambda e: len(e["text"]))

        # Collect all variant names (excluding canonical)
        variants = []
        for entity in cluster_entities:
            if entity["text"] != canonical["text"]:
                variants.append(entity["text"])

        # Merge entity types (use most common type)
        entity_types = [e.get("type") for e in cluster_entities if e.get("type")]
        canonical_type = max(set(entity_types), key=entity_types.count) if entity_types else canonical.get("type")

        # Average confidence scores
        confidences = [e.get("confidence", 0.8) for e in cluster_entities]
        avg_confidence = sum(confidences) / len(confidences)

        # Create merged entity
        merged = {
            "text": canonical["text"],
            "variants": variants,
            "type": canonical_type,
            "label": canonical.get("label", canonical_type),
            "confidence": round(avg_confidence, 2),
            "cluster_size": len(cluster_entities),
            "is_merged": len(cluster_entities) > 1
        }

        if len(cluster_entities) > 1:
            logger.debug(
                f"Merged {len(cluster_entities)} entities: "
                f"{canonical['text']} + variants: {variants}"
            )

        return merged

    def resolve_entities_in_graph(
        self,
        entities: List[Dict[str, Any]],
        embedder
    ) -> List[Dict[str, Any]]:
        """
        Convenience method: Resolve entities by computing embeddings on the fly

        Args:
            entities: List of entity dicts with 'text' field
            embedder: Embedder instance (e.g., JinaEmbedder)

        Returns:
            List of resolved entities with merged variants
        """
        if not entities:
            return []

        # Extract entity texts
        entity_texts = [e["text"] for e in entities]

        # Compute embeddings
        logger.info(f"Computing embeddings for {len(entity_texts)} entities")
        embeddings = embedder.embed(entity_texts)

        # Resolve using embeddings
        return self.resolve_entities(entities, embeddings)
