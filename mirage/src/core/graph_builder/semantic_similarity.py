"""
Semantic Similarity Relationship Extractor
Computes cosine similarity between entity embeddings and creates
SIMILAR_TO relationships for semantically similar entities
"""

from typing import List, Dict, Any
import numpy as np
from loguru import logger


class SemanticSimilarityExtractor:
    """
    Extract semantic similarity relationships between entities
    based on cosine similarity of their embeddings

    Creates SIMILAR_TO relationships for entities with high
    embedding similarity (≥ threshold)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        top_k: int = 5
    ):
        """
        Initialize semantic similarity extractor

        Args:
            similarity_threshold: Minimum cosine similarity (0-1) to create relationship
            top_k: Maximum number of similar entities per entity
        """
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        logger.info(
            f"Initialized SemanticSimilarityExtractor: "
            f"threshold={similarity_threshold}, top_k={top_k}"
        )

    def extract_similarities(
        self,
        entities_with_embeddings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract semantic similarity relationships from entities with embeddings

        Args:
            entities_with_embeddings: List of entities with "name" and "embedding"

        Returns:
            List of similarity relationship dicts with:
            - source: Entity name
            - target: Entity name
            - type: "SIMILAR_TO"
            - source_type: "semantic"
            - score: Cosine similarity (0-1)
            - confidence: Same as score
        """
        # Filter entities that have embeddings
        valid_entities = [
            e for e in entities_with_embeddings
            if e.get("embedding") is not None and len(e.get("embedding", [])) > 0
        ]

        if len(valid_entities) < 2:
            logger.warning(
                f"Not enough entities with embeddings ({len(valid_entities)}) "
                "for similarity computation"
            )
            return []

        logger.info(
            f"Computing semantic similarities for {len(valid_entities)} entities "
            f"with embeddings"
        )

        # Prepare entity names and embeddings matrix
        entity_names = [e["name"] for e in valid_entities]
        embeddings = np.array([e["embedding"] for e in valid_entities])

        # Compute pairwise cosine similarity
        similarity_matrix = self._compute_cosine_similarity(embeddings)

        logger.info(f"Computed {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} similarity matrix")

        # Extract relationships from similarity matrix
        relationships = []
        for i, entity_i in enumerate(entity_names):
            # Get similarities for this entity
            similarities = similarity_matrix[i]

            # Get top-k most similar entities (excluding self)
            # argsort returns indices in ascending order, so we reverse it
            similar_indices = np.argsort(similarities)[::-1]

            # Skip first (self) and take top-k
            count = 0
            for j in similar_indices[1:]:  # Skip self (first element)
                score = similarities[j]

                # Check if meets threshold
                if score < self.similarity_threshold:
                    break  # Since sorted, no more will meet threshold

                # Create relationship
                relationships.append({
                    "source": entity_i,
                    "target": entity_names[j],
                    "type": "SIMILAR_TO",
                    "source_type": "semantic",
                    "score": float(score),  # Convert numpy float to Python float
                    "confidence": float(score)  # Use similarity as confidence
                })

                count += 1
                if count >= self.top_k:
                    break  # Reached top-k limit

        logger.info(
            f"Extracted {len(relationships)} semantic similarity relationships "
            f"(threshold ≥ {self.similarity_threshold})"
        )

        # Log statistics
        if relationships:
            scores = [r["score"] for r in relationships]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            logger.info(
                f"Similarity stats: "
                f"avg_score={avg_score:.3f}, "
                f"min_score={min_score:.3f}, "
                f"max_score={max_score:.3f}"
            )

        return relationships

    def _compute_cosine_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix

        Args:
            embeddings: Array of shape (n_entities, embedding_dim)

        Returns:
            Similarity matrix of shape (n_entities, n_entities)
            where similarity[i][j] = cosine_similarity(embedding[i], embedding[j])
        """
        # Normalize embeddings to unit vectors
        # This makes dot product equivalent to cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)

        normalized_embeddings = embeddings / norms

        # Compute similarity matrix via matrix multiplication
        # (n, d) @ (d, n) = (n, n)
        similarity_matrix = normalized_embeddings @ normalized_embeddings.T

        # Clip to [0, 1] range (numerical errors can cause slightly > 1 or < 0)
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

        return similarity_matrix

    def extract_with_threshold(
        self,
        entities_with_embeddings: List[Dict[str, Any]],
        similarity_threshold: float = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Extract similarities with custom parameters

        Args:
            entities_with_embeddings: List of entities with embeddings
            similarity_threshold: Override default threshold
            top_k: Override default top-k

        Returns:
            List of similarity relationships
        """
        original_threshold = self.similarity_threshold
        original_top_k = self.top_k

        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        if top_k is not None:
            self.top_k = top_k

        try:
            return self.extract_similarities(entities_with_embeddings)
        finally:
            self.similarity_threshold = original_threshold
            self.top_k = original_top_k
